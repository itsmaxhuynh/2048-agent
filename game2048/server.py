"""A local web UI for watching the trained agent.

    python -m game2048.server --weights weights/agent.npy
    # then open http://127.0.0.1:8000

Only the standard library, so there is nothing to install and nothing to go
wrong on stage. The server plays a whole game up front (it takes about a tenth
of a second) and hands the browser every frame at once. The browser then owns
playback -- play, pause, scrub, change speed -- which means the animation can
never stutter because of a slow request mid-demo.
"""

import argparse
import csv
import json
import os
import re
import time
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

from game2048 import fast, patterns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT, "web")


def record_game(weights, seed=None, max_moves=100_000, depth=1):
    """Play one greedy game, capturing what the agent saw at every step."""
    if seed is not None:
        np.random.seed(seed)
    board = fast.as_board(fast.new_game())
    score = 0
    stage = 0
    frames = []
    while len(frames) < max_moves:
        stage = int(fast.advance_stage(board, stage))
        evaluations = []
        for d in range(4):
            after, reward = fast.move(board, d)
            after = fast.as_board(after)
            if after == board:
                evaluations.append(None)
            else:
                value = (float(fast.evaluate(after, weights[min(stage, len(weights) - 1)])) if depth <= 1
                         else float(fast.value_after(after, weights[min(stage, len(weights) - 1)], depth)))
                evaluations.append([int(reward), round(value, 1)])

        best_d, after, reward, _ = fast.best_action_search(board, weights[min(stage, len(weights) - 1)], depth)
        frames.append({
            "cells": [int((int(board) >> (4 * i)) & 0xF) for i in range(16)],
            "score": score,
            "chosen": int(best_d),
            "evals": evaluations,
            "stage": stage,
            "depth": depth,
        })
        if best_d < 0:
            break
        score += int(reward)
        after = fast.as_board(after)
        stage = int(fast.advance_stage(after, stage))
        board = fast.as_board(fast.spawn(after))

    cells = [int((int(board) >> (4 * i)) & 0xF) for i in range(16)]
    return {
        "frames": frames,
        "final": {"cells": cells, "score": score, "moves": len(frames) - 1,
                  "best_tile": 1 << max(cells)},
    }


MILESTONES = (7, 8, 9, 10, 11, 12, 13, 14, 15)      # 128 ... 32768


def aggregate(weights, games, depth=1):
    """Play `games` complete games and report how strong the agent actually is.

    Reach rates matter more than the average here: an average is one number that
    a few lucky games can carry, while "it builds an 8192 in 63% of games" is the
    claim a room can check against its own experience of playing 2048.
    """
    scores = np.empty(games, dtype=np.int64)
    tiles = np.empty(games, dtype=np.int64)
    stages = np.zeros(fast.N_STAGES, dtype=np.int64)
    total_moves = 0
    t0 = time.time()
    for i in range(games):
        score, moves, tile, stage = fast.play_reported(weights, depth)
        scores[i] = int(score)
        tiles[i] = int(tile)
        total_moves += int(moves)
        stages[int(stage)] += 1
    elapsed = max(time.time() - t0, 1e-9)

    reach = [{"tile": 1 << k, "pct": round(100.0 * float((tiles >= k).mean()), 1)}
             for k in MILESTONES if (tiles >= k).any()]
    # a game that reached stage k passed through every stage below it
    reached = [int(stages[k:].sum()) for k in range(fast.N_STAGES)]
    return {
        "games": games,
        "depth": depth,
        "average": int(scores.mean()),
        "median": int(np.median(scores)),
        "best": int(scores.max()),
        "worst": int(scores.min()),
        "moves_per_game": int(total_moves / games),
        "games_per_sec": round(games / elapsed, 2),
        "seconds": round(elapsed, 1),
        "reach": reach,
        "stages": [{"stage": k + 1, "pct": round(100.0 * n / games, 1)}
                   for k, n in enumerate(reached)],
    }


def newest_mtime(path):
    """When any stage of this checkpoint last changed on disk."""
    newest = 0.0
    for k in range(fast.N_STAGES):
        p = fast.stage_path(path, k)
        if os.path.exists(p):
            newest = max(newest, os.path.getmtime(p))
    return newest


def curve_label(path):
    """'talk/cosine-a0.0025-curve-stage0.csv' -> 'cosine-a0.0025'.

    Comparison runs are named by their file, so adding one to --compare is the
    whole ceremony: no matching label argument to keep in step.
    """
    base = re.sub(r"-?curve-stage\d+\.csv$", "", os.path.basename(path))
    return base.strip("-") or "run"


def curve_paths(path):
    """Every stage's curve that sits alongside the one named, in stage order.

    --curve points at one file, but the four stages each write their own, so a
    path with a stage number in it stands for the whole set. Anything else is
    taken literally, which keeps --curve pointed at a one-off log working.
    """
    base = os.path.basename(path)
    if not re.search(r"stage\d+", base):
        return [(0, path)] if os.path.exists(path) else []
    found = []
    for k in range(fast.N_STAGES):
        p = os.path.join(os.path.dirname(path), re.sub(r"stage\d+", f"stage{k}", base))
        if os.path.exists(p):
            found.append((k, p))
    return found


def read_curve(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append({"games": int(row["games"]), "avg": float(row["avg_score"]),
                             "max": int(row["max_score"]),
                             "alpha": float(row.get("alpha") or 0.0)})
            except (KeyError, ValueError):
                continue
    # the CSV is appended across runs; make the x axis cumulative and monotonic
    total = 0
    last = 0
    for r in rows:
        if r["games"] <= last:
            total += last
        last = r["games"]
        r["games"] = total + r["games"]
    return rows


class Handler(BaseHTTPRequestHandler):
    curve_runs = ()
    weights = None
    depth = 1
    reload_each_game = False
    weights_mtime = 0.0
    curve_path = "talk/learning-curve.csv"
    weights_path = ""

    @classmethod
    def refresh_weights(cls):
        """Re-read the checkpoint if the trainer has written a newer one.

        Checkpoints are saved atomically (write to .tmp, then rename), so a load
        always sees a complete file -- which is what makes it safe to watch an
        agent get better while it is still training.
        """
        mtime = newest_mtime(cls.weights_path)
        if mtime <= cls.weights_mtime:
            return False
        try:
            cls.weights = fast.load_weights(cls.weights_path)
        except (FileNotFoundError, AssertionError, ValueError) as exc:
            print(f"  checkpoint not usable yet ({exc}); keeping the loaded one", flush=True)
            return False
        cls.weights_mtime = mtime
        print(f"  reloaded weights from {time.strftime('%H:%M:%S', time.localtime(mtime))}",
              flush=True)
        return True

    def log_message(self, fmt, *args):
        pass                                    # keep the terminal clean during a demo

    def _send(self, code, body, content_type):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)

        if url.path in ("/", "/index.html"):
            path = os.path.join(WEB_DIR, "index.html")
            if not os.path.exists(path):
                return self._send(404, "web/index.html is missing", "text/plain")
            with open(path, "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")

        if url.path == "/api/info":
            return self._send(200, json.dumps({
                "weights": self.weights_path,
                "trained_at": time.strftime("%H:%M:%S", time.localtime(self.weights_mtime)),
                "patterns": patterns.describe(),
                "n_lookups": patterns.N_LOOKUPS,
                "table_mb": round(patterns.N_PATTERNS * patterns.TABLE_SIZE * 4 / 2 ** 20),
            }), "application/json")

        if url.path == "/api/game":
            seed = query.get("seed", [None])[0]
            seed = int(seed) if seed not in (None, "", "null") else None
            # --depth sets the default; the viewer may override it per game so a
            # demo can switch between greedy and search without a restart.
            depth = max(1, min(3, int(query.get("depth", [str(self.depth)])[0])))
            try:
                if self.reload_each_game:
                    Handler.refresh_weights()
                game = record_game(self.weights, seed, depth=depth)
            except Exception as exc:                    # never kill the demo
                return self._send(500, json.dumps({"error": str(exc)}), "application/json")
            return self._send(200, json.dumps(game), "application/json")

        if url.path == "/api/report":
            q = parse_qs(url.query)
            games = max(1, min(2000, int(q.get("games", ["200"])[0])))
            depth = max(1, min(3, int(q.get("depth", [str(self.depth)])[0])))
            try:
                if self.reload_each_game:
                    Handler.refresh_weights()
                report = aggregate(self.weights, games, depth)
            except Exception as exc:                    # never kill the demo
                return self._send(500, json.dumps({"error": str(exc)}), "application/json")
            return self._send(200, json.dumps(report), "application/json")

        if url.path == "/api/curve":
            # One entry per (run, stage). Two runs on one axis is the whole point
            # when the thing being compared is a training schedule.
            series = []
            for run, (label, base) in enumerate(self.curve_runs):
                for k, p in curve_paths(base):
                    rows = read_curve(p)
                    if rows:
                        series.append({"stage": k, "run": run,
                                       "label": label, "rows": rows})
            return self._send(200, json.dumps(series), "application/json")

        return self._send(404, "not found", "text/plain")


def main(argv=None):
    p = argparse.ArgumentParser(description="Web UI for the trained 2048 agent.")
    p.add_argument("--weights", default="weights/agent.npy")
    p.add_argument("--curve", default="talk/curve-stage0.csv")
    p.add_argument("--curve-label", default="step", metavar="NAME")
    p.add_argument("--compare", nargs="*", metavar="PATH",
                   default=["talk/cosine-curve-stage0.csv",
                            "talk/cosine-a0.0025-curve-stage0.csv"],
                   help="further runs to overlay, stage for stage. Each is "
                        "labelled from its filename; missing files are skipped. "
                        "Pass with no values to show one run only")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--open", action="store_true", help="open a browser window")
    p.add_argument("--reload", action="store_true",
                   help="re-read the checkpoint before each new game, so you can "
                        "watch an agent improve while it is still training")
    p.add_argument("--depth", type=int, default=1, metavar="PLIES",
                   help="expectimax decision levels (1 = greedy; 2 is much stronger "
                        "and about 17x slower to record a game)")
    args = p.parse_args(argv)

    # --weights names the base path; the real files carry a stage suffix, so
    # test for stage 0 rather than the base name, which need not exist at all.
    stage0 = fast.stage_path(args.weights, 0)
    if not os.path.exists(stage0):
        print(f"no weights at {stage0} -- train first with:\n"
              f"  python -m game2048.train --games 50000", file=sys.stderr)
        return 1

    print(f"loading {args.weights} "
          f"({os.path.getsize(stage0) / 2**20:.0f} MB per stage) ...", flush=True)
    Handler.weights = fast.load_weights(args.weights)
    Handler.weights_path = args.weights
    Handler.weights_mtime = newest_mtime(args.weights)
    Handler.reload_each_game = args.reload
    Handler.curve_path = args.curve
    Handler.curve_runs = [(args.curve_label, args.curve)]
    for extra in args.compare or []:
        if curve_paths(extra):
            Handler.curve_runs.append((curve_label(extra), extra))
    print("curves: " + ", ".join(
        f"{label} ({len(curve_paths(base))} stages)"
        for label, base in Handler.curve_runs))
    Handler.depth = args.depth

    print("warming up the compiled kernels ...", flush=True)
    record_game(Handler.weights, seed=0, max_moves=5, depth=args.depth)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  2048 agent viewer -> {url}\n  Ctrl+C to stop\n", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
