"""Self-play training, one stage at a time.

Multi-stage TD learning (Yeh et al. 2016) is a *curriculum*, not just a
partitioned value function. Each stage is trained separately, and a later stage
never plays the early game at all: it trains from boards collected at the moment
the previous stage handed over. That is what gives a late stage millions of
games of data instead of the thin tail it would see in one online loop.

    python -m game2048.train --stage 0 --games 5000000
    python -m game2048.train --collect 1 --games 300000
    python -m game2048.train --stage 1 --games 5000000
    ... and so on for stages 2 and 3

Ctrl+C saves and exits cleanly -- important when the thing you are about to demo
has been training for hours.
"""

import argparse
import csv
import math
import os
import signal
import sys
import time

import numpy as np

from game2048 import fast
from game2048.saturation import SaturationMonitor, explain

TILE_MILESTONES = (7, 8, 9, 10, 11, 12, 13, 14, 15)     # 128 ... 32768


class Stats:
    """Running statistics for one block of games."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores = []
        self.moves = 0
        self.tiles = np.zeros(20, dtype=np.int64)
        self.start = time.time()

    def add(self, score, moves, max_tile):
        self.scores.append(score)
        self.moves += moves
        self.tiles[max_tile] += 1

    def summary(self, played):
        n = len(self.scores)
        elapsed = max(time.time() - self.start, 1e-9)
        rates = {}
        for k in TILE_MILESTONES:
            reached = int(self.tiles[k:].sum())
            if reached:
                rates[1 << k] = 100.0 * reached / n
        return {
            "games": played,
            "avg": float(np.mean(self.scores)),
            "max": int(np.max(self.scores)),
            "games_per_sec": n / elapsed,
            "moves_per_sec": self.moves / elapsed,
            "rates": rates,
        }


def format_block(s, alpha):
    reach = "  ".join(f"{tile}:{pct:.0f}%" for tile, pct in s["rates"].items())
    return (f"{s['games']:>9,}  avg {s['avg']:>8,.0f}  best {s['max']:>8,}  "
            f"a={alpha:<8.5f} {s['games_per_sec']:>6.1f} g/s  "
            f"{s['moves_per_sec']:>8,.0f} mv/s   {reach}")


def cosine_alpha(game, total, alpha, alpha_final):
    """Step size on a cosine ramp from `alpha` down to `alpha_final` over `total` games.

    The paper drops alpha once, from 0.0025 to 0.00025, when the curve saturates
    (Yeh et al. 2016, IV.E(b)). That works, but it wastes the games either side of
    the cliff: too large a step for a while, then too small too suddenly. Cosine
    spends most of its time near the two ends and passes quickly through the
    middle, which is the same reasoning that made it standard for neural nets.

    The schedule is a pure function of the *cumulative* game count, not of this
    run's progress, so `--resume` picks the curve back up where it left off
    instead of restarting the ramp. Past `total` it pins to `alpha_final`, so
    overrunning the budget decays rather than wraps.
    """
    t = min(max(game, 0), total) / max(total, 1)
    return alpha_final + 0.5 * (alpha - alpha_final) * (1.0 + math.cos(math.pi * t))


def save(array, path):
    """Write atomically -- a half-written 256 MB checkpoint is worse than none."""
    tmp = path + ".tmp.npy"
    np.save(tmp, array)
    os.replace(tmp, path)


def restarts_path(path, stage):
    root, ext = os.path.splitext(path)
    return f"{root}.restarts{stage}{ext}"


def install_signal_handler(state):
    def on_signal(signum, frame):
        state["stopping"] = True
        print("\ninterrupted -- finishing this game, then saving", flush=True)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)


def do_collect(args):
    """Play the previous stage's weights and snapshot every crossing into `stage`."""
    stage = args.collect
    prev = fast.load_stage(args.out, stage - 1)
    if prev is None:
        print(f"stage {stage - 1} is not trained yet", file=sys.stderr)
        return 1

    starts = np.zeros((1, 2), dtype=np.uint64)
    n_starts = 0
    if stage >= 2:
        src = restarts_path(args.out, stage - 1)
        if not os.path.exists(src):
            print(f"missing {src} -- collect for stage {stage - 1} first", file=sys.stderr)
            return 1
        starts = np.load(src)
        n_starts = len(starts)
        print(f"replaying {n_starts:,} stage-{stage - 1} restart boards")

    print(f"collecting boards at the stage-{stage} boundary "
          f"(requires {fast.STAGE_REQUIREMENTS[stage - 1]})", flush=True)
    state = {"stopping": False}
    install_signal_handler(state)

    out = np.zeros((args.max_restarts, 2), dtype=np.uint64)   # board, score
    fast.collect_boards(prev, starts, n_starts, stage, out, 1, 1)     # compile

    found = games = 0
    chunk = max(1000, args.block)
    t0 = time.time()
    while found < args.max_restarts and games < args.games and not state["stopping"]:
        f, g = fast.collect_boards(prev, starts, n_starts, stage,
                                   out[found:], args.max_restarts - found, chunk)
        found += int(f)
        games += int(g)
        el = time.time() - t0
        print(f"  {games:>9,} games -> {found:>8,} boards  "
              f"({100 * found / max(games, 1):.1f}% reach it, {games / el:.0f} g/s)",
              flush=True)

    dest = restarts_path(args.out, stage)
    save(out[:found], dest)
    scores = out[:found, 1].astype(np.int64)
    print(f"\nsaved {found:,} restart boards from {games:,} games -> {dest}")
    if found:
        print(f"  score on arrival: median {np.median(scores):,.0f}  "
              f"mean {scores.mean():,.0f}  max {scores.max():,}")
    return 0


def do_train(args):
    stage = args.stage
    path = fast.stage_path(args.out, stage)

    weights = fast.load_stage(args.out, stage) if args.resume else None
    if weights is None:
        weights = fast.new_weights()
        print(f"fresh stage-{stage} weights: {weights.nbytes / 2**20:.0f} MB")
    else:
        print(f"resumed stage {stage} from {path}")

    starts = np.zeros((1, 2), dtype=np.uint64)
    if stage > 0:
        src = restarts_path(args.out, stage)
        if not os.path.exists(src):
            print(f"missing {src} -- run --collect {stage} first", file=sys.stderr)
            return 1
        starts = np.load(src)
        if args.restarts_after:
            keep = starts[:, 2] >= args.restarts_after if starts.shape[1] > 2 else None
            if keep is not None:
                before = len(starts)
                starts = starts[keep]
                print(f"restricting to boards harvested after game "
                      f"{args.restarts_after:,}: {len(starts):,} of {before:,}")
        if not len(starts):
            print("no restart boards left after filtering", file=sys.stderr)
            return 1
        print(f"training stage {stage} from {len(starts):,} collected boards")

    state = {"stopping": False}
    install_signal_handler(state)

    log_path = args.log or f"talk/curve-stage{stage}.csv"
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    new_log = not os.path.exists(log_path)
    offset = 0
    if args.resume and not new_log:
        with open(log_path, newline="") as f:
            rows = [r for r in csv.reader(f) if r and r[0].isdigit()]
        if rows:
            offset = int(rows[-1][0])
            print(f"continuing the learning curve from game {offset:,}")
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["games", "avg_score", "max_score", "games_per_sec",
                         "moves_per_sec", "alpha"])

    buffers = fast.new_buffers()
    zero = np.uint64(0)

    if args.harvest.strip().lower() in ("all", "*"):
        harvest = list(range(1, fast.N_STAGES))
    elif args.harvest.strip():
        harvest = sorted({int(x) for x in args.harvest.split(",") if x.strip()})
    else:
        harvest = []
    for k in harvest:
        if not 1 <= k < fast.N_STAGES:
            print(f"--harvest: stage {k} is not in 1..{fast.N_STAGES - 1}", file=sys.stderr)
            return 1

    catch_mask, caught_row, caught_flags = fast.catch_buffers(harvest)
    corpora = {}                       # stage -> (board, score, game) rows
    for k in harvest:
        path_k = restarts_path(args.out, k)
        if os.path.exists(path_k):
            corpora[k] = list(map(tuple, np.load(path_k)))
            print(f"harvesting stage {k} into {path_k} "
                  f"({len(corpora[k]):,} boards already there)")
        else:
            corpora[k] = []
            print(f"harvesting stage {k} into {path_k}")

    def flush_corpora():
        for k, rows in corpora.items():
            if rows:
                save(np.asarray(rows, dtype=np.uint64), restarts_path(args.out, k))

    print("warming up the JIT...", flush=True)
    t0 = time.time()
    fast.learn(weights, args.alpha, args.lam, buffers)
    print(f"compiled in {time.time() - t0:.1f}s\n", flush=True)

    stats = Stats()
    played = 0
    anneal_over = args.anneal_over or args.games
    cosine = args.schedule == "cosine"
    alpha = cosine_alpha(offset, anneal_over, args.alpha, args.alpha_final) \
        if cosine else args.alpha
    decayed = False
    if cosine:
        print(f"cosine schedule: alpha {args.alpha} -> {args.alpha_final} "
              f"over {anneal_over:,} games (now at game {offset:,}, alpha {alpha:.6f})",
              flush=True)
    last_save = time.time()
    monitor = SaturationMonitor(
        window=args.saturation_window, tolerance=args.saturation_tolerance)
    if (args.auto_decay or args.stop_when_saturated) and offset and not new_log:
        # A resumed run already has a curve. Without this the monitor would spend
        # a whole window re-deriving a verdict the log can already answer.
        with open(log_path, newline="") as f:
            past = [r for r in csv.reader(f) if r and r[0].isdigit()]
        for r in past[-args.saturation_window:]:
            monitor.add(int(r[0]), float(r[1]))
        primed = monitor.verdict()
        if primed:
            print(f"primed the saturation monitor from {len(past[-args.saturation_window:])} "
                  f"past blocks:\n" + explain(primed, args.saturation_tolerance) + "\n",
                  flush=True)
    try:
        while played < args.games and not state["stopping"]:
            if stage > 0:
                row = starts[played % len(starts)]
                start, start_score = np.uint64(row[0]), np.int64(row[1])
            else:
                start, start_score = zero, np.int64(0)
            score, moves, tile = fast.play_and_learn(
                weights, alpha, args.lam, start, start_score, *buffers,
                catch_mask, caught_row, caught_flags)
            for k in harvest:
                if caught_flags[k] and len(corpora[k]) < args.max_restarts:
                    corpora[k].append((int(caught_row[k, 0]), int(caught_row[k, 1]),
                                       played + offset))
            stats.add(int(score), int(moves), int(tile))
            played += 1

            if cosine:
                alpha = cosine_alpha(played + offset, anneal_over,
                                     args.alpha, args.alpha_final)

            if args.decay_after and played == args.decay_after:
                alpha = args.alpha_final
                decayed = True
                print(f"  -- alpha {args.alpha} -> {alpha} at the planned game count",
                      flush=True)

            if played % args.block == 0:
                s = stats.summary(played + offset)
                line = format_block(s, alpha)
                if harvest:
                    line += "   [" + " ".join(f"s{k}:{len(corpora[k]):,}"
                                              for k in harvest) + "]"
                print(line, flush=True)

                if args.auto_decay or args.stop_when_saturated:
                    monitor.add(s["games"], s["avg"])
                    v = monitor.verdict()
                    if v and v["saturated"]:
                        if not decayed and args.auto_decay and not cosine:
                            alpha = args.alpha_final
                            decayed = True
                            print("\n  -- saturated; dropping the step size\n" +
                                  explain(v, args.saturation_tolerance) +
                                  f"\n  alpha {args.alpha} -> {alpha}\n", flush=True)
                            monitor.reset()      # the dynamics just changed
                        elif args.stop_when_saturated:
                            print("\n  -- saturated again at the lower step size\n" +
                                  explain(v, args.saturation_tolerance), flush=True)
                            state["stopping"] = True
                writer.writerow([s["games"], f"{s['avg']:.1f}", s["max"],
                                 f"{s['games_per_sec']:.2f}", f"{s['moves_per_sec']:.0f}",
                                 f"{alpha:.6f}"])
                log.flush()
                stats.reset()
                if time.time() - last_save >= args.save_every:
                    save(weights, path)
                    flush_corpora()
                    last_save = time.time()
    finally:
        save(weights, path)
        flush_corpora()
        for k in harvest:
            print(f"harvested {len(corpora[k]):,} stage-{k} boards "
                  f"-> {restarts_path(args.out, k)}")
        log.close()
    print(f"\nplayed {played:,} games -> {path}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Train a 2048 n-tuple agent by self-play.")
    p.add_argument("--stage", type=int, default=0, help="which stage to train (0-3)")
    p.add_argument("--collect", type=int, default=0, metavar="STAGE",
                   help="instead of training, collect restart boards for this stage")
    p.add_argument("--games", type=int, default=1_000_000, help="games this run")
    p.add_argument("--alpha", type=float, default=0.0025, help="per-weight step size")
    p.add_argument("--alpha-final", type=float, default=0.00025,
                   help="step size after --decay-after games")
    p.add_argument("--schedule", choices=("step", "cosine"), default="step",
                   help="how alpha moves: 'step' drops it once (the paper's own "
                        "recipe, driven by --decay-after or --auto-decay), "
                        "'cosine' anneals it smoothly over --anneal-over games")
    p.add_argument("--anneal-over", type=int, default=0, metavar="N",
                   help="games the cosine ramp spans, counting from the start of "
                        "the stage rather than from this run (0 = use --games). "
                        "Set it explicitly whenever you --resume, or the ramp "
                        "will be scaled to the wrong budget")
    p.add_argument("--decay-after", type=int, default=0, metavar="N",
                   help="drop alpha at exactly this many games (0 = use --auto-decay)")
    p.add_argument("--auto-decay", action="store_true",
                   help="drop alpha when the curve stops paying for itself, the way "
                        "the paper describes; see game2048.saturation")
    p.add_argument("--stop-when-saturated", action="store_true",
                   help="end the run once the decayed phase has flattened too")
    p.add_argument("--saturation-window", type=int, default=160, metavar="BLOCKS")
    p.add_argument("--saturation-tolerance", type=float, default=0.02,
                   help="gain worth having, as a fraction of the current average")
    p.add_argument("--lam", type=float, default=0.5, metavar="LAMBDA",
                   help="TD(lambda); 0 selects online TD(0)")
    p.add_argument("--block", type=int, default=1000, help="games between progress lines")
    p.add_argument("--out", default="weights/agent.npy", help="checkpoint stem")
    p.add_argument("--resume", action="store_true", help="continue from an existing stage")
    p.add_argument("--log", default=None, help="CSV of block stats")
    p.add_argument("--save-every", type=float, default=300.0,
                   help="seconds between checkpoints (each writes 256 MB)")
    p.add_argument("--max-restarts", type=int, default=2_000_000,
                   help="cap on collected boards (space, not policy)")
    p.add_argument("--harvest", default="", metavar="STAGES",
                   help="while training, snapshot boards entering these stages -- "
                        "'1', '1,2,3', or 'all'. Free, since the games are played "
                        "anyway. Each board records the game it came from, because "
                        "the weights keep changing as it collects")
    p.add_argument("--restarts-after", type=int, default=0, metavar="GAME",
                   help="use only restart boards harvested after this game, so a "
                        "stage trains on positions from a strong player rather than "
                        "whatever was around early in the harvest")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.schedule == "cosine" and (args.auto_decay or args.decay_after):
        print("--schedule cosine already anneals alpha; drop --auto-decay/"
              "--decay-after", file=sys.stderr)
        return 1

    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    return do_collect(args) if args.collect else do_train(args)


if __name__ == "__main__":
    sys.exit(main())
