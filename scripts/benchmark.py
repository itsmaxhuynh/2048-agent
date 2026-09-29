"""The depth ladder: how much does search buy on top of the trained weights?

Plays complete games with learning off, at each search depth, and reports what
the agent actually builds. This is the number to quote: the training log's
average is measured with the weights moving underneath the player, and at
1 ply, neither of which is how the agent gets deployed.

Each depth runs until it hits its game count or its time budget, so the whole
thing stays bounded even though depth 3 is hundreds of times slower than depth 1.

    .venv/bin/python scripts/benchmark.py [--out talk/benchmark.txt]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game2048 import fast          # noqa: E402

# (plies, max games, seconds budget). Depth 3 gets the loosest game count and
# the longest budget because it is the slowest and the most interesting.
LADDER = ((1, 3000, 300), (2, 400, 600), (3, 60, 900))


def run(weights, plies, max_games, budget):
    scores, tiles, stages = [], [], []
    moves = 0
    t0 = time.time()
    while len(scores) < max_games and time.time() - t0 < budget:
        s, m, tile, stage = fast.play_reported(weights, plies)
        scores.append(int(s)); tiles.append(int(tile)); stages.append(int(stage))
        moves += int(m)
        if len(scores) % max(1, max_games // 20) == 0:
            print(f"\r  depth {plies}: {len(scores)} games, avg {np.mean(scores):,.0f}",
                  end="", flush=True)
    el = time.time() - t0
    print(f"\r  depth {plies}: {len(scores)} games, avg {np.mean(scores):,.0f}"
          f"  ({el:.0f}s)          ", flush=True)
    return {"plies": plies, "scores": np.array(scores), "tiles": np.array(tiles),
            "stages": np.array(stages), "mps": moves / el, "gps": len(scores) / el}


def report(rows, out):
    w = []
    w.append("2048 multi-stage agent, learning off, N-tuple weights from the "
             "4-stage curriculum")
    w.append(f"generated {time.strftime('%Y-%m-%d %H:%M')}\n")
    base = rows[0]["scores"].mean()
    basemps = rows[0]["mps"]
    w.append(f"{'depth':<8}{'games':>7}{'avg score':>12}{'+/- 95%':>10}{'median':>11}"
             f"{'best':>11}{'moves/sec':>12}{'cost':>9}{'strength':>10}")
    for r in rows:
        s = r["scores"]
        ci = 1.96 * s.std(ddof=1) / np.sqrt(len(s))
        w.append(f"{r['plies']:<8}{len(s):>7}{s.mean():>12,.0f}{ci:>10,.0f}"
                 f"{np.median(s):>11,.0f}{s.max():>11,.0f}{r['mps']:>12,.0f}"
                 f"{basemps / r['mps']:>8.0f}x{s.mean() / base:>9.2f}x")
    w.append("\nreach rates (share of games that built this tile)")
    ks = [k for k in range(11, 18)
          if any(float((r["tiles"] >= k).mean()) > 0 for r in rows)]
    w.append(f"{'tile':>8}" + "".join(f"{'depth ' + str(r['plies']):>12}" for r in rows))
    for k in ks:
        w.append(f"{1 << k:>8,}" + "".join(
            f"{100 * float((r['tiles'] >= k).mean()):>11.1f}%" for r in rows))
    w.append("\nfurthest stage reached (share of games)")
    w.append(f"{'stage':>8}" + "".join(f"{'depth ' + str(r['plies']):>12}" for r in rows))
    for st in range(fast.N_STAGES):
        w.append(f"{st:>8}" + "".join(
            f"{100 * float((r['stages'] == st).mean()):>11.1f}%" for r in rows))
    text = "\n".join(w) + "\n"
    print("\n" + text)
    with open(out, "w") as fh:
        fh.write(text)
    print(f"written to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="weights/agent.npy")
    p.add_argument("--out", default="talk/benchmark.txt")
    args = p.parse_args()

    weights = fast.load_weights(args.weights)
    print(f"loaded {weights.shape[0]} stages, "
          f"{weights.nbytes / 2**20:,.0f} MB\nwarming up the JIT ...", flush=True)
    for plies in (1, 2, 3):
        fast.play_reported(weights, plies)      # compile before anything is timed
    print("running\n", flush=True)
    report([run(weights, *cfg) for cfg in LADDER], args.out)


if __name__ == "__main__":
    main()
