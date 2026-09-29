"""Measure playing strength across every core, without paying for the weights twice.

Depth 3 is about 440x the per-move cost of depth 1 and its games run twice as
long, so 10,000 games at 3 ply is roughly 40 hours on one core. This script
spends the other three.

The reason it can: evaluation never writes the weight tables, so the workers are
forked *after* the tables are loaded and share them copy-on-write. Measured with
`Pss_Anon`, four workers sharing a 256 MB table report ~63 MB each rather than
256 MB each. That is why CLAUDE.md's "no room for multiprocessing workers each
holding a copy" does not apply here: it is a statement about training, where the
tables are written and copy-on-write breaks on the first update.

Two things this file must keep doing:

* **Fork, never spawn.** A spawned worker re-imports and re-loads the weights,
  which is 1 GB per worker and an out-of-memory kill on this laptop.
* **Seed every chunk explicitly.** numba's `np.random` is its own state, separate
  from numpy's and not visibly inherited across fork. Leaving it implicit means
  betting a multi-hour benchmark on undocumented behaviour; seeding also makes
  the run reproducible from `--seed`.

    .venv/bin/python scripts/eval_parallel.py --games 10000 --depth 3 --jobs 4
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from game2048 import fast          # noqa: E402

WEIGHTS = None      # set in the parent before forking; inherited, never copied
DEPTH = 1


@njit(cache=True)
def seed_rng(s):
    """Seed numba's internal RNG. numpy.random.seed() does not reach it."""
    np.random.seed(s)


def play_chunk(job):
    """One worker's slice: `n` games under a distinct, recorded seed."""
    index, n, seed = job
    seed_rng(seed)
    scores = np.empty(n, dtype=np.int64)
    tiles = np.empty(n, dtype=np.int64)
    moves = np.empty(n, dtype=np.int64)
    for i in range(n):
        score, mv, tile = fast.play_staged(WEIGHTS, DEPTH)
        scores[i] = int(score)
        tiles[i] = int(tile)
        moves[i] = int(mv)
    return index, scores, tiles, moves


def summarise(scores, tiles, moves, games, depth, jobs, elapsed):
    out = [
        f"  games        {games:,}",
        f"  average      {scores.mean():,.0f}",
        f"  median       {np.median(scores):,.0f}",
        f"  best         {scores.max():,}",
        f"  speed        {games / elapsed:.2f} games/sec  ({jobs} workers)",
        f"  moves/sec    {moves.sum() / elapsed:,.0f}  (all {jobs} workers; "
        f"divide by {jobs} for the per-core figure the ladder quotes)",
        f"  moves/game   {moves.mean():,.0f}",
        f"  wall clock   {elapsed / 60:.1f} min",
        f"  search depth {depth} {'(greedy)' if depth <= 1 else 'plies'}",
        "",
        "  reach rates (share of games that built this tile):",
    ]
    for k in range(7, 17):
        share = float((tiles >= k).mean())
        if share > 0:
            out.append(f"    {1 << k:>6,}  {share * 100:>5.1f}%  {'#' * int(share * 40)}")
    return "\n".join(out)


def main(argv=None):
    global WEIGHTS, DEPTH
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", default="weights/agent.npy")
    p.add_argument("--games", type=int, default=10000)
    p.add_argument("--depth", type=int, default=1, metavar="PLIES")
    p.add_argument("--jobs", type=int, default=4,
                   help="worker processes; default 4, the physical core count")
    p.add_argument("--seed", type=int, default=20480,
                   help="base seed; chunk k runs under seed + k")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="write the summary here as well as to stdout")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="write every game's score and max-tile exponent here")
    args = p.parse_args(argv)

    DEPTH = args.depth
    print(f"loading {args.weights} ...", flush=True)
    WEIGHTS = fast.load_weights(args.weights)
    print(f"loaded {WEIGHTS.shape[0]} stages, {WEIGHTS.nbytes / 2**30:.2f} GB", flush=True)

    # Compile in the parent, so the workers inherit machine code instead of each
    # paying the compile. Also the last chance to fail before we fork.
    seed_rng(args.seed)
    fast.play_staged(WEIGHTS, args.depth)
    print("kernels compiled; forking workers", flush=True)

    # Small chunks keep the workers balanced (game lengths vary a lot at depth)
    # and give the progress line something to count.
    per_chunk = max(1, min(50, args.games // (args.jobs * 8) or 1))
    bounds = list(range(0, args.games, per_chunk)) + [args.games]
    jobs = [(k, bounds[k + 1] - bounds[k], args.seed + 1 + k)
            for k in range(len(bounds) - 1)]

    scores = np.empty(args.games, dtype=np.int64)
    tiles = np.empty(args.games, dtype=np.int64)
    moves = np.empty(args.games, dtype=np.int64)
    done = 0
    running = 0                             # chunks land out of order, so the
    t0 = time.time()                        # progress average cannot slice `scores`
    ctx = mp.get_context("fork")            # never "spawn": see the module docstring
    with ctx.Pool(args.jobs) as pool:
        for index, s, t, m in pool.imap_unordered(play_chunk, jobs):
            lo = bounds[index]
            scores[lo:lo + s.size] = s
            tiles[lo:lo + t.size] = t
            moves[lo:lo + m.size] = m
            done += s.size
            running += int(s.sum())
            rate = done / (time.time() - t0)
            eta = (args.games - done) / rate / 60 if rate else 0
            print(f"\r  {done}/{args.games} games  avg {running / done:,.0f}"
                  f"  {rate:.2f} g/s  eta {eta:.0f} min   ", end="", flush=True)
    elapsed = time.time() - t0
    print()

    report = summarise(scores, tiles, moves, args.games, args.depth, args.jobs,
                       elapsed)
    print("\n" + report)
    if args.csv:
        np.savetxt(args.csv, np.column_stack([scores, tiles]), fmt="%d",
                   delimiter=",", header="score,max_tile_exponent", comments="")
        print(f"\n  per-game data -> {args.csv}")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(f"# {args.games:,}-game evaluation -- {args.depth} ply\n")
            fh.write(f"# weights  {args.weights}\n")
            fh.write(f"# seed     {args.seed} (chunk k under seed + 1 + k)\n")
            fh.write(f"# command  scripts/eval_parallel.py --games {args.games} "
                     f"--depth {args.depth} --jobs {args.jobs} --seed {args.seed}\n\n")
            fh.write(report + "\n")
        print(f"  summary       -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
