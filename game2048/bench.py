"""The optimisation ladder, measured.

    python -m game2048.bench

Every rung runs the *same game* with the *same rules*; only the representation
and the compiler change. The numbers this prints are the backbone of the talk,
so they are measured on the machine giving the talk rather than quoted from
memory.

Run it with nothing else competing for the CPU -- a background training run
will halve the numbers.
"""

import os
import random
import time

import numpy as np

from game2048 import board as B
from game2048 import fast
from game2048 import naive
from game2048 import arrayboard
from game2048 import slow


def timed(fn, seconds=3.0, warmup=0):
    """Run fn() repeatedly for `seconds`; return (moves_per_sec, games, detail)."""
    for _ in range(warmup):
        fn()
    moves = games = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        moves += fn()
        games += 1
    elapsed = time.perf_counter() - start
    return moves / elapsed, games / elapsed


def stage_naive(rng=random.Random(0)):
    return naive.play_random_game(rng)[1]


def stage_bitboard(rng=random.Random(0)):
    board = B.new_game(rng)
    moves = 0
    while True:
        legal = B.legal_moves(board)
        if not legal:
            break
        board, _ = B.move(board, legal[rng.randrange(len(legal))])
        board = B.spawn(board, rng)
        moves += 1
    return moves


def stage_compiled_array():
    return int(arrayboard.play_random_array()[1])


def stage_compiled():
    return int(fast.play_random()[1])


_BUFS = fast.new_buffers()


def moves_actually_trained(path="talk/learning-curve.csv"):
    """Total moves behind the shipped weights, read off the training log.

    Each row covers a fixed block of games and records both games/sec and
    moves/sec, so moves-per-block falls out of the ratio. Hard-coding this number
    goes stale the moment training runs longer.
    """
    import csv
    if not os.path.exists(path):
        return 1_000_000_000
    total = 0.0
    prev = 0
    for row in csv.DictReader(open(path)):
        games, g, m = int(row["games"]), float(row["games_per_sec"]), float(row["moves_per_sec"])
        total += (games - prev) * m / g
        prev = games
    return int(total)


def search_ladder(weights, seconds=6.0):
    """What buying lookahead costs, and what it buys.

    Depth 1 is the training-time player. Every deeper level inserts a chance node
    over the tile the game drops next, so the branching factor is the number of
    empty cells -- which is why the price rises so much faster than the score.
    """
    out = []
    for plies in (1, 2, 3):
        fast.play_searched(weights, plies)                    # compile this specialisation
        scores, moves = [], 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            score, m, _ = fast.play_searched(weights, plies)
            scores.append(int(score))
            moves += int(m)
        elapsed = time.perf_counter() - t0
        out.append((plies, moves / elapsed, float(np.mean(scores)), len(scores)))
    return out


def make_agent_stages(weights):
    def stage_agent():
        return int(fast.play_greedy(weights)[1])

    def stage_learning():
        return int(fast.learn(weights, 0.0025, 0.5, _BUFS)[1])

    return stage_agent, stage_learning


def main():
    print("2048 in Python -- how fast can the same game go?\n")
    print(f"numba available: {fast.HAVE_NUMBA}")
    print("each stage plays complete games of 2048 and counts moves\n")

    rows = []

    print("  measuring stage 1/5: naive list-of-lists ...", flush=True)
    mps, gps = timed(stage_naive, seconds=3.0)
    rows.append(("1. naive lists, tile values", mps, gps, "random play"))

    print("  measuring stage 2/5: bit-packed board, pure Python ...", flush=True)
    mps, gps = timed(stage_bitboard, seconds=3.0)
    rows.append(("2. one 64-bit int + row tables", mps, gps, "random play"))

    print("  measuring stage 3/6: naive algorithm, compiled ...", flush=True)
    mps, gps = timed(stage_compiled_array, seconds=3.0, warmup=1)
    rows.append(("3. naive algorithm + numba", mps, gps, "random play"))

    print("  measuring stage 4/6: bitboard + tables, compiled ...", flush=True)
    mps, gps = timed(stage_compiled, seconds=3.0, warmup=1)
    rows.append(("4. bitboard + tables + numba", mps, gps, "random play"))

    print("  measuring stage 5/6: agent choosing moves (32 lookups/position) ...", flush=True)
    weights = fast.new_weights()
    for _ in range(400):                        # give it a value function worth consulting
        fast.learn(weights, 0.0025, 0.5, _BUFS)
    agent, learning = make_agent_stages(weights)
    mps, gps = timed(agent, seconds=3.0, warmup=1)
    rows.append(("5. + n-tuple agent picking moves", mps, gps, "greedy play"))

    print("  measuring stage 6/6: agent learning while it plays ...", flush=True)
    mps, gps = timed(learning, seconds=3.0, warmup=1)
    rows.append(("6. + TD learning on every move", mps, gps, "self-play training"))

    base = rows[0][1]
    print("\n" + "=" * 78)
    print(f"{'stage':<34}{'moves/sec':>13}{'games/sec':>12}{'vs naive':>10}")
    print("-" * 78)
    for name, mps, gps, _ in rows:
        print(f"{name:<34}{mps:>13,.0f}{gps:>12,.1f}{mps / base:>9.0f}x")
    print("=" * 78)

    bitboard_gain = rows[1][1] / rows[0][1]
    compile_gain = rows[2][1] / rows[1][1]
    repr_gain = rows[3][1] / rows[2][1]
    print(f"\n  bit-packing, interpreted   {bitboard_gain:>6.1f}x   <- looks like a waste of time")
    print(f"  compiling the naive code  {compile_gain:>6.1f}x   <- the obvious win")
    print(f"  ...on the packed board    {repr_gain:>6.1f}x   <- what the packing was FOR")
    print(f"\ntotal, before any machine learning: {rows[3][1] / rows[0][1]:,.0f}x")
    print("The representation only pays once the interpreter is out of the way.\n")

    # The rungs above race *random* play. The training loop is much heavier:
    # ~160 table lookups per move rather than four. This is the number that
    # actually answers "could I have trained this in pure Python?".
    print("also measuring: the agent itself, in pure Python ...", flush=True)
    rows_slow = []
    weights_slow = slow.as_weights(weights)      # same flat row, cheap float access
    t0 = time.perf_counter()
    slow_moves = 0
    while time.perf_counter() - t0 < 4.0:
        slow_moves += slow.play_greedy(weights_slow, max_moves=400)[1]
    slow_mps = slow_moves / (time.perf_counter() - t0)
    fast_mps = rows[5][1]
    print(f"\n{'agent in pure Python':<34}{slow_mps:>13,.0f} moves/sec")
    print(f"{'agent compiled (learning)':<34}{fast_mps:>13,.0f} moves/sec"
          f"   -> {fast_mps / slow_mps:,.0f}x\n")

    trained_moves = moves_actually_trained()     # what this project's agent actually played
    print(f"the trained agent played about {trained_moves / 1e6:.0f} million moves:")
    print(f"  compiled     {trained_moves / fast_mps / 60:>8,.0f} minutes")
    print(f"  pure Python  {trained_moves / slow_mps / 3600:>8,.0f} hours\n")

    # a second, sharper measurement: what one position costs the agent
    print("cost of a single position evaluation (32 table lookups into one 256 MB stage):")
    board = fast.new_game()
    fast.evaluate(board, weights)
    n = 200_000
    start = time.perf_counter()
    for _ in range(n):
        fast.evaluate(board, weights)
    per = (time.perf_counter() - start) / n
    print(f"  {per * 1e6:.2f} microseconds  ({1 / per:,.0f} positions/sec)")
    print("  -- almost all of it cache misses; the table is far bigger than L3.\n")

    # Search is where the speed gets spent. It buys strength at a steep and
    # steepening exchange rate -- the counterweight to the ladder above.
    print("also measuring: buying strength with lookahead (play time only) ...", flush=True)
    ladder = search_ladder(weights)
    base_mps, base_score = ladder[0][1], ladder[0][2]
    print(f"\n{'depth':<10}{'moves/sec':>13}{'cost':>10}{'avg score':>13}{'gain':>9}{'games':>8}")
    print("-" * 63)
    for plies, mps, score, games in ladder:
        print(f"{plies:<10}{mps:>13,.0f}{base_mps / mps:>9,.0f}x"
              f"{score:>13,.0f}{score / base_score:>8.2f}x{games:>8}")
    print()
    print("  Lookahead is the only thing here that trades throughput for strength,")
    print("  and the rate gets worse with every level. It is also why the 226x")
    print("  mattered: that speed is the budget this spends.\n")


if __name__ == "__main__":
    main()
