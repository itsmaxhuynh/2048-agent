"""The agent, in pure Python -- the version that never finished training.

bench.py's early rungs measure *random* play, where a move costs four table
lookups. The real training loop is far heavier: for every move it evaluates all
four candidate afterstates plus the previous one, and each evaluation is 32
pattern lookups. That is ~160 lookups per move, not four.

So "could I have trained this without a compiler?" needs this measurement, not
the random-play one. This module is an honest, reasonably optimised pure-Python
implementation -- shifts precomputed, locals hoisted, no numpy scalar overhead
in the inner loop -- and it is still the answer to that question being "no".
"""

import array
import random

from game2048 import board as B
from game2048.patterns import PATTERN_CELLS, PATTERN_TABLE, N_LOOKUPS, TABLE_SIZE

# hoist everything out of the inner loop, the way you would if this were the plan
SHIFTS = [[4 * int(PATTERN_CELLS[i, j]) for j in range(6)] for i in range(N_LOOKUPS)]
NIBBLES = [4 * j for j in range(6)]
TABLES = [int(t) for t in PATTERN_TABLE]

# The four tables live end to end in one flat row, so a lookup is table offset
# plus local index. Same arithmetic as `fast.PTAB_BASE`, recomputed here so this
# module stays importable without numba.
BASES = [int(t) * TABLE_SIZE for t in PATTERN_TABLE]

# One list of (offset, shifts) rather than two parallel lookups per iteration.
PLAN = list(zip(BASES, SHIFTS))


def as_weights(flat):
    """One stage's flat float32 row, in a form the interpreter reads cheaply.

    Indexing a numpy array hands back a numpy scalar, and adding 32 of those per
    position costs more than the lookup being measured (333 ns against 206 ns
    here). `array` yields a real Python float instead. `frombytes` is a memcpy,
    so this is a one-off 256 MB copy and no per-element conversion.
    """
    w = array.array("f")
    w.frombytes(memoryview(flat).cast("B"))
    return w


def evaluate(board, weights):
    """V(board) with the same 32 tuple lookups the compiled kernel does.

    `weights` is one stage's flat row, as `as_weights` returns it, so this and
    `fast.evaluate` are indexing the identical layout and comparing the same
    work. What it leaves out is the five `_extras` the compiled kernel also
    sums, which makes this side of the race slightly cheap: the ratio it feeds
    into is a lower bound on the real gap, not an inflated one.
    """
    total = 0.0
    for base, shifts in PLAN:
        idx = (((board >> shifts[0]) & 0xF)
               | (((board >> shifts[1]) & 0xF) << 4)
               | (((board >> shifts[2]) & 0xF) << 8)
               | (((board >> shifts[3]) & 0xF) << 12)
               | (((board >> shifts[4]) & 0xF) << 16)
               | (((board >> shifts[5]) & 0xF) << 20))
        total += weights[base + idx]
    return total


def best_action(board, weights):
    best_d, best_total, best_after, best_reward = -1, float("-inf"), board, 0
    for d in range(4):
        after, reward = B.move(board, d)
        if after == board:
            continue
        total = reward + evaluate(after, weights)
        if total > best_total:
            best_d, best_total, best_after, best_reward = d, total, after, reward
    return best_d, best_after, best_reward


def play_greedy(weights, rng=None, max_moves=None):
    """One game, agent-driven, entirely in the interpreter."""
    rng = rng or random.Random()
    board = B.new_game(rng)
    score = moves = 0
    while True:
        d, after, reward = best_action(board, weights)
        if d < 0:
            break
        score += reward
        moves += 1
        board = B.spawn(after, rng)
        if max_moves and moves >= max_moves:
            break
    return score, moves, B.max_tile(board)
