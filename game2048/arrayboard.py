"""The control experiment: the obvious algorithm, compiled.

bench.py shows that bit-packing the board bought almost nothing in pure Python
(~1.5x). That is a fair thing to notice and an easy thing to draw the wrong
conclusion from -- "the clever representation was a waste of time".

This module is how we check. It plays 2048 the way naive.py does -- a 4x4 grid,
gather the non-empty tiles in a line, merge neighbouring pairs, write them back
-- but written so numba can compile it. Comparing this against the compiled
bitboard isolates one variable: same compiler, same machine, different
representation.

The answer is in the benchmark, and it is not 1.5x.
"""

import numpy as np

from game2048.fast import njit


@njit(cache=True, inline="always")
def _coords(direction, line, step):
    """Cell (row, col) of the `step`-th square along `line`, counting from the
    wall the tiles are sliding into. Lets one slide routine serve all four moves."""
    if direction == 3:                  # left
        return line, step
    if direction == 1:                  # right
        return line, 3 - step
    if direction == 0:                  # up
        return step, line
    return 3 - step, line               # down


@njit(cache=True)
def move_array(grid, direction, gathered, result):
    """Slide `grid` (4x4 int8 of exponents) in place. Returns (reward, changed)."""
    reward = 0
    changed = False
    for line in range(4):
        n = 0
        for step in range(4):
            r, c = _coords(direction, line, step)
            v = grid[r, c]
            if v != 0:
                gathered[n] = v
                n += 1
        for k in range(4):
            result[k] = 0
        out = 0
        k = 0
        while k < n:
            if k + 1 < n and gathered[k] == gathered[k + 1] and gathered[k] < 15:
                result[out] = gathered[k] + 1
                reward += 1 << (gathered[k] + 1)
                k += 2
            else:
                result[out] = gathered[k]
                k += 1
            out += 1
        for step in range(4):
            r, c = _coords(direction, line, step)
            if grid[r, c] != result[step]:
                changed = True
                grid[r, c] = result[step]
    return reward, changed


@njit(cache=True)
def spawn_array(grid):
    empty = 0
    for r in range(4):
        for c in range(4):
            if grid[r, c] == 0:
                empty += 1
    if empty == 0:
        return False
    pick = np.random.randint(0, empty)
    seen = 0
    for r in range(4):
        for c in range(4):
            if grid[r, c] == 0:
                if seen == pick:
                    grid[r, c] = 1 if np.random.random() < 0.9 else 2
                    return True
                seen += 1
    return False


@njit(cache=True)
def play_random_array():
    """One random game on the 4x4 array board. Returns (score, moves, max_tile)."""
    grid = np.zeros((4, 4), dtype=np.int8)
    gathered = np.zeros(4, dtype=np.int8)
    result = np.zeros(4, dtype=np.int8)
    scratch = np.zeros((4, 4), dtype=np.int8)
    spawn_array(grid)
    spawn_array(grid)

    score = 0
    moves = 0
    while True:
        legal = np.empty(4, dtype=np.int64)
        n = 0
        for d in range(4):
            for r in range(4):
                for c in range(4):
                    scratch[r, c] = grid[r, c]
            _, changed = move_array(scratch, d, gathered, result)
            if changed:
                legal[n] = d
                n += 1
        if n == 0:
            break
        reward, _ = move_array(grid, legal[np.random.randint(0, n)], gathered, result)
        score += reward
        moves += 1
        spawn_array(grid)

    best = 0
    for r in range(4):
        for c in range(4):
            if grid[r, c] > best:
                best = grid[r, c]
    return score, moves, best
