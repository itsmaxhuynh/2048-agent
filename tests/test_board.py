"""The safety net: the fast board must agree with the obvious one, always.

Every optimisation in this project is only allowed because these tests pass.
Run with:  python -m tests.test_board
"""

import random
import sys

from game2048 import board as B
from game2048 import naive


def grid_of(board):
    return B.to_grid(board)


def check_transpose(trials=2000, seed=1):
    rng = random.Random(seed)
    for _ in range(trials):
        b = rng.getrandbits(64)
        want = [[0] * 4 for _ in range(4)]
        cells = B.cells(b)
        for r in range(4):
            for c in range(4):
                want[c][r] = cells[r * 4 + c]
        got = B.cells(B.transpose(b))
        assert [got[r * 4 + c] for r in range(4) for c in range(4)] == \
               [want[r][c] for r in range(4) for c in range(4)], f"transpose failed for {b:016x}"
    return trials


def check_moves(trials=20000, seed=2):
    """Random reachable-ish positions: same result from both implementations."""
    rng = random.Random(seed)
    for _ in range(trials):
        # bias toward small exponents so positions look like real games
        grid = [[0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                if rng.random() < 0.7:
                    grid[r][c] = 1 << rng.randint(1, 6)
        board = B.from_grid(grid)
        for d in range(4):
            n_grid, n_reward, n_changed = naive.move(grid, d)
            f_board, f_reward = B.move(board, d)
            assert grid_of(f_board) == n_grid, (
                f"move {B.DIRECTION_NAMES[d]} mismatch\n{grid}\n{n_grid}\n{grid_of(f_board)}")
            assert f_reward == n_reward, f"reward mismatch {f_reward} != {n_reward}"
            assert (f_board != board) == n_changed, "legality mismatch"
    return trials


def check_games(games=200, seed=3):
    """Play both implementations in lockstep with a shared RNG stream."""
    for g in range(games):
        rng_a = random.Random(1000 + g)
        rng_b = random.Random(1000 + g)
        grid = [[0] * 4 for _ in range(4)]
        naive.spawn(grid, rng_a)
        naive.spawn(grid, rng_a)
        board = B.new_game(rng_b)
        assert grid_of(board) == grid, "initial spawn diverged"

        score_a = score_b = 0
        while True:
            legal_a = naive.legal_moves(grid)
            legal_b = B.legal_moves(board)
            assert legal_a == legal_b, f"legal moves diverged: {legal_a} vs {legal_b}"
            if not legal_a:
                break
            d = legal_a[rng_a.randrange(len(legal_a))]
            rng_b.randrange(len(legal_b))  # keep the two streams aligned
            grid, ra, _ = naive.move(grid, d)
            board, rb = B.move(board, d)
            score_a += ra
            score_b += rb
            naive.spawn(grid, rng_a)
            board = B.spawn(board, rng_b)
            assert grid_of(board) == grid, "boards diverged mid-game"
        assert score_a == score_b, f"scores diverged {score_a} != {score_b}"
    return games


def check_score_identity(games=50, seed=4):
    """score_of(board) reconstructs the game score from the tiles alone.

    Every tile on the board was built by merges worth (k-1)*2^k points, except
    that each spawned 4 arrives already worth 4 points nobody scored. Track
    those and the two totals must agree exactly -- which independently checks
    the merge scores baked into ROW_SCORE_LEFT/RIGHT.
    """
    rng = random.Random(seed)

    def spawn_tracked(board):
        after = B.spawn(board, rng)
        # the one cell that changed is the spawn; a 4 has exponent 2
        for i in range(16):
            if B.cell(board, i) != B.cell(after, i):
                return after, 4 if B.cell(after, i) == 2 else 0
        return after, 0

    for _ in range(games):
        board, bonus = spawn_tracked(0)
        board, b2 = spawn_tracked(board)
        bonus += b2
        score = 0
        while True:
            legal = B.legal_moves(board)
            if not legal:
                break
            board, r = B.move(board, legal[rng.randrange(len(legal))])
            score += r
            board, b = spawn_tracked(board)
            bonus += b
        assert B.score_of(board) == score + bonus, \
            f"{B.score_of(board)} != {score} + {bonus}"
    return games


if __name__ == "__main__":
    checks = [
        ("transpose", check_transpose),
        ("single moves vs naive", check_moves),
        ("full games in lockstep", check_games),
        ("score identity", check_score_identity),
    ]
    failed = False
    for name, fn in checks:
        try:
            n = fn()
            print(f"  PASS  {name} ({n} cases)")
        except AssertionError as e:
            failed = True
            print(f"  FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
