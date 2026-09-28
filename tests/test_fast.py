"""The compiled kernels must agree with the pure-Python board, bit for bit.

board.py is already checked against naive.py in test_board.py, so this closes
the chain: naive lists -> bit-packed ints -> compiled machine code.
"""

import random
import sys

import numpy as np

from game2048 import board as B
from game2048 import fast
from game2048 import patterns
from game2048 import arrayboard


def check_move(trials=50000, seed=11):
    rng = random.Random(seed)
    for _ in range(trials):
        grid = [[0] * 4 for _ in range(4)]
        for r in range(4):
            for c in range(4):
                if rng.random() < 0.7:
                    grid[r][c] = 1 << rng.randint(1, 11)
        b = B.from_grid(grid)
        for d in range(4):
            want_board, want_reward = B.move(b, d)
            got_board, got_reward = fast.move(np.uint64(b), d)
            assert int(got_board) == want_board, (
                f"dir {d} on {b:016x}: {int(got_board):016x} != {want_board:016x}")
            assert int(got_reward) == want_reward, f"reward {got_reward} != {want_reward}"
    return trials


def check_transpose_and_terminal(trials=20000, seed=12):
    rng = random.Random(seed)
    for _ in range(trials):
        b = rng.getrandbits(64)
        assert int(fast.transpose(np.uint64(b))) == B.transpose(b), "transpose"
        assert bool(fast.is_terminal(np.uint64(b))) == B.is_terminal(b), "terminal"
        assert int(fast.max_tile(np.uint64(b))) == B.max_tile(b), "max tile"
    return trials


def addresses(board):
    """Every weight slot evaluate() reads for this board, in order.

    Duplicated deliberately: if two symmetries land on the same entry, that
    entry really is read twice.
    """
    out = []
    for i in range(patterns.N_LOOKUPS):
        idx = sum(((board >> (4 * int(patterns.PATTERN_CELLS[i, j]))) & 0xF) << (4 * j)
                  for j in range(6))
        out.append(int(patterns.PATTERN_TABLE[i]) * patterns.TABLE_SIZE + idx)
    cells = [(board >> (4 * i)) & 0xF for i in range(16)]
    counts = [min(cells.count(e), patterns.LARGE_TILE_CAP)
              for e in patterns.LARGE_TILE_EXPONENTS]
    lt = 0
    for c in counts:
        lt = lt * 8 + c
    empty = cells.count(0)
    distinct = len({c for c in cells if c})
    right = [1,2,3,-1, 5,6,7,-1, 9,10,11,-1, 13,14,15,-1]
    down = [4,5,6,7, 8,9,10,11, 12,13,14,15, -1,-1,-1,-1]
    same = ladder = 0
    for i in range(16):
        v = cells[i]
        if not v:
            continue
        for j in (right[i], down[i]):
            if j < 0 or cells[j] == 0:
                continue
            u = cells[j]
            if u == v:
                same += 1
            elif abs(u - v) == 1:
                ladder += 1
    out.append(patterns.OFF_LARGE_TILE + lt)
    out.append(patterns.OFF_EMPTY + empty)
    out.append(patterns.OFF_DISTINCT + distinct)
    out.append(patterns.OFF_MERGEABLE + min(same, 24))
    out.append(patterns.OFF_NEIGHBOUR + min(ladder, 24))
    return out


def check_stage_events(trials=3000, seed=18):
    """advance_stage must fire exactly on the paper's events, and never go back.

    T16k needs a 16384; T16+8k needs a 16384 and an 8192 *together*; T16+8+4k
    adds a 4096. Because it is a moment in a game rather than a property of a
    board, merging the 8192 away afterwards must not drop the stage.
    """
    def want(board, stage):
        cells = [(board >> (4 * i)) & 0xF for i in range(16)]
        has = lambda e: e in cells
        s = stage
        if s == 0 and has(14):
            s = 1
        if s == 1 and has(14) and has(13):
            s = 2
        if s == 2 and has(14) and has(13) and has(12):
            s = 3
        return s

    rng = random.Random(seed)
    for _ in range(trials):
        b = rng.getrandbits(64)
        for st in range(4):
            got = int(fast.advance_stage(np.uint64(b), st))
            assert got == want(b, st), f"advance_stage({b:016x}, {st}) = {got}"
            assert got >= st, "stage went backwards"

    # a board with 16384+8192 promotes to 2; merging the 8192 away must not demote
    board = (14 << 0) | (13 << 4) | (12 << 8)
    st = int(fast.advance_stage(np.uint64(board), 0))
    assert st == 3, f"expected stage 3, got {st}"
    gone = np.uint64(14 << 0)
    assert int(fast.advance_stage(gone, st)) == 3, "stage must not drop when tiles merge away"
    return f"{trials} boards x 4 stages, plus the merge-away case"


def check_search(games=6, seed=19):
    """Search must agree with the greedy player at depth 1, and beat it deeper.

    Depth 1 is not "approximately" the greedy player -- it is the same
    arithmetic, so it must pick the same move every time. Anything else means
    the chance node is leaking into the leaf.
    """
    np.random.seed(seed)
    w = fast.new_weights()
    bufs = fast.new_buffers()
    for _ in range(400):                  # teach it something to have opinions about
        fast.learn(w, 0.0025, 0.5, bufs)

    np.random.seed(seed)
    board = fast.as_board(fast.new_game())
    checked = 0
    while checked < 400:
        greedy = fast.best_action(board, w)
        searched = fast.best_action_search(board, w, 1)
        assert greedy[0] == searched[0], f"depth 1 chose {searched[0]}, greedy {greedy[0]}"
        assert int(greedy[1]) == int(searched[1]), "depth 1 afterstate differs"
        assert abs(float(greedy[3]) - float(searched[3])) < 1e-2, "depth 1 value differs"
        if greedy[0] < 0:
            break
        board = fast.as_board(fast.spawn(fast.as_board(greedy[1])))
        checked += 1

    for _ in range(50):                   # value_after at depth 1 is just evaluate()
        b = np.uint64(random.Random(seed).getrandbits(64))
        assert abs(float(fast.value_after(b, w, 1)) - float(fast.evaluate(b, w))) < 1e-2

    np.random.seed(seed)
    flat = [int(fast.play_searched(w, 1)[0]) for _ in range(games * 4)]
    np.random.seed(seed)
    deep = [int(fast.play_searched(w, 2)[0]) for _ in range(games)]
    assert np.mean(deep) > np.mean(flat), (
        f"depth 2 ({np.mean(deep):.0f}) did not beat depth 1 ({np.mean(flat):.0f})")
    return (f"depth 1 == greedy over {checked} moves; "
            f"depth 2 {np.mean(deep):,.0f} > depth 1 {np.mean(flat):,.0f}")


def check_value_function(trials=300, seed=13):
    """A hand-rolled sum over every slot must match the compiled evaluate()."""
    rng = np.random.default_rng(seed)
    w = fast.new_weights()
    py_rng = random.Random(seed)
    boards = [py_rng.getrandbits(64) for _ in range(trials)]
    for b in boards:
        for a in addresses(b):
            w[a] = rng.standard_normal(dtype=np.float32)
    for b in boards:
        want = float(sum(float(w[a]) for a in addresses(b)))
        got = float(fast.evaluate(np.uint64(b), w))
        assert abs(got - want) < 1e-2, f"evaluate {got} != {want}"
    return f"{trials} boards, {patterns.N_FEATURES} slots each"


def check_update_moves_value(seed=14):
    """update(b, d) adds d to every slot evaluate(b) reads -- including repeats.

    So V moves by d * sum(count^2) over distinct addresses, not d * 37: a slot
    hit twice gets 2d and is then read twice. Getting this wrong is how a
    learning rate silently becomes something else.
    """
    w = fast.new_weights()
    rng = random.Random(seed)
    d = 0.01
    for _ in range(200):
        b = rng.getrandbits(64)
        addrs = addresses(b)
        expected = d * sum(addrs.count(a) for a in addrs)
        before = float(fast.evaluate(np.uint64(b), w))
        fast.update(np.uint64(b), d, w)
        after = float(fast.evaluate(np.uint64(b), w))
        assert abs((after - before) - expected) < 1e-2, \
            f"update moved value by {after - before}, expected {expected}"
    return 200


def check_learning_reduces_error(games=300, seed=15):
    """Sanity: a short training run must beat random play. If this fails the
    learning rule is wrong, however fast it runs."""
    np.random.seed(seed)
    w = fast.new_weights()
    bufs = fast.new_buffers()
    zero, zscore = np.uint64(0), np.int64(0)
    random_scores = [fast.play_random()[0] for _ in range(games)]
    for _ in range(games):
        fast.learn(w, 0.0025, 0.5, bufs)
    trained = [fast.play_greedy(w)[0] for _ in range(50)]
    base, now = np.mean(random_scores), np.mean(trained)
    assert now > base * 1.3, f"learning did not help: random {base:.0f} -> trained {now:.0f}"
    return f"random {base:.0f} -> after {games} games {now:.0f}"


def check_array_board(trials=20000, seed=17):
    """The compiled 4x4-array board must agree with the bitboard.

    It is the control in the benchmark, so if it were subtly wrong the headline
    "representation matters" number would be measuring the wrong thing.
    """
    rng = random.Random(seed)
    gathered = np.zeros(4, dtype=np.int8)
    result = np.zeros(4, dtype=np.int8)
    for _ in range(trials):
        cells = [rng.randint(1, 11) if rng.random() < 0.7 else 0 for _ in range(16)]
        b = 0
        for i, k in enumerate(cells):
            b |= k << (4 * i)
        for d in range(4):
            grid = np.array(cells, dtype=np.int8).reshape(4, 4)
            reward, changed = arrayboard.move_array(grid, d, gathered, result)
            want_board, want_reward = B.move(b, d)
            want_cells = [(want_board >> (4 * i)) & 0xF for i in range(16)]
            assert list(grid.reshape(16)) == want_cells, (
                f"array board disagrees on {B.DIRECTION_NAMES[d]}:\n"
                f"  got  {list(grid.reshape(16))}\n  want {want_cells}")
            assert reward == want_reward, f"reward {reward} != {want_reward}"
            assert changed == (want_board != b), "legality mismatch"
    return trials


def check_python_boundary(trials=1500, seed=16):
    """Regression: how boards are allowed to cross into the compiled kernels.

    numba hands a uint64 back to Python as a plain int, and passing that int
    back in types it as int64. `int64 & uint64` promotes to float64 in numpy's
    rules, which would silently destroy the board -- so every Python-facing
    kernel re-asserts np.uint64 on entry.

    Above 2^63 a Python int cannot be typed as int64 at all and numba raises
    OverflowError. That is the reason callers outside this package always wrap
    boards in np.uint64() rather than trusting a bare int.
    """
    rng = random.Random(seed)
    w = fast.new_weights()
    big_rejected = 0
    for _ in range(trials):
        b = rng.getrandbits(64)

        # np.uint64 is the supported way in, for every board
        assert int(fast.transpose(np.uint64(b))) == B.transpose(b), "transpose boundary"
        assert int(fast.max_tile(np.uint64(b))) == B.max_tile(b), "max_tile boundary"
        for d in range(4):
            got, reward = fast.move(np.uint64(b), d)
            want, want_r = B.move(b, d)
            assert int(got) == want, f"move boundary: {int(got):016x} != {want:016x}"
            assert int(reward) == want_r, "reward boundary"
        assert float(fast.evaluate(np.uint64(b), w)) == 0.0, "evaluate boundary"

        if b < 2 ** 63:
            # a bare int in int64 range must give identical answers, not float64 mush
            assert int(fast.transpose(b)) == B.transpose(b), "bare int transpose"
            assert int(fast.move(b, 3)[0]) == B.move(b, 3)[0], "bare int move"
        else:
            try:
                fast.transpose(b)
                raise AssertionError("expected OverflowError for a board >= 2**63")
            except OverflowError:
                big_rejected += 1
    assert big_rejected > 0, "test never exercised the >= 2**63 path"
    return f"{trials} boards, {big_rejected} correctly rejected as bare ints"


if __name__ == "__main__":
    checks = [
        ("compiled move vs board.py", check_move),
        ("transpose / terminal / max tile", check_transpose_and_terminal),
        ("stage events vs the paper rule", check_stage_events),
        ("evaluate matches hand-rolled sum", check_value_function),
        ("update shifts value by delta", check_update_moves_value),
        ("compiled array board vs bitboard", check_array_board),
        ("board survives the Python boundary", check_python_boundary),
        ("learning beats random", check_learning_reduces_error),
        ("expectimax vs greedy", check_search),
    ]
    failed = False
    for name, fn in checks:
        try:
            print(f"  PASS  {name} ({fn()})")
        except AssertionError as e:
            failed = True
            print(f"  FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
