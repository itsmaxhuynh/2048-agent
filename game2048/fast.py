"""Stage 2 -- the same algorithm, compiled.

Everything here is deliberately written in the subset of Python that numba can
turn into machine code: scalars, loops, and flat numpy arrays. No objects, no
lists, no exceptions. `@njit` then compiles each function on first call.

If numba is not installed the `njit` decorator below degrades to a no-op, so
the module still runs -- just roughly 50x slower. That fallback is what makes
the benchmark in bench.py an apples-to-apples comparison.

The board is a numpy uint64. Every shift amount and mask is *also* a uint64:
mixing uint64 with a plain Python int makes numpy (and numba) promote the whole
expression to float64, which silently destroys the low bits of the board. That
is the single sharpest edge in this file.
"""

import os

import numpy as np

from game2048.board import ROW_LEFT, ROW_RIGHT, ROW_SCORE_LEFT, ROW_SCORE_RIGHT
from game2048.patterns import (
    PATTERN_CELLS, PATTERN_TABLE, N_LOOKUPS, TABLE_SIZE, N_PATTERNS,
    LARGE_TILE_CAP, LARGE_TILE_EXPONENTS, TOTAL_WEIGHTS, N_FEATURES,
    OFF_LARGE_TILE, OFF_EMPTY, OFF_DISTINCT, OFF_MERGEABLE, OFF_NEIGHBOUR,
)

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

# --- uint64 constants ------------------------------------------------------
U0, U1, U2, U3, U4, U12, U16, U24 = (np.uint64(v) for v in (0, 1, 2, 3, 4, 12, 16, 24))
U32, U48 = np.uint64(32), np.uint64(48)
MASK_CELL = np.uint64(0xF)
MASK_ROW = np.uint64(0xFFFF)

NIBBLE_HI = np.uint64(0x8888888888888888)     # bit 3 of every cell
NIBBLE_LO = np.uint64(0x1111111111111111)     # bit 0 of every cell
PTAB_BASE = (PATTERN_TABLE * TABLE_SIZE).astype(np.int64)   # flat offset of each table
LT_EXP = np.array(LARGE_TILE_EXPONENTS, dtype=np.uint64)
LT_CAP = np.int64(LARGE_TILE_CAP)
# cell i's right neighbour, then cell i's down neighbour; -1 where there is none
RIGHT_OF = np.array([1,2,3,-1, 5,6,7,-1, 9,10,11,-1, 13,14,15,-1], dtype=np.int64)
DOWN_OF = np.array([4,5,6,7, 8,9,10,11, 12,13,14,15, -1,-1,-1,-1], dtype=np.int64)

TA1, TA2, TA3 = (np.uint64(v) for v in (0xF0F00F0FF0F00F0F, 0x0000F0F00000F0F0, 0x0F0F00000F0F0000))
TB1, TB2, TB3 = (np.uint64(v) for v in (0xFF00FF0000FF00FF, 0x00FF00FF00000000, 0x00000000FF00FF00))

CELL_SHIFT = (np.arange(16) * 4).astype(np.uint64)
# (32, 6) bit offsets: PSHIFT[i, j] is where pattern instance i's j-th cell lives
PSHIFT = CELL_SHIFT[PATTERN_CELLS].copy()
NIBBLE = (np.arange(6) * 4).astype(np.int64)           # where each cell sits in the index
PTABLE = PATTERN_TABLE.copy()


def as_board(value):
    """The supported way to hand a board to these kernels from Python.

    Always use this at the boundary. A bare Python int works only while it fits
    in int64; the moment a 32768 tile lands in cell 15 the top bit is set and
    numba raises OverflowError. See tests/test_fast.py::check_python_boundary.
    """
    return np.uint64(value)


N_STAGES = 4
"""Yeh et al. 2016, Strategy 2: split at T16k, T16+8k and T16+8+4k.

A stage boundary is a *moment in a game*, not a property of a board: T16+8k is
"the first time both a 16384 and an 8192 have been on the board together". Once
passed you stay past it, even if the 8192 is later merged away. So the stage is
episode state that only ever increases -- see `advance_stage`.
"""

STAGE_REQUIREMENTS = (
    (14,),                  # T16k      -- a 16384 exists
    (14, 13),               # T16+8k    -- 16384 and 8192 together
    (14, 13, 12),           # T16+8+4k  -- and a 4096
)


def new_weights():
    """One stage's zeroed weights: 256 MB, tuple tables and extra features in one row.

    Training always works on a single stage at a time, so this is flat. Play
    time stacks the trained stages into a 2-D array -- see `load_weights`.
    """
    return np.zeros(TOTAL_WEIGHTS, dtype=np.float32)


def stage_path(path, stage):
    """weights/agent.npy + stage 2 -> weights/agent.stage2.npy"""
    root, ext = os.path.splitext(path)
    return f"{root}.stage{stage}{ext}"


def load_stage(path, stage):
    """One stage's flat weight row, or zeros if it has not been trained yet."""
    p = stage_path(path, stage)
    if not os.path.exists(p):
        return None
    w = np.load(p)
    assert w.dtype == np.float32, f"expected float32 weights, got {w.dtype}"
    assert w.shape == (TOTAL_WEIGHTS,), (
        f"{p} has shape {w.shape}, expected ({TOTAL_WEIGHTS},) -- the feature set "
        f"changed, so old checkpoints cannot be loaded")
    return w


def load_weights(path, stages=N_STAGES):
    """Stack the trained stages for play, stopping at the last one that exists.

    Only the stages actually trained are materialised. Each is 256 MB, and four
    copies of the same untrained fallback is a gigabyte of nothing. Play clamps
    the stage index to what was loaded, so a game running past the last trained
    stage keeps using the deepest weights available -- which is the right
    fallback, since a stage left at zero would teach the stage below it that
    crossing the boundary leads somewhere worthless.
    """
    rows = []
    for k in range(stages):
        w = load_stage(path, k)
        if w is None:
            break
        rows.append(w)
    if not rows:
        raise FileNotFoundError(
            f"no weights at {stage_path(path, 0)} -- train stage 0 first")
    if len(rows) < stages:
        print(f"stages {len(rows)}-{stages - 1} not trained; "
              f"play falls back to stage {len(rows) - 1}")
    return np.ascontiguousarray(np.stack(rows))


# --- board mechanics# --- board mechanics -------------------------------------------------------
@njit(cache=True, inline="always")
def transpose(board):
    b = np.uint64(board)
    a1 = b & TA1
    a2 = b & TA2
    a3 = b & TA3
    a = a1 | (a2 << U12) | (a3 >> U12)
    b1 = a & TB1
    b2 = a & TB2
    b3 = a & TB3
    return b1 | (b2 >> U24) | (b3 << U24)


@njit(cache=True, inline="always")
def _slide(board, table, score):
    r0 = board & MASK_ROW
    r1 = (board >> U16) & MASK_ROW
    r2 = (board >> U32) & MASK_ROW
    r3 = (board >> U48) & MASK_ROW
    out = (np.uint64(table[r0])
           | (np.uint64(table[r1]) << U16)
           | (np.uint64(table[r2]) << U32)
           | (np.uint64(table[r3]) << U48))
    gained = score[r0] + score[r1] + score[r2] + score[r3]
    return out, gained


@njit(cache=True)
def move(board, direction):
    """0=up 1=right 2=down 3=left. Returns (board, reward); unchanged = illegal."""
    b = np.uint64(board)
    if direction == 3:
        return _slide(b, ROW_LEFT, ROW_SCORE_LEFT)
    if direction == 1:
        return _slide(b, ROW_RIGHT, ROW_SCORE_RIGHT)
    t = transpose(b)
    if direction == 0:
        out, gained = _slide(t, ROW_LEFT, ROW_SCORE_LEFT)
    else:
        out, gained = _slide(t, ROW_RIGHT, ROW_SCORE_RIGHT)
    return transpose(out), gained


@njit(cache=True)
def is_terminal(board):
    b = np.uint64(board)
    for d in range(4):
        nb, _ = move(b, d)
        if nb != b:
            return False
    return True


@njit(cache=True)
def max_tile(board):
    b = np.uint64(board)
    best = U0
    for i in range(16):
        v = (b >> CELL_SHIFT[i]) & MASK_CELL
        if v > best:
            best = v
    return best


@njit(cache=True, inline="always")
def advance_stage(board, stage):
    """The stage this game is in, given where it was and what is on the board.

    Monotone by construction: each test only fires while the game is still below
    that stage, and nothing ever moves it back down.
    """
    b = np.uint64(board)
    s = stage
    if s == 0 and has_tile(b, 14):
        s = 1
    if s == 1 and has_tile(b, 14) and has_tile(b, 13):
        s = 2
    if s == 2 and has_tile(b, 14) and has_tile(b, 13) and has_tile(b, 12):
        s = 3
    return s


@njit(cache=True)
def spawn(board):
    """A 2 with probability 0.9, a 4 with probability 0.1, uniform empty cell."""
    b = np.uint64(board)
    n = 0
    for i in range(16):
        if ((b >> CELL_SHIFT[i]) & MASK_CELL) == U0:
            n += 1
    if n == 0:
        return b
    pick = np.random.randint(0, n)
    seen = 0
    for i in range(16):
        if ((b >> CELL_SHIFT[i]) & MASK_CELL) == U0:
            if seen == pick:
                value = U1 if np.random.random() < 0.9 else U2
                return b | (value << CELL_SHIFT[i])
            seen += 1
    return b


@njit(cache=True)
def new_game():
    return spawn(spawn(np.uint64(0)))


@njit(cache=True, inline="always")
def has_tile(board, exponent):
    """Is there a cell holding exactly this exponent?

    XOR every cell with the target, so a matching cell becomes zero, then use
    the classic has-zero-nibble test: a nibble is zero exactly when subtracting
    one borrows into its top bit while that bit was clear.
    """
    x = np.uint64(board) ^ (np.uint64(exponent) * NIBBLE_LO)
    return ((x - NIBBLE_LO) & ~x & NIBBLE_HI) != U0


@njit(cache=True, inline="always")
def _extras(board):
    """All five whole-board features in two sweeps instead of five.

    Written naively -- one loop per feature -- these cost more than the 32 table
    lookups they are meant to assist. Counting empties, distinct values and the
    large-tile histogram share a single pass over the cells; only the two
    adjacency counts need their own, because they read pairs.

    Returns (large_tile_index, empty, distinct, mergeable_pairs, ladder_pairs).
    """
    b = np.uint64(board)
    empty = np.int64(0)
    seen = np.uint64(0)
    n11 = np.int64(0); n12 = np.int64(0); n13 = np.int64(0)
    n14 = np.int64(0); n15 = np.int64(0)

    for i in range(16):
        v = (b >> CELL_SHIFT[i]) & MASK_CELL
        if v == U0:
            empty += 1
            continue
        seen |= U1 << v
        if v >= np.uint64(11):
            if v == np.uint64(11):
                n11 += 1
            elif v == np.uint64(12):
                n12 += 1
            elif v == np.uint64(13):
                n13 += 1
            elif v == np.uint64(14):
                n14 += 1
            elif v == np.uint64(15):
                n15 += 1

    distinct = np.int64(0)
    for k in range(16):
        if (seen >> np.uint64(k)) & U1:
            distinct += 1

    if n11 > LT_CAP: n11 = LT_CAP
    if n12 > LT_CAP: n12 = LT_CAP
    if n13 > LT_CAP: n13 = LT_CAP
    if n14 > LT_CAP: n14 = LT_CAP
    if n15 > LT_CAP: n15 = LT_CAP
    lt = ((((n11 * 8 + n12) * 8 + n13) * 8 + n14) * 8) + n15

    same = np.int64(0)
    ladder = np.int64(0)
    for i in range(16):
        v = (b >> CELL_SHIFT[i]) & MASK_CELL
        if v == U0:
            continue
        for which in range(2):
            j = RIGHT_OF[i] if which == 0 else DOWN_OF[i]
            if j < 0:
                continue
            u = (b >> CELL_SHIFT[j]) & MASK_CELL
            if u == U0:
                continue
            if u == v:
                same += 1
            elif u == v + U1 or v == u + U1:
                ladder += 1
    if same > 24:
        same = 24
    if ladder > 24:
        ladder = 24
    return lt, empty, distinct, same, ladder


# --- the value function ----------------------------------------------------
@njit(cache=True, inline="always")
def _index(board, i):
    """Pack the 6 cells that pattern instance i reads into one 24-bit index."""
    idx = np.int64(0)
    for j in range(6):
        cell = np.int64((np.uint64(board) >> PSHIFT[i, j]) & MASK_CELL)
        idx |= cell << NIBBLE[j]
    return idx


@njit(cache=True)
def evaluate(board, w):
    """V(board): 32 tuple lookups plus 5 whole-board features, all from `w`.

    `w` is one stage's flat weight row. Which stage that is belongs to the
    caller -- it depends on what the game has already done, not on what the
    board looks like now. See `Stages` in CLAUDE.md.
    """
    b = np.uint64(board)
    total = np.float32(0.0)
    for i in range(N_LOOKUPS):
        total += w[PTAB_BASE[i] + _index(b, i)]
    lt, empty, distinct, same, ladder = _extras(b)
    total += w[OFF_LARGE_TILE + lt]
    total += w[OFF_EMPTY + empty]
    total += w[OFF_DISTINCT + distinct]
    total += w[OFF_MERGEABLE + same]
    total += w[OFF_NEIGHBOUR + ladder]
    return total


@njit(cache=True)
def update(board, delta, w):
    """Apply one correction to every weight that produced the estimate.

    `delta` is already the step for a single weight (alpha * error), matching
    the paper's per-weight alpha of 0.0025 rather than a total divided by the
    feature count. Changing that changes the effective learning rate by ~37x.
    """
    b = np.uint64(board)
    d = np.float32(delta)
    for i in range(N_LOOKUPS):
        w[PTAB_BASE[i] + _index(b, i)] += d
    lt, empty, distinct, same, ladder = _extras(b)
    w[OFF_LARGE_TILE + lt] += d
    w[OFF_EMPTY + empty] += d
    w[OFF_DISTINCT + distinct] += d
    w[OFF_MERGEABLE + same] += d
    w[OFF_NEIGHBOUR + ladder] += d


@njit(cache=True)
def best_action(board, w):
    """Greedy 1-ply: the move maximising immediate reward + V(afterstate).

    Returns (direction, afterstate, reward, value_of_afterstate).
    direction is -1 when the game is over.
    """
    b = np.uint64(board)
    best_d = -1
    best_score = -np.inf
    best_after = b
    best_reward = np.int64(0)
    best_value = np.float32(0.0)
    for d in range(4):
        after, reward = move(b, d)
        if after == b:
            continue
        value = evaluate(after, w)
        total = reward + value
        if total > best_score:
            best_score = total
            best_d = d
            best_after = after
            best_reward = reward
            best_value = value
    return best_d, best_after, best_reward, best_value


# --- expectimax search -----------------------------------------------------
@njit(cache=True)
def _search(board, w, plies):
    """Best achievable value from `board` with `plies` decision levels left.

    A max node over the four moves; between two max nodes sits a *chance* node,
    because the tile that drops is not ours to choose -- so its children are
    averaged by probability rather than maximised. That alternation is what
    makes this expectimax rather than minimax.

    `plies == 1` is the greedy player: evaluate each afterstate and stop.

    The chance node is written inline rather than as its own function because
    numba compiles direct self-recursion but not mutual recursion.
    """
    best = -np.inf
    for d in range(4):
        after, reward = move(np.uint64(board), d)
        if after == np.uint64(board):
            continue
        if plies <= 1:
            value = np.float64(evaluate(after, w))
        else:
            expected = 0.0
            empty = 0
            for i in range(16):
                if ((after >> CELL_SHIFT[i]) & MASK_CELL) == U0:
                    empty += 1
                    expected += 0.9 * _search(after | (U1 << CELL_SHIFT[i]), w, plies - 1)
                    expected += 0.1 * _search(after | (U2 << CELL_SHIFT[i]), w, plies - 1)
            value = expected / empty if empty > 0 else 0.0
        total = reward + value
        if total > best:
            best = total
    if best == -np.inf:
        return 0.0                                  # no legal move: the game is over
    return best


@njit(cache=True)
def value_after(after, w, plies):
    """What an afterstate is worth with `plies` decision levels of lookahead.

    plies=1 is the plain table lookup. Deeper, it is the chance node: the average
    over every tile the game could drop next, weighted 0.9 for a 2 and 0.1 for a
    4, of the best the agent can then do. This is the number the viewers put on
    screen, so what you read is what the agent actually compared.
    """
    if plies <= 1:
        return np.float64(evaluate(after, w))
    a = np.uint64(after)
    expected = 0.0
    empty = 0
    for i in range(16):
        if ((a >> CELL_SHIFT[i]) & MASK_CELL) == U0:
            empty += 1
            expected += 0.9 * _search(a | (U1 << CELL_SHIFT[i]), w, plies - 1)
            expected += 0.1 * _search(a | (U2 << CELL_SHIFT[i]), w, plies - 1)
    if empty == 0:
        return 0.0
    return expected / empty


@njit(cache=True)
def best_action_search(board, w, plies):
    """`best_action`, but looking `plies` decisions ahead over the tile spawns.

    Same return shape as `best_action` so callers can swap one for the other.
    plies=1 is exactly `best_action` and costs 4 evaluations; plies=2 costs a few
    hundred, which is fine at play time and ruinous during training.
    """
    b = np.uint64(board)
    best_d = -1
    best_score = -np.inf
    best_after = b
    best_reward = np.int64(0)
    best_value = np.float32(0.0)
    for d in range(4):
        after, reward = move(b, d)
        if after == b:
            continue
        value = value_after(after, w, plies)
        total = reward + value
        if total > best_score:
            best_score = total
            best_d = d
            best_after = after
            best_reward = reward
            best_value = np.float32(value)
    return best_d, best_after, best_reward, best_value


@njit(cache=True)
def play_searched(w, plies):
    """One game played with search and no learning -- how strong it really is."""
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    while True:
        d, after, reward, _ = best_action_search(board, w, plies)
        if d < 0:
            break
        score += reward
        moves += 1
        board = spawn(after)
    return score, moves, np.int64(max_tile(board))


# --- learning --------------------------------------------------------------
LAMBDA_WEIGHTS = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625], dtype=np.float64)
"""Yeh et al. formula (6): a truncated TD(lambda), lambda=0.5, n-step returns
n=1..5, weights summing to 1. No eligibility traces -- with millions of
features and thousands of moves per game, traces cost more than they earn, so
the whole trajectory is recorded and the updates applied once the game ends."""

MAX_MOVES = 60000


def new_buffers(size=MAX_MOVES):
    """Trajectory buffers for TD(lambda). Allocate once, reuse every game."""
    return (np.zeros(size, dtype=np.uint64),
            np.zeros(size, dtype=np.int64),
            np.zeros(size, dtype=np.float64))


NO_MASK = np.zeros(N_STAGES, dtype=np.int64)
NO_CAUGHT = np.zeros((N_STAGES, 2), dtype=np.uint64)
NO_FLAGS = np.zeros(N_STAGES, dtype=np.int64)
"""Placeholders for callers that are training, not harvesting."""


def catch_buffers(stages):
    """(mask, scratch board/score rows, scratch flags) for harvesting `stages`."""
    mask = np.zeros(N_STAGES, dtype=np.int64)
    for k in stages:
        mask[k] = 1
    return mask, np.zeros((N_STAGES, 2), dtype=np.uint64), np.zeros(N_STAGES, dtype=np.int64)


def learn(w, alpha, lam, buffers):
    """`play_and_learn` for the common case: fresh game, nothing collected."""
    return play_and_learn(w, alpha, lam, np.uint64(0), np.int64(0), *buffers,
                          NO_MASK, NO_CAUGHT, NO_FLAGS)


@njit(cache=True)
def play_and_learn(w, alpha, lam, start_board, start_score, afters, rewards, values,
                   catch_mask, caught, caught_flags):
    """One self-play game, learning as it goes. Returns (score, moves, max_tile).

    Learns the value of *afterstates* -- the board right after your slide,
    before the random tile appears (Szubert & Jaskowski 2014). That removes the
    spawn randomness from the thing being learned.

    `start_board` of 0 begins a fresh game; anything else resumes from a
    collected position, which is how a later stage gets trained on millions of
    games that all begin where that stage actually applies.

Wherever `catch_mask[k]` is set, the game snapshots the first board that enters
    stage k into `caught[k]` and records it in `caught_flags[k]`, without
    interrupting play -- the game trains to the end as normal. One game can fire
    several of these, since it passes through the stages in order.

    That harvests restart positions out of games being played anyway, instead of
    paying for a separate collection pass. Two caveats, both real: the weights
    keep changing, so the corpus mixes positions from players of different
    strengths (the game index is stored alongside, so a later stage can restrict
    itself to a strong window); and boards for stage k+1 ideally come from the
    trained stage-k player, which does not exist yet while stage 0 is training.
    The deeper the stage, the more that matters.

    `lam` <= 0 selects online TD(0):

        V(s'_t)  <-  V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]

    Otherwise the trajectory is recorded and a truncated lambda-return is
    applied after the game ends, with the weights held still for the whole
    episode so every target is computed against the same value function.
    """
    if start_board == np.uint64(0):
        board = new_game()
        score = np.int64(0)
    else:
        board = np.uint64(start_board)
        score = np.int64(start_score)

    moves = np.int64(0)
    have_prev = False
    prev_after = np.uint64(0)
    stage = 0
    watching = False
    for k in range(len(catch_mask)):
        caught_flags[k] = 0
        if catch_mask[k] != 0:
            watching = True

    while moves < MAX_MOVES:
        d, after, reward, value = best_action(board, w)
        if d < 0:
            break
        if watching:
            new_stage = advance_stage(after, stage)
            if new_stage > stage:
                # one move can cross more than one boundary
                for k in range(stage + 1, new_stage + 1):
                    if catch_mask[k] != 0 and caught_flags[k] == 0:
                        caught[k, 0] = after
                        caught[k, 1] = np.uint64(score + reward)
                        caught_flags[k] = 1
                stage = new_stage
        if lam > 0.0:
            afters[moves] = after
            rewards[moves] = reward
        elif have_prev:
            error = (reward + value) - evaluate(prev_after, w)
            update(prev_after, alpha * error, w)
        prev_after = after
        have_prev = True
        score += reward
        moves += 1
        board = spawn(after)

    if lam <= 0.0:
        if have_prev:
            update(prev_after, alpha * (0.0 - evaluate(prev_after, w)), w)
        return score, moves, np.int64(max_tile(board))

    # --- truncated TD(lambda), applied now that the game is over -------------
    n = moves
    for t in range(n):
        values[t] = evaluate(afters[t], w)

    for t in range(n):
        target = 0.0
        acc = 0.0
        full = 0.0                      # the return once the horizon runs past the end
        have_full = False
        for k in range(5):
            j = t + k + 1
            if j < n:
                acc += rewards[j]
                rn = acc + values[j]
            elif j == n:
                rn = acc                # V of a terminal afterstate is 0
                full = rn
                have_full = True
            else:
                rn = full if have_full else acc
            target += LAMBDA_WEIGHTS[k] * rn
        update(afters[t], alpha * (target - values[t]), w)

    return score, moves, np.int64(max_tile(board))


@njit(cache=True)
def collect_boards(w, starts, n_starts, stage, out, limit, max_games):
    """Play with frozen weights, snapshotting the moment `stage` is entered.

    `out` is (limit, 2): column 0 the board, column 1 the score so far. The
    score matters as much as the board -- Yeh et al. carry it into the next
    stage so a stage-2 game is scored from the start of the game that produced
    its opening position, not from that position. Without it a later stage's
    learning curve is not comparable with an earlier one, and that comparison is
    the entire basis of the paper's evaluation.

    The game is abandoned once the snapshot is taken: the next stage plays
    forward from here, so anything after the boundary is wasted work.

    A game resumed from `starts` inherits that entry's score, so scores
    accumulate correctly across a chain of collections.

    Returns (found, games).
    """
    found = np.int64(0)
    games = np.int64(0)
    while found < limit and games < max_games:
        if n_starts > 0:
            idx = games % n_starts
            board = np.uint64(starts[idx, 0])
            score = np.int64(starts[idx, 1])
        else:
            board = new_game()
            score = np.int64(0)
        games += 1
        moves = np.int64(0)
        while moves < MAX_MOVES:
            d, after, reward, _ = best_action(board, w)
            if d < 0:
                break
            score += reward
            moves += 1
            if advance_stage(after, stage - 1) >= stage:
                out[found, 0] = after
                out[found, 1] = np.uint64(score)
                found += 1
                break
            board = spawn(after)
    return found, games


# --- playing ---------------------------------------------------------------
@njit(cache=True)
def play_staged(weights, plies):
    """One game using the right stage's weights at every point. No learning."""
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    stage = 0
    top = weights.shape[0] - 1
    while moves < MAX_MOVES:
        stage = advance_stage(board, stage)
        d, after, reward, _ = best_action_search(
            board, weights[stage if stage < top else top], plies)
        if d < 0:
            break
        score += reward
        moves += 1
        stage = advance_stage(after, stage)
        board = spawn(after)
    return score, moves, np.int64(max_tile(board))


@njit(cache=True)
def play_reported(weights, plies):
    """`play_staged`, also reporting the furthest stage the game reached.

    Which stage a game gets to is the thing worth watching in a multi-stage
    agent: a stage almost no game reaches is a table that never gets trained.
    """
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    stage = 0
    top = weights.shape[0] - 1
    while moves < MAX_MOVES:
        stage = advance_stage(board, stage)
        d, after, reward, _ = best_action_search(
            board, weights[stage if stage < top else top], plies)
        if d < 0:
            break
        score += reward
        moves += 1
        stage = advance_stage(after, stage)
        board = spawn(after)
    return score, moves, np.int64(max_tile(board)), np.int64(stage)


@njit(cache=True)
def play_greedy(w):
    """One game with a single stage's weights, greedily -- the training player."""
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    while moves < MAX_MOVES:
        d, after, reward, _ = best_action(board, w)
        if d < 0:
            break
        score += reward
        moves += 1
        board = spawn(after)
    return score, moves, np.int64(max_tile(board))


@njit(cache=True)
def play_searched(w, plies):
    """One game with a single stage's weights and `plies` of lookahead."""
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    while moves < MAX_MOVES:
        d, after, reward, _ = best_action_search(board, w, plies)
        if d < 0:
            break
        score += reward
        moves += 1
        board = spawn(after)
    return score, moves, np.int64(max_tile(board))


@njit(cache=True)
def play_random():
    """No value function at all -- the floor everything is measured against."""
    board = new_game()
    score = np.int64(0)
    moves = np.int64(0)
    while True:
        legal = np.empty(4, dtype=np.int64)
        n = 0
        for d in range(4):
            nb, _ = move(board, d)
            if nb != board:
                legal[n] = d
                n += 1
        if n == 0:
            break
        after, reward = move(board, legal[np.random.randint(0, n)])
        score += reward
        moves += 1
        board = spawn(after)
    return score, moves, np.int64(max_tile(board))
