"""Stage 1 -- the whole board in one 64-bit integer.

A 2048 board has 16 cells and every cell holds a power of two. Instead of
storing the tile value (2, 4, 8, ... 32768) we store its *exponent* (1, 2, 3,
... 15), which fits in 4 bits. Sixteen cells x 4 bits = 64 bits = one integer.

    cell index i  ->  bits 4*i .. 4*i+3
    row r         ->  bits 16*r .. 16*r+15     (a row is a uint16)

    index layout          bit layout (low bits on the right)
     0  1  2  3
     4  5  6  7           [ cell15 ][ cell14 ] ... [ cell1 ][ cell0 ]
     8  9 10 11
    12 13 14 15

Because a row is only 16 bits wide there are just 65536 possible rows, so we
can precompute the answer to "what does this row look like after sliding left,
and how many points did that score?" for *every* row, once, at import time.
A move then costs four array lookups instead of four Python loops.

Directions are 0=up 1=right 2=down 3=left throughout the package ("URDL").
"""

import numpy as np

ROW_MASK = 0xFFFF
CELL_MASK = 0xF

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRECTION_NAMES = ("up", "right", "down", "left")


def _slide_line_left(line):
    """The naive row slide, run once per possible row at import time."""
    tiles = [v for v in line if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1] and tiles[i] < 15:
            out.append(tiles[i] + 1)          # exponents merge by adding one
            score += 1 << (tiles[i] + 1)      # ...but the score is the tile value
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (4 - len(out)))
    return out, score


def _reverse_row(row):
    return ((row >> 12) & 0xF) | ((row >> 4) & 0xF0) | ((row << 4) & 0xF00) | ((row << 12) & 0xF000)


def _build_tables():
    """One pass over all 65536 rows fills every table we will ever need."""
    left = np.zeros(65536, dtype=np.uint16)
    right = np.zeros(65536, dtype=np.uint16)
    score_left = np.zeros(65536, dtype=np.int32)
    score_right = np.zeros(65536, dtype=np.int32)
    for row in range(65536):
        line = [(row >> (4 * c)) & CELL_MASK for c in range(4)]
        out, gained = _slide_line_left(line)
        packed = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)
        rev = _reverse_row(row)
        left[row] = packed
        score_left[row] = gained
        # sliding row X right is sliding reverse(X) left, then reversing back
        right[rev] = _reverse_row(packed)
        score_right[rev] = gained
    return left, right, score_left, score_right


ROW_LEFT, ROW_RIGHT, ROW_SCORE_LEFT, ROW_SCORE_RIGHT = _build_tables()


def transpose(board):
    """Swap rows and columns with bit masks -- no loop, ~12 machine ops.

    Works as two interleaved swaps: first exchange the 4-bit diagonals inside
    each 2x2 block of cells, then exchange the 2x2 blocks themselves.
    """
    a1 = board & 0xF0F00F0FF0F00F0F
    a2 = board & 0x0000F0F00000F0F0
    a3 = board & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def _slide_rows(board, table, score_table):
    result = 0
    reward = 0
    for r in range(4):
        row = (board >> (16 * r)) & ROW_MASK
        result |= int(table[row]) << (16 * r)
        reward += int(score_table[row])
    return result, reward


def move(board, direction):
    """Apply a move. Returns (new_board, reward). new_board == board means illegal."""
    if direction == LEFT:
        return _slide_rows(board, ROW_LEFT, ROW_SCORE_LEFT)
    if direction == RIGHT:
        return _slide_rows(board, ROW_RIGHT, ROW_SCORE_RIGHT)
    # up/down are left/right on the transposed board
    t = transpose(board)
    if direction == UP:
        moved, reward = _slide_rows(t, ROW_LEFT, ROW_SCORE_LEFT)
    else:
        moved, reward = _slide_rows(t, ROW_RIGHT, ROW_SCORE_RIGHT)
    return transpose(moved), reward


def cell(board, i):
    return (board >> (4 * i)) & CELL_MASK


def cells(board):
    return [(board >> (4 * i)) & CELL_MASK for i in range(16)]


def set_cell(board, i, value):
    return (board & ~(CELL_MASK << (4 * i))) | (value << (4 * i))


def empty_cells(board):
    return [i for i in range(16) if not (board >> (4 * i)) & CELL_MASK]


def max_tile(board):
    return max(cells(board))


def score_of(board):
    """Total merge score implied by the tiles on the board.

    Building a 2^k tile costs (k-1) merges of total value (k-1)*2^k, which is
    exactly the score 2048 would have shown. Handy for sanity checks.
    """
    return sum((k - 1) << k for k in cells(board) if k >= 2)


def spawn(board, rng):
    """Place a 2 (90%) or a 4 (10%) in a random empty cell."""
    empty = empty_cells(board)
    if not empty:
        return board
    i = empty[rng.randrange(len(empty))]
    value = 1 if rng.random() < 0.9 else 2
    return set_cell(board, i, value)


def new_game(rng):
    board = spawn(0, rng)
    return spawn(board, rng)


def legal_moves(board):
    return [d for d in range(4) if move(board, d)[0] != board]


def is_terminal(board):
    for d in range(4):
        if move(board, d)[0] != board:
            return False
    return True


def to_grid(board):
    """16-cell board -> 4x4 list of tile values, for printing."""
    return [[(1 << k) if k else 0 for k in cells(board)[r * 4:r * 4 + 4]] for r in range(4)]


def from_grid(grid):
    board = 0
    for r in range(4):
        for c in range(4):
            v = grid[r][c]
            k = v.bit_length() - 1 if v else 0
            board |= k << (4 * (r * 4 + c))
    return board


def render(board):
    lines = ["+------+------+------+------+"]
    for row in to_grid(board):
        lines.append("|" + "|".join(f"{v or '':^6}" for v in row) + "|")
        lines.append("+------+------+------+------+")
    return "\n".join(lines)
