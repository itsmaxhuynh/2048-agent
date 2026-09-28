"""What the agent actually looks at.

We never learn a value for a whole board -- there are far too many. Instead we
learn values for small *patterns* of cells (n-tuples) and add them up:

    V(board) = sum over patterns of  weight_table[pattern][cells the pattern sees]

Four 6-cell patterns is the set from Wu et al., "Multi-stage Temporal Difference
Learning for 2048-like Games" (2014). Six cells x 4 bits = a 24-bit index, so
each pattern needs a table of 16^6 = 16.7M floats. Four of those is 256 MB --
big, but it fits in RAM, and every lookup is O(1).

2048 does not care which way up the board is, so each pattern is also read
under all 8 symmetries of the square (4 rotations x 2 reflections). All 8 share
one weight table, which is what makes the network learn 8x faster from the same
game: one observed position teaches eight.
"""

import numpy as np

# Cell indices, as laid out in board.py:
#    0  1  2  3
#    4  5  6  7
#    8  9 10 11
#   12 13 14 15
BASE_PATTERNS = [
    (0, 1, 2, 3, 4, 5),        # top two rows, left-leaning
    (4, 5, 6, 7, 8, 9),        # middle band
    (0, 1, 2, 4, 5, 6),        # 3x2 block, top-left
    (4, 5, 6, 8, 9, 10),       # 3x2 block, centre
]

N_CELLS = 6
TABLE_SIZE = 16 ** N_CELLS     # 16,777,216 entries per pattern
N_PATTERNS = len(BASE_PATTERNS)


def _identity_grid():
    return [[r * 4 + c for c in range(4)] for r in range(4)]


def _rotate(grid):
    """Rotate the coordinate grid 90 degrees clockwise."""
    return [[grid[3 - c][r] for c in range(4)] for r in range(4)]


def _flip(grid):
    """Mirror the coordinate grid left-to-right."""
    return [row[::-1] for row in grid]


def symmetry_grids():
    """The 8 symmetries of the square, as coordinate remappings."""
    grids = []
    g = _identity_grid()
    for _ in range(4):
        grids.append(g)
        grids.append(_flip(g))
        g = _rotate(g)
    return grids


def build_pattern_table():
    """Returns (cells, table_id).

    cells    : (32, 6) int64 -- the board cells each pattern instance reads
    table_id : (32,)   int64 -- which of the 4 weight tables it reads them from
    """
    grids = symmetry_grids()
    cells, table_id = [], []
    for p, pattern in enumerate(BASE_PATTERNS):
        for grid in grids:
            flat = [grid[i // 4][i % 4] for i in range(16)]
            cells.append([flat[c] for c in pattern])
            table_id.append(p)
    return np.array(cells, dtype=np.int64), np.array(table_id, dtype=np.int64)


PATTERN_CELLS, PATTERN_TABLE = build_pattern_table()
N_LOOKUPS = len(PATTERN_CELLS)          # 4 patterns x 8 symmetries = 32

# --- extra features (Yeh et al. 2016, section IV.A and IV.E) ----------------
#
# The tuple tables see six cells at a time and cannot express anything global.
# These five features do: how crowded the board is, how much of it is
# mergeable, and -- the important one -- exactly which large tiles are present,
# which is what tells the network that a position is *difficult* rather than
# merely high-scoring.
#
# All the weights live in one flat array per stage so there is a single
# checkpoint and a single numba array to pass around. The tuple tables occupy
# the first N_PATTERNS * TABLE_SIZE slots; each feature below owns the block
# starting at its offset.

LARGE_TILE_CAP = 7                      # counts saturate here; 8 values each
LARGE_TILE_EXPONENTS = (11, 12, 13, 14, 15)      # 2048, 4096, 8192, 16384, 32768
LARGE_TILE_SIZE = 8 ** len(LARGE_TILE_EXPONENTS)  # 32,768
EMPTY_SIZE = 17                         # 0..16 empty cells
DISTINCT_SIZE = 17                      # 0..16 distinct tile values
MERGEABLE_SIZE = 25                     # 0..24 adjacent equal pairs
NEIGHBOUR_SIZE = 25                     # 0..24 adjacent (v, 2v) pairs

TUPLE_WEIGHTS = N_PATTERNS * TABLE_SIZE          # 67,108,864
OFF_LARGE_TILE = TUPLE_WEIGHTS
OFF_EMPTY = OFF_LARGE_TILE + LARGE_TILE_SIZE
OFF_DISTINCT = OFF_EMPTY + EMPTY_SIZE
OFF_MERGEABLE = OFF_DISTINCT + DISTINCT_SIZE
OFF_NEIGHBOUR = OFF_MERGEABLE + MERGEABLE_SIZE
TOTAL_WEIGHTS = OFF_NEIGHBOUR + NEIGHBOUR_SIZE   # 67,141,716 -> 268 MB float32

N_EXTRA = 5
N_FEATURES = N_LOOKUPS + N_EXTRA        # lookups summed for one position


def describe():
    lines = [
        f"{N_PATTERNS} patterns x 8 symmetries = {N_LOOKUPS} lookups per position",
        f"{N_PATTERNS} tables x {TABLE_SIZE:,} float32 = "
        f"{N_PATTERNS * TABLE_SIZE * 4 / 2**20:.0f} MB of weights",
        f"+ {N_EXTRA} global features ({TOTAL_WEIGHTS - TUPLE_WEIGHTS:,} more weights)",
        f"= {N_FEATURES} lookups per position, {TOTAL_WEIGHTS * 4 / 2**20:.0f} MB per stage",
        "",
    ]
    for p, pattern in enumerate(BASE_PATTERNS):
        grid = ["." * 4 for _ in range(4)]
        marks = [list("....") for _ in range(4)]
        for cellno in pattern:
            marks[cellno // 4][cellno % 4] = "#"
        lines.append(f"pattern {p}: cells {pattern}")
        lines.extend("    " + "".join(row) for row in marks)
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
