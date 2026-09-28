"""Stage 0 -- 2048 the obvious way.

This is the implementation almost everyone writes first: the board is a list of
lists holding the *tile values* you see on screen (0, 2, 4, 8, ...), and a move
is a loop over rows.

It is correct, it is readable, and it is far too slow to train a reinforcement
learning agent with. It exists so that `bench.py` can put a real number on
"far too slow", and so the talk has something to optimise away from.

Nothing else in the package imports this module.
"""

import random

SIZE = 4


def new_board():
    board = [[0] * SIZE for _ in range(SIZE)]
    spawn(board)
    spawn(board)
    return board


def spawn(board, rng=random):
    """Drop a 2 (90%) or a 4 (10%) into a uniformly chosen empty cell."""
    empty = [(r, c) for r in range(SIZE) for c in range(SIZE) if board[r][c] == 0]
    if not empty:
        return False
    r, c = empty[rng.randrange(len(empty))]
    board[r][c] = 2 if rng.random() < 0.9 else 4
    return True


def _slide_row_left(row):
    """Compact a single row toward index 0, merging each tile at most once."""
    tiles = [v for v in row if v]
    out, score, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] * 2
            out.append(merged)
            score += merged
            i += 2          # both tiles are consumed, so the new tile cannot merge again
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (SIZE - len(out)))
    return out, score


def move(board, direction):
    """Apply a move. direction: 0=up 1=right 2=down 3=left.

    Returns (new_board, reward, changed).
    """
    rows = _to_left_rows(board, direction)
    new_rows, reward = [], 0
    for row in rows:
        out, gained = _slide_row_left(row)
        new_rows.append(out)
        reward += gained
    new_board = _from_left_rows(new_rows, direction)
    return new_board, reward, new_board != board


def _to_left_rows(board, direction):
    """Re-view the board as four rows that all want to slide toward index 0."""
    if direction == 3:                                  # left
        return [row[:] for row in board]
    if direction == 1:                                  # right
        return [row[::-1] for row in board]
    if direction == 0:                                  # up -> columns, top first
        return [[board[r][c] for r in range(SIZE)] for c in range(SIZE)]
    return [[board[r][c] for r in reversed(range(SIZE))] for c in range(SIZE)]


def _from_left_rows(rows, direction):
    if direction == 3:
        return [row[:] for row in rows]
    if direction == 1:
        return [row[::-1] for row in rows]
    if direction == 0:
        return [[rows[c][r] for c in range(SIZE)] for r in range(SIZE)]
    return [[rows[c][SIZE - 1 - r] for c in range(SIZE)] for r in range(SIZE)]


def legal_moves(board):
    return [d for d in range(4) if move(board, d)[2]]


def is_terminal(board):
    return not legal_moves(board)


def play_random_game(rng=random):
    """Returns (score, moves, max_tile)."""
    board = new_board()
    score = moves = 0
    while True:
        options = legal_moves(board)
        if not options:
            break
        board, reward, _ = move(board, options[rng.randrange(len(options))])
        spawn(board, rng)
        score += reward
        moves += 1
    return score, moves, max(max(row) for row in board)


if __name__ == "__main__":
    s, m, t = play_random_game()
    print(f"random game: score={s} moves={m} max tile={t}")
