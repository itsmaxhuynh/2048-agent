"""Watch the agent play, in the terminal.

    python -m game2048.play --weights weights/agent.npy --delay 0.08
    python -m game2048.play --weights weights/agent.npy --eval 200     # no display

The terminal view exists for a reason a prettier UI does not serve: next to the
board it prints what the agent thinks each of the four moves is worth. When the
agent does something baffling, this is the screen that tells you whether the
value function is wrong or the move choice is. Every bug in this project was found
here.

    up      +8   V  41210.3   total  41218.3   <-
    right   +0   V  39880.1   total  39880.1
    down    +4   V  40551.7   total  40555.7
    left     -   illegal
"""

import argparse
import os
import sys
import time

import numpy as np

from game2048 import fast
from game2048.board import DIRECTION_NAMES

# xterm-256 background/foreground pairs, roughly the colours of the real game
TILE_COLOURS = {
    0: (236, 240), 1: (223, 236), 2: (222, 236), 3: (215, 236), 4: (209, 231),
    5: (203, 231), 6: (196, 231), 7: (220, 231), 8: (214, 231), 9: (208, 231),
    10: (202, 231), 11: (226, 231), 12: (33, 231), 13: (27, 231), 14: (21, 231),
    15: (55, 231),
}
RESET = "\033[0m"


def paint(exponent):
    bg, fg = TILE_COLOURS.get(exponent, (55, 231))
    text = "" if exponent == 0 else str(1 << exponent)
    return f"\033[48;5;{bg}m\033[38;5;{fg}m{text:^7}{RESET}"


def blank(exponent):
    """A coloured spacer row, so each tile is three terminal rows tall."""
    bg, _ = TILE_COLOURS.get(exponent, (55, 231))
    return f"\033[48;5;{bg}m{'':^7}{RESET}"


def render(board, score, moves, evaluations=None, chosen=-1, stage=0):
    cells = [int((int(board) >> (4 * i)) & 0xF) for i in range(16)]
    largest = 1 << max(cells) if max(cells) else 0

    lines = [
        f"  score {score:<10,}  moves {moves:<7,}  best tile {largest:<7,}"
        f"  stage {stage + 1}/{fast.N_STAGES}",
        "",
    ]
    for r in range(4):
        row = cells[r * 4:r * 4 + 4]
        lines.append("  " + "".join(blank(v) for v in row))
        lines.append("  " + "".join(paint(v) for v in row))
        lines.append("  " + "".join(blank(v) for v in row))
    lines.append("")

    if evaluations is not None:
        for d in range(4):
            entry = evaluations[d]
            name = DIRECTION_NAMES[d]
            if entry is None:
                lines.append(f"  {name:<7} {'-':>5}   illegal" + " " * 34)
            else:
                reward, value = entry
                mark = "  <- chosen" if d == chosen else " " * 11
                lines.append(f"  {name:<7} {reward:>+5}   V {value:>12,.1f}   "
                             f"total {reward + value:>12,.1f}{mark}")
    return "\n".join(lines)


def evaluate_moves(board, weights, depth=1, stage=0):
    """What the agent sees: reward + value for each legal direction.

    At depth 1 the value is a straight table lookup, V(afterstate). Deeper, it is
    the expectimax value -- averaged over every tile the game could drop next --
    so the numbers on screen are the ones the agent actually compared.
    """
    out = []
    b = fast.as_board(board)
    for d in range(4):
        after, reward = fast.move(b, d)
        after = fast.as_board(after)      # bit 63 is set whenever cell 15 holds >= 256
        if after == b:
            out.append(None)
        elif depth <= 1:
            out.append((int(reward), float(fast.evaluate(after, weights[min(stage, len(weights) - 1)]))))
        else:
            out.append((int(reward), float(fast.value_after(after, weights[min(stage, len(weights) - 1)], depth))))
    return out


def watch(weights, delay=0.08, seed=None, depth=1):
    if seed is not None:
        np.random.seed(seed)
    board = fast.as_board(fast.new_game())
    score = moves = stage = 0
    print("\033[2J", end="")                     # clear once, then repaint in place
    while True:
        stage = int(fast.advance_stage(board, stage))
        evaluations = evaluate_moves(board, weights, depth, stage)
        best_d, after, reward, _ = fast.best_action_search(board, weights[min(stage, len(weights) - 1)], depth)
        if best_d < 0:
            break
        print("\033[H" + render(board, score, moves, evaluations, best_d, stage), flush=True)
        time.sleep(delay)
        score += int(reward)
        moves += 1
        after = fast.as_board(after)
        stage = int(fast.advance_stage(after, stage))
        board = fast.as_board(fast.spawn(after))
    print("\033[H" + render(board, score, moves, stage=stage) + "\n\n  game over\n", flush=True)
    return score, moves, int(fast.max_tile(board))


def evaluate_strength(weights, games, depth=1):
    """Play silently and report how strong the agent actually is."""
    scores, tiles = [], []
    t0 = time.time()
    if depth > 1:
        print(f"  searching {depth} plies deep -- expect this to be slow", flush=True)
    for i in range(games):
        score, _, tile = fast.play_staged(weights, depth)
        scores.append(int(score))
        tiles.append(int(tile))
        if (i + 1) % max(1, games // 20) == 0:
            done = i + 1
            print(f"\r  {done}/{games} games  avg {np.mean(scores):,.0f}", end="", flush=True)
    print()
    scores = np.array(scores)
    tiles = np.array(tiles)
    print(f"\n  games        {games:,}")
    print(f"  average      {scores.mean():,.0f}")
    print(f"  median       {np.median(scores):,.0f}")
    print(f"  best         {scores.max():,}")
    print(f"  speed        {games / (time.time() - t0):.2f} games/sec")
    print(f"  search depth {depth} {'(greedy)' if depth <= 1 else 'plies'}\n")
    print("  reach rates (share of games that built this tile):")
    for k in range(7, 17):
        share = float((tiles >= k).mean())
        if share > 0:
            bar = "#" * int(share * 40)
            print(f"    {1 << k:>6,}  {share * 100:>5.1f}%  {bar}")
    return scores, tiles


def main(argv=None):
    p = argparse.ArgumentParser(description="Watch or measure a trained 2048 agent.")
    p.add_argument("--weights", default="weights/agent.npy")
    p.add_argument("--delay", type=float, default=0.08, help="seconds between moves")
    p.add_argument("--eval", type=int, default=0, metavar="N",
                   help="play N games with no display and report strength")
    p.add_argument("--games", type=int, default=1, help="games to watch")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--depth", type=int, default=1, metavar="PLIES",
                   help="expectimax decision levels (1 = greedy, the training-time "
                        "player; 2-3 are much stronger and much slower)")
    args = p.parse_args(argv)

    if not os.path.exists(args.weights):
        print(f"no weights at {args.weights} -- train first:\n"
              f"  python -m game2048.train --games 50000", file=sys.stderr)
        return 1
    print(f"loading {args.weights} ...", flush=True)
    weights = fast.load_weights(args.weights)

    if args.eval:
        evaluate_strength(weights, args.eval, args.depth)
        return 0

    for g in range(args.games):
        score, moves, tile = watch(weights, args.delay, args.seed, args.depth)
        print(f"  game {g + 1}: score {score:,}  moves {moves:,}  best tile {1 << tile:,}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
