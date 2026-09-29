# 2048, learned in Python

A 2048 agent that teaches itself to play, written for a talk at
[Python Adelaide](https://www.meetup.com/en-au/pythonadelaide/events/316478346/).
No neural network: the value function is four lookup tables, and learning is
adding a number to 32 array cells.

After 146,000 self-play games on a laptop it reaches the 2048 tile in 93% of
games, 4096 in 80%, and 8192 in 26%, averaging ~81,000 points.

## Quick start

```sh
python3 -m venv .venv && ./.venv/bin/pip install numpy numba

./.venv/bin/python -m tests.test_board      # correctness net
./.venv/bin/python -m tests.test_fast

./.venv/bin/python -m game2048.bench        # the optimisation ladder
./.venv/bin/python -m game2048.train --games 50000   # train (checkpoints as it goes)
```

`numba` is optional -- everything runs without it, about 78x slower.

## Training

Training follows the paper's multi-stage recipe. Stage 0 learns from empty
boards. Each later stage starts from boards where the stage below first reached
a milestone (16384; then 16384 + 8192; then 16384 + 8192 + 4096), so it spends
its games on the endgame instead of replaying the opening. Each stage is one
256 MB file, `<stem>.stage0.npy` to `<stem>.stage3.npy`, under `weights/`.

### The whole ladder, one command

```sh
./scripts/curriculum.sh           # the paper's step schedule -> weights/agent.*
./scripts/curriculum-cosine.sh    # cosine annealing           -> weights/cosine.*
```

These run for many hours. Progress goes to `talk/curriculum*.log` and the
learning curves to `talk/*curve-stage*.csv`. You can kill them at any point:
a checkpoint is written every 10 minutes, and re-running the script resumes
from where it stopped.

`curriculum.sh` begins at stage 1, so train stage 0 first (next section).
`curriculum-cosine.sh` does all four stages itself.

The strongest checkpoint, `cosine2m`, is the cosine script at 2M games per stage.
It took about 20 hours on a 4-core laptop:

```sh
STEM=weights/cosine2m.npy ALPHA_FINAL=0.0001 \
G0=2000000 G1=2000000 G2=2000000 G3=2000000 \
    ./scripts/curriculum-cosine.sh
```

### One stage at a time

```sh
# stage 0 from scratch; the paper's schedule drops alpha once the curve flattens
./.venv/bin/python -m game2048.train --stage 0 --games 2000000 \
    --auto-decay --stop-when-saturated --log talk/curve-stage0.csv

# play the finished stage 0, saving the boards that cross into stage 1
./.venv/bin/python -m game2048.train --collect 1 --max-restarts 50000

# train stage 1 from those boards; repeat collect + train for stages 2 and 3
./.venv/bin/python -m game2048.train --stage 1 --games 2000000 \
    --auto-decay --stop-when-saturated --log talk/curve-stage1.csv
```

Useful flags:

* `--out weights/NAME.npy` sets the checkpoint stem. The default is
  `weights/agent.npy`.
* `--resume` continues an existing stage instead of starting it from zero.
* `--schedule cosine --anneal-over N` anneals alpha smoothly over N games,
  replacing the single drop. The ramp follows the stage's total game count, so
  pass `--anneal-over` again whenever you `--resume`.
* `--alpha` is the step size for each weight (default 0.0025, the paper's
  value), not a total spread across the 32 lookups.

A run holds just one stage's 256 MB table in memory: training loads the stage
it trains, and collecting loads the stage below. Ctrl-C finishes the current
game, saves, and exits; on its own, `train` checkpoints every 5 minutes
(`--save-every`, in seconds).

Check a run with `scripts/eval_parallel.py`. `play.py --eval` won't work here:
it looks for the stem file itself, and staged training only writes the
`.stageK` files.

## Watching it play

Two viewers, same agent. Both take `--weights` (the *base* name: the real files
carry a stage suffix, so `weights/cosine2m.npy` loads `cosine2m.stage0.npy`
through `cosine2m.stage3.npy`) and `--depth` (expectimax decision levels; 1 is
the greedy player training uses, 2 is the good default, 3 is very slow).

### In the terminal

```sh
# one game, 2-ply search, tiles drawn in ANSI colour
./.venv/bin/python -m game2048.play --weights weights/cosine2m.npy --depth 2

# slow it down so the board is readable, and play three in a row
./.venv/bin/python -m game2048.play --weights weights/cosine2m.npy --depth 2 \
    --delay 0.3 --games 3

# no display at all: play N games and print reach rates and averages
./.venv/bin/python -m game2048.play --weights weights/cosine2m.npy --depth 2 --eval 100
```

A 2-ply game is about 11,000 moves and takes a few seconds; `--delay 0` plays
as fast as the terminal can scroll. For a serious measurement use
`scripts/eval_parallel.py` instead, which forks four workers over the shared
tables:

```sh
./.venv/bin/python scripts/eval_parallel.py --games 1000 --depth 2 --jobs 4
```

### In a browser

```sh
./.venv/bin/python -m game2048.server --weights weights/cosine2m.npy --depth 2 \
    --curve talk/cosine2m-curve-stage0.csv --curve-label cosine --open
```

It plays a complete game up front, then serves it back as a scrubbable replay on
<http://127.0.0.1:8000>, so nothing on screen is a live search.

* **Play / Back / Step / New game**, or the left and right arrow keys and space.
* **search** picks 1, 2 or 3 plies for the *next* game. Recording one costs
  about 0.2 s at 1 ply, 4 s at 2, and a minute at 3, so New game shows a timer.
* The right-hand column tabs between **Thinking** (what the agent scored each
  move at, for the move on screen), **Strength** (reach rates over a batch of
  games it plays on demand) and **Curve** (the training curve from `--curve`,
  a CSV that training writes under `talk/`; leave it out if you haven't trained).

Useful extras: `--port 8137` to move it, `--host 0.0.0.0` to reach it from
another machine, `--compare talk/step2m-curve-stage0.csv` to overlay a second
run on the curve, and `--reload` to re-read the checkpoint before every game so
you can watch an agent improve while it is still training.

Other trained checkpoints in `weights/`: `step2m.npy` (the paper's step
schedule, 2M games), `cosine2m.npy` (cosine annealing, 2M games, the strongest),
`cosine.npy` and `agent.npy` (earlier, shorter runs).

## What is in here

| file | what it is |
|---|---|
| `game2048/naive.py` | 2048 the obvious way: list of lists, tile values. The baseline. |
| `game2048/board.py` | The same game in one 64-bit int, with a 65536-entry row table. |
| `game2048/arrayboard.py` | The naive algorithm, compiled. The control in the benchmark. |
| `game2048/patterns.py` | The four 6-cell patterns and their 8 symmetries. |
| `game2048/fast.py` | Compiled kernels: moves, value function, TD learning. |
| `game2048/slow.py` | The agent in pure Python -- the version that never finished training. |
| `game2048/train.py` | Self-play training loop with checkpoints and a CSV learning curve. |
| `game2048/play.py` | Terminal viewer, including what the agent thinks each move is worth. |
| `game2048/server.py` + `web/` | Browser viewer: playback, scrubbing, evaluations, learning curve. |
| `game2048/bench.py` | Measures every rung of the optimisation ladder. |
| `tests/` | The fast versions must agree with the obvious one, always. |

## How it learns

The agent learns the value of **afterstates** -- the board right after your
slide, before the random tile appears -- with TD(0):

```
V(s'_t)  <-  V(s'_t) + alpha * [ r_{t+1} + V(s'_{t+1}) - V(s'_t) ]
```

Learning the afterstate rather than the state takes the spawn randomness out of
the thing being learned, which is why it converges in thousands of games rather
than millions.

`V` is the sum of 32 table lookups: four 6-cell patterns, each read under all 8
symmetries of the square, all 8 sharing one table. That sharing is what lets a
single game teach the network eight times over.

## The numbers behind the talk

Measured on an i5-8250U laptop, playing complete games:

| stage | moves/sec | vs naive |
|---|---:|---:|
| naive lists, tile values | 32,000 | 1x |
| one 64-bit int + row tables | 48,000 | 1.5x |
| naive algorithm + numba | 1,319,000 | 41x |
| bitboard + tables + numba | 7,219,000 | 226x |

Bit-packing bought 1.5x in the interpreter and looked like wasted effort. Under
a compiler the same change was worth another 5.5x. Optimisations multiply, and
some are worth nothing until you have done the other one first.

Training the agent took ~374 million moves: about 10 minutes compiled, or an
estimated 13 hours in pure Python.

## Origin

A Python re-telling of a C++ project from the CGI Lab at NCTU, which applied
"Multi-Stage Temporal Difference Learning for 2048-like Games"
([arXiv:1606.07374](https://arxiv.org/pdf/1606.07374.pdf)) to Threes!.
