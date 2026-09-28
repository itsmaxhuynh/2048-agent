# 2048, learned in Python

A 2048 agent that teaches itself to play, written for a talk at Python Adelaide.
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
./.venv/bin/python -m game2048.play --weights weights/agent.npy      # watch in the terminal
./.venv/bin/python -m game2048.server --open                          # watch in a browser
```

`numba` is optional -- everything runs without it, about 78x slower.

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
| `talk/` | Speech plan, benchmark output, learning curve. |

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
