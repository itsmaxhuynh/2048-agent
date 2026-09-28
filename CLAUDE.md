# 2048 in Python -- talk project

A self-teaching 2048 agent (n-tuple network + TD(0) on afterstates), written to
be *explained on stage*. The audience is Python Adelaide; the talk is about the
optimisation journey as much as the reinforcement learning.

That goal drives the design: every module is meant to be readable aloud, the
slow versions are kept deliberately, and the benchmark ladder is part of the
product, not scaffolding.

## Environment

```sh
./.venv/bin/python -m ...          # always; numba lives in .venv, not system Python
```

Laptop constraints that shaped the code: 8 GB RAM (~3 GB free) and 4 cores.
The weight tables are 256 MB per stage (768 MB for the three stages), so there
is no room for multiprocessing workers each holding a copy.

## Commands

```sh
./.venv/bin/python -m tests.test_board     # naive vs bitboard
./.venv/bin/python -m tests.test_fast      # bitboard vs compiled kernels
./.venv/bin/python -m game2048.bench       # the optimisation ladder (needs an idle CPU)
./.venv/bin/python -m game2048.train --stage K --games N --resume
./.venv/bin/python -m game2048.train --collect K --max-restarts 50000
./.venv/bin/python -m game2048.play --weights weights/agent.npy [--eval N]
./.venv/bin/python scripts/eval_parallel.py --games N --depth D --jobs 4
./.venv/bin/python -m game2048.server --open
./scripts/curriculum.sh                    # the whole ladder, step schedule
./scripts/curriculum-cosine.sh             # ditto, cosine, to weights/cosine.*
```

`play.py --eval` stats `--weights` itself, which staged training never writes
(only `agent.stageK.npy`), so it exits with "train first" even when the stages
are there. `scripts/eval_parallel.py` loads the stages directly and is what to
reach for; it also forks workers that share the 1 GB of tables copy-on-write,
which is safe only because evaluation never writes them.

Run both test modules after touching `board.py`, `fast.py`, or `arrayboard.py`.
They are the reason the optimisations are safe: `test_board` plays 200 complete
games in lockstep between the naive and bit-packed implementations, and
`test_fast` closes the chain from bitboard to machine code.

Pause any background training before benchmarking (`kill -STOP` / `-CONT`) --
a training run halves the numbers.

## The sharp edge: uint64 at the numba boundary

The board is a `uint64`. numba hands one back to Python as a plain `int`, and
passing that int back in types it as `int64`. Under numpy promotion rules
`int64 & uint64` becomes **float64**, which silently destroys the board rather
than raising. Above 2^63 -- any board with a 256-or-larger tile in cell 15 --
numba instead raises `OverflowError: int too big to convert`.

Consequences, all load-bearing:

* Every Python-facing kernel in `fast.py` re-asserts `b = np.uint64(board)` on
  entry, into a **new local name**. Reassigning the parameter itself would make
  numba unify `int64` and `uint64` to `float64` and reintroduce the bug.
* Callers outside `game2048/` use `fast.as_board(...)` at the boundary.
* Every uint64 constant is pre-cast at module level (`U4`, `MASK_CELL`, `TA1`...).
  A bare Python int literal in a shift or mask promotes the expression.
* `tests/test_fast.py::check_python_boundary` locks this down. Do not delete it.

## Board encoding

16 cells x 4 bits in one 64-bit int. A cell holds the tile's **exponent**
(1 = tile 2, 11 = tile 2048), not its value. Cell `i` is at bits `4*i`; row `r`
is the 16 bits at `16*r`. Directions are `0=up 1=right 2=down 3=left` ("URDL")
everywhere, matching the C++ ancestor of this project.

`board.py` builds `ROW_LEFT/ROW_RIGHT/ROW_SCORE_LEFT/ROW_SCORE_RIGHT` at import
(~0.4 s for all 65536 rows). Up and down are left and right on the transposed
board. `fast.py` imports those tables as module globals; numba freezes them as
compile-time constants, which is why `cache=True` works there.

## Value function

Four 6-cell patterns x 8 symmetries = 32 lookups per position, into 4 tables of
16^6 float32. All 8 symmetries of a pattern share one table -- that weight
sharing is the whole data-efficiency story, so changing it changes what the agent
learns, not just how it is indexed.

On top of the tuples sit five whole-board features (`_extras`): the large-tile
histogram from the paper, plus empty-cell, distinct-value, mergeable-pair and
neighbouring-`v`/`2v` counts. They are 37 more weights in the same flat row, so
one stage is `TOTAL_WEIGHTS = 67,141,716` float32 = 256 MB, not `4 x 16^6`
exactly. `_extras` computes all five in two sweeps because one loop per feature
costs more than the 32 lookups they assist.

`update()` applies `delta` to every contributing weight **as given**. It does not
divide by the feature count, so `--alpha` in `train.py` is the paper's *per
weight* step (0.0025), not a total. Reintroducing a divide-by-32 changes the
effective learning rate by ~37x.

`--schedule cosine` anneals alpha from `--alpha` to `--alpha-final` over
`--anneal-over` games (`cosine_alpha`), instead of the paper's single drop at
saturation. The ramp is a function of the *cumulative* game count, so `--resume`
rejoins the curve rather than restarting it, which is why `--anneal-over` must be
passed explicitly on a resumed run.

### Stages

`N_STAGES = 4`: the paper's Strategy 2, splitting at T16k, T16+8k and T16+8+4k
(`fast.STAGE_REQUIREMENTS`).

**A stage boundary is a moment in a game, not a property of a board.** T16+8k is
"the first time a 16384 and an 8192 have been on the board together"; merging the
8192 away afterwards must not drop the stage back. So the stage is episode state
that only ever increases, and `fast.advance_stage(board, stage)` takes the
current stage and returns the new one. There is no `stage_of(board)` and there
cannot be one. `tests/test_fast.py::check_stage_events` pins this, including the
merge-away case.

Consequences worth not breaking:

* **`evaluate(board, w)` and `update(board, delta, w)` take the weight row from
  the caller** and do not derive a stage. They cannot: the board alone does not
  determine the stage. Every caller threads `advance_stage` through its own loop
  and is responsible for handing both the same `w`.
* **A fresh stage starts at zero** (`fast.new_weights()`), which is safe here
  precisely because stages are trained *sequentially and in isolation*: the stage
  below is finished and frozen before the next one begins, so a zero table above
  can never teach it that crossing the boundary is worthless. That argument fails
  under concurrent training, so do not merge the stages into one online loop.
* **`fast.load_weights()` stops at the last stage that exists** and play clamps
  the index to it, so an untrained top stage falls back to the deepest trained
  one rather than to zeros.

Training a stage is a curriculum, not a slice of one long game: `--collect K`
plays stage K-1 and snapshots every crossing into K, then `--stage K` trains from
those boards replayed round-robin. `scripts/curriculum.sh` runs the whole ladder
on the paper's step schedule; `scripts/curriculum-cosine.sh` runs it on cosine to
its own stem.

## Search

`best_action` is the greedy 1-ply player and is what training uses -- always.
`best_action_search(board, w, plies)` is expectimax with the same return shape;
`plies=1` must reproduce `best_action` exactly, which
`tests/test_fast.py::check_search` asserts move by move.

`_search` is written as direct self-recursion with the chance node inlined,
because numba compiles direct recursion but **not** mutual recursion -- splitting
the chance node into its own function will fail to compile. `value_after()` is
the one exception: it is the chance node as a leaf, used by the viewers so the
numbers on screen are the ones the agent compared.

Search is play-time only (`--depth` on `play.py` and `server.py`). Depth 2 costs
about 17x the compute for ~1.5x the score, depth 3 about 438x for ~2x; training
at depth 2 would take weeks. Do not wire `--depth` into `train.py`.

Raising `N_STAGES` costs 256 MB each and is safe: `load_weights` simply stops at
the last trained stage and play clamps to it. Lowering it will not load an
existing checkpoint.

Weight files are plain `np.save` of one stage's flat `(67141716,)` float32 row,
one file per stage (`agent.stage0.npy` ...); `load_weights` stacks them. Changing
`BASE_PATTERNS` or `N_CELLS` silently invalidates every existing checkpoint --
there is no header or version in the file.

## Conventions

Modules are ordered by the talk's narrative, and `naive.py`, `arrayboard.py`
and `slow.py` exist **only** to be benchmarked against. They are not dead code;
deleting them removes the evidence for the talk's central claim. Keep them
correct -- the tests check them.

Docstrings explain *why* a thing is the way it is, because they double as
speaker notes.
