#!/usr/bin/env bash
# The full curriculum, every stage from scratch, on a cosine step-size schedule.
#
# Trains to its own stem (weights/cosine.*) so the step-schedule agent in
# weights/agent.* stays intact and playable. That is the point: the two stems are
# a controlled A/B of the only thing that changed, the alpha schedule.
#
#   ./scripts/curriculum-cosine.sh
#   G0=1000000 ./scripts/curriculum-cosine.sh      # shorter stage 0
#   STEM=weights/cosine2m.npy G0=2000000 ... ./scripts/curriculum-cosine.sh
#
# The stem names the run. Weights, curves, log and closing eval all derive from
# it, so a longer-budget rerun does not read the shorter run's curves and
# conclude every stage is already finished.
#
# Cosine needs its budget up front, since the ramp is a function of the total.
# So unlike curriculum.sh there is no --stop-when-saturated here: a stage runs
# its full allocation and lands on alpha_final exactly as it finishes.
#
# Safe to kill and rerun. Each stage passes --resume, and --anneal-over pins the
# ramp to the stage's whole budget rather than to whatever is left, so a resumed
# stage picks the curve up where it stopped instead of restarting the ramp.
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEM=${STEM:-weights/cosine.npy}
BOARDS=${BOARDS:-50000}       # restart boards per stage
GAMECAP=${GAMECAP:-600000}    # give up collecting after this many games

# Every output path hangs off the stem, so two runs with different budgets can
# live side by side. The default stem reproduces the original names exactly.
NAME=$(basename "${STEM%.npy}")
LOG=talk/curriculum-${NAME}.log
curve() { echo "talk/${NAME}-curve-stage$1.csv"; }

# Per-stage budgets, defaulting to what the step-schedule run actually spent, so
# the comparison is like for like.
SAVE_EVERY=${SAVE_EVERY:-600}   # seconds between checkpoints; each writes 256 MB
ALPHA=${ALPHA:-0.0025}          # peak step size, where the ramp starts
ALPHA_FINAL=${ALPHA_FINAL:-0.00025}   # floor, reached exactly as a stage ends

G0=${G0:-1900000}
G1=${G1:-750000}
G2=${G2:-675000}
G3=${G3:-640000}

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Games already logged for a stage, so a rerun finishes the budget instead of
# starting a fresh one on top of it. --games counts *this run*, not the total.
done_games() {
  local f n; f=$(curve "$1")
  [ -f "$f" ] || { echo 0; return; }
  n=$(tail -1 "$f" | cut -d, -f1)
  case "$n" in (*[!0-9]*|'') echo 0;; (*) echo "$n";; esac
}

train() {   # stage, budget
  local stage=$1 games=$2
  local done left
  done=$(done_games "$stage")
  left=$(( games - done ))
  if [ "$left" -le 0 ]; then
    say "stage ${stage} already has ${done} of ${games} games -- skipping"
    return
  fi
  [ "$done" -gt 0 ] && say "stage ${stage} resuming at ${done}; ${left} games left of ${games}"
  say "training stage ${stage} for ${left} games (cosine, budget ${games})"
  $PY -m game2048.train --stage "$stage" \
      --out "$STEM" \
      --games "$left" --anneal-over "$games" --schedule cosine \
      --alpha "$ALPHA" --alpha-final "$ALPHA_FINAL" \
      --block 2000 --save-every "$SAVE_EVERY" --resume \
      --log "$(curve "$stage")" \
      >> "$LOG" 2>&1 || { say "train ${stage} failed"; exit 1; }
  say "stage ${stage} done"
}

stage_file() { echo "${STEM%.npy}.stage$1.npy"; }
restart_file() { echo "${STEM%.npy}.restarts$1.npy"; }

say "=== cosine curriculum starting: stem ${STEM}, alpha ${ALPHA} -> ${ALPHA_FINAL}, budgets ${G0}/${G1}/${G2}/${G3}"
train 0 "$G0"

for stage in 1 2 3; do
  eval "games=\$G${stage}"
  if [ ! -f "$(restart_file "$stage")" ]; then
    say "collecting stage-${stage} restart boards"
    $PY -m game2048.train --collect "$stage" --out "$STEM" \
        --max-restarts "$BOARDS" --games "$GAMECAP" --block 2000 \
        >> "$LOG" 2>&1 || { say "collect ${stage} failed"; exit 1; }
  else
    say "stage-${stage} restart boards already on disk -- skipping collection"
  fi
  train "$stage" "$games"
done

say "=== cosine curriculum complete"
$PY scripts/eval_parallel.py --weights "$STEM" --games 2000 --depth 1 --jobs 4 \
    --out "talk/eval-${NAME}-1ply-2000.log" >> "$LOG" 2>&1 \
  && say "1-ply sanity eval -> talk/eval-${NAME}-1ply-2000.log"
