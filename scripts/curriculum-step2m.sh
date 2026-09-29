#!/usr/bin/env bash
# The full curriculum on the paper's *single alpha drop*, at a fixed game count.
#
#   ./scripts/curriculum-step2m.sh
#   P1=500000 P2=500000 ./scripts/curriculum-step2m.sh     # a shorter rehearsal
#   STEM=weights/probe.npy ./scripts/curriculum-step2m.sh  # somewhere else
#
# Every stage gets 1,000,000 games at alpha 0.0025, then 1,000,000 more at
# 0.00025. That is improvement (b) from Yeh et al., except that the drop happens
# at a planned game count rather than when the saturation detector fires.
#
# This is the third arm of the schedule experiment, and it is the one that makes
# the other two readable:
#
#   weights/agent.*     0.0025 -> 0.00025, drop at saturation, ~1.9M/750k/675k/640k
#   weights/cosine2m.*  0.0025 -> 0.0001,  cosine ramp,         2M per stage
#   weights/step2m.*    0.0025 -> 0.00025, drop at 1M,          2M per stage
#
# Against cosine2m it holds the budget fixed and changes the shape of the
# schedule. Against agent.* it holds the shape fixed and changes when the drop
# happens and how many games follow it. Neither comparison is available from the
# two runs we already have, because they differ in both at once.
#
# ---------------------------------------------------------------------------
# Why two invocations per stage rather than one with --decay-after
#
# --decay-after fires on `played`, the games *this run* (train.py:296), while
# --anneal-over is a function of `played + offset`, the cumulative count. So
# cosine survives a kill and a --resume and a step drop does not: restart a 2M
# run after 1.2M games and --decay-after 1000000 would drop alpha a second time,
# 1M games into the *remainder*, at cumulative game 2.2M that never arrives.
#
# So each stage runs as two explicit phases, and phase 2 simply passes the low
# alpha as --alpha with no decay flag at all, which keeps it constant. Both
# phases resume off the cumulative game count in the curve CSV, so killing this
# script at any moment and rerunning it resumes the correct phase at the correct
# step size. That is worth the extra ten lines: these runs get killed.
# ---------------------------------------------------------------------------
#
# Safe to kill and rerun. Checkpoints every SAVE_EVERY seconds, 256 MB a time.
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEM=${STEM:-weights/step2m.npy}
BOARDS=${BOARDS:-50000}       # restart boards per stage
GAMECAP=${GAMECAP:-600000}    # give up collecting after this many games

# Every output path hangs off the stem, so this run cannot read another run's
# curves and conclude a stage is already finished.
NAME=$(basename "${STEM%.npy}")
LOG=talk/curriculum-${NAME}.log
curve() { echo "talk/${NAME}-curve-stage$1.csv"; }

SAVE_EVERY=${SAVE_EVERY:-600}          # seconds between checkpoints
ALPHA=${ALPHA:-0.0025}                 # phase 1: the paper's per-weight step
ALPHA_FINAL=${ALPHA_FINAL:-0.00025}    # phase 2: the paper's 10x drop
P1=${P1:-1000000}                      # games before the drop, every stage
P2=${P2:-1000000}                      # games after it
TOTAL=$(( P1 + P2 ))

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Games already logged for a stage. train.py takes its own offset from the last
# row of this same CSV (train.py:205), so the two agree by construction.
done_games() {
  local f n; f=$(curve "$1")
  [ -f "$f" ] || { echo 0; return; }
  n=$(tail -1 "$f" | cut -d, -f1)
  case "$n" in (*[!0-9]*|'') echo 0;; (*) echo "$n";; esac
}

# One phase of one stage: `left` more games at a constant step size.
phase() {   # stage, label, alpha, games-left
  local stage=$1 label=$2 alpha=$3 left=$4
  say "stage ${stage} ${label}: ${left} games at alpha ${alpha}"
  $PY -m game2048.train --stage "$stage" \
      --out "$STEM" \
      --games "$left" --schedule step --alpha "$alpha" \
      --block 2000 --save-every "$SAVE_EVERY" --resume \
      --log "$(curve "$stage")" \
      >> "$LOG" 2>&1 || { say "train ${stage} ${label} failed"; exit 1; }
}

train() {   # stage
  local stage=$1 done
  done=$(done_games "$stage")

  if [ "$done" -ge "$TOTAL" ]; then
    say "stage ${stage} already has ${done} of ${TOTAL} games -- skipping"
    return
  fi
  [ "$done" -gt 0 ] && say "stage ${stage} resuming at ${done} of ${TOTAL}"

  # Phase 1, only if the drop point is still ahead of us.
  if [ "$done" -lt "$P1" ]; then
    phase "$stage" "phase 1" "$ALPHA" "$(( P1 - done ))"
    done=$(done_games "$stage")     # re-read: the phase may have been cut short
  fi

  # Phase 2. A phase-1 run that stopped early leaves `done` below P1, in which
  # case the guard above has just finished it, so this is the honest remainder.
  if [ "$done" -lt "$TOTAL" ]; then
    phase "$stage" "phase 2" "$ALPHA_FINAL" "$(( TOTAL - done ))"
  fi
  say "stage ${stage} done"
}

stage_file() { echo "${STEM%.npy}.stage$1.npy"; }
restart_file() { echo "${STEM%.npy}.restarts$1.npy"; }

say "=== step curriculum starting: stem ${STEM}, alpha ${ALPHA} for ${P1} games then ${ALPHA_FINAL} for ${P2}, every stage"

# STAGES exists so the ladder can be run a rung at a time (`STAGES=0 ...`), and
# so a short rehearsal of this script can stop before the collection step, which
# on an untrained stage 0 would spend GAMECAP games finding no boards at all.
for stage in ${STAGES:-0 1 2 3}; do
  if [ "$stage" -gt 0 ] && [ ! -f "$(restart_file "$stage")" ]; then
    say "collecting stage-${stage} restart boards"
    $PY -m game2048.train --collect "$stage" --out "$STEM" \
        --max-restarts "$BOARDS" --games "$GAMECAP" --block 2000 \
        >> "$LOG" 2>&1 || { say "collect ${stage} failed"; exit 1; }
  elif [ "$stage" -gt 0 ]; then
    say "stage-${stage} restart boards already on disk -- skipping collection"
  fi
  train "$stage"
done

say "=== step curriculum complete"
$PY scripts/eval_parallel.py --weights "$STEM" --games 2000 --depth 1 --jobs 4 \
    --out "talk/eval-${NAME}-1ply-2000.log" >> "$LOG" 2>&1 \
  && say "1-ply sanity eval -> talk/eval-${NAME}-1ply-2000.log"
