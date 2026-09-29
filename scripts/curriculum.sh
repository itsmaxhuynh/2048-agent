#!/usr/bin/env bash
# Run the multi-stage curriculum the way the paper does it: train a stage to
# saturation, THEN harvest the boards that cross into the next one, then train
# that one. Collecting after training (rather than during) means every restart
# board comes from the finished player for that stage, not from whatever the
# weights happened to be halfway through the run.
#
#   ./scripts/curriculum.sh [wait-for-pid]
#
# Kill it at any time; each step leaves its own checkpoint behind and the next
# invocation picks up from the last stage that has weights.
set -u

cd "$(dirname "$0")/.."
PY=.venv/bin/python
BOARDS=50000          # restart boards to gather per stage
GAMECAP=600000        # give up collecting after this many games
LOG=talk/curriculum.log

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait out a trainer that is already running, if we were handed its pid.
if [ $# -ge 1 ]; then
  say "waiting for pid $1 to finish stage 0"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  say "pid $1 exited"
fi

for stage in 1 2 3; do
  prev=$((stage - 1))
  if [ ! -f "weights/agent.stage${prev}.npy" ]; then
    say "stage ${prev} has no weights -- stopping"; exit 1
  fi

  if [ ! -f "weights/agent.restarts${stage}.npy" ]; then
    say "collecting stage-${stage} restart boards"
    $PY -m game2048.train --collect "$stage" \
        --max-restarts "$BOARDS" --games "$GAMECAP" --block 2000 \
        >> "$LOG" 2>&1 || { say "collect ${stage} failed"; exit 1; }
  else
    say "stage-${stage} restart boards already on disk -- skipping collection"
  fi

  say "training stage ${stage}"
  $PY -m game2048.train --stage "$stage" \
      --games 2000000 --block 2000 --save-every 600 --resume \
      --auto-decay --stop-when-saturated \
      --log "talk/curve-stage${stage}.csv" \
      >> "$LOG" 2>&1 || { say "train ${stage} failed"; exit 1; }
  say "stage ${stage} done"
done

say "curriculum complete"
