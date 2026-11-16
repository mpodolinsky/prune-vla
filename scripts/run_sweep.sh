#!/usr/bin/env bash
# Sequential sweep over pruning settings on LIBERO-10. Usage:
#   N_EPISODES=5 scripts/run_sweep.sh            # all conditions below, 10 tasks x N episodes each
#   TASK_IDS="0 4 7" N_EPISODES=10 scripts/run_sweep.sh
# Results land in outputs/<tag>/ (eval_info.json, prune_stats.json); summarise with scripts/summarize.py.
set -euo pipefail
cd "$(dirname "$0")/.."
N_EPISODES="${N_EPISODES:-5}"
TASK_ARGS=()
[[ -n "${TASK_IDS:-}" ]] && TASK_ARGS=(--task-ids $TASK_IDS)

run() { uv run python -m prune_vla.eval_libero --n-episodes "$N_EPISODES" --profile "${TASK_ARGS[@]}" "$@"; }

run --method none                                              # pi0.5 as shipped (968-slot prefix)
run --method none --compact                                    # same model, padding slots dropped (fair baseline)
for k in 128 64 32 16; do
  run --method divprune --tokens-per-camera "$k" --compact
  run --method random   --tokens-per-camera "$k" --compact
done
run --method pool --tokens-per-camera 64 --compact
run --method pool --tokens-per-camera 16 --compact
