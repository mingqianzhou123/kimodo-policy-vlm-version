#!/usr/bin/env bash
set -euo pipefail
SETUP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CROSS=$(cd "$SETUP/../.." && pwd)
CODE=$(cd "$CROSS/.." && pwd)
WORK=$(cd "$CODE/.." && pwd)
export PATH="$WORK/envs/kimodo/bin:$PATH"
export PYTHONPATH="$CROSS/data/cache/python:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CODE"
RUN_ID=${1:?Pass exactly one run ID}
[[ "$RUN_ID" =~ ^box-[a-z0-9-]+$ ]] || exit 2
[[ -f "$CROSS/runs/$RUN_ID/run.json" ]] || exit 2
# Each independently scheduled job owns a separate scheduler log.
set -o noclobber
exec python -u "$CROSS/scripts/run_box_screen.py" "$RUN_ID" > "$CROSS/runs/$RUN_ID/scheduler.log" 2>&1
