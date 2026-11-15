#!/usr/bin/env bash
set -euo pipefail
RUN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CROSS=$(cd "$RUN/../.." && pwd)
CODE=$(cd "$CROSS/.." && pwd)
WORK=$(cd "$CODE/.." && pwd)
export PATH="$WORK/envs/kimodo/bin:$PATH"
export PYTHONPATH="$CROSS/data/cache/python:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$CODE"
set -o noclobber
exec python -u "$CROSS/scripts/run_box_scratch.py" "$(basename "$RUN")" > "$RUN/scheduler.log" 2>&1
