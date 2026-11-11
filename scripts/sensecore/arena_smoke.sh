#!/usr/bin/env bash
# Short GPU check of Arena training: text cache for every task, 200 steps of configs/arena.yaml on
# HOI_double_desk/sonic from pt_419h, then the new checkpoint answers one Arena request.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
H=$KVLA_HOME
python $KVLA_CODE/scripts/cache_text_embeddings.py --out $H/cache/arena --arena-data $KVLA_ARENA_DATA
export ARENA_PREPARED=$H/data/arena_prepared ARENA_TEXT_CACHE=$H/cache/arena ARENA_RUNS=$H/runs/arena
RUN=$H/runs/arena/smoke_$(date +%m%d_%H%M%S)
python $KVLA_CODE/scripts/train.py --config $KVLA_CODE/configs/arena.yaml --task HOI_double_desk/sonic \
  --init-checkpoint $KVLA_WEIGHTS/Pre_Train/pt_419h_gbs1024_100w_controlnet4_detach_true_mse/2026-08-26_17-06-35/checkpoint_1000000 \
  --run-dir $RUN main.stop_after=200 main.log_every=20 main.checkpoint_every=200
python $H/arena_serve_check.py $RUN $H/cache/arena
echo ARENA_SMOKE_DONE
