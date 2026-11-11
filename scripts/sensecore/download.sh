#!/usr/bin/env bash
# Weights, Arena data and the candidate VLM. Log: $H/logs/download.log
# kimodo-policy weights and the Arena dataset go through fetch.py + manifest.tsv (file list with sizes built
# on the Mac), because hf-mirror kept answering the hf CLI's listing calls for the dataset with 403.
# Order: what the GPU self-check needs (171 rows; .bak files dropped: hf-mirror answers 403 for them), then the dataset, then the rest of the weights.
set -uxo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd $KVLA_HOME
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
for i in $(seq 20); do hf download Qwen/Qwen3-VL-2B-Instruct --local-dir $KVLA_VLM && break; sleep 120; done
for i in 1 2 3; do python fetch.py manifest.tsv --first 171 && break; done
echo SELFCHECK_INPUTS_READY
for i in 1 2 3; do python fetch.py manifest.tsv && break; done
du -sh $KVLA_WEIGHTS $KVLA_VLM $KVLA_ARENA_DATA
echo DOWNLOAD_DONE
