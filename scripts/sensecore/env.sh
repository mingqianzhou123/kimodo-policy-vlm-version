# Kimodo-VLA on SenseCore. Source this in the dev instance and at the top of every ACP job.
# Uses no /root or /app path, so it works in any job image that mounts /data. KVLA_HOME is the folder
# this file is in (scripts/sensecore/ of the repo is copied there).
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KVLA_HOME=$H
export KVLA_CODE=$H/kimodo-policy-vlm-version
export PATH=$H/envs/kimodo/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com          # huggingface.co is unreachable from cn-sh-01
export HF_HOME=$H/hf_cache
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
# Weights and data live outside the code folder, so replacing the code keeps them.
export KVLA_WEIGHTS=$H/weights/kimodo-policy               # HF YunhengWang/kimodo-policy @ aed6ac7
export KVLA_ARENA_DATA=$H/data/HumanoidArena_dataset_v3_1  # HF dataset WilliamWang16/HumanoidArena_dataset_v3_1
export KVLA_VLM=$H/weights/Qwen3-VL-2B-Instruct            # candidate VLM, not decided
[ -e $KVLA_CODE/checkpoints ] || ln -s $KVLA_WEIGHTS/checkpoints $KVLA_CODE/checkpoints 2>/dev/null
cd $KVLA_CODE 2>/dev/null || true
