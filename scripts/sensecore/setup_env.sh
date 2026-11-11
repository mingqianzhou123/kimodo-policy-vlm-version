#!/usr/bin/env bash
# Build the Python environment on /data (re-runnable). Log: $H/logs/setup_env.log
set -euxo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E=$H/envs/kimodo
CODE=$H/kimodo-policy-vlm-version
ALI=https://mirrors.aliyun.com/pypi/simple/
CF=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
source /root/anaconda3/etc/profile.d/conda.sh
[ -x $E/bin/python ] || conda create -y -p $E --override-channels -c $CF python=3.11 git git-lfs
P=$E/bin/python
$P -m pip install -U pip -i $ALI
# Same torch pairing as the L40S policy venv, which runs the repo.
$P -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 -f https://mirrors.aliyun.com/pytorch-wheels/cu126/ -i $ALI
# BiGym from a GitHub tarball (git clone of github.com times out here), same commit as pyproject.
$P -m pip install "bigym @ https://codeload.github.com/swirl-uk/BiGym2/tar.gz/d71501359b215b9d78e27dcb70b220412d474dfa" -i $ALI
$P -m pip install -e "$CODE[toolbench,dev]" pillow -i $ALI
# VLM side: LoRA, multi-GPU, data tools, downloads.
$P -m pip install accelerate qwen-vl-utils "huggingface_hub[cli]" pyarrow pandas opencv-python-headless tensorboard -i $ALI
$P - <<'PY'
import torch, transformers, peft, kimodo_vla, PIL
print("OK torch", torch.__version__, "cuda build", torch.version.cuda, "transformers", transformers.__version__, "peft", peft.__version__)
PY
echo SETUP_ENV_DONE
