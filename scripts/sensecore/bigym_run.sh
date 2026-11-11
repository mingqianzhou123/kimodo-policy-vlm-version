#!/bin/bash
# One BiGym training run on SenseCore (an ACP job, all its GPUs), then its validation evaluation.
#
#   bash $H/acp_job.sh <job name> bash $H/bigym_run.sh <group> <config> <task> <seed> <stop after|full> <eval steps,comma> [overrides...]
#   e.g. bash $H/bigym_run.sh stability/recipe_lr05 configs/bigym2_recipe_lr05.yaml move_plate 1000 30000 25000,30000
#
# Run directory: $H/runs/bigym2/<group>/<task>_3cam_s<seed>. A rerun of the same arguments continues from resume.pt
# and skips checkpoints already evaluated. Evaluation: EVAL_EPISODES (50) on the validation seed block EVAL_SEED_BASE
# (700000; never the official 620000 block while tuning), one checkpoint per GPU at a time. The per-GPU micro-batch
# is batch_size / GPUs (no accumulation) unless an override sets main.micro_batch_size.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
H=$KVLA_HOME
cd $KVLA_CODE || exit 1
export BIGYM2_PREPARED=$H/data/bigym2/prepared84
# The ACP images have no system libEGL: MuJoCo's EGL backend needs the glvnd loader from $H/envs/egl_loader plus
# NVIDIA's libEGL_nvidia.so.0 from the driver. Only evaluation renders; the text cache and training import BiGym without it.
EGL_ENV=(LD_LIBRARY_PATH=$H/envs/egl_loader/lib:${LD_LIBRARY_PATH:-} __EGL_VENDOR_LIBRARY_DIRS=$H/envs/egl_loader/vendor_nvidia MUJOCO_GL=egl)
export MUJOCO_GL=disable
GROUP=$1 CONFIG=$2 TASK=$3 SEED=$4 STOP=$5 EVAL=$6
shift 6
RUN=$H/runs/bigym2/$GROUP/${TASK}_3cam_s$SEED
NGPU=$(nvidia-smi -L | wc -l)
BATCH=$(python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; from run_config import load_yaml; print(load_yaml(Path('$CONFIG')).main.batch_size)")
OVERRIDES=("main.micro_batch_size=$((BATCH / NGPU))" "main.seed=$SEED")
[ "$STOP" != full ] && OVERRIDES+=("main.stop_after=$STOP")
OVERRIDES+=("$@")
echo "run $RUN on $NGPU GPUs, batch $BATCH, overrides ${OVERRIDES[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The task's 60 prepared demos must all be there and readable (a run must never train on part of the data).
python - <<EOF || { echo BIGYM_DATA_INCOMPLETE; exit 1; }
import numpy as np, torch
from pathlib import Path
files = sorted(Path("$BIGYM2_PREPARED/$TASK").glob("*.frames.npy"))
assert len(files) == 60, f"{len(files)} prepared demos of $TASK, expected 60"
for f in files:
    np.load(f, mmap_mode="r")[-1].sum()
    torch.load(f.with_name(f.name.replace(".frames.npy", ".pt")), map_location="cpu", weights_only=False)
print("data ok:", len(files), "demos")
EOF
# BiGym instruction embeddings: the first job builds them (LLM2Vec on its GPU); others wait for that.
if [ ! -f data/cache/bigym2/$TASK.pt ]; then
    if mkdir $H/locks_textcache 2>/dev/null; then
        python scripts/cache_text_embeddings.py --out data/cache/bigym2; rmdir $H/locks_textcache
    else
        for i in $(seq 120); do [ -f data/cache/bigym2/$TASK.pt ] && break; sleep 30; done
    fi
fi
[ -f data/cache/bigym2/$TASK.pt ] || { echo BIGYM_TEXT_CACHE_MISSING; exit 1; }

# GPU rendering check (does not stop training): evaluation runs only if MuJoCo gets an NVIDIA EGL context.
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"; ls /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/local/nvidia/lib64 2>/dev/null | grep -E "libEGL_nvidia|libGLX_nvidia" | sort -u
env "${EGL_ENV[@]}" python $H/render_probe.py 2>&1 | grep -v Warning | tail -3
RENDER_OK=${PIPESTATUS[0]}; [ $RENDER_OK -eq 0 ] && echo BIGYM_RENDER_OK || echo "BIGYM_RENDER_UNAVAILABLE (training continues; evaluation skipped)"

if [ $NGPU -gt 1 ]; then TRAIN=(python -m torch.distributed.run --standalone --nproc_per_node=$NGPU); else TRAIN=(python); fi
LAST=$( [ "$STOP" != full ] && echo $STOP || python -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; from run_config import load_yaml, checkpoint_steps; print(max(checkpoint_steps(load_yaml(Path('$CONFIG')).main)))")
if [ -d $RUN/checkpoint_$LAST ]; then
    echo "training already finished (checkpoint_$LAST)"
elif [ -f $RUN/resume.pt ]; then
    "${TRAIN[@]}" scripts/train.py --resume $RUN || { echo BIGYM_TRAIN_FAILED; exit 1; }
else
    mkdir -p $(dirname $RUN)
    "${TRAIN[@]}" scripts/train.py --task $TASK --config $CONFIG --run-dir $RUN "${OVERRIDES[@]}" || { echo BIGYM_TRAIN_FAILED; exit 1; }
fi
echo BIGYM_TRAIN_DONE
[ $RENDER_OK -eq 0 ] || { echo "BIGYM_EVAL_SKIPPED: no GPU EGL here; rerun the same command where rendering works"; exit 0; }

i=0
for STEP in ${EVAL//,/ }; do
    env "${EGL_ENV[@]}" CUDA_VISIBLE_DEVICES=$((i % NGPU)) python scripts/eval_run.py --run-dir $RUN --steps $STEP --parallel 1 \
        --episodes ${EVAL_EPISODES:-50} --seed-base ${EVAL_SEED_BASE:-700000} --execution-frames 50 &
    i=$((i + 1))
    [ $((i % NGPU)) -eq 0 ] && wait
done
wait
ls $RUN/val${EVAL_SEED_BASE:-700000}/
echo BIGYM_RUN_DONE
