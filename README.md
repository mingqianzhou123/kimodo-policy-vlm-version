# Kimodo-VLA

Kimodo-Policy（人形机器人行走 + 操作的 VLA）的新代码库。模型只认一种通用观测和一种通用动作，
每个 benchmark 只写一个薄适配器；同一个模型可以在 BiGym 2.0、HumanoidToolBench、HumanoidArena、SIMPLE
之间切换，只换适配器。论文结果的复现版本在原仓库
[kimodo-policy/kimodo-policy](https://github.com/kimodo-policy/kimodo-policy)，本仓库不改动它。

- 代码怎么分层、接新 benchmark 写哪几个文件：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 每个 benchmark 的结论、做过的决定和待办：[BiGym 2.0](docs/benchmarks/bigym2.md)、
  [HumanoidToolBench](docs/benchmarks/toolbench.md)、[HumanoidArena 和 SIMPLE](docs/benchmarks/arena_simple.md)

## 安装

```bash
uv venv --python 3.11 && uv pip install -e ".[bigym2,toolbench,dev]"
```

GPU 服务器上先按 CUDA 版本装 torch（如 `--index-url https://download.pytorch.org/whl/cu126`），再装本包。
BiGym 用 MuJoCo 离屏渲染，需要 `MUJOCO_GL=egl`。如果机器上没有 `libEGL.so.1`（只有
`libEGL_nvidia`），可以在无 root 权限的情况下补上：从 Ubuntu 镜像下载与系统 `libglvnd0` 同版本的
`libegl1` deb，用 `dpkg-deb -x` 解压，把其中的 `libEGL.so.1*` 所在目录加进 `LD_LIBRARY_PATH`。
HumanoidToolBench、Arena、SIMPLE 的仿真各在自己的环境里跑，本仓库只起策略服务，不需要装它们。

权重放在仓库根目录的 `checkpoints/` 下（与原仓库相同的布局，来自
[Hugging Face](https://huggingface.co/YunhengWang/kimodo-policy) 的 `checkpoints/`）：

```
checkpoints/
  Kimodo-G1-RP-v1/
  dinov3-vitl16-pretrain-lvd1689m/
```

## 每个 benchmark 怎么跑

| benchmark | 准备训练数据 | 检查动作接口（演示回放） | 评测 |
|---|---|---|---|
| BiGym 2.0 | `prepare_bigym2.py` | `check_bigym2_roundtrip.py` | `eval_bigym2.py`（同一进程，官方评测器） |
| HumanoidToolBench | `prepare_toolbench.py` | `check_toolbench_roundtrip.py`（在 ToolBench 的环境里跑） | `serve_toolbench.py`，再让 ToolBench 的评测器连过来 |
| HumanoidArena | 旧仓库已有 | — | `serve_arena.py`，Arena 的评测器连过来 |
| SIMPLE | 旧仓库已有 | — | `serve_simple.py`（HTTP）；同进程评测还没接上 |

```bash
# BiGym 2.0
python scripts/prepare_bigym2.py --data-dir <bigym-g1-native60> --out <prepared dir> --workers 16
MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
    --checkpoint <checkpoint 目录> --text-cache <文本 embedding 目录> --out outputs/move_plate.json

# HumanoidToolBench
python scripts/prepare_toolbench.py --data-dir <ToolBook>/raw --out <prepared dir> --workers 16
python scripts/serve_toolbench.py --checkpoint <checkpoint 目录> --text-cache <文本 embedding 目录> --port 21000

# HumanoidArena / SIMPLE
python scripts/serve_arena.py --checkpoint <checkpoint 目录> --text-cache <文本 embedding 目录> --controller sonic
python scripts/serve_simple.py --checkpoint <checkpoint 目录> --text-cache <文本 embedding 目录>
```

训练目前按 BiGym 配置：`train.py`（单卡或多卡）、`submit_bigym2.py`（提交到 Slurm 集群，训完自动评测）、
`eval_run.py`（评一个 run 的若干 checkpoint）、`summarize_bigym2.py`（汇总成论文主表格式）、`gpu_queue.py`
（不走集群时排到单机的空闲 GPU）、`cache_text_embeddings.py`（任务指令的文本 embedding）；
用法见 [BiGym 文档](docs/benchmarks/bigym2.md) 的训练一节。

## 测试

```bash
pytest                                       # 不需要旧仓库的全部测试
KIMODO_ORIGINAL_REPO=<旧仓库路径> pytest      # 再加上和旧仓库逐项对比的测试
```
