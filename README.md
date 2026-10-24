# Kimodo-VLA

Kimodo-Policy（humanoid loco-manipulation VLA）的新代码库：模型与 benchmark 解耦，
每个 benchmark 只写一个薄适配器。论文结果的复现版本在原仓库
[kimodo-policy/kimodo-policy](https://github.com/kimodo-policy/kimodo-policy)，本仓库不改动它。

先读 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安装

```bash
uv venv --python 3.11 && uv pip install -e ".[bigym2,dev]"
```

GPU 服务器上先按 CUDA 版本装 torch（如 `--index-url https://download.pytorch.org/whl/cu126`），再装本包。
BiGym 用 MuJoCo 离屏渲染，需要 `MUJOCO_GL=egl`。如果机器上没有 `libEGL.so.1`（只有
`libEGL_nvidia`），可以在无 root 权限的情况下补上：从 Ubuntu 镜像下载与系统 `libglvnd0` 同版本的
`libegl1` deb，用 `dpkg-deb -x` 解压，把其中的 `libEGL.so.1*` 所在目录加进 `LD_LIBRARY_PATH`。

权重放在仓库根目录的 `checkpoints/` 下（与原仓库相同的布局，来自
[Hugging Face](https://huggingface.co/YunhengWang/kimodo-policy) 的 `checkpoints/`）：

```
checkpoints/
  Kimodo-G1-RP-v1/
  dinov3-vitl16-pretrain-lvd1689m/
```

## 测试

```bash
pytest
```

## 在 BiGym 2.0 上评测

```bash
MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
    --checkpoint <checkpoint 目录> --text-cache <文本 embedding 目录> \
    --out outputs/move_plate.json
```
