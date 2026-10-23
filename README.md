# Kimodo-VLA

Kimodo-Policy（humanoid loco-manipulation VLA）的新代码库：模型与 benchmark 解耦，
每个 benchmark 只写一个薄适配器。论文结果的复现版本在原仓库
[kimodo-policy/kimodo-policy](https://github.com/kimodo-policy/kimodo-policy)，本仓库不改动它。

先读 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安装

```bash
uv venv --python 3.11 && uv pip install -e ".[bigym2,dev]"
```

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
