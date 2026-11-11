# 交给 Codex：在 8×RTX 5090 上跑 BiGym 稳定性实验

背景先读 `docs/benchmarks/bigym2.md` 第 9、10 节。一句话：我们的 BiGym 从零训练（101k 步 × 每步 256）经常中途发散，
成功率被拖垮；现在要验证一组「稳定组合」设置能不能让训练不发散、成功率上去。GPU 有限：**只跑一个种子，不做多种子、不扫参数**。

## 1. 代码和环境

- 代码：`git clone https://github.com/mingqianzhou123/kimodo-policy-vlm-version.git`（main，f89158d 或更新）。
- Python 3.11。**5090 是 Blackwell（sm_120），必须装 CUDA 12.8 及以上编译的 torch**（例如
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`），先装 torch 再
  `pip install -e ".[bigym2,dev]"`（BiGym 固定在 d715013，mujoco < 3.9）。装完确认
  `torch.cuda.get_device_capability()` 是 `(12, 0)`，并在 8 张卡上各跑一次矩阵乘法。
- 跑测试：`MUJOCO_GL=disable python -m pytest -q`（参考：Mac 上 62 passed、4 skipped）。

## 2. 画图（评测要用）

- 设 `MUJOCO_GL=egl`，用 `python scripts/sensecore/render_probe.py` 检查：要打印 `GL_RENDERER NVIDIA ...`、退出码 0。
  打印 `llvmpipe` 是 CPU 软件渲染，比物理慢约 400 倍，评测跑不动，不能用。
- 缺 `libEGL.so.1` 时的补法见 README「安装」一节，或参考 `scripts/sensecore/setup_egl_loader.sh`。

## 3. 权重、文字特征、数据

- 权重（约 2.3 GB）：Hugging Face `YunhengWang/kimodo-policy`（revision aed6ac7）里的
  `checkpoints/Kimodo-G1-RP-v1/` 和 `checkpoints/dinov3-vitl16-pretrain-lvd1689m/`，放到仓库根目录的 `checkpoints/` 下。
- 文字特征：用户会给 `bigym2_text_cache.tar.gz`（商汤上算好的 20 个任务），在仓库根目录 `tar xzf` 到 `data/cache/`，
  得到 `data/cache/bigym2/<task>.pt`。这样不需要下 15 GB 的 Llama-3-8B 去重算。
- 演示数据：Hugging Face 数据集 `SWIRL-Lab/bigym-g1-native60`，revision `4c3e4414aaf74540899de0b9e82239ef927c9710`
  （和之前所有结果同一版）。这次只需要 `dishwasher_load_cups` 和 `move_plate` 两个任务的文件夹。处理：
  `MUJOCO_GL=egl python scripts/prepare_bigym2.py --data-dir <下载目录> --out data/bigym2/prepared84 --tasks dishwasher_load_cups move_plate --workers 16`。
  检查每个任务正好 60 个 `*.frames.npy`，而且每个都能读（参考 `scripts/sensecore/bigym_run.sh` 开头的数据检查）。
  训练读 `BIGYM2_PREPARED` 环境变量指的目录（默认 `data/bigym2/prepared84`）。

## 4. 先做两个短检查

1. 单卡 200 步 + 评 2 回合（只验证流程，成功率 0 正常）：
   ```bash
   python scripts/train.py --task move_plate --config configs/bigym2_recipe_lr05.yaml --run-dir log/bigym2/check_1gpu \
       training.optimizer.betas=[0.9,0.95] training.spike_skip.factor=5 \
       main.batch_size=32 main.micro_batch_size=32 main.stop_after=200 main.checkpoint_every=200 main.log_every=20
   MUJOCO_GL=egl python scripts/eval_run.py --run-dir log/bigym2/check_1gpu --steps 200 --parallel 1 --episodes 2 --seed-base 700000 --execution-frames 50
   ```
   参考：A800 上单卡每步 0.39 秒，损失从约 0.63 降到 0.2。
2. 8 卡 200 步，用正式设置（每步 256、每卡 32）：
   ```bash
   torchrun --standalone --nproc_per_node=8 scripts/train.py --task move_plate --config configs/bigym2_recipe_lr05.yaml \
       --run-dir log/bigym2/check_8gpu training.optimizer.betas=[0.9,0.95] training.spike_skip.factor=5 \
       main.micro_batch_size=32 main.stop_after=200 main.checkpoint_every=200 main.log_every=20
   ```
   看三件事：多卡能正常跑完；**每卡显存放得下**（5090 只有 32 GB，A800 是 80 GB；放不下就改 `main.micro_batch_size=16`，
   梯度会自动累加，结果等价）；每步多少秒，据此估 30k 步要多久。

## 5. 正式实验（两个训练，一个种子）

设置：`configs/bigym2_recipe_lr05.yaml`（学习率减半）+ `training.optimizer.betas=[0.9,0.95]` + `training.spike_skip.factor=5`，
种子 1000，按 101k 步的学习率曲线训到 **30k 步停**（`main.stop_after=30000`），任务 `dishwasher_load_cups` 和 `move_plate`：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py --task <task> --config configs/bigym2_recipe_lr05.yaml \
    --run-dir log/bigym2/stability/stable_combo/<task>_3cam_s1000 \
    training.optimizer.betas=[0.9,0.95] training.spike_skip.factor=5 \
    main.micro_batch_size=32 main.seed=1000 main.stop_after=30000
```

两个训练一个接一个跑（各用 8 卡），或者各用 4 卡同时跑（`--nproc_per_node=4`、`main.micro_batch_size=64` 或按显存改小），看第 4 步测出的速度和显存定。
训完评 25k 和 30k，**只用验证种子 700000，每个 50 回合；不要用官方种子（620000 起），官方种子留到最后只评一次**：

```bash
MUJOCO_GL=egl python scripts/eval_run.py --run-dir <run> --steps 25000 30000 --parallel 2 --episodes 50 --seed-base 700000 --execution-frames 50
```

## 6. 怎么判断、报告什么

- 发散：`python scripts/diag/stability_summary.py log/bigym2/stability/stable_combo`（连续 500 步运动损失中位数超过 0.5 算发散）；
  `scripts/diag/loss_spikes.py`、`scripts/diag/plot_loss_curves.py` 看损失曲线和尖峰。报告：有没有发散、损失最高值、
  被跳过的尖峰步数、25k / 30k 的验证成功率。
- 对照（不重跑原设置）：原设置第二套里装杯子 15k 步起发散、成功率 0；move_plate 重训时 5k 步起发散。
- **没发散**：同一个训练接着跑满 101k——把 run 目录 `config.json` 里的 `main.stop_after` 改成 `null`，再
  `torchrun ... scripts/train.py --resume <run>`；然后评 80k、85k、90k、95k、100k（验证种子 700000，50 回合），
  和第二套正式结果比（装杯子 0%、move_plate 23%）。**发散了就停**，不再加卡时。
- 结果写进 `docs/benchmarks/bigym2.md` 第 10 节；提交和推送 GitHub 前先问用户；不要把服务器路径、IP、密钥写进仓库（仓库是公开的）。
