# Kimodo VLM v6.1

独立实验版本：在 HumanoidArena 搬箱子场景里，用 Qwen 从图像定位目标，冻结的 Kimodo 生成双腕参考，SONIC 执行。夹爪和接触门控还没启用，夹爪保持打开；这是"视觉标记到达"实验，不是自然物体抓取。

## 当前结论（2026-11-23）

五个原失败案例从稳定双支撑重新开始后，双腕实际位置相对参考和真值标记的严格达标数由 0/5 提升到 5/5，根部三维平均跟踪误差由 2.785 cm 降到 0.810 cm，最大真值标记误差 2.445 cm。新参考自身还有约 2.055° 前倾。详见 [docs/settled-start-20261123.md](docs/settled-start-20261123.md)。

## 怎么跑

在本目录下运行，并把本目录放在 `PYTHONPATH` 最前。不要 `pip install -e .`，否则会覆盖共享环境里主项目的安装。开发机上的解释器是 `../cross_tracker/data/cache/envs/policy/bin/python`，本地是 `../.venv/bin/python`。

```bash
# 1. 测试（不需要权重和仿真器）
python -m pytest tests -q

# 2. 跑当前最佳配置的五个案例（需要一张空闲 GPU）
python scripts/run_settled_suite.py --gpu 1

# 3. 汇总并和旧结果对比
python scripts/report_settled.py --suite settled-20261123-r1
```

`run_settled_suite.py` 只是把当前最佳配置的开关拼好后调用 `run_oracle.py`。`run_oracle.py` 的默认值仍是最早的原始路线，直接运行它得到的不是上面的结果。

第 2、3 步读取旧实验的输出作为输入：`runs/vlm-refined-20261118-r1-<案例>/` 提供目标，`runs/dynamics-actuator-20261119-r1-<案例>/` 提供对比基线。这两组目录不能删。部分测试也读 `runs/` 里的文件。

## 目录

| 位置 | 内容 |
|---|---|
| `kimodo_vla/` | 模型与算法。从主项目包复制而来，新增 `oracle/`（目标、连续性、IK 投影、支撑约束）、`vlm/`（图像定位）、`model/sparse_conditions.py`、`model/native_policy.py`、`controllers/gripper_gate.py` |
| `arena/` | 接入仿真器的代码，在仿真进程内运行 |
| `scripts/` | 入口和工具，见下表 |
| `tests/` | 接口与几何检查，不替代真权重和仿真测试 |
| `configs/` | 执行器与载荷对照用的参数 |
| `docs/` | 每个阶段的实验记录 |
| `runs/` | 实验输出，不进 git |
| `archives/` | 旧代码快照和阶段记录，不进 git |
| `checkpoints/` | 本版本自己的模型权重，不进 git |

## 脚本

| 脚本 | 用途 |
|---|---|
| `run_settled_suite.py` | **主入口**：按当前最佳配置跑五个案例 |
| `report_settled.py` | 汇总一组结果并和旧基线对比 |
| `run_oracle.py` | 底层启动器：起策略服务和仿真，检查 GPU，保存权重与源码哈希 |
| `serve_oracle.py`、`serve_fixed_reference.py`、`serve_recorded_reference.py` | 三种参考来源的策略服务，由 `run_oracle.py` 启动 |
| `*_sim_bootstrap.py` | 仿真进程的启动钩子，由 `run_oracle.py` 按开关选择 |
| `infer_vlm_points.py`、`infer_vlm_refined.py` | Qwen 图像定位（粗定位 / 两尺度精定位），由 `kimodo_vla/vlm/policy.py` 调用 |
| `analyze_root_dynamics.py`、`analyze_sonic_replay.py` | `report_settled.py` 用到的误差分解 |
| `prepare_vlm_model.py`、`cache_text_embeddings.py`、`cache_vlm_motion_text.py` | 一次性准备：下载 Qwen 权重、缓存文本向量 |

## 历史

`docs/` 里早期阶段的记录会提到已经删除的分析脚本。它们保存在 git 提交 `13f8200`（清理前的完整源码）里，需要时用 `git show 13f8200:"vlm version/scripts/<名字>.py"` 取回。更早的 11 份代码快照在 `archives/code-snapshots-20261117-20261123.tar.gz`。
