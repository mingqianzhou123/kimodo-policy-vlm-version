# Agentic 线

零训练：开源 VLM 给出意图，冻结的 Kimodo 按约束生成全身参考运动，冻结的 SONIC 以 50 Hz 跟踪执行。场景是 HumanoidArena。

模型和算法在仓库根目录的 `kimodo_vla/` 里，和训练线共用同一份；本目录只放仿真接入、入口脚本、测试和实验记录。

## 现在到哪了（2026-11-23）

- 已能用指定帧的腕位置和朝向驱动 Kimodo，由 SONIC 执行。五个固定案例从稳定双支撑开始后，双腕实际位置相对参考和真值标记的严格达标数为 5/5，根部三维平均跟踪误差 0.810 cm，最大真值标记误差 2.445 cm。[记录](docs/settled-start-20261123.md)
- Qwen3-VL-4B 的图像定位在九组彩色标记条件下跑通，左右平均误差 0.49 / 0.48 cm。自然物体还没测。[记录](docs/vlm-refinement-20261118.md)
- 走路、单手到达和前倾（实验二）：按计划的线底座不可用。到达 18 点只用 Kimodo 0 个达标，加手臂 IK 1 个；向前 0.5 m 到位 2/5，侧步和后退 0/10；髋带动的前倾 30° 做得到。SONIC 不看根部位置。[记录](docs/exp2-reach-coverage-20261124.md)
- 夹爪一直保持打开，还没有抓取成功率。

## 每台机器配置一次

```bash
cp agentic/env.example.sh agentic/env.local.sh   # 填上本机的三个位置，这个文件不进 git
source agentic/env.local.sh
```

| 变量 | 内容 | 默认 |
|---|---|---|
| `KVLA_SIM_CACHE` | 仿真器环境、HumanoidArena 源码、SONIC 权重 | 无，必须设置 |
| `KVLA_CHECKPOINTS` | 模型权重 | `<仓库>/checkpoints` |
| `KVLA_AGENTIC_RUNS` | 实验输出 | `agentic/runs`，建议指到不会丢的盘 |

代码里的所有位置都由 [scripts/paths.py](scripts/paths.py) 从这三个变量得出。

## 怎么跑

在仓库根目录下：

```bash
# 测试（不需要权重和仿真器；缺少旧实验数据的几项会自动跳过）
pytest agentic/tests -q

# 跑五个案例（需要一张空闲 GPU）
python agentic/scripts/run_settled_suite.py --gpu 1

# 汇总并和旧结果对比
python agentic/scripts/report_settled.py --suite settled-20261123-r1
```

`run_oracle.py` 的默认值就是 11 月 17 日以来所有实验共用的后端。加上 `--settled-goal <目标文件>` 是上面那个结果的完整配置。更早的变体要显式写开关，例如 `--variant original --support-constraints none --hand-constraints keyframes --trajectory-projection none`；`offset` 和 `box` 两种模式必须这样写。

正式实验只在已提交的代码上跑：启动时记录提交号，有未提交的改动会拒绝启动，调试时加 `--allow-dirty`。每次运行结束后会删掉自己的仿真器缓存。

第二、三步读旧实验的输出作为输入：`vlm-refined-20261118-r1-<案例>/` 提供目标，`dynamics-actuator-20261119-r1-<案例>/` 提供对比基线。

## 目录

| 位置 | 内容 |
|---|---|
| `arena/` | 接入仿真器的代码，在仿真进程内运行 |
| `scripts/` | 入口和工具，见下表 |
| `tests/` | 接口与几何检查，不替代真权重和仿真测试 |
| `configs/` | 执行器与载荷对照用的参数 |
| `docs/` | 每个阶段的实验记录 |

相关的包内模块：`kimodo_vla/oracle/`（目标、连续性、IK 投影、支撑约束）、`kimodo_vla/vlm/`（图像定位）、`kimodo_vla/model/sparse_conditions.py`、`kimodo_vla/model/native_policy.py`、`kimodo_vla/controllers/gripper_gate.py`。

## 脚本

| 脚本 | 用途 |
|---|---|
| `run_settled_suite.py` | **主入口**：对五个案例各调用一次 `run_oracle.py` |
| `report_settled.py` | 汇总一组结果并和旧基线对比 |
| `overview.py` | 生成实验进度页：按 `configs/experiments.json` 里的计划给每个实验算进度和成功率，每个实验目录可点开看误差曲线和视频。写到 `<实验输出>/overview/index.html`；加 `--serve 8765` 后在浏览器打开 `http://localhost:8765/overview/index.html`。页面模板是同名的 `overview.html` |
| `run_oracle.py` | 启动器：起策略服务和仿真，检查 GPU，记录提交号、权重与源码哈希 |
| `serve_oracle.py`、`serve_fixed_reference.py`、`serve_recorded_reference.py` | 三种参考来源的策略服务，由 `run_oracle.py` 启动 |
| `*_sim_bootstrap.py` | 仿真进程的启动钩子，由 `run_oracle.py` 按开关选择 |
| `analyze_root_dynamics.py`、`analyze_sonic_replay.py` | `report_settled.py` 用到的误差分解 |
| `run_motion_suite.py` | 实验二的入口：按 `configs/exp2-motions.json` 里的组（向前、侧步、后退、前倾、18 点到达）逐个调用 `run_oracle.py --motion-goal` |
| `analyze_motion.py` | 给走动 / 到达 / 前倾的运行打分：跌倒、完成比例、手腕误差（目标→参考、参考→实际）、前倾角；`--markdown` 生成记录里的表 |
| `rollout_kimodo.py` | 不开仿真，只让 Kimodo 把一个目标连续生成出来，半分钟看步态；`--against <运行>` 给出 3 s、5 s 的连续预测偏差 |
| `analyze_prediction.py` | 只读已有运行：把每次规划里 Kimodo 的原始预测和 0.5 / 1.0 / 1.5 s 后的实测状态对比，并给出“假设身体不动”的对照。写到 `<实验输出>/<组名>/prediction.json` |
| `prepare_vlm_model.py`、`cache_vlm_motion_text.py` | 一次性准备：下载 Qwen 权重、缓存运动短语的文本向量 |
| `paths.py` | 所有位置的唯一来源 |

Qwen 定位本身是包内的两个可执行模块：`python -m kimodo_vla.vlm.infer_points` 和 `python -m kimodo_vla.vlm.infer_refined`。任务指令的文本向量用仓库根目录的 `scripts/cache_text_embeddings.py`。

## 历史

这条线在 2026-11-23 之前位于 `vlm version/`，带着一份复制的包。`docs/` 里早期阶段的记录会提到已删除的分析脚本和旧路径，它们保存在提交 `13f8200` 里，用 `git show 13f8200:"vlm version/scripts/<名字>.py"` 取回。
