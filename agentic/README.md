# Agentic 线

零训练：开源 VLM 给出意图，冻结的 Kimodo 按约束生成全身参考运动，冻结的 SONIC 以 50 Hz 跟踪执行。场景是 HumanoidArena。

模型和算法在仓库根目录的 `kimodo_vla/` 里，和训练线共用同一份；本目录只放仿真接入、入口脚本、测试和实验记录。

## 现在到哪了（2026-11-23）

- 已能用指定帧的腕位置和朝向驱动 Kimodo，由 SONIC 执行。五个固定案例从稳定双支撑开始后，双腕实际位置相对参考和真值标记的严格达标数为 5/5，根部三维平均跟踪误差 0.810 cm，最大真值标记误差 2.445 cm。[记录](docs/settled-start-20261123.md)
- Qwen3-VL-4B 的图像定位在九组彩色标记条件下跑通，左右平均误差 0.49 / 0.48 cm。自然物体还没测。[记录](docs/vlm-refinement-20261118.md)
- 走路、单手到达和前倾（实验二）：按计划的线底座不可用。到达 18 点只用 Kimodo 0 个达标，加手臂 IK 1 个；向前 0.5 m 到位 2/5，侧步和后退 0/10；髋带动的前倾 30° 做得到。SONIC 不看根部位置。[记录](docs/exp2-reach-coverage-20261124.md)
- 自然物体的指点测试（实验三）已有 80 帧图像集、统一提示词和评测器；Qwen 接触点 3D 正确率 78.75%、延迟中位数 1.37 s，GPT-6.2 Astra 100%、3.88 s，Gemini 4 Argon 92.5%、7.66 s。[记录](docs/vlm-pointing-20261123.md)
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

箱子抓取（实验一）用同一个启动器，加 `--grasp-loop`：

```bash
# 一个回合：每轮锁存箱子位姿、等待、走近、到位合手、夹起，失败重试，成功即停
python agentic/scripts/run_oracle.py --run-id <名字> --grasp-loop --seed 1000 --sim-gpu 2 --policy-gpu 2 --share-policy-sim-gpu

# 一组种子（debug 或 heldout，见 configs/grasp-loop.json），多张卡并行，然后汇总
python agentic/scripts/run_grasp_suite.py --group debug --tag <标签> --gpus 2 3 4
python agentic/scripts/report_grasp.py 'dbg-exp1-<标签>-*'
```

静态抓取门槛的结果（30 个预留种子成功 17 个，没到 26 个的门槛）和失败归因见 `docs/exp1-static-grasp-20261124.md`。

`run_oracle.py` 的默认值就是 11 月 17 日以来所有实验共用的后端。加上 `--settled-goal <目标文件>` 是上面那个结果的完整配置。更早的变体要显式写开关，例如 `--variant original --support-constraints none --hand-constraints keyframes --trajectory-projection none`；`offset` 和 `box` 两种模式必须这样写。

正式实验只在已提交的代码上跑：启动时记录提交号，有未提交的改动会拒绝启动，调试时加 `--allow-dirty`。每次运行结束后会删掉自己的仿真器缓存。

第二、三步读旧实验的输出作为输入：`vlm-refined-20261118-r1-<案例>/` 提供目标，`dynamics-actuator-20261119-r1-<案例>/` 提供对比基线。

## 目录

| 位置 | 内容 |
|---|---|
| `arena/` | 接入仿真器的代码，在仿真进程内运行 |
| `scripts/` | 入口和工具，见下表 |
| `tests/` | 接口与几何检查，不替代真权重和仿真测试 |
| `configs/` | 执行器与载荷对照用的参数；实验计划；箱子抓取的参数和种子分组 |
| `docs/` | 每个阶段的实验记录 |

相关的包内模块：`kimodo_vla/oracle/`（目标、连续性、IK 投影、支撑约束、箱子抓取的回合逻辑 `grasp_policy.py`）、`kimodo_vla/vlm/`（图像定位）、`kimodo_vla/model/sparse_conditions.py`、`kimodo_vla/model/native_policy.py`、`kimodo_vla/controllers/gripper_gate.py`。

## 脚本

| 脚本 | 用途 |
|---|---|
| `run_settled_suite.py` | **主入口**：对五个案例各调用一次 `run_oracle.py` |
| `report_settled.py` | 汇总一组结果并和旧基线对比 |
| `run_grasp_suite.py` | 箱子抓取：一组种子各跑一个回合，仿真器崩溃的回合自动重跑 |
| `report_grasp.py` | 箱子抓取：成功数、逐轮标记、接触率、夹住后滑落率 |
| `overview.py` | 生成实验进度页：按 `configs/experiments.json` 里的计划给每个实验算进度和成功率，每个实验目录可点开看误差曲线和视频。写到 `<实验输出>/overview/index.html`；加 `--serve 8765` 后在浏览器打开 `http://localhost:8765/overview/index.html`。页面模板是同名的 `overview.html` |
| `run_oracle.py` | 启动器：起策略服务和仿真，检查 GPU，记录提交号、权重与源码哈希 |
| `serve_oracle.py`、`serve_fixed_reference.py`、`serve_recorded_reference.py` | 三种参考来源的策略服务，由 `run_oracle.py` 启动 |
| `*_sim_bootstrap.py` | 仿真进程的启动钩子，由 `run_oracle.py` 按开关选择 |
| `analyze_root_dynamics.py`、`analyze_sonic_replay.py` | `report_settled.py` 用到的误差分解 |
| `run_motion_suite.py` | 实验二的入口：按 `configs/exp2-motions.json` 里的组（向前、侧步、后退、前倾、18 点到达）逐个调用 `run_oracle.py --motion-goal` |
| `analyze_motion.py` | 给走动 / 到达 / 前倾的运行打分：跌倒、完成比例、手腕误差（目标→参考、参考→实际）、前倾角；`--markdown` 生成记录里的表 |
| `rollout_kimodo.py` | 不开仿真，只让 Kimodo 把一个目标连续生成出来，半分钟看步态；`--against <运行>` 给出 3 s、5 s 的连续预测偏差 |
| `capture_pointing_frames.py` | 采指点测试的第一视角 RGB-D 图像集：按 `configs/pointing-frames.json` 摆物体和机器人站位，每个场景启动一次仿真；模型输入写到 `<数据集>/inputs`，只供评测的真值写到 `<数据集>/truth`，`manifest.json` 列出每帧的覆盖类别、实测可见度和文件哈希。`pointing_sim_bootstrap.py` 是它的仿真启动钩子 |
| `run_pointing.py` | `run`：用一个后端（`qwen`、`openai`、`gemini`）跑完图像集，逐帧记录输入、原始输出、延迟和 token，只读 `inputs/`；`evaluate`：对照真值打分。闭源后端的模型版本和单价写在 `configs/pointing-backends.json`，密钥从环境变量 `OPENAI_API_KEY`、`GEMINI_API_KEY` 读 |
| `analyze_prediction.py` | 只读已有运行：把每次规划里 Kimodo 的原始预测和 0.5 / 1.0 / 1.5 s 后的实测状态对比，并给出“假设身体不动”的对照。写到 `<实验输出>/<组名>/prediction.json` |
| `prepare_vlm_model.py`、`cache_vlm_motion_text.py` | 一次性准备：下载 Qwen 权重、缓存运动短语的文本向量 |
| `paths.py` | 所有位置的唯一来源 |

红蓝标记的 Qwen 定位是包内的两个可执行模块：`python -m kimodo_vla.vlm.infer_points` 和 `python -m kimodo_vla.vlm.infer_refined`。自然物体的指点用 `kimodo_vla/vlm/pointing.py`（提示词、输出格式、单次与两尺度两种做法）、`backends.py`（三个后端）和 `pointing_eval.py`（真值几何与打分）。任务指令的文本向量用仓库根目录的 `scripts/cache_text_embeddings.py`。

## 历史

这条线在 2026-11-23 之前位于 `vlm version/`，带着一份复制的包。`docs/` 里早期阶段的记录会提到已删除的分析脚本和旧路径，它们保存在提交 `13f8200` 里，用 `git show 13f8200:"vlm version/scripts/<名字>.py"` 取回。
