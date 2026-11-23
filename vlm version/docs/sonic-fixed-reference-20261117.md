# SONIC 固定合法参考对照 — 2026-11-17

固定参考已在开发机 HumanoidArena 跑完。保持与小幅伸手各 400 步、8 秒，均无生成模型参与。双腕平均跟踪误差约 2–3 cm；上一轮 Kimodo 参考下为 21–31 cm。主要排查方向应转到生成目标误差和生成后关节轨迹的可执行性，而不是将大误差归结为 SONIC 的普遍精度下限。

## 实验与口径

- 实现、运行产物均位于开发机 `/root/workspace/humanoidVLA/vlm version`，共享 Arena 源码未修改。
- 保持：启动时实测 29 关节与 root 位姿固定。伸手：双侧 shoulder pitch 减少 0.15 rad，1.2 秒五次曲线过渡，之后保持；目标腕位移 5.638 cm。
- 两端关节都在 XML 限位内，插值保持限位，端点速度和加速度为零。这里“合法”指运动学限位与连续性，不预先声称动力学可行。
- root 和目标只在启动时确定，不跟随实测漂移。50 Hz、每次 HTTP 发送 25 帧；不开夹爪，不使用 IK 修正。
- 与上一轮同任务、seed 1000、SONIC 权重路径及配置路径；初始 root 和全部 29 关节逐值一致。当前权重 SHA256 保存在 `run.json`，实际 Arena 三个关键文件 SHA256/mtime 保存在 `arena-source-audit.json`，源码副本在 `runs/sonic-source-audit-20261117`。旧运行未保存共享 Arena 哈希，因此这是与上一轮的历史对照，不声称完成了同时锁定全部依赖的重新配对实验。
- 主表使用开始后 `(1.2, 4.0]` 秒，共 140 步，与上一轮对齐。括号内为左/右腕，位置单位 cm。误差是不同向量的范数，不能直接相加或相减。

## 分阶段实测

| 指标，均值 | Kimodo 保持（旧） | 固定保持 | Kimodo 伸手（旧） | 固定伸手 |
|---|---:|---:|---:|---:|
| 已发送参考腕位置 → 目标 | 14.69 / 25.00 | 0 / 0 | 18.99 / 22.63 | 0 / 0 |
| 已发送参考 → Arena 消费参考 | 0.35 / 0.37 | <0.000001 | 0.48 / 0.49 | <0.000001 |
| Arena 消费参考 → 实测腕位置 | 28.54 / 31.25 | **2.41 / 2.05** | 21.07 / 21.11 | **2.52 / 2.59** |
| 实测腕位置 → 目标 | 27.85 / 45.01 | 2.41 / 2.05 | 21.36 / 27.78 | 2.52 / 2.59 |
| 双腕同时 ≤3 cm 的帧占比 | 0% | **100%** | 0% | **95%** |

固定保持的腕跟踪误差 p95 为 2.81 / 2.15 cm；固定伸手为 2.99 / 2.81 cm。后续 `(4,8]` 秒 200 步中，保持均值 2.37 / 2.06 cm，伸手 2.29 / 2.79 cm，两组双腕 ≤3 cm 帧占比均为 100%。这是单种子短时姿态测试，不是抓取成功率。

主表中 Kimodo 的生成误差来自实际发送的 50 Hz 帧，和上一报告中“未来稀疏关键帧约 13–17 cm”的统计对象不同。原始位置特征及旋转 FK 在投影前已有明显误差，故问题也不只发生在 tracker 之后。

## 固定参考下 tracker 的剩余误差

把同一参考关节放到实测 root 位姿上做 FK，得到中间腕位置，分别计算姿态与 root 的贡献：

| `(1.2,4]` 秒均值 | 保持 | 伸手 |
|---|---:|---:|
| 实测腕 → 实测 root 上的参考 FK，cm | 2.28 / 1.80 | 2.65 / 2.70 |
| 上述 FK → 世界系参考腕，cm | 2.32 / 2.51 | 2.20 / 2.20 |
| root 位置误差，cm | 2.14 | 2.11 |
| root 完整旋转误差，度 | 2.49 | 0.51 |
| 参考 → 实测手臂关节 RMSE，rad | 0.0678 | 0.0632 |
| SONIC decoder 输出 → 实测手臂关节 RMSE，rad | 0.0582 | 0.0904 |

两种腕误差向量的和严格等于总跟踪误差向量，但范数存在抵消。decoder 输出是 PD 位置目标，并非应当无偏复现的参考姿态；这项差值含力矩补偿和控制时序，不能直接判作 decoder 精度差或电机故障。独立 XML FK 与仿真腕位置差均值 <1 微米，未发现厘米级 FK/关节顺序错误。

实际运行的 `action_provider_sonic.py:5077` 将最新 joint29 参考重复成 encoder 窗口；root 高度输入置零，encoder/decoder 均没有世界系 root XY 位置误差输入（见 5068、5199、5270 附近）。因此此接口不能直接闭环纠正绝对 XY 目标；不过本次 root 位置误差只有约 2.1 cm，不能用此限制解释所有旧运行的大误差。

## 已找到的生成后参考问题

审计实际发出的前 200 帧关节标量，不修改或平滑数据：

| 指标 | Kimodo 保持 | Kimodo 伸手 | 固定保持 | 固定伸手 |
|---|---:|---:|---:|---:|
| 至少一个关节超 XML 限位的帧 | **108/200** | **104/200** | 0/200 | 0/200 |
| 最大单步关节标量变化，度 | 254.49 | 338.66 | 0 | 0.268 |
| 重规划交接的最大单关节变化范围，度 | 40.46–115.59 | 42.26–143.37 | 0 | 0–0.257 |

超限主要发生在踝关节；伸手最大单步来自 right_hip_yaw 从 -2.832 到 3.078 rad。部分大跳变可能包含旋转表示的分支切换，不能解释成机器人真的瞬间转了这么多。但 joint29 接口消费的是关节标量，并直接做相邻差分计算速度（`vla_robot_current_local_runtime_v3.py:421`），所以这不是可忽略的日志现象。当前矩阵到关节角的转换、角度连续性、限位与重规划交接均需处理。

结论分三层：

1. **生成层已有目标误差**：参考进入 tracker 之前就没有到达目标。
2. **生成后关节轨迹已有可执行性问题**：超限和突变使旧运行的“tracker 误差”混入参考质量与分布匹配问题；尚不能量化各问题分别贡献多少。
3. **SONIC 能跟随这两种温和合法参考，但仍有约 2–3 cm 残差**：这足以支持继续修复 A，不足以证明接触精度或复杂搬运能力。

下一步建议先保留本对照不变，独立修复矩阵到合法连续关节角的转换和每 0.5 秒重规划交接，再重新运行 A；分别报告目标精度损失与跟踪改善。不要用简单夹限后腕目标偏移掩盖生成误差。完成这一步后再决定 VLM 或视觉注入，当前不需要训练 B/C。

## 复现与产物

在开发机 `vlm version` 目录运行，GPU 编号须选空闲卡：

```bash
../cross_tracker/data/cache/envs/policy/bin/python scripts/run_oracle.py --run-id fixed-hold-new --reference-source fixed --mode hold --steps 400 --sim-gpu 2
../cross_tracker/data/cache/envs/policy/bin/python scripts/run_oracle.py --run-id fixed-reach-new --reference-source fixed --mode reach --steps 400 --sim-gpu 3
PYTHONPATH=. ../cross_tracker/data/cache/envs/policy/bin/python scripts/summarize_fixed.py runs/fixed-hold-new
PYTHONPATH=. ../cross_tracker/data/cache/envs/policy/bin/python scripts/audit_reference_joints.py runs/fixed-hold-new
```

- 本次运行：`runs/sonic-fixed-hold-20261117-r1`、`runs/sonic-fixed-reach-20261117-r1`。
- 每组保存 `commands.jsonl`、`execution.jsonl`、`fixed-aligned.jsonl`、`fixed-summary.json`、`joint-reference-audit.json`、配置、日志、录像及源代码哈希。完整运行目录在开发机，本地同步了指标、日志、配置与录像。
- 新增固定参考源、CPU HTTP 服务、运行入口开关、root/肢体/decoder 指标及两项分析脚本。原生 Kimodo 仍是默认入口。
- 本地与开发机 22 项测试通过；两组仿真均退出码 0、400 步完成，所用 GPU 2/3 已释放。Arena 搬箱任务显示 timeout 是预期结果：没有执行抓取或搬运，不能称任务成功。
