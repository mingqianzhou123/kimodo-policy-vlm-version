# Kimodo VLM v6.1

独立实验版本，开发机目录为 `/root/workspace/humanoidVLA/vlm version/`。
第一阶段是实验 A：在 HumanoidArena 搬箱子场景中，用仿真真值手部目标测试冻结 Kimodo 与 SONIC。

已实现独立源码、未来稀疏条件输入、真值状态历史、原生冻结权重推理、SONIC 执行和逐步误差记录。
**已增加 Qwen 图像目标接入，保留 oracle 对照；这是视觉标记到达实验，不等于自然物体抓取。接触和夹爪门控未启用，夹爪保持打开。**
现有 `serve_arena.py` 仍是基线入口，不能把运行它的结果当作 v6.1。

**两尺度 VLM 精定位对照已完成：9/9 执行，双腕视觉误差均值 0.49 / 0.48 cm；4/9 组到达后全程双腕 3 cm 内（旧版 1/9）。后端冻结，右腕实际跟踪仍有明显残差。**

**固定失败参考的 SONIC 独立回放已完成：5 组 × 400 帧，消费参考及实际双腕轨迹均与原实验逐帧相同。误差定位到参考后的执行链路：根部平移 2.25–2.64 cm，右腕根部相对误差 2.71–3.22 cm；以持续偏差为主。VLM、Kimodo、IK 均未在回放中加载。**

**SONIC → 执行器静差审计已完成：同五组、8,000 个物理步，目标传递误差为 0，实际轨迹与原版完全相同。右肩 pitch/roll 静差与重力平衡预测的平均差最大 0.055°。发现 SONIC 增益与 Arena 限幅/armature 混用，以及四个未显式声明质量的传感器刚体在 PhysX 中各为 1 kg；本轮未改 PD、动力学、VLM 或夹爪。**

**载荷/执行器独立对照已完成：五组左/右腕均值，当前 1.66/3.61 cm、只改载荷 2.65/2.72 cm、只改执行器 2.53/2.90 cm；全程双腕 3 cm 分别 0/5、3/5、0/5。执行器组腕相对根部误差降到 1.04/0.61 cm，但根部平移增至 2.79 cm。载荷组的传感器 1 g 是诊断近似，不是实物标定。默认后端不变，原版保存在 `snapshots/actuator-before-dynamics-20261119`。**

**2026-11-20 归因更正：冻结参考的脚底距原地面约 2.083 cm。初始未落地姿态被保存为根部/双脚支撑目标，因此固定回放复现误差不能证明 reference 动力学可行，也不能把下沉全部归咎于 SONIC。已完成固定 reference 与已对齐执行器的载荷 × 支撑面四格诊断：减载/支撑面/组合的根部均值为 2.56/2.38/2.10 cm，双腕全程 3 cm 为 2/5、2/5、3/5。支撑面消除了大部分下沉，但留下约 2 cm 的水平偏移，主要表现为固定双脚上方的根部前倾。支撑面组属于环境反事实，不计作原场景修复。**

**2026-11-23 原地面落地后初始化**：五个原失败案例保持同一视觉世界目标，稳定双支撑后重新生成参考。双腕实际相对消费参考和真值标记均由严格 0/5 提升至 5/5。根部三维平均跟踪误差由 2.785 降至 0.810 cm，最大真值标记误差 2.445 cm。新参考自身保留约 2.055° 前倾。使用独立的 `--settled-goal` 入口。

- [五案验收、前倾分解与归档](docs/settled-start-20261123.md)
- [落地判据与世界目标保持协议](docs/settled-start-protocol-20261123.md)
- [根部误差归因：载荷、支撑高度与全身姿态](docs/root-dynamics-20261120.md)
- [根部载荷/支撑对照协议](docs/root-dynamics-protocol-20261120.md)
- [权重核对与载荷/执行器独立对照：结果和限制](docs/dynamics-comparison-20261119.md)
- [动力学对照协议](docs/dynamics-protocol-20261119.md)
- [执行器静差、增益与载荷审计：五组固定参考](docs/actuator-audit-20261119.md)
- [失败组固定参考回放：根部、腰部和右臂误差分解](docs/sonic-replay-20261118.md)
- [VLM 两尺度精定位与异常核验：九组配对结果](docs/vlm-refinement-20261118.md)
- [VLM 定位稳定性：固定后端九组对照](docs/vlm-stability-20261117.md)
- [VLM 第一阶段：图像目标接入](docs/vlm-integration-20261117.md)
- [IK 保持阶段冗余关节稳定](docs/kimodo-ik-hold-20261117.md)
- [连续手部目标与 IK 投影对照](docs/kimodo-continuous-ik-20261117.md)
- [根部与双脚约束对照](docs/kimodo-support-20261117.md)
- [历史条件、重规划交接与关节连续性对照](docs/kimodo-continuity-20261117.md)
- [Kimodo 约束与关节转换诊断：56 次真实权重对照](docs/kimodo-constraint-diagnosis-20261117.md)
- [SONIC 固定参考对照：误差分层与关节轨迹问题](docs/sonic-fixed-reference-20261117.md)
- [实验 A 记录与结论](docs/oracle-a-20261117.md)
- [具体实现计划](docs/implementation-plan.md)
- `kimodo_vla/vlm/policy.py`、`scripts/infer_vlm_points.py`：真实 Qwen 图像定位、深度反投影、连续手部目标，复用冻结的 IK-hold 后端。
- `scripts/infer_vlm_refined.py`、`kimodo_vla/vlm/refinement.py`：两尺度 VLM 选点与颜色核验；以 `--vlm-localizer refined` 启用，`baseline` 保留旧版。
- `scripts/compare_vlm_stability.py`：新旧条件配对及同一 RGB-D 帧的粗点/精定位比较。
- `scripts/review_vlm.py`：分别统计视觉定位、投影后参考、实际跟踪和真实目标误差。
- `scripts/run_oracle.py`：启动独立 native policy 与 Arena，检查空闲 GPU，保存权重/源码哈希和源码快照。
- `kimodo_vla/oracle/policy.py`：固定世界目标、实测历史、30 Hz 生成、50 Hz 参考；关闭图像分支。
- `arena/oracle_bridge.py`：进程内接入真实状态与实际消费参考，不修改共享 Arena。
- `scripts/summarize_oracle.py`、`scripts/audit_oracle.py`：分别统计关键帧误差、对齐后的参考传递和跟踪误差。
- `kimodo_vla/oracle/continuity.py`：实测边界条件、参考交接和有限关节角连续投影。
- `kimodo_vla/oracle/hand_path.py`：按绝对时间生成合法 oracle 手部路径，`--hand-constraints continuous` 条件覆盖实际执行区间。
- `kimodo_vla/oracle/projection.py`：`--trajectory-projection ik-hold` 在目标不变且参考已到位时保持已消费关节，跨规划包抑制冗余运动；`--trajectory-projection ik` 启用有速度/限位约束的双腕与双脚 IK，保留模型姿态先验、修正前后参考和求解日志。
- `scripts/review_ik_hold.py`：对照原 IK 与保持版的逐帧双腕精度、关节运动和跨规划保持状态。
- `scripts/summarize_continuous.py`：核对连续条件值与 mask，按逐帧双腕实际误差验收 3 cm。
- `kimodo_vla/oracle/support.py`：固定初始根部/双脚的原生未来条件，`--support-constraints none|root|root-feet|root-feet-keyframes`，默认关闭。
- `scripts/summarize_support.py`：手部、根部、脚部的生成、传递、跟踪和相对固定目标的漂移。
- `scripts/summarize_continuity.py`：新变体的原始生成、修正参考、传递、跟踪与到位误差对齐统计。
- `scripts/verify_oracle_targets.py`：离线验证目标存在限位内的 IK 解；不修正模型输出。
- `kimodo_vla/model/kimodo_policy.py`：`predict_future` 新增未来条件与 bool mask。
- `kimodo_vla/model/sparse_conditions.py`：root 与双手关键帧转换，先归一化再 mask。
- `kimodo_vla/controllers/gripper_gate.py`：按实测到位执行、保持、超时、去重。
- `tests/`：77 项接口、CFG、独立 FK、目标可达性、执行器重力与干预隔离检查；不替代真权重与仿真测试。
- `--dynamics-profile sensor-load|actuator|actuator-load`：仅在固定参考回放且开启 `--actuator-audit` 时允许，通过 `arena/dynamics_override.py` 在当前仿真进程应用单因素参数，默认关闭；`scripts/verify_sonic_release.py` 核对发布权重/配置，`scripts/report_dynamics.py` 核对实际生效值和相同参考后生成比较。
- `--actuator-audit`：只允许固定参考回放，附加 `arena/actuator_audit.py` 的只读 PhysX 采样。`scripts/analyze_actuator_audit.py` 与 `scripts/report_actuator_audit.py` 独立核验目标、真实生效参数、重力平衡及五组原版一致性；`scripts/inspect_actuator_asset.py` 对相同 SHA256 的 USD 做只读检查。
- `scripts/run_sonic_replay_suite.py`：冻结五组原动作块、CPU 只读回放、核验 400 帧消费参考；`scripts/analyze_sonic_replay.py` 分解根部/腰部/双臂并统计右臂七关节，`scripts/report_sonic_replay.py` 从通过核验的结果生成报告。
- `scripts/serve_fixed_reference.py`、`scripts/summarize_fixed.py`：固定合法参考的 CPU 服务与分阶段误差统计；运行时使用 `--reference-source fixed`。
- `scripts/audit_reference_joints.py`：发送关节的 XML 限位与重规划交接审计。
- `source-provenance.json`：源版本来源；`deployment-manifest.json`：部署文件哈希。

本地验证，在本目录运行：

```bash
../.venv/bin/python -m pytest tests -q
```

开发机验证，在本目录运行：

```bash
../cross_tracker/data/cache/envs/policy/bin/python -m pytest tests -q
../cross_tracker/data/cache/envs/policy/bin/python scripts/run_oracle.py \
  --run-id oracle-hold-new --mode hold --steps 200 --sim-gpu 1 --policy-gpu 2
../cross_tracker/data/cache/envs/policy/bin/python scripts/summarize_oracle.py runs/oracle-hold-new
PYTHONPATH=. ../cross_tracker/data/cache/envs/policy/bin/python scripts/audit_oracle.py runs/oracle-hold-new
```

`--variant original` 保留原版默认路径；`handoff` 启用短历史与 0.2 s 交接；`bounded` 再加入 XML 限位与默认 2 rad/s 变化上限。新变体使用 `scripts/summarize_continuity.py` 统计执行结果，不能使用旧 `audit_oracle.py` 的原始参考假设。

默认 `--text-mode none --constraint-cfg 2 --diffusion-steps 50 --root-anchor reference`。
`--mode reach` 从初始姿态将双侧 shoulder pitch 减少 0.15 rad，通过 FK 生成双腕位置、
姿态和虚拟手端目标，并保存对应的合法关节配置。`hold` 保持初始实测腕位姿。
目标均在启动时确定，到达期限固定为启动后 1.2 秒；之后继续保持。
`offset` 为历史探索对照：双腕平移 5 cm 且锁定姿态，尚未找到固定根姿态下的精确限位内解，
不作为严格可达目标结果。`box` 只提供未标定模板，不作为抓取评测。
`task` 文本模式只用于复现旧冒烟对照，使用已有搬箱 instruction 缓存，无约束 CFG；
不能视作正确的阶段 motion text。完整 VLM 版本须补齐冻结 LLM2Vec 的运动短语编码。

调用此版本须以本目录为工作目录并将本目录放在 `PYTHONPATH` 最前，不要 `pip install -e .`
覆盖共享环境里原项目的安装。模型权重以后按 manifest 指向共享只读权重，新的 VLM/LoRA
放在本目录自己的 `checkpoints/`；不要复制旧训练输出或在此启动基线正式评测。
