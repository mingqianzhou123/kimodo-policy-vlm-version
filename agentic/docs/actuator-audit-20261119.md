# SONIC → 执行器静差审计（2026-11-19）

结论：五组固定参考的执行器目标传递无误；肩部静差与实际重力负载 / 有限位置刚度的平衡偏差高度一致。当前集成混用了 SONIC 增益与 Arena 的执行器限幅、armature 和机器人载荷配置。这轮只诊断，没有调 PD、修改动力学、VLM 或夹爪。

## 对照与测量边界

- 保留原版快照 `snapshots/sonic-before-actuator-audit-20261119`，147 个原文件已按原 manifest 校验。
- 原始五组消费参考、目标、场景、tracker 权重和原始桥接逻辑均通过逐项核验。共 2,000 控制帧 / 8,000 个 5 ms 物理步；控制周期 20 ms。五组实际双腕轨迹相对原版的最大差均为 0。
- SONIC 解码目标 → PhysX 位置目标最大差 0 rad；provider Kp/Kd → PhysX Kp/Kd 最大差 0。未加载 VLM、Kimodo 或在线 IK。
- 统计静差窗口为每组第 4–8 秒。下表重力对照采用每个控制周期最后一个物理步，与 link 位姿严格同步；完整 200 Hz 统计另存 JSON。关节静差定义为实际关节角减 SONIC 解码目标；腕跟踪误差定义为实际腕位置减冻结执行参考，不是 VLM 选点误差。
- r1 首次采样因 Arena 重置后缓存 PhysX view 失效而停止；失败目录保留。r2 每步获取当前 view，完整五组均通过，控制路径不变。

## 五组结果

| 组别 | 右腕跟踪均值（到达后，cm） | 右肩 pitch 静差 / 重力预测（°） | 右肩 roll 静差 / 重力预测（°） |
|---|---:|---:|---:|
| base-center | 4.495 | 7.573 / 7.540 | 7.439 / 7.414 |
| base-right | 3.479 | 7.215 / 7.190 | 7.566 / 7.539 |
| far-center | 3.577 | 10.179 / 10.129 | 7.700 / 7.663 |
| far-left | 3.167 | 9.847 / 9.809 | 7.619 / 7.593 |
| far-right | 3.340 | 10.025 / 9.970 | 7.650 / 7.620 |

右肩 pitch/roll 的平均静差与重力预测之差最大 **0.055°**。重力矩分别通过 PhysX API 和实际质量、COM、link 位姿、USD 关节轴独立计算，两者全关节最大差 **0.000043 N·m**。

`q_actual − q_target ≈ −τ_gravity_comp / Kp`。位置控制必须保持一定角度偏差才能提供托住手臂的力矩。SONIC 解码目标本身可以学到提前量，所以该偏差不等于相对原始参考的净误差；最后几厘米仍包含 tracker 目标补偿残差和根部误差，提前量可能过大，也可能过小。

例如 base-center 右肩 roll：SONIC 目标相对参考 -13.34°，实际相对 SONIC 目标 +7.44°，相加才是实际相对参考 -5.90°。让实际角度完全贴住 SONIC 输出不等于消除参考跟踪误差，直接加重力补偿可能重复补偿。

## 实际生效的配置

| 右臂关节 | 实际 Kp / 官方当前 Kp | 实际 Kd / 官方当前 Kd | 实际力矩上限 / 官方当前上限（N·m） | 实际 / 官方当前 armature（kg·m²） |
|---|---:|---:|---:|---:|
| shoulder_pitch | 14.223 / 14.251 | 0.454 / 0.907 | 60 / 25 | 0.000000 / 0.003610 |
| shoulder_roll | 34.223 / 14.251 | 0.454 / 0.907 | 60 / 25 | 0.000000 / 0.003610 |
| shoulder_yaw | 24.223 / 14.251 | 0.454 / 0.907 | 60 / 25 | 0.000000 / 0.003610 |
| elbow | 19.223 / 14.251 | 0.454 / 0.907 | 60 / 25 | 0.000000 / 0.003610 |
| wrist_roll | 14.223 / 14.251 | 0.454 / 0.907 | 60 / 25 | 0.000000 / 0.003610 |
| wrist_pitch | 16.779 / 16.778 | 0.534 / 1.068 | 60 / 5 | 0.000000 / 0.004250 |
| wrist_yaw | 16.779 / 16.778 | 0.534 / 1.068 | 60 / 5 | 0.000000 / 0.004250 |

- SONIC 在位置控制模式只改写 PhysX Kp/Kd，没有同步替换 Arena 的力矩上限和 armature。肩 roll/yaw、elbow 与 ankle pitch 的 Kp 还有本地手动加值；不是一整套一致的官方配置。
- Provider 的动作缩放与默认角度也按关节名重排核对：相对当前官方定义最大差分别为 1.89e-08 和 2.96e-08 rad，未发现此处顺序或尺度错误。
- 全身 29 个控制关节的附加转子惯量（armature）为 0，不代表机器人 link 本体惯量为 0；关节摩擦系数为 0。运行时没有输出延迟、外加力或力矩，速度目标及前馈力矩均为 0。重力启用，自碰撞关闭，位置/速度求解迭代为 8/4。
- 五组实际角度及位置目标均未超限；全程双臂使用实际 Kp/Kd 估算的 PD 力矩最大仅达实际上限的 23.6%。接触传感器在手臂/手部记录为 0，未抓箱，排除本轮持物或桌面接触解释。该估算不等于隔离测得的电机力矩。
- `IsaacLab applied_torque` 在隐式执行器中按旧缓存增益估算，不能当实际驱动力矩。其 arms 缓存仍为 40/5 或 20/1，而 PhysX 已采用上表增益。报告保留旧估算、实际增益 PD 估算和 PhysX projected joint force，明确区分三者。

## 载荷与惯量

运行时整机 **38.61423 kg**；官方 URDF 显式 inertial 质量合计 **34.39423 kg**。共同有惯量声明的 link，其质量最大差 2.10e-07 kg、COM 最大差 7.24e-09 m、惯量矩阵最大差 6.25e-09 kg·m²，未发现主体惯量抄错。

| link | 当前质量（kg） | 官方 URDF 状态 |
|---|---:|---|
| imu_in_pelvis | 1.00 | 存在但未声明 inertial |
| d435_link | 1.00 | 未包含此 link |
| imu_in_torso | 1.00 | 存在但未声明 inertial |
| mid360_link | 1.00 | 存在但未声明 inertial |
| left_hand_camera_base_link | 0.11 | 未包含此 link |
| right_hand_camera_base_link | 0.11 | 未包含此 link |

对运行中相同 SHA256 的 USD 做了独立只读检查：这四个 1 kg sensor body 的 `physics:mass` 均 **没有 authored value**，USD 读取的是默认值 0；它们进入 PhysX 后的实际质量均为 1 kg。不是源码明确指定了四个 1 kg 传感器，问题出在未明确声明质量后的导入结果。证据为 `runs/actuator-audit-20261119/asset-body-properties-verified.json`。

**不能将 4.22 kg 全部断言为相对原始训练环境的多余载荷。** 官方 URDF 也有三个无 inertial 的 sensor link，尚无该 checkpoint 原始转换后 USD 的质量快照；它们是否在训练导入时成为默认质量仍需核对。可以确定当前新增的 D435 与双手相机合计 1.22 kg；四个 1 kg sensor body 都确实在当前 PhysX 中存在。

双手相机各 0.11 kg 对肩部的重力矩贡献已单独离线计算，见各 `actuator-analysis.json` 的 `hand_camera_gravity_mean_nm`。躯干/骨盆传感器不直接属于肩部关节的子树，影响整机支撑与根部动力学，不能用它们直接解释全部手臂静差。

## 下一步（本轮未执行）

1. 先建立与当前 SONIC 权重对应的可追溯动力学基线：核对原始导入选项、无惯量传感器默认质量和 checkpoint 版本。当前官方源码只作为有版本记录的参照，不冒充该权重精确训练配置。
2. 保留本轮五组冻结参考，分别做“载荷一致化”和“完整执行器配置一致化”的单因素对照；不能同时改质量、增益、armature 后把改善归于某一项。自碰撞若需调整也应单独测。
3. 若配置核对后仍有净腕偏差，再比较重力前馈或目标补偿。SONIC 可能已经学到部分重力补偿，直接再叠加会重复补偿；本轮重力平衡证据不是直接加力矩的授权结论，也不证明某组新 PD 能稳定双腕 3 cm。

## 证据与复现

- 结果：`runs/actuator-audit-20261119-r2/actuator-summary.json`、`configuration-comparison.json`。
- 每组：`runs/actuator-audit-20261119-r2-<case>/actuator-config.json`、`actuator-physics.jsonl`、`actuator-controls.jsonl`、`actuator-usd-joints.json`、`actuator-analysis.json`。
- 源码与 API 证据：`runs/actuator-audit-20261119/environment-source`、`api-source`、`official`；官方文件均保存 URL、固定 revision 和 SHA256。
- 执行：`PYTHONPATH=. <policy-python> scripts/run_sonic_replay_suite.py --suite <new-id> --actuator-audit --sim-gpu 7 --gpu-min-free-mib 16384`。
- 重建报告：`PYTHONPATH=. <policy-python> scripts/report_actuator_audit.py --suite actuator-audit-20261119-r2`。

官方对照：[当前固定版本机器人配置](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/envs/manager_env/robots/g1.py)、[部署参数](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp)、[URDF](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf)。

[当前 root SONIC 权重的发布配置](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_release/config.yaml) 保存了质量随机化、初始关节偏移和地面材料随机化等训练设置，因此以上名义配置差异不能单独证明因果。
