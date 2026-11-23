# SONIC 载荷与执行器独立对照（2026-11-19）

同五组固定参考，对照当前版、只改载荷、只改执行器。VLM、Kimodo、IK、夹爪控制及共享 Arena 文件均未修改。新两组共 10 次仿真、4,000 控制帧、16,000 物理步；每组消费参考与原版逐帧核验通过。

## 结论

- **还没有全部修好**：到达后全部记录帧的双腕 3 cm 达标组数，当前 0/5、载荷 3/5、执行器 0/5。两套实验配置均保留为可选项，没有替换默认后端。
- **只改载荷改善右腕、牺牲左腕**：右腕五组均值 3.61 → 2.72 cm；左腕 1.66 → 2.65 cm。腕相对根部和净关节误差基本没变，不能解释为手臂控制整体变准。载荷会改变全身姿态及各误差向量的抵消关系。
- **公开执行器配置显著改善腕相对根部跟踪，但根部更差**：左/右相对误差 3.17/2.89 → 1.04/0.61 cm；根部平移 2.42 → 2.79 cm。全局腕均值仍有左右权衡。这支持执行器配置不匹配是局部跟踪问题的重要因素，但不能归因到某一个增益，也不能说所有剩余误差都只有根部平移一个来源。
- **不应追求把实际−SONIC 指令静差直接调到零**：执行器组该误差变大，实际−参考的左右臂 RMSE 却平均下降约 10%。SONIC 输出本身含有相对参考的偏置，直接消除静差可能破坏两者的补偿关系。
- 后续优先验证根部/全身动力学：以执行器对照为一个已知分支，先补齐导入后质量、固定手结构与碰撞等训练/部署差异的证据；若测试载荷和执行器组合，应作为独立新组与本轮两组对照，不能预设改善会相加。现阶段没有证据要求重训 VLM 或继续改其选点。

执行器组使用 GPU 1，载荷组使用 GPU 7。先在 GPU 1 重跑不干预的 base-center：400 帧双腕位置与原版最大差 0.000000 cm，满足预先设定的 0.01 cm 等价阈值；未以更换 GPU 后的偏差冒充配置改善。

## 权重与训练配置核对

- 当前 encoder/decoder 的 SHA256 与固定 HF revision 的 LFS SHA256 完全一致，属于 default SONIC；发布 manifest 明确对应 `sonic_release/config.yaml` 与 `sonic_release/last.pt`。配置文件的 Git blob ID 也已核验。
- 配置指定 `g1_model_12_dex`、physics dt 0.005 s、decimation 4；公开训练代码映射到 `G1_CYLINDER_MODEL_12_DEX_CFG`。执行器参数取首次公开训练代码 commit `daf389964fa4a4545218e8405f24eb55f4912453`，不是按实验结果试调增益。
- 独立解析公开源码的常数、公式和关节正则，145 个执行器参数转换为 PhysX float32 后全部相同；Python float64 的最大差为 5.7e-9，来自公开源码截断 π，未影响实际生效值。记录在 `runs/dynamics-20261119/profile-source-audit.json`。
- 核实到的是发布对应关系与公开名义配置；精确训练 commit、Isaac Sim/IsaacLab 版本、历史转换 USD 和每次随机化样本没有恢复，不能宣称完整复原了训练现场。公开 URDF 的手部关节全部 fixed；Arena 有 14 个可动手指关节。自碰撞、导入合并、手部结构等差异本轮保留，避免混入这两类对照。
- 当前 IsaacLab 2.2 导入默认 `merge_fixed_joints=True`、`link_density=0`。这与 Arena 已导入的多刚体结构不同；不能仅用原始 URDF 的质量和代替训练时 PhysX 实际质量。

## 单因素干预

- **载荷组**：只把 imu_in_pelvis、imu_in_torso、d435_link、mid360_link 的默认 1 kg 各降为 0.001 kg，并按质量比缩放对应惯量，COM、相机位姿、显式定义的双手相机及手部质量、全部执行器配置不变。1 g 是诊断用正质量近似，不是真实传感器标定或已验证的训练参数。
- **执行器组**：质量/COM/本体惯量不变；按公开训练配置对齐 29 个身体关节的 Kp、Kd、力矩上限、速度上限、附加转子惯量。14 个手指执行器、控制频率、参考、初始状态、碰撞设置不变。这是一个执行器参数组对照，不能把效果单独归功于其中 Kp 或 Kd。
- 两组都在第一控制周期、SONIC 设置完成后、第一物理步之前应用一次。保存干预前后数据，确认仅预定字段变化；没有同时改两组参数。当前版完整保存在 `snapshots/actuator-before-dynamics-20261119`（154 个文件）。

## 双腕跟踪误差

单位 cm；统计到达后 1.2–8 s，参考为冻结执行轨迹。L/R 为左/右腕；“全程”指 50 Hz 日志中全部到达后控制帧，不额外推断帧间峰值。均值降低不等于全程双腕 3 cm 达标。

| 组别 | 当前 L/R 均值 | 只改载荷 L/R 均值 | 只改执行器 L/R 均值 | 全程双腕 3 cm（当前/载荷/执行器） |
|---|---:|---:|---:|---|
| base-center | 1.057 / 4.495 | 2.686 / 2.791 | 1.951 / 3.396 | 否 / 是 / 否 |
| base-right | 1.442 / 3.479 | 2.543 / 2.664 | 1.976 / 3.250 | 否 / 否 / 否 |
| far-center | 1.928 / 3.577 | 2.492 / 2.711 | 3.297 / 2.480 | 否 / 是 / 否 |
| far-left | 2.031 / 3.167 | 3.022 / 2.707 | 2.664 / 2.703 | 否 / 否 / 否 |
| far-right | 1.851 / 3.340 | 2.528 / 2.711 | 2.740 / 2.667 | 否 / 是 / 否 |

| 五组平均 | 左/右腕均值（cm） | 双腕达标组数 | 根部平移均值（cm） | 腕相对根部误差 L/R（cm） |
|---|---:|---:|---:|---:|
| 当前 | 1.662 / 3.612 | 0/5 | 2.421 | 3.166 / 2.890 |
| 只改载荷 | 2.654 / 2.717 | 3/5 | 2.164 | 3.164 / 2.917 |
| 只改执行器 | 2.526 / 2.899 | 0/5 | 2.785 | 1.035 / 0.608 |

## 关节静差与净跟踪误差

第 4–8 s，单位 °；每侧七个手臂关节跨帧 RMSE，再对五组取平均。实际−SONIC 输出是执行器静差；实际−冻结参考是净关节跟踪误差，两者必须分开。

| 配置 | 实际−SONIC 输出 L/R RMSE | 实际−参考 L/R RMSE |
|---|---:|---:|
| 当前 | 5.093 / 4.690 | 3.099 / 2.929 |
| 只改载荷 | 5.076 / 4.707 | 3.102 / 2.921 |
| 只改执行器 | 7.749 / 7.382 | 2.771 / 2.636 |

各关节有符号偏差、双腕 p95/最大误差、3 cm 帧占比、根部相对腕误差、实际 PhysX 参数与重力矩均保存在 `runs/dynamics-20261119/comparison.json`。

## 复现与证据

- 权重/发布配置核验：`runs/dynamics-20261119/training-provenance.json`；参数：`configs/sonic-dynamics.json`。
- 每次运行：`dynamics-intervention.json` 保存前后参数；`actuator-config.json` 保存独立采样到的实际生效值，原始物理步、控制步和消费参考日志全部保留。
- 本地与开发机均通过 73 项测试；跨 GPU 不干预复现的实际双腕位置差为 0。`dynamics-final-manifest.json` 与 `dynamics-evidence-manifest.json` 分别记录最终代码/文档和原始证据哈希，旧版 154 文件快照另行核验。
- 运行：`PYTHONPATH=. <policy-python> scripts/run_sonic_replay_suite.py --suite <new-id> --actuator-audit --dynamics-profile sensor-load --sim-gpu <idle-gpu>`；执行器组改为 `--dynamics-profile actuator`。
- 报告：`PYTHONPATH=. <policy-python> scripts/report_dynamics.py --load-suite dynamics-load-20261119-r1 --actuator-suite dynamics-actuator-20261119-r1`。

[发布权重与配置映射](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/config.json)；[对应训练 YAML](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_release/config.yaml)；[固定版本机器人配置](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/daf389964fa4a4545218e8405f24eb55f4912453/gear_sonic/envs/manager_env/robots/g1.py)。
