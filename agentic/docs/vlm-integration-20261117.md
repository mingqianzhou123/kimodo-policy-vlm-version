# VLM 第一阶段接入：视觉目标 → Kimodo → IK → SONIC

## 本轮边界

冻结已通过双腕 3 cm 验收的 IK-hold 后端，把目标来源换成真实 Qwen 图像定位。不训练 VLM，不加 ControlNet，不改 Kimodo 权重/采样、IK 权重/速度或 SONIC；夹爪保持打开。

这是带颜色标记的受控到达实验，不是自然物体抓取。相机位姿来自仿真的标定传感器，尚不具备真机外参估计。一次 episode 只做一次 VLM 推理，推理期间仿真暂停；没有实现实时异步重规划。

## 首轮真实执行结果

`runs/vlm-reach-20261117-r1` 已完成：seed 1000、头部相机、真实 Qwen BF16 权重、16 次原生 Kimodo 重规划、400 个 50 Hz 控制帧（8 秒）。到达期限后 1.2–8 s 共 340 帧用于下表。**通路接入成功，双腕实际 3 cm 验收未通过。**

| 误差层 | 左腕 | 右腕 |
|---|---:|---:|
| VLM 指点相对真实标记中心（像素） | 8.13 px | 17.39 px |
| 反投影视觉目标相对真实标记（cm） | 1.31 | 2.85 |
| 原始生成参考相对视觉目标，平均（cm） | 2.20 | 2.17 |
| IK 后参考相对视觉目标，最大（cm） | 0.0000211 | 0.0000166 |
| 实际双腕相对视觉目标，平均（cm） | 3.42 | 1.95 |
| 实际双腕相对视觉目标，最大（cm） | 3.76 | 2.94 |
| 实际双腕相对真实标记，平均（cm） | 4.11 | 3.00 |
| 实际双腕相对真实标记，最大（cm） | 4.40 | 3.39 |

实际双腕同时在真实标记 3 cm 内的帧占比 **0%**；相对视觉目标则为 4.12%。期限后的参考关节变化范围仍为 **0°**。这说明 IK 准确落实了视觉目标，并成功保持参考，但视觉定位与物理跟踪都还有残差，不能把 IK 的微小 FK 残差当作端到端成功。

Qwen 原始输出为 `left=[250,830]`、`right=[750,830]`，对应原图浮点坐标 `[159.75,397.57]`、`[479.25,397.57]`；实际取深度的像素为 `[160,398]`、`[479,398]`。右侧预测点位于蓝色方块旁的机器人手部，深度约 0.44425 m，而标记中心深度约 0.41735 m。深度数值有效不代表语义位置正确，本轮没有做颜色吸附或用真值修正预测。

首轮单图生成 47 tokens，推理 **2.053 s**，模型加载 **3.719 s**；这是共享 GPU 上的一次观测，不是实时吞吐保证。输入图、原始输出、标点图、模型文件指纹、完整参考及执行轨迹均已归档。另在不含标记的旧帧上做了一次真实模型负例：Qwen 返回不可见，解析器拒绝执行。

与上一轮 oracle 的稳定 3 cm 结果不能直接作单变量精度比较：本轮腕目标来自视觉估计，路径改为 Cartesian 插值，腕朝向保持初始值；上一轮路径及末端朝向来自合法 witness 的 FK。后端求解器未改，但目标条件不同，因此不能把所有新增实际误差都归因于 VLM。

![真实 Qwen 的目标预测，右点偏在蓝色标记旁](../runs/vlm-reach-20261117-r1/vlm-intent.points.png)

## 模型与接口

- 模型：[Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，BF16、SDPA、确定性生成，最多 128 个新 token。
- 模型保存在本子目录 `checkpoints/Qwen3-VL-4B-Instruct/`。下载逐文件校验官方 LFS SHA256 或 Git blob SHA1，再保存本地 SHA256 清单；没有修改共享 Python 环境。
- 输入：头部相机完整 640×480 RGB，没有真值目标坐标、关节 witness、物体状态或带答案的标注图。
- 输出：`left`、`right` 为 0–1000 图像坐标；`motion_text` 限定为 `Reach both hands toward the targets and hold still.`。标记不可见时应输出 null 并拒绝执行。
- 本轮默认 `--text-mode none`，motion_text 记录但不参与 Kimodo 文本条件，用于隔离视觉目标这一变量。已提供原版 LLM2Vec 短语缓存脚本及严格缓存键接入；尚未取得该短语的实际编码时，不冒用搬箱任务 embedding，也不把 Qwen 隐状态作为替代。

## 执行链

1. Arena 生成红色（左腕）、蓝色（右腕）的 35 mm 视觉方形标记。中心取先前合法短距离伸手配置的 FK，作为场景布置；marker 不参与碰撞或动力学。
2. 真值只写入 `visual-evaluation-truth.json` 供离线评估。策略没有读取该文件。实际腕目标只由 Qwen 图像坐标 + 深度 + 相机参数计算。
3. 采集同一暂停时刻的 RGB、`distance_to_image_plane`（米制 optical Z）、K、ROS optical-to-world 位姿，写入 `vlm-capture.npz`。相机约定为 +Z 向前、+X 向右、+Y 向下。
4. 图像点恢复到原图像素，读取该像素深度，以 `K^-1 [u,v,1] * depth` 反投影并转世界系。无效深度、明显深度边缘、格式错误、不可见目标或单腕位移超过 15 cm 均报错；没有裁剪或替换成 oracle 的后门。
5. 保留初始腕朝向和虚拟 hand-end 相对偏移，用 1.2 s 五次平滑插值生成视觉目标路径。没有用目标 witness 关节路径生成动作。根部和双脚目标沿用初始站姿。
6. 同一路径分别写入 Kimodo 的实际执行区间手部条件和 IK 投影。到达后继续使用 ik-hold，目标不变时保持已消费参考。
7. 保存原始生成、转换结果、IK 后参考、实际执行。离线分别计算像素/世界定位误差、参考相对视觉目标误差、实际跟踪参考误差和实际相对真实标记误差。

## 相机首帧处理

[IsaacLab CameraCfg](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py) 默认不更新相机位姿。本实验明确开启 `update_latest_camera_pose`，刷新运动学与渲染；[SensorBase.update](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab/isaaclab/sensors/sensor_base.py) 即使传 `force_recompute=True`，也只刷新标记为过期的 buffer。因此暂停时间下仅 `update(0)` 仍会返回旧图。本实验在获取一次新 intent 前重置相机缓存，不推进物理时间，再一起读取图像、深度和外参。前三次采集诊断保留，不能当作有效标记定位结果。

## 运行

在开发机 `/root/workspace/humanoidVLA/vlm version` 下，先确认模型清单存在。选择空闲 GPU；若显式共用其他常驻服务的 GPU，可追加 `--gpu-min-free-mib 16384`，启动前检查每张卡至少有 16 GiB 空余，不终止其他进程。policy 与 sim 使用不同卡。

```bash
PYTHONPATH=. ../cross_tracker/data/cache/envs/policy/bin/python scripts/run_oracle.py \
  --run-id vlm-new-reach --mode hold --steps 400 \
  --variant bounded --joint-speed 4 --support-constraints root \
  --hand-constraints continuous --trajectory-projection ik-hold \
  --policy-gpu 6 --sim-gpu 1 --visual-markers \
  --vlm-model checkpoints/Qwen3-VL-4B-Instruct

PYTHONPATH=. ../cross_tracker/data/cache/envs/policy/bin/python scripts/review_vlm.py \
  runs/vlm-new-reach
```

`--mode hold` 在此只选择原后端的站立初始化，视觉目标会替换双腕目标；run.json 标明 `target_source=qwen_rgb_depth`、`oracle=false`。不传 `--vlm-model` 时旧版行为不变。

## 保留与验证

- 原 IK-hold 的 118 个文件逐一校验并保存于 `snapshots/ik-hold-before-vlm-20261117/`，本地/开发机均保留。
- 原生 `oracle/policy.py`、IK、连续性处理和模型源码均未修改。旧文件只在 Arena 桥接、服务入口和启动器增加显式视觉选项。
- 本地与开发机 50 项测试通过，包括坏 JSON/不可见点、边界像素、旋转相机的 optical Z 反投影、深度缺失/边缘拒绝、世界目标路径、暂停时缓存刷新与原有 40 项检查。
- 完整物理执行结果以 run 内 `vlm-summary.json` 为准，单元测试不能替代真实 Qwen 和 Arena 执行。
