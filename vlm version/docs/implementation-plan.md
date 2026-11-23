# Kimodo v6.1 在 HumanoidArena 的实现计划

2026 年 11 月 17 日更新。当前已授权顺序：先做 A（真值稀疏目标 → 冻结 Kimodo → SONIC），
按阶段测误差；再决定 VLM 和视觉注入。VLM 选型、B 轻量 cross-attention、C 完整 ControlNet
均未定案，也不默认都要训练。下文的 VLM/夹爪部分为后续设计，不能当成已实现状态。
第一轮使用 `Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody` 场景，仅诊断双手到达，夹爪打开。
具体运行、误差和限制见 [实验 A 记录](oracle-a-20261117.md)。

## 当前已完成和缺口

开发机 T2M-PLUS 实时检查显示 8 张 RTX 5090，检查时 GPU 1–6 空闲；0 与 7 有进程。
`humanoidVLA` 已有 Kimodo-G1-RP-v1、DINOv3、独立 policy 与 IsaacLab 环境。
6 组跨 tracker 基线运行记录为 completed；这只证明基线环境可用，不代表 v6.1 成功。
项目 `checkpoints/` 中未发现 VLM 或 LLM2Vec；其他目录是否有可复用副本仍需检查。
资源调度时必须重查显存和进程，不固定占用检查时空闲的卡。

| 项目 | 当前状态 | 下一步 |
|---|---|---|
| 未来稀疏条件入口 | 已接入独立 native A，真权重生成并记录分阶段误差 | 根据 A 的误差定位决定下一步 |
| motion text | 原模型支持任意文本 embedding | 补充运动短语缓存，随后接在线 LLM2Vec |
| 夹爪门控 | 独立纯逻辑原型，接口测试通过 | 接到 Arena 每个控制步和真实状态 |
| VLM 与视觉替换 | 待实现 | 固定权重版本，冻结视觉，LLM LoRA |
| Arena 新观测和双时钟 | A 已增加实测 root、关节、腕位姿及时间戳；RGB-D 标定待做 | VLM 接入前补齐视觉观测 |
| A 仿真闭环 | 已完成真权重与 SONIC 到达诊断；结果单独归档 | 先定位生成和跟踪误差，不将诊断完成标记为抓取成功 |

独立目录为 `/root/workspace/humanoidVLA/vlm version/`。独立源码快照避免在线修改
原版评测使用的模型和适配器。现有服务脚本仍按基线工作，不能通过改目录名宣称已运行 VLM。

## VLM 输出什么

先使用一个结构化意图包，不让 VLM 输出 29 个关节角。每包包含下列字段：

| 字段 | 内容和职责 |
|---|---|
| motion_text | 英文运动描述，如双手前伸、抱起箱子、走向桌子；随当前阶段更新 |
| root_goal | 站位像素、朝向或朝向目标、到达时间 |
| hand_goals | 左右手接触点像素、抓取姿态模板、到达时间；双手任务有两个目标 |
| compensation | 手部、站位、时间补偿；首轮固定为零 |
| gripper_events | 每手的 open/close 请求与到位触发方式，可组成双手同步事件 |
| phase | continue/retry/done；不把 command=close 解释为抓取成功 |

`episode_id`、`intent_id`、图像捕获时间、标定版本、坐标系、有效期限由运行时加上，
不是让 VLM 自己编造。到达时间采用相对捕获时刻的离散档位，转成绝对仿真时间；
同一个意图重规划时，到达时间不会一直向后顺延。

MVP 中抓取姿态先由示教标定少量模板，VLM 选择模板。模板必须包括腕部朝向和腕到
夹持中心的偏移。物体表面的像素点不是 wrist joint 的位置：由接触目标、手朝向和
固定工具偏移求出 wrist 与 hand-end 两个目标。以后再消融直接预测旋转的收益。
像素不可见或深度无效时标为无效，不用背景深度假装有效目标。

VLM 暂不定型。A 诊断之后再对两个 Qwen 4B 候选做小规模 Arena 指点误差和延迟测试，
固定 revision 和 processor。尚无本项目实测结果，不能预先保证 1–2 Hz。

## 稀疏约束如何进入 Kimodo

### 坐标和时间转换

1. VLM 在原图或明确的 resize/crop 图上给出像素点。先逆变换回原始图像像素。
2. 用同一捕获帧的深度和内参反投影。若深度是 optical Z，使用
   `p_camera = depth * inverse(K) * [u, v, 1]`；若是 ray distance 则先转换，不能混用。
3. 用该捕获时刻的相机位姿转换到稳定世界系。记录 optical frame 与 Isaac camera frame
   的轴约定，不用最新机器人姿态解释旧图。
4. 在每次 Kimodo 重规划时，把所有目标及实测历史转换到同一个窗口坐标系。
   世界系 Z-up 到当前代码 Kimodo Y-up 的轴映射是 `[x,y,z] -> [y,z,x]`，还要包含窗口
   平移和朝向旋转。高度相对任务地面保留，不能把 pelvis 高度减成零。
5. 沿用现有 episode 时间网格：`cut = round(sim_time * 30)`，
   `j = round(arrival_absolute_time * 30) - cut`，写在完整画布的 `H+j`，H=100。
   合法未来索引为 0–49。过期意图失效；过远目标留给上层后续规划或生成明确中间目标，
   不把所有远处目标截到第 49 帧。

现有 Arena adapter 将 root_pos 与 hand 都填零；`KimodoFeatures.history` 也主动丢弃 root
平移，仅约束 root heading 和全身旋转。因此原图中“100 帧实测运动全部已知”目前不成立。
新版本需要扩展 root 位姿与手反馈，并为 oracle 实验使用完整实测姿态历史生成正确 mask；
完整位置前缀相对原策略的变化要单独记录。若最终只用本体估计而无全局定位，要明确估计
坐标系和漂移，不能悄悄使用仿真真值作为部署输入。

### 画布与特征

每个样本的画布是 `[100+50, 417]`，另有同形状的逐特征 mask。root 2D waypoint 是
平滑 root 的地面 XZ，区别于真实 pelvis 轨迹。G1 手部使用 `*_wrist_yaw_skel` 与
`*_hand_roll_skel` 的目标位置，以及 wrist 的全局旋转。

同一关键帧指定平滑 root `[rx, rz]` 和手部绝对位置 `[px, py, pz]` 后，位置特征写成：

```text
local_joints_positions[joint] = [px-rx, py, pz-rz]
```

这里“local”只减水平平移，不能再按 root heading 旋转一次。root heading 若需要约束，
使用表示自身的 `[cos(theta), sin(theta)]` 定义，不能直接塞 Arena yaw。
当前 `hand_keyframe()` 只实现 root XZ、腕/手端位置和 wrist rotation；heading 后续显式加。

`future_condition` 是未归一化的 `[B,50,417]`，`future_condition_mask` 是 bool。
使用 checkpoint 的分块 stats 归一化，然后把未知特征清零；不可先 mask 再直接使用
未清零的归一化结果，因为物理零对应的标准化值可能不为零。

独立版本已把这两项写入 `observed_motion[:,100:]` 和 `motion_mask[:,100:]`。
原 `TwostageDenoiser.forward` 会替换已知特征并将 mask 拼入 Root；Body 使用预测 root
转换出的 local root，以及带稀疏手约束的 body 输入和同一 mask。这条通道在每个去噪步存在。
当前同时传 RTC 会明确报错，避免未经验证的旧参考与新目标冲突；首轮 RTC 关闭。
训练函数还会清空未来条件，必须在训练 VLM 视觉 ControlNet 前同步修改训练侧的 mask 构造。

### 不能省略的 FK 检查

模型预测的位置特征和旋转特征可能不一致。当前 inverse 默认由旋转重建 posed joints，
最终下发也是关节角；因此“mask 中的腕点正确”不等于“机器人手到了”。每次生成分别报告：

1. 未做最终覆盖的生成位置特征误差；
2. Kimodo 旋转 FK 的 wrist/TCP 误差；
3. 转成 29 个关节角后的 G1 FK 误差；
4. tracker 实测的手位置误差。

后处理前后分别保存指标，不用最后直接覆盖坐标制造零误差。如果位置条件仍不够，先补齐
官方末端约束格式、增加一个合理中间关键帧，再评估带关节限位的轻量 IK/优化修正。
这是待评估的工程步骤，不能声称最终修正仍完全由先验保证。
官方明确说明末端约束包含平滑 root、腕位置与旋转和手端位置，也说明严格到达需要后处理。
[Kimodo 官方约束说明](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html)

## motion text 如何进入 Kimodo

VLM 输出自然语言字符串，冻结的原版 LLM2Vec 把它变成 `[B,1,4096]`，传入
`predict_future(text_feat=..., text_length=...)`。同一 embedding 同时进入 Root 和 Body 的
文本条件，继续沿用原模型的 text projection 与 mask。

不要把 Qwen 最后隐藏状态直接当成 LLM2Vec embedding：维度相同也不代表分布和语义对齐。
现有 wrapper 已完成 4096 维编码；缺的是权重和动态字符串缓存。

第一轮固定少量运动短语并预计算 embedding，例如接近箱子、双手前伸、抱起、转身搬运、
放下和松手。文本缓存键使用规范化后的字符串与编码器 revision 的哈希，只有文本变化
时重编码。已有 task instruction embedding 不是这些阶段短语的替代品。
后续自由文本通过独立编码服务生成；8B LLM2Vec 不必与 VLM、Kimodo 常驻同一卡。

运动文本负责动作类型与风格；稀疏约束负责到达哪里、朝向和时间。实验对比 text only、
constraints only、两者结合，检查相互冲突时的生成质量。

## VLM 视觉分支与 ControlNet

是否需要视觉注入由 A 的失败类型决定；只有障碍物/接近方向等视觉信息不足时再做 B/C。
B 与 C 应先使用相同冻结 DINO 特征，并同时输入未来稀疏条件（训练时随机丢弃），避免
把适配器结构与特征来源两个变量混在一起。B 应使用逐层独立注意力或投影。
以后若换成 VLM 冻结视觉编码器，需保留空间 token。动态分辨率不能直接当成旧版固定 196：
固定图像预处理与网格，或增加保持布局的重采样层，并改 image_feat_dim/token_count。

VLM 视觉 token 的语义和维度都不同，旧 DINO ControlNet 不能直接视为兼容的训练完成权重。
保留形状兼容的运动侧初始化，重建并训练视觉投影/融合层；语义 LoRA 与视觉 ControlNet
可分别训练。视觉编码器及共享视觉 merger 冻结，避免一条分支训练改变另一条分支的输入。

当前 A 关闭 ControlNet、DINO 和 hand head；使用原始冻结权重及原生条件通道。
视觉分支训练尚未启动。若尝试 v6.1 的前 k 步注入，要按 DDIM 的实际高噪声顺序和训练
base timestep 映射实现，测试 k=0、部分步、全部步，不能认为关闭后几步天然保证流形或稳定性。
普通基线构造器仍加载 DINO；A 使用新增 `NativeKimodoPolicy`，不加载视觉分支。

## 夹爪训练和执行

### 学习事件而不学习门控规则

门控本身不训练网络。VLM 学习左/右手什么时候请求 open、close、hold，以及请求所绑定
的目标与阶段。门控阈值用训练/验证场景标定，评测前固定。

从 SONIC 示教中的 action 最后两维提取 0→1 与 1→0 转移，先去抖、合并邻近转移，保留
大量 hold 样本并平衡事件样本。只把开合转移当作命令标签，不当作抓到物体标签。
以事件前若干观测帧作为 VLM 输入，输出未来事件、目标和运动文本，通过输出 token 的
监督微调训练 LoRA；prompt/image 不计语言目标损失。必要时对少见事件字段加权。

几何标签必须区分“示教发出的参考”和“实际执行结果”。当前公开 state64 不提供实际
root 平移和夹爪读数，仅靠它无法重建完整的实测抓取目标。首轮通过仿真示教回放收集
相机 RGB-D、相机位姿、root/手实际状态和命令。未来接触点应投影到输入时刻的相机，
不能投影到未来相机后当成当前像素。对于移动物体和不可见点须单独处理。

### 执行侧逐步检查

策略服务器仍返回 body action_chunk，另加结构化 gripper_events 元数据。Arena 当前 HTTP
client 只取 action_chunk，需扩展解析并保存事件。在 action provider 每个控制步读取实测
TCP 位姿/速度，门控决定左右手开合，覆盖最终送给 Dex3 的手指命令。不能只在 0.5 秒一次
的 infer 请求上检查，也不能按预测的 body 轨迹提前填入整段 hand 动作。

原型条件是位置误差、姿态误差、速度均达标并连续驻留，再执行一次。当前默认 3 cm、
约 14°、0.1 m/s、驻留 0.1 s，仅为待标定初值。事件使用固定 deadline、单调 episode
时间和唯一 ID；重传不重置驻留，过期不触发，超时保持当前命令并返回 timed_out。
释放也应绑定放置到位事件；取消未完成事件不会自动松开已握持物体。

当前实现为每手一个 `ArrivalGripperGate`。搬箱子的双手同步还需一个上层 barrier：
左右手都 ready 后同一步 fire，记录两手到位误差；不能让先到的一只手先夹走箱子。
门控使用与目标同帧的实测 TCP，目标是腕点时不可拿物体中心来比较。

抓取确认单独处理。仿真中先用 contact、物体抬升和随手移动作诊断/训练标签；部署反馈
用实际可得的夹爪位置/电流/接触与视觉。完全闭合不必然空抓，中间闭合也不必然抓成功。
VLM 输入 close-fired、timeout、grasp-unconfirmed 等结果；失败后重新给目标和事件。
保持 body 在接触附近、等待抓取确认再输出 lift，防止门控尚未完成身体就继续离开。

## 训练顺序和验证关卡

| 阶段 | 具体工作 | 进入下一阶段的证据 |
|---|---|---|
| M0 接口 | 当前 CPU 原型，随后真权重离线文本与稀疏点生成；完整坐标/时间转换 | 三层 FK 误差、文本切换、边界帧正确，无 NaN |
| M1 Oracle Arena | 已知可达站位、双手目标、固定 motion text，冻结 Kimodo，SONIC，执行侧门控 | 单场景 reach/close/lift/place/open 的完整录像、事件与实测日志 |
| M2 VLM 最小训练 | 小规模 SONIC 数据做回放重标注；LLM LoRA 输出上述字段 | 留出 episode 的坐标、事件和阶段质量；不能只看训练 loss |
| M3 完整 v6.1 | 视觉替换与 ControlNet 训练，异步 VLM、动态文本、失败重试 | 固定小场景集较 oracle 的差距、延迟和失败归因 |
| M4 迁移与补偿 | 换 TWIST2；再考虑 zoo/上下文补偿 | 分开生成误差与跟踪误差，报告闭环成功和重试成本 |

M1 的真值对象/目标只供接口诊断，单独标记 oracle；M3 的成功率不准继续输入对象真值。
“跑通”至少意味着一个完整物理任务成功，不能用 HTTP 200、50 步无崩溃或 token 合法替代。

首轮不启动长训练。先选 1 条示教、1 个可重复场景，检查 root 与腕点画面叠加；然后小规模
固定场景验证。不满足精度时先调整约束表达/姿态与转换，而非扩大 VLM 训练掩盖接口错误。

## 双时钟和日志

VLM 目标频率 1–2 Hz，Kimodo 30 Hz 表示、约每 0.5 s 重规划 50 帧，只执行前 15 帧；
tracker 与夹爪门控使用 50 Hz 控制步。每个预测包带捕获时间和有效期限。VLM 返回慢时
不冻结执行线程：只用未过期的最新目标，过期后进入明确的等待/重新感知流程。
仿真同步 HTTP 会暂停仿真时间，因此任务跑通不证明墙钟实时；分别报告仿真与真实时间延迟。

每轮保存 g（原始目标）、c（补偿后的约束）、r（生成参考 FK）、o（实际执行结果），
坐标变换、原始 VLM 包、约束 mask、文本哈希、事件触发/超时、推理延迟、视频和环境版本。
首轮 c=g。未来补偿不要直接把 c-o 当作正确新命令标签：优先使用实际达到结果反标目标的
逆映射监督，并单独处理生成器与 tracker 两部分误差。

## 文件落点

| 文件 | 职责 |
|---|---|
| `kimodo_vla/model/sparse_conditions.py` | 已有：关键帧和归一化；后续补充多关键帧冲突检测 |
| `kimodo_vla/model/kimodo_policy.py` | 已有：推理未来条件入口；待补训练 mask 和视觉/采样开关 |
| `kimodo_vla/controllers/gripper_gate.py` | 已有：每手门控；待接双手 barrier 与真实 actuator |
| `kimodo_vla/model/vlm_planner.py` | 待建：VLM 输出、坐标转换、时间缓存与异步更新 |
| `kimodo_vla/benchmarks/arena/vlm_adapter.py` | 待建：扩展观测、事件协议，不改变旧协议默认行为 |
| `scripts/serve_arena_vlm.py` | 待建：独立服务入口，模式区分 oracle/VLM |
| `scripts/prepare_vlm_labels.py` | 待建：事件分段、几何标签与数据划分 |
| `scripts/train_vlm.py` | 待建：固定视觉、LLM LoRA 与字段监督 |
| `arena/` | 待建：固定 Arena 版本的执行侧接入与补丁，保留基线源文件 |

计划明确以 v6.1 为主线；本目录的未实现项不作为完成的功能或实验结果。

## 2026-11-17 固定参考对照后的优先级

固定参考双腕平均跟踪误差约 2–3 cm；旧 Kimodo 发送参考存在超限与重规划突变。详见 [对照报告](sonic-fixed-reference-20261117.md)。下一步先处理合法连续关节角与交接，重新测 A 的生成、传递和跟踪误差；VLM 选型、B/C 注入训练继续后置。后续已在独立变体中实现，见下方连续性对照。

## 约束诊断补充

[56 次真实权重对照](kimodo-constraint-diagnosis-20261117.md)确认未来条件生效，历史条件组织影响精度。超限大部分已在原始旋转中出现，跨窗口跳变也在转换前存在，±π 分支另有放大作用。下一步应分别验证历史稀疏化、连续边界和合法关节处理；尚未改变正式默认路径。

## 历史与连续性实现

已加入 `original`、`handoff`、`bounded` 三组独立选项，默认仍为原版。实测历史只保留最近一帧条件，未来零帧条件使用当前实测姿态，发送参考从上一消费参考平滑交接；第二阶段使用有限关节角与每步速度区间的联合投影。实现细节、真实 Arena 对照及剩余误差见 [连续性报告](kimodo-continuity-20261117.md)。生成原始输出与修正参考分别保留，不能用执行变稳推断 Kimodo 已精确满足目标。

## 根部与双脚条件诊断

已在独立 bounded 变体上加入 `--support-constraints none|root|root-feet|root-feet-keyframes`，默认 none 保留上轮修复版。根部与脚目标固定于初始站姿，未来零帧仍用当前实测姿态。根部条件显著减少转身和站位漂移；密集双脚条件没有带来一致的手部收益。详见 [根脚约束对照](kimodo-support-20261117.md)。后续继续区分指定未来手部关键帧与真正执行的整段手轨迹误差，不把少数关键帧约 1 cm 的结果等同于持续到位。VLM 继续后置。

## 连续执行区间与 IK 投影

已完成根部版、连续手目标版和 IK 版的保持/伸手六组对照，并保存前两阶段完整快照。连续目标显著改善生成精度，但实际双腕 3 cm 达标帧只有 55% / 70%，因此按约定停止软条件调参，转入显式 IK。IK 参考手位达到数值精度，实际伸手在 4–8 s 双腕全程达标；保持仍未达标。剩余工作集中在投影后冗余关节运动、物理跟踪偏差及实时性能，不把 IK 精度归功于原生 Kimodo。详见 [完整报告](kimodo-continuous-ik-20261117.md)。VLM 继续后置。

## IK 保持阶段姿态稳定

保留原 IK 源码快照与 8 秒执行基线，新增 `ik-hold`：目标静止且上一已消费参考满足手脚约束时保持关节值，未到位继续收敛，目标改变恢复原移动目标 IK。加入跨规划包的数值容差判定，避免相同目标的单帧/批量 FK 差异错误释放保持。此阶段只核验精度与关节运动，不接 VLM、不做速度优化。详见 [保持稳定性报告](kimodo-ik-hold-20261117.md)。

最终 r2 保持、伸手在 1.2–8 s 各 340 帧均达到实际双腕 ≤3 cm，最大误差分别 2.89/2.15 cm、2.38/2.50 cm；保持区间参考关节变化均为零。该结论仅覆盖当前初态与合法短距离目标，尚不代表一般抓取能力。

## VLM 第一阶段接入（2026-11-17）

旧 IK-hold 已冻结保存。新增真实 Qwen3-VL-4B-Instruct → 同帧 RGB-D 反投影 → 连续目标 → Kimodo → IK-hold → SONIC 通路；先做视觉标记到达，保持夹爪打开，默认文本条件关闭以隔离视觉目标变量。不是 ControlNet 训练，也不是自然物体抓取。模型指纹、相机同步、执行结果及限制见 [接入报告](vlm-integration-20261117.md)。本段更新取代上文“VLM 继续后置”的历史阶段状态。
