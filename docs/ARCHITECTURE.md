# Kimodo-VLA 代码架构

> 目标读者：新加入的合作者。读完这一页应该知道：代码分几层、每层管什么、
> 要接一个新 benchmark 需要写哪几个文件。

## 1. 一句话

模型只认一种「通用观测」和一种「通用动作」；每个 benchmark 写一个薄适配器，
负责把自己的观测翻译成通用观测、把通用动作翻译回自己的动作。
模型、推理循环都不知道自己在跑哪个 benchmark。

```
 benchmark 环境 ──原始观测──▶ Adapter.to_observation ──Observation──▶ Agent ──▶ Planner(模型)
       ▲                                                                │
       └──环境动作── Adapter.to_env_action ◀──ActionChunk(控制频率)──────┘
```

## 2. 目录

```
kimodo_vla/
  core/
    types.py        # Observation、ActionChunk、Planner 接口：全仓库唯一的约定
    agent.py        # Agent：保留最近观测，计划用完就重新规划（与 benchmark、模型都无关）
  model/
    features.py     # 位姿轨迹 → Kimodo 特征：评测（planner）和训练数据共用同一份转换
    planner.py      # KimodoPlanner：把 KimodoPolicy 包成 Planner（以后 VLM 版本并列放这里）
    kimodo_policy.py, modules/   # 从旧仓库原样迁入，只改了 import 路径
  motion/ skeleton/ utils/       # G1 运动表示、正运动学、几何工具（原样迁入）
  controllers/
    groot_wbc.py    # GR00T 系下肢控制器：骨盆轨迹 ↔ 底盘速度/转向角速度/高度指令（双向）
  data/
    episodes.py     # 训练文件格式 + KimodoWindowDataset（与 benchmark 无关）
    build.py        # 一条演示 → 训练文件（历史、目标、手、帧对齐；与 benchmark 无关）
  benchmarks/
    base.py         # BenchmarkAdapter、Demonstration 接口 + AgentPolicy（把 Agent 和适配器拼成 policy）
    replay.py       # DemoPlanner / DirectPlanner：把演示当成「完美预测」回放，检查动作接口
    bigym2/
      adapter.py    # BiGym 2.0 观测/动作的双向翻译（动作只管 BiGym 的布局和归一化）
      demos.py      # 读 BiGym 演示：实际执行的动作、操作员指令、图像（BiGym2Demo）
      policy.py     # make_policy / make_kimodo_policy：交给 BiGym 官方评测器
scripts/
  prepare_bigym2.py         # 演示 → 训练文件（Kimodo 特征 + 三路官方 84×84 图像）
  check_bigym2_roundtrip.py # 动作来回转换检查
  eval_bigym2.py            # 按官方协议评测一个 checkpoint
tests/              # 见第 7 节
```

依赖方向只有一条：`benchmarks → core ← model`，`benchmarks → controllers、data`。
`core/`、`model/`、`controllers/`、`data/` 里不出现任何 benchmark 名字。

## 3. 三个约定（`core/types.py`）

统一约定：MuJoCo 世界坐标系、z 朝上、米和弧度；关节向量按 `G1_JOINT_NAMES`
（29 个 G1 关节，顺序与已发布 checkpoint 的 40 维动作一致）；手的闭合度 0=开、1=合。

```python
@dataclass
class Observation:                   # 一个控制步看到的东西
    images: dict[str, np.ndarray]    # 相机名 → (H, W, 3) uint8
    root_pos: np.ndarray             # (3,) 骨盆位置
    root_rot: np.ndarray             # (3, 3) 骨盆朝向
    joint_pos: np.ndarray            # (29,)
    joint_mask: np.ndarray           # (29,) bool，benchmark 不提供的关节为 False
    hand: np.ndarray                 # (2,)
    instruction: str
    step: int = 0                    # 回合内第几步，由 Agent 填

@dataclass
class ActionChunk:                   # 控制频率下的 T 帧参考轨迹，第 0 帧是下一步的目标
    root_delta_xy: np.ndarray        # (T, 2) 每帧骨盆位移，在该帧骨盆坐标系下
    root_height: np.ndarray          # (T,)   骨盆绝对高度
    root_rot: np.ndarray             # (T, 3, 3) 世界系朝向
    joint_pos: np.ndarray            # (T, 29)
    hand: np.ndarray                 # (T, 2)
    fps: float

class Planner(Protocol):             # 模型侧唯一需要实现的接口
    history_steps: int
    def reset(self, seed=None): ...
    def plan(self, history: list[Observation]) -> ActionChunk: ...
```

`ActionChunk.from_array40 / to_array40` 和旧仓库的 40 维动作互转，旧数据和旧 checkpoint 可以直接用。

## 4. 接一个新 benchmark

两个接口在 `benchmarks/base.py`：

```python
class BenchmarkAdapter(Protocol):     # 评测：原始观测 → Observation，ActionChunk → 环境动作
    control_fps: float
    def reset(self) -> None: ...
    def to_observation(self, raw_obs) -> Observation: ...
    def to_env_action(self, target: ActionChunk, obs: Observation) -> np.ndarray: ...

class Demonstration(Protocol):        # 训练数据：一条演示
    instruction: str
    control_fps: float
    cameras: Sequence[str]
    def observations(self) -> list[Observation]: ...   # 每个控制步过一遍 to_observation
    def commanded_motion(self) -> DemoMotion: ...       # 操作员的指令（不是机器人实际执行的动作，见第 10 节）
    def images(self, steps) -> np.ndarray: ...          # [len(steps), 相机, H, W, 3]
```

`AgentPolicy(agent, adapter)` 把 Agent 和适配器拼成 `policy(raw_obs) -> env_action`，带 `reset()`。

清单（不需要改 `core/`、`model/`、`controllers/`、`data/`）：

1. **`benchmarks/<name>/adapter.py`**：实现 `BenchmarkAdapter`。动作里底盘那部分调控制器模块，
   比如 GR00T 系（GR00T-WBC、GR00T Balance/Walk）就用 `controllers.groot_wbc.BaseCommander`，
   适配器自己只管动作布局和归一化。以后换底层控制器，只改 `to_env_action` 里这一行调用。
2. **`benchmarks/<name>/demos.py`**：读这个 benchmark 的演示，每条包成一个 `Demonstration`。
   如果演示里存的是底盘速度指令，用 `controllers.groot_wbc.pelvis_path` 积分成骨盆轨迹。
3. **两个薄脚本**：`scripts/prepare_<name>.py`（加载演示 → `data.build.training_episode` →
   `save_episode`）和 `scripts/check_<name>_roundtrip.py`（用 `benchmarks.replay` 的
   `DemoPlanner` / `DirectPlanner` 回放演示，统计这个 benchmark 自己的成功判定）。
   评测脚本照 `eval_<name>.py` 写一个入口即可。不做跨 benchmark 的统一入口。
4. **`pyproject.toml`**：在 `[project.optional-dependencies]` 加一组 `<name> = [...]`，锁定版本。
5. **必需的测试**：`tests/test_<name>_adapter.py` 在真实环境里检查观测读数和仿真器一致
   （参考 `test_bigym2_adapter.py`）；再在真实演示上跑一次来回转换检查，结果记到本文档
   （参考第 10 节）。

## 5. BiGym 2.0 适配器具体做什么

| 方向 | 内容 |
|---|---|
| 观测 | `low_dim_obs` 按官方布局取出：骨盆 x/y/z/yaw、14 个手臂关节、（开启躯干俯仰的 14 个任务）3 个腰关节、2 个夹爪。腿部关节和（其余任务的）腰关节 `joint_mask=False`、值填 0。实测 WBC 接管腰部时，腰俯仰也只在 0.06 rad 以内。骨盆横滚和俯仰角在官方观测里拿不到，按 0 处理。`rgb_obs` 拆成 head / right_wrist / left_wrist 三张图 |
| 动作 | 底盘指令由 `controllers.groot_wbc` 算：`vx, vy` = 第 0 帧位移 × 50Hz；`wz` = 上一步目标朝向到第 0 帧目标朝向的变化率（和位移一样用向后差分，纯前馈，不做朝向闭环）；`height` 直接给。适配器负责其余部分：`torso pitch` 取目标腰俯仰关节；14 个手臂关节直接给；手闭合度 ≥ 0.5 → 夹爪合。最后按 `env.wholebody_action_layout()` 归一化到 [-1, 1] |
| 语言 | 用 BiGym 自带的 `task_instructions.yaml`，和演示数据里的指令一致 |

BiGym 是纯 Python + MuJoCo，模型和环境在同一个进程里跑，不需要 HTTP 服务。
注意 BiGym 锁定 `mujoco<3.9`；`kimodo_vla` 本身不依赖 mujoco，所以评测环境里只装 BiGym 要的版本即可。

```bash
MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
    --checkpoint <ckpt 目录> --text-cache <文本 embedding 目录> --out outputs/move_plate.json
```

## 6. KimodoPlanner 和旧仓库的关系

`model/planner.py` 是旧仓库 `evaluation/humanoidarena_server.py` 里推理流程的移植，
去掉了 HTTP、调试开关和事务快照。所有关节都可见时，它送进模型的输入和输出的动作
与旧代码一致：`tests/test_port_equivalence.py` 用同一条 420 步轨迹对比三种设置
（每次执行 50 帧；15 帧；15 帧 + RTC），动作最大差 6e-6。

RTC（实时分块融合：新一段动作的开头向上一段没执行完的尾巴靠拢，让两段衔接更平滑）保留为
`KimodoPlanner(rtc=True, rtc_overlap_frames=12, rtc_frozen_frames=1, rtc_ramp_power=1.0)`，**默认关**。
所有正式评测都是关着的；评测脚本不传这个参数就是关。

和旧代码不同的两点：

1. 历史特征的掩码按 `joint_mask` 逐关节生成：没上报的关节，其全局旋转特征被遮住。
   所有关节都上报时，掩码和旧代码完全相同。
2. 50Hz 观测重采样到 30Hz 时，时间网格固定从回合开始算起（训练时整条轨迹也是这样重采样的），
   所以 Agent 只保留最近约 3.4 秒观测，结果也和保留整条历史一致。

目前只支持二值手部控制（BiGym 是两指夹爪）；连续手部控制是 SIMPLE 用的，迁移 SIMPLE 时再加。

接入 VLM：新模型实现同一个 `Planner` 接口即可，Agent、适配器、评测脚本一行不改。

## 7. 测试

```bash
pip install -e ".[bigym2,dev]"
pytest                                                    # 类型、Agent、历史编码、训练文件、GR00T 指令互逆、BiGym 适配器
KIMODO_ORIGINAL_REPO=<旧仓库路径> pytest tests/test_port_equivalence.py   # 与旧推理流程对比
```

`test_bigym2_adapter.py` 在真实 BiGym 环境里检查：观测和仿真器关节值一致；归一化与 BiGym 互逆；
保持初始姿态不摔、不漂；给 0.3 m/s 后退位移，3 秒退 0.73 米（实测稳定速度约 0.28 m/s）；给左转目标会左转。
（测的是后退：有些场景里正前方是台面，往前走会被挡住。）

## 8. 和旧仓库相比刻意不做的事

1. **一个 benchmark 一个两千行的服务文件**。旧 `simple_server.py` 有 2208 行，靠动态拼类、
   临时替换另一个文件里的函数来复用 Arena 的推理逻辑。新结构里推理只在 `model/planner.py` 写一次。
2. **推理循环藏在某个 benchmark 文件里**。旧代码的历史缓存、分块执行都在 `humanoidarena_server.py` 里。
3. **每个实验一对 sh + yaml**。旧 `scripts/` 下约 50 对，彼此只差一两个字段。训练迁过来时只保留一份基础配置。
4. **数据源的专用小函数堆在公共文件里**。旧 `data/common.py` 有 1300 行。

旧仓库保持不动，作为论文结果的复现版本。

## 9. 还没做的

- [x] 训练数据：`scripts/prepare_bigym2.py` + `data/episodes.py`，见第 11 节。
- [ ] 训练脚本 `train.py`。
- [x] 动作来回转换检查（`scripts/check_bigym2_roundtrip.py`，见第 10 节）。
- [ ] 文本 embedding：用 LLM2Vec 给 20 个任务指令预先算好。
- [x] 输入图像：只用头部相机，官方 84×84，由 DINOv3 预处理在模型内部放大到 224。
      BiGym 作者澄清：所有方法拿到的都是同样的 84×84 原始数据，π0.5 的 224 也只是模型内部放大，
      不是重新渲染。评测用官方相机设置，所以官方 runner 会正常出排行榜记录。
- [x] 训练方式：只用 BiGym 演示从零训练 ControlNet 和手部头，不用我们的 VLA 预训练权重；
      Kimodo-G1、DINOv3、LLM2Vec 保持冻结。ControlNet 的注入层初始为零，所以训练从原版 Kimodo-G1 起步。
      用 pt_419h 初始化的一组作为对照。
- [x] 手腕相机（代码）：`KimodoPolicyConfig.num_cameras` 设成几，模型就接几路图像。每路单独过同一个
      冻结的 DINOv3，各得 196 个 token，按相机顺序拼起来（三路 588 个）送进 ControlNet 和手部头。
      ControlNet 里把 token 压到 50 个的那层随路数变宽，所以三路模型不能加载一路的 ControlNet 权重
      （pt_419h 对照只能用一路）。checkpoint 的 `config.json` 里 `model.cameras`（如
      `["head", "right_wrist", "left_wrist"]`）决定 planner 取哪几路、按什么顺序；没写就是只用 `head`。
      一路时的计算和以前完全一样。`tests/test_multicamera.py` 只测了图像编码和 planner；
      需要真实权重的完整检查是 `scripts/check_multicamera.py`（见下一条）。
- [ ] 手腕相机（实验）：一路/三路消融。BiGym 作者确认 π0.5 三路相机都用了，ACT、DP 等也是三路，
      所以三路作为主结果，只用头部的一路作为消融（2026-10-25 定）。
      `scripts/check_multicamera.py` 已在 A800 上用真实权重和 move_plate 数据跑通：训练、
      三路相机各自影响预测、官方环境里闭环控制都正常。

## 10. 动作来回转换检查的结论

做法：把演示里的动作当成「完美的模型预测」，走和正式评测完全相同的流程
（Agent → 30Hz Kimodo 动作块 → 重采样 → 40 维编码 → 适配器 → BiGym），看还剩多少成功率。
5 个任务 × 60 条演示，2026-10-24 在 A800 上跑。

| 任务 | 原样回放 | 实际执行的动作当目标 | 操作员指令当目标（30Hz 流程） | 指令直接给 50Hz |
|---|---|---|---|---|
| drawer_top_open | 100% | 0% | 97% | 100% |
| flip_cup | 100% | 2% | 97% | 98% |
| move_plate | 100% | 7% | 97% | 100% |
| pick_box | 100% | 0% | 80% | 90% |
| reach_target_single | 100% | 95% | 100% | 100% |

结论：

1. **训练目标必须用操作员的指令，不能用机器人实际执行出来的动作。** 把实际动作当目标，
   手臂会再叠加一次重力下垂（平均误差 0.02 rad），除了够取任务以外基本全部失败。
   `bigym2/demos.commanded_motion` 就是训练要用的目标（对应 `Demonstration.commanded_motion`）。
2. **朝向闭环有害，已关闭。** 增益为 2 时 move_plate 掉到 23%，pick_box 掉到 2%：
   它会去纠正 WBC 本来就有的朝向漂移，结果走出和演示不一样的路线。
3. **转向角速度要和平移一样用向后差分。** 原来用的是向前差分，比操作员提前一步，
   pick_box 因此少了约 12 个点（50Hz 直接给：78% → 90%）。
4. **Kimodo 的 30Hz 表示带来的损失**：大部分任务只有 0–3 个点，pick_box 有 10 个点（90% → 80%），
   它转身和走动多，30Hz 重采样会把路线磨平。所以这套接口的成功率上限约为 94%（5 个任务平均）。
5. pick_box 直接给 50Hz 也只有 90%：抱 3kg 箱子对接触很敏感，指令上的微小差别就可能失败。



## 11. 训练数据

**流程**：`prepare_bigym2.py` 把每条演示包成 `BiGym2Demo`，交给 `data/build.training_episode` 转成两个文件，`KimodoWindowDataset` 再从中切窗口：

- `<演示名>.pt`（30Hz）：`observed_motion` 是模型看到的历史，由官方观测经 `to_observation`
  和 `KimodoFeatures.history` 得到，和评测走同一段代码；`target_motion` 是要学的目标，
  即操作员指令（见第 10 节）；`hand` 是夹爪指令。
- `<演示名>.frames.npy`：每个 30Hz 帧对应的三路相机图像（头部、右腕、左腕），直接取数据集里
  记录的官方 84×84 图像。`KimodoWindowDataset(cameras=...)` 选用哪几路：一路时 `egoview` 是
  `[3, H, W]`，多路时是 `[路数, 3, H, W]`，和 `KimodoPolicyConfig.num_cameras` 对应（见第 9 节）。

每个样本 = 100 帧历史 + 50 帧未来 + 切点处的图像，键名和约定与旧仓库的训练数据完全一致，
`KimodoPolicy.training_kimodo_policy_controlnet` 可以直接用。

**训练和评测一致性**（2026-10-25，A800 上用 move_plate、pick_box 的真实演示验证）：
- 训练窗口的历史条件与评测时 planner 构造的完全相同：分别对比 22 个和 74 个切点，差值为 0
  （`tests/test_episodes.py` 用合成数据做同样的检查）。
- 训练文件里的三路图像与官方环境评测时实时给出的观测逐像素相同（差为 0）。

**规模**：5 个任务 300 条演示，三路 84×84 图像共 8GB，可切出 112,683 个训练窗口（pick_box 演示最长，独占 57,643 个）。
20 个任务按这 5 个任务的平均估算，约 32GB。窗口数和演示长度成正比，多任务训练时要考虑按任务均衡采样。
\n
## 12. 四个 benchmark 共用一套接口（设计草案，待确认）

目标：HumanoidArena、SIMPLE（旧仓库）、BiGym 2.0、HumanoidToolBench 共用同一个 Agent + Planner，
换 benchmark 只换适配器。来源：旧仓库 `evaluation/humanoidarena_server.py`、`simple_server.py`
和两份 `*_eval.md`；ToolBench 来自 SNU-PI/HumanoidToolBench（3fec9bd）的 `docs/PUBLIC_EVALUATION.md`、
`data/README.md`、`src/humanoidtoolbench/actions/g1.py`，以及 ToolBook 数据集的元数据。

| | HumanoidArena | SIMPLE | BiGym 2.0 | HumanoidToolBench |
|---|---|---|---|---|
| 运行方式 | 仿真（Isaac Lab）在另一个进程，HTTP 调我们：`/health`、`/reset {seed}`、`/infer` | 官方评测在同一进程（`--eval`：SIMPLE 自己的 venv 里跑 MuJoCo 物理 + Isaac 渲染，直接调 runtime）；另有 HTTP `/act` | 同一进程，官方 `evaluate(policy)` 每步调一次 | HTTP：它们的评测器连我们。`POST /act`（带 NumPy 编码的 JSON），可选 `GET /info`、`POST /release`，一个服务只接一个评测器。原因：它锁定 transformers 4.57.1，我们要 5.1.0，装不进同一个环境 |
| 每次请求给什么 | 任务名；`front` 图像（HWC，base64）；`state_history` [T, 64]：上次请求以来每个 50Hz 步的状态（根朝向 rot6d + 29 关节角 + 29 关节速度；SIMPLE 那边的换算表明根朝向相对回合首帧朝向） | 同样的 64 维（由 `floating_base_pose` 四元数和 `body_q` 换算）；`head_stereo_left` 图像；手的测量值（Dex3 关节 q/dq/力矩） | 一步的 `low_dim_obs`（骨盆 x/y/z/yaw、14 手臂关节、部分任务的腰、2 夹爪）+ 三路 84×84 图像 | 头部相机 `rgb_head_stereo_left` 360×640；`state` [1, 32]：两手各 7、两臂各 7、腰 roll/pitch/yaw 测量值、上一次发出的骨盆高度指令；指令文本（6 句固定）；首次请求带 `history.reset`。**没有骨盆位置和朝向，没有腿** |
| 每次要我们回什么 | 一整段 40 维全身参考 [T, 40]（50Hz） | 一整段 36 维 WBC 指令 [T, 36] | 一步的归一化动作：vx、vy、高度、转向角速度、[躯干俯仰]、14 手臂、2 夹爪 | 一整段 36 维指令 [T, 36]，**和 SIMPLE 的 36 维布局相同**：两只 Dex3 手 14 + 手臂 14 + 躯干 roll/pitch/yaw + 骨盆高度 + vx、vy + 转向开关（绝对值 ≥ 0.1 才跟踪朝向）+ 目标朝向（绝对角，不是角速度）。评测器把整段执行完才再请求 |
| 底层控制器 | SONIC 或 TWIST2，在仿真那一侧，直接吃 40 维参考 | decoupled WBC（SONIC 系），在仿真那一侧 | GR00T-WBC：速度 + 转向角速度 + 高度 | NVIDIA decoupled WBC（GR00T Balance/Walk）：速度 + 目标朝向（误差/0.5 秒，最快 1 rad/s）+ 高度；vx、vy、开关全为 0 时站立 |
| 手 | 二值，2 维 | Dex3 七自由度手。二值：闭合姿态，加原代码里的防抖状态机；连续：闭合度 × 闭合姿态，历史用测量到的闭合度 | 两指夹爪，二值 | Dex3 七自由度手，关节目标连续；需要「2 个闭合度 → 14 个关节」的映射，同 SIMPLE 连续模式 |
| 相机 | `front` 一路 | `head_stereo_left` 一路 | head + 两个手腕，84×84 | 头部一路，360×640（手腕相机只在数据里有，评测时不给） |
| 观测到的关节 | 29 个全部 + 根朝向 | 29 个全部 + 根朝向 | 手臂（+ 部分任务的腰）+ 根 yaw | 手臂 + 腰；**没有根朝向** |
| 控制频率 / 每次执行 | 50Hz；执行 15 个模型帧（文档里正式评测的参数） | 50Hz；执行 50 个模型帧 | 50Hz；50 个模型帧 | 50Hz；整段 |
| 演示数据 | 旧仓库已有训练数据 | 旧仓库已有训练数据 | 每步存了操作员指令（第 10、11 节） | LeRobot 50fps，存了操作员指令（`teleop.navigate_command` [vx, vy, 角速度, 目标朝向]、高度指令、手的指令），还有骨盆位姿（只在数据里有）和三路 360×640 图像 |

### 共用件和每个 benchmark 各写什么

1. **一个 Agent，两种运行方式。**
   - 同进程（BiGym、SIMPLE）：现有 `AgentPolicy`，每步 `to_observation → Agent.step → to_env_action`。
   - HTTP（Arena、ToolBench，以及 SIMPLE 的 `/act`）：新建一个通用服务 `benchmarks/server.py`，包住 Agent：
     收到请求后，适配器把它解析成「上次以来的一串 Observation + 当前图像」，全部记进 Agent 的历史，
     立即规划一次，再由适配器把**整段** ActionChunk 换成回复；一把锁保证同一时间只处理一个请求。
     路由和编码因 benchmark 而异（Arena：`/reset`、`/infer`，base64 图像；ToolBench 和 SIMPLE：`/act`，
     NumPy 编码的 JSON，外加 ToolBench 的 `/info`、`/release`），所以每个适配器声明自己的路由表和编解码，
     服务本身只管锁、重置和调用 Agent。
   - 适配器接口相应分成两种：逐步的 `to_env_action(chunk, obs) -> 一步动作`（BiGym）和整段的
     `to_response(chunk, obs) -> 回复`（Arena、SIMPLE、ToolBench）。
   - 需要给 `core/agent.py` 加两个小方法（只记录观测、立即规划），现有 `step` 不变。这是唯一要动 `core/` 的地方。
2. **`controllers/` 按控制器家族分，适配器挑一个用。**
   - 40 维全身参考（SONIC、TWIST2，控制器在 Arena 那一侧）：就是 `ActionChunk.to_array40`，不需要新模块。
   - GR00T 系（`groot_wbc.py`）：核心是「ActionChunk 每帧 → 底盘速度、朝向、高度」，下面两种编码：
     转向角速度（BiGym，已完成）；目标朝向 + 转向开关的 36 维 decoupled WBC 布局（SIMPLE 和 ToolBench 共用，
     移植旧代码 `arena_action_to_simple`）。反方向（指令 → 骨盆轨迹，训练目标用）同样两种：BiGym 积分角速度，
     ToolBench 直接用记录的目标朝向。
   - 注意：旧 SIMPLE 代码里转向开关一直填 0（等于不跟踪朝向），ToolBench 要开开关才会转身，见问题 3。
3. **ToolBench 没有根朝向，需要专门处理。** Kimodo 的历史条件里一直有根朝向。两个办法：
   (a) 适配器用我们自己发出的目标朝向推算当前朝向，当作观测；
   (b) 历史里把根朝向遮掉。
   不管选哪种，训练数据必须用同一种（演示里虽然有真实骨盆位姿，也不能拿来做历史），否则训练和评测看到的不一样。
4. **Planner 要补的东西**（等价测试需要，全部做成参数，默认行为不变）：
   - 每次执行帧数：已有（Arena 15、其余 50）。
   - 第一帧位移从哪里算起（根位置的起点）：Arena 默认用模型生成的历史末端；TWIST2 正式评测用 `--anchor-root 1`
     （第一帧预测）；SIMPLE 用外推 `2·r0 − r1`。三种都做成选项。
   - 连续手部控制（SIMPLE、ToolBench）：目前 `KimodoPlanner` 只支持二值；历史改用 `Observation.hand` 里的测量闭合度。
   - SIMPLE 二值手的防抖状态机、1.5 m/s 导航速度保护：属于 SIMPLE 适配器，不进 planner。
   - RTC（实时分块融合）：已作为 planner 选项加上，默认关（正式评测都关着），见第 6 节。
   - 360×640 的图像由 DINOv3 预处理缩到 224×224（和 BiGym 84×84 放大到 224 是同一条路径），宽高比会变，模型里不用改。
5. **训练数据**：ToolBench 演示存了操作员指令，BiGym 那套「用指令当训练目标」可以照用：写一个
   `benchmarks/toolbench/demos.py` 实现 `Demonstration`，复用 `data/build.py`。
6. **等价测试**：仿照 `tests/test_port_equivalence.py`，同一串请求分别喂给旧 runtime 和新的「服务 + 适配器」，
   对比回复（Arena 比 40 维，SIMPLE 比 36 维）。需要 `KIMODO_ORIGINAL_REPO` 指向旧仓库，不跑 Isaac 仿真。

### 已确认（2026-10-25）

1. Arena、SIMPLE 只做等价测试（同样输入 → 同样输出），不跑 Isaac 仿真。
2. RTC 现在就作为 planner 的选项加上（和 benchmark 无关），默认关，评测时不开。已完成，见第 6 节。
3. 转向开关做成每个 benchmark 自己的设置：SIMPLE 保持旧代码的「关」（等价测试要求）；ToolBench 先看操作员
   演示里开关是怎么用的（`teleop.navigate_command` 第 3 项就是这个开关），在演示回放检查里比较「照操作员的用法」
   和「一直开」，选成功率高的。要注意：开关一开，控制器就一直用行走策略，不会切回站立策略。
4. TWIST2 的 `--anchor-root 1` 和 SIMPLE 的根起点外推都是必须的，做成 planner 选项，各适配器指定。
5. ToolBench 请求里确实没有骨盆位姿（`docs/PUBLIC_EVALUATION.md` 的请求表和 `policies/remote_humanoid.py` 只发
   32 维状态），用方案 a：适配器拿我们上一次发出的目标朝向当作当前朝向。生成训练数据时同样用操作员记录的目标
   朝向，不用演示里的真实骨盆位姿。

### 还没确定

6. ToolBench 的 vx、vy 用哪个坐标系：由 NVIDIA 的 `decoupled_wbc` 包决定，不在 ToolBench 仓库里，接入时查源码
   或在仿真里给一个速度看机器人往哪走。
