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
    agent.py        # Agent：保留最近观测，计划用完就重新规划；也可「记下观测、立即规划」（按请求要整段的 benchmark）
  model/
    features.py     # 位姿轨迹 → Kimodo 特征：评测（planner）和训练数据共用同一份转换
    planner.py      # KimodoPlanner：把 KimodoPolicy 包成 Planner（以后 VLM 版本并列放这里）
    kimodo_policy.py, modules/   # 从旧仓库原样迁入，只改了 import 路径
  motion/ skeleton/ utils/       # G1 运动表示、正运动学、几何工具（原样迁入）
  controllers/
    groot_wbc.py    # GR00T-WBC（BiGym）：骨盆轨迹 ↔ 底盘速度/转向角速度/高度指令（双向）
    decoupled_wbc.py # NVIDIA decoupled WBC（SIMPLE、ToolBench）：40 维参考 → 36 维指令（速度 + 目标朝向）
  data/
    episodes.py     # 训练文件格式 + KimodoWindowDataset（与 benchmark 无关）
    build.py        # 一条演示 → 训练文件（历史、目标、手、帧对齐；与 benchmark 无关）
  benchmarks/
    base.py         # BenchmarkAdapter、Demonstration 接口 + AgentPolicy（把 Agent 和适配器拼成 policy）
    replay.py       # DemoPlanner / DirectPlanner：把演示当成「完美预测」回放，检查动作接口
    server.py       # ChunkPolicy + make_server：按请求要整段动作的 benchmark 共用的 HTTP 服务
    arena/          # HumanoidArena：adapter（64 维状态 → Observation，回 40 维）+ policy（路由、入口）
    simple/         # SIMPLE：adapter + hands（Dex3 手、二值手防抖）+ policy（同进程调用或 HTTP）
    toolbench/      # HumanoidToolBench：adapter（32 维状态，补全历史）+ policy（/act、/info、/release）
    bigym2/
      adapter.py    # BiGym 2.0 观测/动作的双向翻译（动作只管 BiGym 的布局和归一化）
      demos.py      # 读 BiGym 演示：实际执行的动作、操作员指令、图像（BiGym2Demo）
      policy.py     # make_policy / make_kimodo_policy：交给 BiGym 官方评测器
scripts/
  prepare_bigym2.py         # 演示 → 训练文件（Kimodo 特征 + 三路官方 84×84 图像）
  cache_text_embeddings.py  # 20 个任务指令的 LLM2Vec embedding
  train.py                  # 训练一个任务（第 12 节），单卡或 torch.distributed.run 多卡
  run_config.py             # 读配置、列 checkpoint，不依赖模型代码
  submit_bigym2.py          # 把主表 27 个训练和随后的评测提交到 Slurm 集群
  eval_run.py               # 评测一个 run 的若干 checkpoint（同一张卡上并行）
  summarize_bigym2.py       # 汇总成论文主表格式（旁边列 π0.5）
  gpu_queue.py              # 不走集群时：把一批任务排到单机上没人用的 GPU
  check_bigym2_roundtrip.py # 动作来回转换检查
  eval_bigym2.py            # 按官方协议评测一个 checkpoint
  serve_arena.py / serve_simple.py / serve_toolbench.py   # 起 HTTP 服务，等对方评测器来调
configs/
  bigym2.yaml               # 训练的唯一一份基础配置（和 π0.5 同口径）
  bigym2_recipe.yaml        # 在基础配置上只改几项：和 ACT/DP 同口径（见第 12 节）
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

先看运行方式：仿真和我们在同一个进程、每步调一次（BiGym），用 `benchmarks/base.py` 的
`BenchmarkAdapter`；仿真在别的进程、每次请求要一整段动作（Arena、SIMPLE、ToolBench），用
`benchmarks/server.py` 的 `RequestAdapter`：

```python
class RequestAdapter(Protocol):
    control_fps: float
    def reset(self) -> None: ...
    def to_observations(self, request) -> list[Observation]: ...   # 上次请求以来的每个控制步，最新一步带图像
    def to_response(self, chunk: ActionChunk, request): ...        # 整段动作换成这个 benchmark 的格式
```

`ChunkPolicy(agent, adapter)` 处理一次请求：观测逐个记进 Agent、立即规划一次、翻译整段；
`make_server(routes, host, port, codec)` 把它放到 HTTP 后面，路由和编码（普通 JSON 或带 NumPy 的 JSON）
由每个 benchmark 的 `policy.py` 给出。

训练数据和逐步运行的接口在 `benchmarks/base.py`：

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

1. **`benchmarks/<name>/adapter.py`**：实现 `BenchmarkAdapter` 或 `RequestAdapter`。动作里控制器那部分
   调 `controllers/` 里对应的模块（GR00T-WBC：`groot_wbc.BaseCommander`；decoupled WBC：`decoupled_wbc.encode`；
   仿真那一侧自己跟踪 40 维参考的 SONIC、TWIST2：直接 `to_array40`），适配器自己只管布局和归一化。
   以后换底层控制器，只改这一处调用。HTTP 的 benchmark 再写 `policy.py` 里的路由。
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
去掉了 HTTP、调试开关和事务快照。所有关节都可见时，它送进模型的输入和旧代码一致。

输出的动作有两种取帧方式（`timing`）：
- `"exact"`（默认）：按每个控制步的真实时间读预测（第 12 节末尾的规划器时间修正）。旧代码每段慢一个模型帧，
  所以默认的动作和旧代码有意不同；`tests/test_port_equivalence.py` 只对比送进模型的输入（差 1.9e-6）。
- `"original"`：照旧代码的方式取帧，用来和旧代码逐项对比、复现旧仓库的评测结果。同一条 420 步轨迹、三种设置
  （每次执行 50 帧；15 帧；15 帧 + RTC）下动作最大差 6e-6；Arena、SIMPLE 的对比测试也用它。
  Arena、SIMPLE 的入口和 `serve_*.py` 默认用 `exact`，`--timing original` 切回旧方式。

RTC（实时分块融合：新一段动作的开头向上一段没执行完、时间相同的那几帧靠拢，让两段衔接更平滑）用
`KimodoPlanner(rtc_overlap_frames=12, rtc_frozen_frames=1)` 打开（旧仓库的设置），**默认关**；指令一变就重新开始。
和固定噪声、平滑过渡一起见第 12 节末尾。

和旧代码不同的两点：

1. 历史特征的掩码按 `joint_mask` 逐关节生成：没上报的关节，其全局旋转特征被遮住。
   所有关节都上报时，掩码和旧代码完全相同。
2. 50Hz 观测重采样到 30Hz 时，时间网格固定从回合开始算起（训练时整条轨迹也是这样重采样的），
   所以 Agent 只保留最近约 3.4 秒观测，结果也和保留整条历史一致。

手部控制两种都支持：二值（BiGym、Arena、SIMPLE 二值权重），连续（SIMPLE 连续权重、ToolBench；历史条件用
`Observation.hand` 里测到的闭合度）。另外两个按 benchmark 设的选项：`root_boundary`（每段第一帧位移从哪里算起，
见第 13 节）和 `hand_filter`（SIMPLE 二值手的防抖）。

接入 VLM：新模型实现同一个 `Planner` 接口即可，Agent、适配器、评测脚本一行不改。

## 7. 测试

```bash
pip install -e ".[bigym2,dev]"
pytest                                                    # 类型、Agent、历史编码、训练文件、GR00T 指令互逆、BiGym 适配器
KIMODO_ORIGINAL_REPO=<旧仓库路径> pytest   # 另外跑与旧代码的对比：推理流程、Arena 服务、SIMPLE（含手部防抖）
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
- [x] 训练脚本 `scripts/train.py` + `configs/bigym2.yaml`，见第 12 节。
- [x] 动作来回转换检查（`scripts/check_bigym2_roundtrip.py`，见第 10 节）。
- [x] 文本 embedding：`scripts/cache_text_embeddings.py` 给 `task_instructions.yaml` 里 20 个任务的指令
      各算一份 LLM2Vec embedding，存成 `<任务名>.pt`（格式同旧仓库，`planner.load_text_embeddings` 直接读）。
      A800 上已生成在 `data/cache/bigym2`。新环境（transformers 5.1、peft 0.18）算出的结果和旧仓库缓存的
      embedding 完全一致（同一句指令，最大差 0）；加载时打印的 LoRA「MISSING」是第一遍加载的提示，可以忽略。
      这 20 句指令和论文附录里 π0.5 用的逐字相同。
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
前 5 个任务 × 60 条演示，2026-10-24 在 A800 上跑；主表其余 6 个任务 2026-10-25 补跑（没测「实际执行」那一列）。

「修正前」一列是 2026-10-25 之前的规划器：它把预测帧从上一段末尾开始摊到控制步上，每段都比实际时间
慢约一个模型帧（33 毫秒），见第 12 节末尾。「修正后」是现在的规划器（按每个控制步的真实时间取帧），
同一批演示，每 50 帧和每 10 帧重新预测一次结果完全相同。

| 任务 | 原样回放 | 实际执行的动作当目标 | 操作员指令当目标（修正前） | 操作员指令当目标（修正后） | 指令直接给 50Hz |
|---|---|---|---|---|---|
| drawer_top_open | 100% | 0% | 97% | 97% | 100% |
| flip_cup | 100% | 2% | 97% | 90% | 98% |
| move_plate | 100% | 7% | 97% | 100% | 100% |
| pick_box | 100% | 0% | 80% | 90% | 90% |
| reach_target_single | 100% | 95% | 100% | 100% | 100% |
| drawer_top_close | 100% | – | 100% | 100% | 100% |
| move_two_plates | 100% | – | 92% | 93% | 100% |
| reach_target_multi_modal | 100% | – | 100% | 100% | 100% |
| reach_target_dual | 100% | – | 98% | 100% | 100% |
| dishwasher_close | 100% | – | 83% | 87% | 98% |
| dishwasher_load_cups | 100% | – | 97% | 93% | 100% |

修正前每 10 帧重新预测时，drawer_top_open 和 move_plate 只剩 67%；修正后和每 50 帧一样。
修正后手臂跟踪误差从约 0.007 rad 降到 0.001 rad。

结论：

1. **训练目标必须用操作员的指令，不能用机器人实际执行出来的动作。** 把实际动作当目标，
   手臂会再叠加一次重力下垂（平均误差 0.02 rad），除了够取任务以外基本全部失败。
   `bigym2/demos.commanded_motion` 就是训练要用的目标（对应 `Demonstration.commanded_motion`）。
2. **朝向闭环有害，已关闭。** 增益为 2 时 move_plate 掉到 23%，pick_box 掉到 2%：
   它会去纠正 WBC 本来就有的朝向漂移，结果走出和演示不一样的路线。
3. **转向角速度要和平移一样用向后差分。** 原来用的是向前差分，比操作员提前一步，
   pick_box 因此少了约 12 个点（50Hz 直接给：78% → 90%）。
4. **规划器必须按控制步的真实时间取帧。** 以前 pick_box 少的 10 个点（80% vs 90%）原以为是 30Hz 表示
   把路线磨平，其实是规划器慢一帧；修正后 pick_box 和直接给 50Hz 一样是 90%。
   修正后仍和 50Hz 有差距的是 dishwasher_close（87% vs 98%）和 flip_cup（90% vs 98%），
   flip_cup 和 dishwasher_load_cups 修正后反而低了 4–7 个点（每个任务 60 条，误差约 ±4 个点）。
   论文主表 9 个任务上，这套接口的成功率上限约为 96%（修正前约 94%；π0.5 是 75%）。
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

## 12. 训练

**协议**（2026-10-25 定，和论文里 π0.5 的做法对齐，见 BiGym 2.0 论文附录 D-C、Table A5）：
每个任务单独训一个模型，只用该任务的 60 条演示；30k 步，每步 48 个样本，前 1k 步学习率线性升温，
之后余弦降到 0.1 倍；保存 5k、10k……和最后 5 个（26k–30k，间隔 1k），评测这最后 5 个。
主表是论文的 9 个任务：pick_box、reach_target_multi_modal、reach_target_dual、drawer_top_close、
drawer_top_open、move_plate、move_two_plates、dishwasher_close、dishwasher_load_cups（π0.5 平均 75%）。
这 9 个任务的训练数据都已在 A800 的 `prepared84` 里。

```bash
python scripts/train.py --task move_plate                    # 三路相机（主结果）
python scripts/train.py --task move_plate --cameras head     # 只用头部（消融）
python scripts/train.py --task move_plate --cameras head --init-checkpoint <pt_419h checkpoint>   # 对照
python scripts/train.py --resume log/bigym2/<run>             # 中断后接着训
```

训练数据目录取环境变量 `BIGYM2_PREPARED`（A800 上由 `kimodo-vlm-env.sh` 设置）。其余设置都在 `configs/bigym2.yaml`，命令行可以用 `main.max_steps=200` 这种写法临时改。
模型结构和损失权重沿用 pt_419h（4 层 ControlNet、root/body 梯度分开、root 损失权重 4、梯度裁剪 1.0），
所以 pt_419h 对照和从零训练只差起点。学习率分组沿用旧仓库：从 Kimodo 复制来的 ControlNet 层 3e-5，
新加的投影层和手部头 1e-4。样本是该任务所有训练窗口均匀打乱，不做图像增强（π0.5 也没写用了）。

**输出**：`log/bigym2/<任务>_<路数>cam_<时间>/` 下有 `config.json`、`train.log`、`metrics.jsonl`、
`resume.pt`（最新的优化器状态，用来接着训）和各个 `checkpoint_<步数>/`（只存 ControlNet 和手部头权重）。
checkpoint 里的 `config.json` 写了 `model.cameras`，评测时 planner 据此取相机，直接给 `eval_bigym2.py` 用。

**实测**（A800，move_plate，三路，每步 48）：显存 21.5GB，约 0.6 秒/步，30k 步约 5 小时。

**批量跑**：`scripts/gpu_queue.py` 读一个任务清单（每行 `名字 命令`），一张卡跑一个，
只用没人占的卡（显存占用低于 500MB；别人的进程即使闲着也占显存），有卡空出来就接着排。
每个任务的输出、所在 GPU、退出码都记在日志目录里；队列本身挂了也不影响已经在跑的训练，
重启队列会跳过已经结束和还在跑的。失败的不自动重试。

**集群（Slurm）**：A800 登录节点上 `sbatch` 可以提交到 30 个计算节点（每个 8 张 A800）。
`scripts/submit_bigym2.py` 给每个（任务，设置）提交 1 个训练作业（1 张卡），训练成功后自动接 2 个评测作业：
最后 5 个 checkpoint（主表分数）和 5k–25k（学习曲线），每个作业在 1 张卡上同时评 5 个 checkpoint
（每个评测进程约 3.3GB 显存），每个 checkpoint 100 回合。结果写在 `<run>/eval/checkpoint_<步数>.json`，
`scripts/summarize_bigym2.py <目录>` 汇总成表。
每个 checkpoint 评两种设置，在表里并列：默认执行完整个 50 帧（1.67 秒）再重新预测，结果在 `eval/`；每 10 帧（0.33 秒，接近 π0.5 的 0.32 秒）重新预测一次，结果在 `eval_exec10/`（`eval_run.py --execution-frames 10`）。重跑提交脚本只补缺的部分：还在队列里的同名作业直接沿用，训完的不重训，没训完的接着训，
评过的 checkpoint 跳过。

计算节点上没有登录节点本地那份 Python（`envs/kimodo` 指向登录节点的 `/var/tmp`），所以 A800 上另建了
`.venv-cluster`：同一套已安装的包，换用共享盘上的 `envs/kimodo.shared`。`kimodo-vlm-env.sh` 会自动选。

**2026-10-25 提交的正式训练**：主表 9 个任务 × 3 组（三路从零 = 主结果；只用头部从零 = 消融；
只用头部从 pt_419h 起步 = 对照），共 27 个训练、108 个评测作业（两种重新预测间隔各 54 个），输出在 A800 的 `log/bigym2/main9/`。

**规划器时间修正（2026-10-25）**：查「每 0.33 秒重新预测几乎全败」时发现，规划器把预测帧从上一段末尾开始
摊到控制步上，每段都比实际时间慢约一个模型帧，10 帧还被拉成 17 步。现在按每个控制步的真实时间取帧
（预测第 0 帧在模型帧 round(步数 × 0.6)，之前的时间用历史最后一帧的根位置），每段不超出最后一个预测帧，
手部历史也按模型帧对齐。`tests/test_planner_timing.py` 在旧代码上会失败。修正前的评测结果保留在各 run 的
`old_timing_eval*/` 里，主表全部用修正后的代码重新评测。修正只解决了一部分：真实模型每 0.33 秒重新预测
仍然大幅掉分，原因是前后两次预测对不上（每次重新预测手臂指令跳动是平时的 5–20 倍），这是模型侧的问题。

**频繁重新预测的三个开关（2026-10-26）**：`KimodoPlanner` 增加了固定噪声（`noise="fixed"`）、RTC
（`rtc_overlap_frames`，模型自带的实时分块衔接，旧仓库推理程序默认开启 12 帧，移植时漏掉了）和平滑过渡
（`blend_steps`），默认全关，`eval_bigym2.py` 有对应参数。够取两个任务第 30k 步、各 50 回合的对比：
每 0.33 秒重新预测时，固定噪声和平滑过渡基本无效；RTC 越强越好，重叠 40 帧时 multi_modal 从 12% 回到 50%、
dual 从 6% 回到 18%，但只追平默认的每 1.67 秒（54%、26%），没有超过。每 0.83 秒、不加开关也只有 16%：
中途根据新观测重新预测并不比坚持原计划好，说明模型对新观测（尤其图像）的利用是瓶颈。这些对比用的是官方评测种子，
只作分析，不用于挑选主结果的设置。

**和 ACT/DP 同口径的一套（2026-10-26 开训）**：`configs/bigym2_recipe.yaml` 在基础配置上加 BiGym 官方的
基线训练配方（图像随机平移 8 像素、以 0.2 的概率只保留历史里的根朝向和骨盆朝向），训练量改成 101k 步 × 256，
每 5000 步存一次，主分数取 80k–100k 的 5 个 checkpoint（论文对 ACT/DP 的做法）。每个任务 1 个种子
（ACT/DP 是 3 个，要披露）。手部历史不遮：它是过去的夹爪指令，不是测量值。每个 run 用 8 张卡
（`python -m torch.distributed.run`；计算节点上的 `torchrun` 脚本指向登录节点的 Python，不能用），
实测约 0.45 秒/步，101k 步约 12–14 小时。评测只评默认设置，`eval_run.py --watch` 和训练同时启动，
每个 checkpoint 存好就评。输出在 A800 的 `log/bigym2/main9_recipe/`。

```bash
python scripts/submit_bigym2.py --out log/bigym2/main9_recipe --config configs/bigym2_recipe.yaml \
    --variants 3cam --gpus-per-train 8 --execution-frames 50 --eval-watch --eval-parallel 8
```

## 13. 四个 benchmark 共用一套接口

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

6. ToolBench 的 vx、vy 在机器人自身坐标系下：仿真回放里实际走的方向和指令方向一致（见下面的回放检查）。

### 实现情况（2026-10-26）

已完成（都不跑仿真，只在 Mac 上测）：

| | 代码 | 和旧代码 / 协议的对照 |
|---|---|---|
| 通用服务 | `benchmarks/server.py`、`Agent.observe/replan` | `test_server.py`：真实 HTTP，两种编码、错误返回 |
| Arena | `benchmarks/arena/`、`scripts/serve_arena.py` | `test_arena_port.py`：同一串请求喂给旧服务和新服务，SONIC、SONIC + RTC、TWIST2 三种设置，40 维回复差 < 1e-5 |
| SIMPLE | `benchmarks/simple/`、`controllers/decoupled_wbc.py`、`scripts/serve_simple.py` | `test_simple_port.py`：二值手、连续手两种权重，40 维和 36 维回复差 < 1e-5；手部防抖状态机用 400 次随机重规划逐帧对比，完全一致 |
| ToolBench | `benchmarks/toolbench/`、`scripts/serve_toolbench.py`、`scripts/prepare_toolbench.py` | `test_toolbench_adapter.py`：32 维状态读取、补全的历史、36 维布局、会话规则（409、`/release`、`/info`）。`test_toolbench_demos.py`：训练数据里的历史和评测时适配器报告的逐步相同；从演示的指令轨迹重建出的 36 维指令就是记录下来的指令。没有旧代码可比 |

ToolBench 每次请求只给一个状态，评测器却在两次请求之间执行完整段动作。适配器把中间这些步用我们发出的
目标补上（朝向也用我们发出的目标朝向），这样历史仍然每个控制步一条。训练数据必须按同样方式构造。

和旧代码不同的一处：SIMPLE 的速度保护拒绝一段动作时，旧代码会把内部状态回滚，新代码不回滚
（回合本来就在这里报错结束）。

还没做：

- SIMPLE 同进程评测：旧代码的 `--eval` 在 SIMPLE 自己的环境里跑仿真和模型。新代码的 `ChunkPolicy` 可以直接调用，
  但还需要在 SIMPLE 的评测循环里接上（它的环境依赖和我们的不同，要实际装环境才能做）。
- ToolBench 的 `root_boundary`（每段第一帧位移从哪里算起）：回放检查用的是完美预测，比不出差别，
  要等有训练好的 ToolBench 模型再比。现在按 SIMPLE 用外推。

### ToolBench 训练数据（2026-10-26）

读 ToolBook（`snupilab/humanoidtoolbench-teleop`，revision 8b2cd31e）的 `raw/` 录制，`scripts/prepare_toolbench.py`
生成和 BiGym 同格式的训练文件。看了 12 条真实演示后定下的几点：

- **训练目标是操作员指令**（和 BiGym 一样）：手臂、躯干、手来自 `action`，骨盆高度和行走来自 `teleop.*`。
  录制的第 t 行是第 t+1 步的目标。HumanoidToolBench 官方 ACT 模型的数据说明也是这么转换的。
- **转向开关照操作员的用法**：操作员只在转身时打开开关（0–5% 的帧），值就是转身角速度，其余时间为 0，
  朝向不被跟踪、会自然漂移约 ±0.3 rad。所以默认 `turn="operator"`：规划出的朝向在转、且角速度 ≥ 0.1 rad/s
  时才开；一直开等于朝向闭环，BiGym 上证明有害。12 条演示里我们和操作员的开关在 98.8–100% 的帧上一致。
- **目标朝向是世界系的绝对角**：三个场景都让机器人以 -90° 出生，适配器加上这个固定偏移。
- **手的闭合姿态是 ToolBench 自己的**：操作员的手指令基本是开/合两种，中间有很短的过渡；闭合姿态和 SIMPLE 不同
  （右手有一部分录制用了 SIMPLE 的姿态，换算后拇指最多差 0.5 rad）。闭合度按连续值存。
- **历史和评测一致**：评测器每段动作只给一次状态，所以训练历史也是每 83 步（50 个模型帧）一个实测状态、
  中间用指令补齐，朝向一律用指令朝向。
- **图像**：头部相机 360×640，训练数据和评测时都先缩到 224×224（同一个函数），所以 DINOv3 看到的像素相同。
  一条 21 秒的演示约 97MB，全部 ToolBook 估计 350–400GB。

把演示的指令经适配器换回 36 维指令（不跑仿真）：速度、朝向、手臂与记录的差 < 3e-6。

### ToolBench 仿真回放检查（2026-10-26）

`scripts/check_toolbench_roundtrip.py`，在 A800 上 ToolBench 自己的环境里跑（安装方法见下）。

**测什么**：ToolBench 公开版的工具是程序生成的、台面布局也和录制用的内部工具不同，录制时的场景复原不了，
所以不看任务成功率，只看机器人动作对不对。机器人放在录制时的出生位置，用评测器自己的 agent、稳定流程和
decoupled WBC 执行整条演示，每条演示跑五种方式：照录制的指令（参照）、我们的 50Hz 换算、经过 Kimodo 30Hz
（评测时的路径，已含规划器时间修正）、转向开关一直开、录制指令加 3e-6 m/s 噪声（看这个检查的下限）。

**结果**（和「照录制指令」那次的终点位置差、朝向最大差；只算所有方式都跑成的演示。一部分运行因登录节点上
EGL 离屏渲染出错没跑成，未计入。没有一次摔倒）：

| 方式 | 演示数 | 终点位置差 均值 / 中位数 | 朝向最大差 均值 |
|---|---|---|---|
| 我们的 50Hz 换算 | 37 | 0.069 / 0.008 m | 0.056 rad |
| 经过 Kimodo 30Hz | 37 | 0.082 / 0.032 m | 0.080 rad |
| 转向开关一直开 | 37 | 0.244 / 0.163 m | 0.219 rad |
| 录制指令加 3e-6 m/s 噪声（下限） | 15 | 0.067 / 0.004 m | 0.059 rad |

均值被少数几条拉高：带接触的长时间行走对极小的差别很敏感，只加 3e-6 的噪声也有一条偏了 0.54 m。
我们的换算和录制指令只差浮点舍入（速度 3e-6、朝向 1e-7），外加一处：有几位操作员右手用的是 SIMPLE 的闭合姿态，
按 ToolBench 姿态换算后拇指差 0.5 rad（大约一半闭合帧是这种），这几条偏得比噪声多。

结论：
1. 仿真本身完全可复现（同样的指令跑两次，差为 0）。
2. 转向开关照操作员用法（只在转身时开）是对的；一直开偏得多 3 倍以上。
3. 开关要用「下一步」的朝向变化：操作员第 t 行的开关值等于目标朝向从 t 到 t+1 的变化率。最早用的是向后差分，
   每条演示的第一步都会误开一下、转身前后也错一步（14,258 行里错 21 行），改成向前差分后 13,778 行全对，
   位置差从 0.072 降到 0.063 m、朝向差从 0.069 降到 0.051 rad。
4. ToolBench 的走路控制器只走出指令速度的约 1/3（原始录制里也是这样），所以实际路线比「指令积分出的路线」短。
   训练目标按指令、评测按同一方式换回去，和 BiGym 一样不受影响。
5. Kimodo 30Hz 表示的额外损失：位置中位数约 2.5 cm、朝向约 0.02 rad。
6. 手的闭合姿态：录制里右手有两种闭合姿态，各占约一半；模型只输出闭合度，评测时只能选一种（现在用 ToolBench 自己那种）。

**ToolBench 环境（A800，无 GitHub）**：在 Mac 上 clone HumanoidToolBench（3fec9bd）和它锁定版本的 gear_sonic、
decoupled_wbc，拷到 `third_party/HumanoidToolBench-3fec9bd`；用系统 Python 3.10 建 `.venv`，按 `uv.lock`
导出的 `requirements-a800.txt` 从阿里云 PyPI 镜像和 download.pytorch.org 安装，另加 pyarrow（读录制用）；
场景网格经 hf-mirror.com 下载（`fetch_assets_mirror.py`，需要 `SSL_CERT_FILE` 指向 certifi）。
回放不渲染图像，所以缺 `libOpenGL.so.0` 不影响；正式评测要渲染头部相机，需要补这个库。

