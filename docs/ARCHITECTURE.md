# Kimodo-VLA 代码架构

> 给新加入的合作者：代码分几层、每层管什么、接一个新 benchmark 要写哪几个文件。
> 这里只写现在的规则。每个 benchmark 的实验结论、做过的决定和待办在 `docs/benchmarks/` 下：
> [BiGym 2.0](benchmarks/bigym2.md)、[HumanoidToolBench](benchmarks/toolbench.md)、
> [HumanoidArena 和 SIMPLE](benchmarks/arena_simple.md)。

## 1. 一句话

模型只认一种「通用观测」和一种「通用动作」；每个 benchmark 写一个薄适配器，负责把自己的观测翻译成通用观测、
把通用动作翻译回自己的动作。模型、推理循环都不知道自己在跑哪个 benchmark。

```
 benchmark ──原始观测──▶ 适配器 ──Observation──▶ Agent ──▶ Planner（模型）
     ▲                                                         │
     └──环境动作 / 回复── 适配器 ◀──ActionChunk（控制频率）─────┘
```

## 2. 目录

```
kimodo_vla/
  core/            types.py：Observation、ActionChunk、Planner（全仓库唯一的约定）
                   agent.py：Agent，`step` 每步交一个观测、计划用完就重新规划；`observe` + `replan` 先记下观测、再立即规划
  model/           planner.py：KimodoPlanner（KimodoPolicy 包成 Planner）
                   planner_original.py：OriginalTimingPlanner（旧仓库的取帧方式，只用于对比和复现）
                   features.py：位姿轨迹 → Kimodo 特征，评测和训练数据共用
                   kimodo_policy.py、modules/：从旧仓库原样迁入
  motion/ skeleton/ utils/   G1 运动表示、正运动学、几何工具（原样迁入）
  controllers/     groot_wbc.py：GR00T-WBC（速度 + 转向角速度 + 高度）
                   decoupled_wbc.py：NVIDIA decoupled WBC（36 维：速度 + 目标朝向 + Dex3 手）
  data/            episodes.py：训练文件格式 + KimodoWindowDataset
                   build.py：一条演示 → 训练文件
  benchmarks/      base.py：BenchmarkAdapter、Demonstration、AgentPolicy
                   server.py：RequestAdapter、ChunkPolicy、make_server（按请求要整段动作的 benchmark）
                   replay.py：DemoPlanner / DirectPlanner（把演示当完美预测回放）
                   bigym2/  arena/  simple/  toolbench/：每个 benchmark 的适配器、读演示、入口
scripts/           每个 benchmark 几个薄脚本（见 README）；train.py 及其集群脚本目前按 BiGym 配置
configs/           训练配置（bigym2.yaml 为基础，其它配置只写不同的几项）
tests/             见第 8 节
```

依赖方向只有一条：`benchmarks → core ← model`，`benchmarks → controllers、data`。
`core/`、`model/`、`controllers/`、`data/` 里不出现任何 benchmark 名字。

## 3. 三个约定（`core/types.py`）

MuJoCo 世界坐标系、z 朝上、米和弧度；关节向量按 `G1_JOINT_NAMES`（29 个 G1 关节，顺序与已发布 checkpoint
的 40 维动作一致）；手的闭合度 0 = 开、1 = 合。

```python
@dataclass
class Observation:                   # 一个控制步看到的东西
    images: dict[str, np.ndarray]    # 相机名 → (H, W, 3) uint8
    root_pos: np.ndarray             # (3,) 骨盆位置
    root_rot: np.ndarray             # (3, 3) 骨盆朝向
    joint_pos: np.ndarray            # (29,)
    joint_mask: np.ndarray           # (29,) bool，benchmark 不提供的关节为 False
    hand: np.ndarray                 # (2,) 测到的闭合度
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
接入 VLM：新模型实现同一个 `Planner` 接口即可，Agent、适配器、评测脚本一行不改。

## 4. 两种运行方式

| | 逐步（BiGym） | 按请求要整段（Arena、SIMPLE、ToolBench） |
|---|---|---|
| 谁调谁 | 仿真和我们在同一进程，每个控制步调一次 | 仿真（通常在别的进程）每次请求要一整段动作 |
| 适配器接口 | `BenchmarkAdapter`（`benchmarks/base.py`）：`to_observation`、`to_env_action` | `RequestAdapter`（`benchmarks/server.py`）：`to_observations`（上次以来的每一步）、`to_response`（整段） |
| 拼成 policy | `AgentPolicy(agent, adapter)`：`policy(raw_obs) -> env_action`，内部 `Agent.step` | `ChunkPolicy(agent, adapter)`：观测逐个 `Agent.observe`，`Agent.replan` 一次，翻译整段；同一时间只处理一个请求 |
| 放到 HTTP 后 | 不需要 | `make_server(routes, host, port, codec)`；路由和编码（普通 JSON 或带 NumPy 的 JSON）由各 benchmark 的 `policy.py` 给出 |

## 5. 规划器（`model/planner.py`）

`KimodoPlanner` 是旧仓库 `evaluation/humanoidarena_server.py` 推理流程的移植，去掉了 HTTP、调试开关和事务快照。
所有关节都可见时，送进模型的输入和旧代码一致。它按每个控制步的真实时间读预测（预测第 0 帧在模型帧
round(步数 × 0.6)，之前的时间用历史最后一帧的根位置），每段不超出最后一个预测帧。

和旧代码有意不同的三点：
1. 取帧时间：旧代码每段比实际时间慢约一个模型帧（33 毫秒），10 帧会被拉成 17 步（见 BiGym 文档的回放结论）。
2. 历史特征的掩码按 `joint_mask` 逐关节生成：没上报的关节，其全局旋转特征被遮住；全部上报时和旧代码相同。
3. 50Hz 观测重采样到 30Hz 时，时间网格从回合开始算起（训练时整条轨迹也这样重采样），所以 Agent 只保留
   最近约 3.4 秒观测，结果和保留整条历史一致；手部历史也按模型帧对齐。

选项（默认值就是旧仓库 Arena + SONIC 的行为）：

| 选项 | 作用 | 谁在用 |
|---|---|---|
| `execution_frames` | 每次执行多少个模型帧（0 = 整段 50 帧） | Arena 15；其余 50 |
| `cameras` | 取哪几路相机、按什么顺序；默认读 checkpoint `config.json` 里的 `model.cameras`，没写就是 `head` | BiGym 三路为主结果 |
| `noise="fixed"` | 一个回合里每次规划用同一份扩散噪声 | 频繁重新预测的实验 |
| `rtc_overlap_frames`、`rtc_frozen_frames` | RTC（实时分块融合）：新一段的开头向上一段没执行完、时间相同的那几帧靠拢；指令一变就重新开始。旧仓库用 12 和 1 | 默认关 |
| `blend_steps` | 新一段的前几个控制目标从上一段平滑过渡 | 默认关 |
| `root_boundary` | 每段第一帧位移从哪里算起：`"history"`（模型生成的历史末端）、`"first_prediction"`（第一帧预测）、`"extrapolate"`（2·r0 − r1） | Arena+SONIC、Arena+TWIST2、SIMPLE 和 ToolBench |
| `hand_filter` | 发出去之前改写二值手的执行段（并记进手部历史） | SIMPLE 的手部防抖 |
| 连续手 | checkpoint 的 `hand_control_mode="continuous"`：历史用 `Observation.hand` 里测到的闭合度，输出 [0, 1] 的闭合度 | SIMPLE 连续权重、ToolBench |

`OriginalTimingPlanner`（`model/planner_original.py`）保留旧仓库的取帧方式和手部历史，只用于和旧代码逐项对比、
复现旧仓库的评测；Arena、SIMPLE 的 `serve_*.py --timing original` 切过去。

## 6. 控制器（`controllers/`）

| 控制器 | 用在 | 我们发什么 | 模块 |
|---|---|---|---|
| SONIC、TWIST2（在仿真那一侧） | Arena | 40 维全身参考 | 不需要，`ActionChunk.to_array40` |
| GR00T-WBC | BiGym | vx、vy（该帧骨盆坐标系）、转向角速度（向后差分、纯前馈）、高度 | `groot_wbc.BaseCommander`；反方向 `pelvis_path` |
| decoupled WBC（GR00T Balance/Walk） | SIMPLE、ToolBench | 36 维：Dex3 两手、手臂、躯干、高度、vx、vy、转向开关、目标朝向（绝对角） | `decoupled_wbc.encode`；反方向 `pelvis_path` |

适配器只管自己的动作布局和归一化；以后换底层控制器，只改适配器里调控制器的那一行。

## 7. 训练数据（`data/`）

每个 benchmark 把一条演示包成 `Demonstration`（`benchmarks/base.py`），`data/build.training_episode` 统一生成：

- `<演示名>.pt`（30Hz）：`observed_motion` 是模型看到的历史，**由评测时同一个适配器产生的观测**经
  `KimodoFeatures.history` 得到；`target_motion` 是要学的目标，**是操作员的指令，不是机器人实际执行出来的动作**
  （实际动作里叠加了一次手臂下垂，当目标几乎全失败，见 BiGym 文档）；`hand` 是手的指令；
  还有 `task`、`instruction`、`fps`、`cameras`。
- `<演示名>.frames.npy`：每个 30Hz 帧对应的各路相机图像，`[T, 相机, H, W, 3]`。

`KimodoWindowDataset` 从中切窗口：每个样本 = 100 帧历史 + 50 帧未来 + 切点处的图像，键名和约定与旧仓库的
训练数据一致，`KimodoPolicy.training_kimodo_policy_controlnet` 可以直接用。`cameras=(...)` 选用哪几路：
一路时 `egoview` 是 `[3, H, W]`，多路时是 `[路数, 3, H, W]`，和 `KimodoPolicyConfig.num_cameras` 对应。

## 8. 接一个新 benchmark

不需要改 `core/`、`model/`、`controllers/`、`data/`：

1. **`benchmarks/<name>/adapter.py`**：实现 `BenchmarkAdapter` 或 `RequestAdapter`；控制器那部分调
   `controllers/` 里对应的模块。按请求的 benchmark 再写 `policy.py` 里的路由。
2. **`benchmarks/<name>/demos.py`**：读演示，每条包成一个 `Demonstration`；训练目标用操作员指令，
   历史要和评测时适配器报告的一模一样。
3. **几个薄脚本**：`prepare_<name>.py`（演示 → `data.build.training_episode` → `save_episode`）、`check_<name>_roundtrip.py`（用 `benchmarks.replay`
   回放演示，统计这个 benchmark 自己的判定），评测入口（`eval_<name>.py` 或 `serve_<name>.py`）。不做跨
   benchmark 的统一入口。
4. **`pyproject.toml`**：在 `[project.optional-dependencies]` 加一组 `<name> = [...]`，锁定版本。
5. **测试**：`tests/test_<name>_adapter.py` 检查观测读数（能在真实环境里查就在真实环境里查），再在真实演示上
   跑一次来回转换检查；有旧实现的，加一个「同样输入 → 同样输出」的对比测试。结论写进 `docs/benchmarks/<name>.md`。

## 9. 测试

```bash
pip install -e ".[bigym2,toolbench,dev]"
pytest                                       # 不需要旧仓库的全部测试；没装 BiGym 时跳过 BiGym 环境测试
KIMODO_ORIGINAL_REPO=<旧仓库路径> pytest      # 再加上与旧代码的对比：推理流程、Arena 服务、SIMPLE（含手部防抖）
```

| 测试 | 管什么 |
|---|---|
| `test_types.py`、`test_agent.py`、`test_server.py` | 通用类型、Agent、HTTP 服务（真实端口、两种编码、错误返回） |
| `test_planner_history.py`、`test_planner_timing.py`、`test_multicamera.py` | 规划器的历史编码、取帧时间和各选项、多路相机 |
| `test_episodes.py` | 训练文件格式，训练窗口的历史和评测时规划器构造的相同 |
| `test_groot_wbc.py` | GR00T 指令和骨盆轨迹互逆 |
| `test_bigym2_adapter.py` | 真实 BiGym 环境里的观测和动作（BiGym 文档有细节） |
| `test_port_equivalence.py`、`test_arena_port.py`、`test_simple_port.py` | 与旧仓库逐项对比（需要 `KIMODO_ORIGINAL_REPO`） |
| `test_toolbench_adapter.py`、`test_toolbench_demos.py` | ToolBench 协议、状态读取，训练历史和评测一致、指令能还原 |

## 10. 和旧仓库相比刻意不做的事

1. **一个 benchmark 一个两千行的服务文件。** 旧 `simple_server.py` 有 2208 行，靠动态拼类、临时替换另一个文件里的
   函数来复用 Arena 的推理逻辑。新结构里推理只在 `model/planner.py` 写一次。
2. **推理循环藏在某个 benchmark 文件里。** 旧代码的历史缓存、分块执行都在 `humanoidarena_server.py` 里。
3. **每个实验一对 sh + yaml。** 旧 `scripts/` 下约 50 对，彼此只差一两个字段。训练只保留一份基础配置。
4. **数据源的专用小函数堆在公共文件里。** 旧 `data/common.py` 有 1300 行。

旧仓库保持不动，作为论文结果的复现版本。
