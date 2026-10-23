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
    planner.py      # KimodoPlanner：把 KimodoPolicy 包成 Planner（以后 VLM 版本并列放这里）
    kimodo_policy.py, modules/   # 从旧仓库原样迁入，只改了 import 路径
  motion/ skeleton/ utils/       # G1 运动表示、正运动学、几何工具（原样迁入）
  benchmarks/
    base.py         # BenchmarkAdapter 接口 + AgentPolicy（把 Agent 和适配器拼成 policy）
    bigym2/
      adapter.py    # BiGym 2.0 观测/动作的双向翻译
      policy.py     # make_policy / make_kimodo_policy：交给 BiGym 官方评测器
scripts/
  eval_bigym2.py    # 按官方协议评测一个 checkpoint
tests/              # 见第 7 节
```

依赖方向只有一条：`benchmarks → core ← model`。`core/` 和 `model/` 里不出现任何 benchmark 名字。

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

## 4. Benchmark 适配器（`benchmarks/base.py`）

```python
class BenchmarkAdapter(Protocol):
    control_fps: float
    def reset(self) -> None: ...
    def to_observation(self, raw_obs) -> Observation: ...
    def to_env_action(self, target: ActionChunk, obs: Observation) -> np.ndarray: ...
```

`AgentPolicy(agent, adapter)` 把两者拼成 `policy(raw_obs) -> env_action`，带 `reset()`。

**接一个新 benchmark = 新建 `benchmarks/<name>/adapter.py`（实现上面 3 个函数）
+ 一个入口函数**。不需要改 `core/`、`model/`。

## 5. BiGym 2.0 适配器具体做什么

| 方向 | 内容 |
|---|---|
| 观测 | `low_dim_obs` 按官方布局取出：骨盆 x/y/z/yaw、14 个手臂关节、（开启躯干俯仰的 14 个任务）3 个腰关节、2 个夹爪。腿部关节和（其余任务的）腰关节 `joint_mask=False`、值填 0。实测 WBC 接管腰部时，腰俯仰也只在 0.06 rad 以内。骨盆横滚和俯仰角在官方观测里拿不到，按 0 处理。`rgb_obs` 拆成 head / right_wrist / left_wrist 三张图 |
| 动作 | `vx, vy` = 第 0 帧位移 × 50Hz（前馈）；`wz` = 目标朝向的变化率 + 2.0 × 朝向误差（闭环）；`height` 直接给；`torso pitch` 取目标腰俯仰关节；14 个手臂关节直接给；手闭合度 ≥ 0.5 → 夹爪合。最后按 `env.wholebody_action_layout()` 归一化到 [-1, 1] |
| 语言 | 用 BiGym 自带的 `task_instructions.yaml`，和演示数据里的指令一致 |

BiGym 是纯 Python + MuJoCo，模型和环境在同一个进程里跑，不需要 HTTP 服务。
注意 BiGym 锁定 `mujoco<3.9`；`kimodo_vla` 本身不依赖 mujoco，所以评测环境里只装 BiGym 要的版本即可。

```bash
MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
    --checkpoint <ckpt 目录> --text-cache <文本 embedding 目录> --out outputs/move_plate.json
```

## 6. KimodoPlanner 和旧仓库的关系

`model/planner.py` 是旧仓库 `evaluation/humanoidarena_server.py` 里推理流程的移植，
去掉了 HTTP、RTC、调试开关和事务快照。所有关节都可见时，它送进模型的输入和输出的动作
与旧代码一致：`tests/test_port_equivalence.py` 用同一条 420 步轨迹对比，最大差 2e-6。

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
pytest                                                    # 类型、Agent、历史编码、BiGym 适配器
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

- [ ] 训练：`data/sources/bigym2.py`（演示 → Kimodo 训练样本，观测要走同一个 `to_observation`，
      保证训练和评测的掩码一致）和 `train.py`。
- [ ] 动作来回转换检查：演示动作 → `ActionChunk` → BiGym 动作，回放看成功率，判断 `vx, vy` 是否需要闭环。
- [ ] 文本 embedding：用 LLM2Vec 给 20 个任务指令预先算好。
- [ ] 待确认：84×84 还是 224×224（官方 runner 对相机分辨率改动不出排行榜记录，需要问作者 π0.5 是怎么报的）；手腕相机是否接进模型。
