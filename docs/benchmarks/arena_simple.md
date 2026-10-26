# HumanoidArena 和 SIMPLE

两者都来自旧仓库（`kimodo-policy`，`evaluation/humanoidarena_server.py`、`simple_server.py` 和两份 `*_eval.md`），
共用旧仓库的 64 维状态格式，所以放在一起。代码：`kimodo_vla/benchmarks/arena/`、`kimodo_vla/benchmarks/simple/`
（`adapter.py`、`policy.py`，SIMPLE 另有 `hands.py`）、`controllers/decoupled_wbc.py`；脚本：`serve_arena.py`、`serve_simple.py`。
通用规则见 [ARCHITECTURE](../ARCHITECTURE.md)。

## 1. 接口

| | HumanoidArena | SIMPLE |
|---|---|---|
| 运行方式 | 仿真（Isaac Lab）在另一个进程，HTTP 调我们：`/health`、`/reset {seed}`、`/infer` | 官方评测在同一进程（旧代码 `--eval`：SIMPLE 自己的 venv 里跑 MuJoCo 物理 + Isaac 渲染，直接调 runtime）；另有 HTTP `/infer`（40 维）、`/act`（36 维） |
| 每次请求给什么 | 任务名；`front` 图像（HWC，base64）；`state_history` [T, 64]：上次请求以来每个 50Hz 步的状态（根朝向 rot6d，相对回合首帧朝向 + 29 关节角 + 29 关节速度） | 同样的 64 维（由 `floating_base_pose` 四元数和 `body_q` 换算）；`head_stereo_left` 图像；二值手要手的测量值（Dex3 关节 q/dq/力矩），连续手要 `hand_closure_history` |
| 要我们回什么 | 一整段 40 维全身参考 [T, 40]（50Hz） | 40 维参考，或换成 36 维 decoupled-WBC 指令 [T, 36] |
| 底层控制器 | SONIC 或 TWIST2，在仿真那一侧，直接吃 40 维参考 | decoupled WBC（SONIC 系），在仿真那一侧 |
| 手 | 二值，2 维 | Dex3 七自由度手。二值：闭合姿态，加防抖状态机；连续：闭合度 × 闭合姿态，历史用测量到的闭合度 |
| 相机 | `front` 一路 | `head_stereo_left` 一路 |
| 正式评测的参数 | 50Hz；每次执行 15 个模型帧；RTC 关；每回合带种子（确定性噪声） | 50Hz；每次执行 50 个模型帧；RTC 关 |

## 2. 已定的事（2026-10-25）

- **只做「同样输入 → 同样输出」的对比测试，不跑 Isaac 仿真。**
- **每段第一帧位移的起点**：Arena + SONIC 用模型生成的历史末端；Arena + TWIST2 用第一帧预测（旧代码的
  `--anchor-root 1`）；SIMPLE 用外推 2·r0 − r1（它的状态里没有根位置）。两种修正都是必须的，做成规划器选项
  `root_boundary`，由各自的入口指定。
- **SIMPLE 的转向开关保持旧代码的 0**（不跟踪朝向），等价测试要求一致。
- **SIMPLE 的手部防抖、1.5 m/s 导航速度保护属于 SIMPLE 适配器**，不进规划器。
- **取帧方式**：入口默认用新规划器（按真实时间取帧）；`--timing original` 换成 `OriginalTimingPlanner`，
  和旧代码逐项一致，用来复现旧仓库的评测结果。

## 3. 和旧代码的对比

| | 测试 | 结果 |
|---|---|---|
| 推理流程 | `test_port_equivalence.py`：同一条 420 步轨迹，每次执行 50 帧、15 帧、15 帧 + RTC | `OriginalTimingPlanner` 动作最大差 6e-6；新规划器只比较送进模型的输入，差 1.9e-6 |
| Arena 服务 | `test_arena_port.py`：同一串请求喂给旧服务和新服务，SONIC、SONIC + RTC、TWIST2 | 40 维回复差 < 1e-5 |
| SIMPLE | `test_simple_port.py`：二值手、连续手两种权重 | 40 维和 36 维回复差 < 1e-5；手部防抖状态机用 400 次随机重规划逐帧对比，完全一致 |

这些测试需要 `KIMODO_ORIGINAL_REPO=<旧仓库路径>`，用一个记录输入、按输入算输出的假模型，不需要 checkpoint。

和旧代码不同的一处：SIMPLE 的速度保护拒绝一段动作时，旧代码会把内部状态回滚，新代码不回滚
（回合本来就在这里报错结束）。

## 4. 待办

- SIMPLE 的同进程评测：新代码的 `ChunkPolicy` 可以直接调用，但还没接进 SIMPLE 自己的评测循环
  （它的环境依赖和我们的不同，要实际装环境才能做）。
- 两者都还没在真实 Isaac 仿真里跑过。
