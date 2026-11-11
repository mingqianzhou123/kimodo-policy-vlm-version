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
- **SIMPLE 的转向开关保持旧代码的 0**（不跟踪朝向），等价测试要求一致。40 维 → 36 维由
  `controllers/decoupled_wbc.encode` 完成，逐行复现旧代码的 `arena_action_to_simple`。
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

## 4. HumanoidArena 训练（2026-11-05）

数据：Hugging Face `WilliamWang16/HumanoidArena_dataset_v3_1`（revision ca04c2d5），每个任务 × tracker
（`sonic`、`twist2`、`mimic_lite`）一份 LeRobot v3 数据，50Hz，每份 100 条左右。代码：`benchmarks/arena/demos.py`、
`scripts/prepare_arena.py`、`configs/arena.yaml`；文本 embedding 用 `cache_text_embeddings.py --arena-data <数据集>`。

```bash
python scripts/prepare_arena.py --data-dir <数据集> --out $ARENA_PREPARED --tasks HOI_double_desk --trackers sonic
python scripts/cache_text_embeddings.py --out $ARENA_TEXT_CACHE --arena-data <数据集>
torchrun --standalone --nproc_per_node=4 scripts/train.py --config configs/arena.yaml \
    --task HOI_double_desk/sonic --init-checkpoint <pt_419h checkpoint_1000000>
```

- **历史**：`observation.state` 的 64 维行，经过 Arena 服务处理请求时用的同一个函数（`adapter.state64_observations`）。
- **目标**：`action` 的 40 维全身参考（tracker 实际跟踪的东西），根位置按旧解码器的方式积分；第 `t` 帧的目标是第 `t` 行。
  手的历史用指令里的手（Arena 的状态里没有手），和旧仓库一致。
- **和旧仓库逐项一致**：`test_arena_demos.py` 把同一条轨迹交给旧仓库的 `HumanoidArenaAdapter.load_episode` 和新代码，
  历史、目标、手部差 < 1e-4（需要 `KIMODO_ORIGINAL_REPO`）。
- **图像**：存成 224×224（DINOv3 本来就把每张图缩到 224×224）。按它的顺序先转成 0–1 浮点再缩放、最后取整，
  和直接喂 480×640 原图相比每个像素最多差半个灰度级（平均 0.17）；录像本身的压缩误差远大于此。每条约 110 MB，
  一个任务 × tracker 约 11 GB；准备一条约 4 秒（单核）。
- **采样**：已发布的微调用 `episode_uniform`：先等概率选一条演示，再在其中等概率选切点（短演示的窗口被抽得更勤）。
  `main.sampling: episodes` 复现它；BiGym 仍是每个窗口每轮一次（`windows`）。
- **配置**：`arena.yaml` 照 ft_419h 的实际 `config.json`：从 pt_419h 初始化，20 万步，全局 batch 64（4 卡 × 16），
  warmup 2000，余弦降到 0.1 倍，seed 42，只用头部相机；结构、损失、学习率和 `bigym2.yaml` 相同。
  不加 `--init-checkpoint` 就是从头训（旧仓库的 `humanoidarena_single_*`）。
- **多任务**：`--task` 可以给几个文件夹，所有窗口合成一个数据集。下载里的 merged 数据集只是各任务的拷贝，
  不必另外准备；旧仓库的多任务训练不含 `HOI_grap_cup`（不在官方评测里），`prepare_arena.py --tasks evaluated` 同样跳过它。

## 5. 待办

- SIMPLE 的同进程评测：新代码的 `ChunkPolicy` 可以直接调用，但还没接进 SIMPLE 自己的评测循环
  （它的环境依赖和我们的不同，要实际装环境才能做）。
- Arena：已在 L40S 的 Isaac 仿真里跑通一局（2026-11-05，已发布的 ft_419h 权重）；SIMPLE 还没在真实仿真里跑过。
- Arena 训练：新代码训出的权重还没在 Arena 里评测过；完整复现 ft_419h（4 卡 20 万步）之前先跑短程检查。
