# BiGym 2.0

代码：`kimodo_vla/benchmarks/bigym2/`（`adapter.py`、`demos.py`、`policy.py`）；脚本：`prepare_bigym2.py`、
`check_bigym2_roundtrip.py`、`eval_bigym2.py`、`train.py`、`submit_bigym2.py`、`eval_run.py`、`summarize_bigym2.py`、
`gpu_queue.py`、`cache_text_embeddings.py`、`check_multicamera.py`；配置：`configs/bigym2.yaml`、`configs/bigym2_recipe.yaml`。
通用规则见 [ARCHITECTURE](../ARCHITECTURE.md)。

## 1. 接口

BiGym 是纯 Python + MuJoCo，模型和环境在同一个进程里跑：官方 `bigym.loco.eval.evaluate(policy)` 每步调一次
`policy(timestep)`（`BenchmarkAdapter` + `AgentPolicy`，入口 `bigym2/policy.make_kimodo_policy`），不需要 HTTP 服务。
BiGym 锁定 `mujoco<3.9`；`kimodo_vla` 本身不依赖 mujoco，所以评测环境里只装 BiGym 要的版本即可。

| 方向 | 内容 |
|---|---|
| 观测 | `low_dim_obs` 按官方布局取出：骨盆 x/y/z/yaw、14 个手臂关节、（开启躯干俯仰的 14 个任务）3 个腰关节、2 个夹爪。腿部关节和（其余任务的）腰关节 `joint_mask=False`、值填 0。实测 WBC 接管腰部时，腰俯仰也只在 0.06 rad 以内。骨盆横滚和俯仰角在官方观测里拿不到，按 0 处理。`rgb_obs` 拆成 head / right_wrist / left_wrist 三张图 |
| 动作 | 底盘指令由 `controllers.groot_wbc` 算：`vx, vy` = 第 0 帧位移 × 50Hz；`wz` = 上一步目标朝向到第 0 帧目标朝向的变化率（和位移一样用向后差分，纯前馈，不做朝向闭环）；`height` 直接给。适配器负责其余部分：`torso pitch` 取目标腰俯仰关节；14 个手臂关节直接给；手闭合度 ≥ 0.5 → 夹爪合。最后按 `env.wholebody_action_layout()` 归一化到 [-1, 1] |
| 语言 | 用 BiGym 自带的 `task_instructions.yaml`，和演示数据里的指令一致 |

```bash
MUJOCO_GL=egl python scripts/eval_bigym2.py --task move_plate \
    --checkpoint <ckpt 目录> --text-cache <文本 embedding 目录> --out outputs/move_plate.json
```

`test_bigym2_adapter.py` 在真实 BiGym 环境里检查：观测和仿真器关节值一致；归一化与 BiGym 互逆；
保持初始姿态不摔、不漂；给 0.3 m/s 后退位移，3 秒退 0.73 米（实测稳定速度约 0.28 m/s）；给左转目标会左转。
（测的是后退：有些场景里正前方是台面，往前走会被挡住。）

## 2. 已定的事

- **输入图像：三路相机（头部 + 两个手腕）是主结果，只用头部的一路是消融**（2026-10-25 定）。BiGym 作者确认
  π0.5 三路相机都用了，ACT、DP 等也是三路。图像用官方 84×84，由 DINOv3 预处理在模型内部放大到 224。
  BiGym 作者澄清：所有方法拿到的都是同样的 84×84 原始数据，π0.5 的 224 也只是模型内部放大，不是重新渲染。
  评测用官方相机设置，所以官方 runner 会正常出排行榜记录。
- **训练方式**：只用 BiGym 演示从零训练 ControlNet 和手部头，不用我们的 VLA 预训练权重；Kimodo-G1、DINOv3、
  LLM2Vec 保持冻结。ControlNet 的注入层初始为零，所以训练从原版 Kimodo-G1 起步。用 pt_419h 初始化的一组作为对照。
- **多路相机的实现**：`KimodoPolicyConfig.num_cameras` 设成几，模型就接几路图像。每路单独过同一个冻结的 DINOv3，
  各得 196 个 token，按相机顺序拼起来（三路 588 个）送进 ControlNet 和手部头。ControlNet 里把 token 压到 50 个的
  那层随路数变宽，所以三路模型不能加载一路的 ControlNet 权重（pt_419h 对照只能用一路）。checkpoint 的
  `config.json` 里 `model.cameras`（如 `["head", "right_wrist", "left_wrist"]`）决定 planner 取哪几路、按什么顺序；
  没写就是只用 `head`。一路时的计算和以前完全一样。`tests/test_multicamera.py` 只测了图像编码和 planner；
  需要真实权重的完整检查是 `scripts/check_multicamera.py`，已在 A800 上用真实权重和 move_plate 数据跑通：
  训练、三路相机各自影响预测、官方环境里闭环控制都正常。
- **文本 embedding**：`scripts/cache_text_embeddings.py` 给 `task_instructions.yaml` 里 20 个任务的指令各算一份
  LLM2Vec embedding，存成 `<任务名>.pt`（格式同旧仓库，`planner.load_text_embeddings` 直接读）。A800 上已生成在
  `data/cache/bigym2`。新环境（transformers 5.1、peft 0.18）算出的结果和旧仓库缓存的 embedding 完全一致
  （同一句指令，最大差 0）；加载时打印的 LoRA「MISSING」是第一遍加载的提示，可以忽略。这 20 句指令和论文附录里
  π0.5 用的逐字相同。

## 3. 动作来回转换检查（`check_bigym2_roundtrip.py`）

做法：把演示里的动作当成「完美的模型预测」，走和正式评测完全相同的流程
（Agent → 30Hz Kimodo 动作块 → 按控制步取帧 → 40 维编码 → 适配器 → BiGym），看还剩多少成功率。
前 5 个任务 × 60 条演示，2026-10-24 在 A800 上跑；主表其余 6 个任务 2026-10-25 补跑（没测「实际执行」那一列）。

「修正前」一列是 2026-10-25 之前的规划器：它把预测帧从上一段末尾开始摊到控制步上，每段都比实际时间
慢约一个模型帧（33 毫秒），见第 5 节末尾。「修正后」是现在的规划器（按每个控制步的真实时间取帧），
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

## 4. 训练数据（`prepare_bigym2.py`）

每条演示包成 `BiGym2Demo`，交给 `data/build.training_episode`。历史由官方观测经 `to_observation` 和
`KimodoFeatures.history` 得到，和评测走同一段代码；目标是操作员指令（第 3 节）；`hand` 是夹爪指令；
图像是每个 30Hz 帧对应的三路相机（头部、右腕、左腕），直接取数据集里记录的官方 84×84 图像。

**训练和评测一致性**（2026-10-25，A800 上用 move_plate、pick_box 的真实演示验证）：
- 训练窗口的历史条件与评测时 planner 构造的完全相同：分别对比 22 个和 74 个切点，差值为 0
  （`tests/test_episodes.py` 用合成数据做同样的检查）。
- 训练文件里的三路图像与官方环境评测时实时给出的观测逐像素相同（差为 0）。
- 2026-10-26 抽出共用的训练数据代码后，在 A800 上重新生成 move_plate，120 个文件和原来逐字节相同。

**规模**：5 个任务 300 条演示，三路 84×84 图像共 8GB，可切出 112,683 个训练窗口（pick_box 演示最长，独占 57,643 个）。
20 个任务按这 5 个任务的平均估算，约 32GB。窗口数和演示长度成正比，多任务训练时要考虑按任务均衡采样。

## 5. 训练

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
每个 checkpoint 评两种设置，在表里并列：默认执行完整个 50 帧（1.67 秒）再重新预测，结果在 `eval/`；每 10 帧
（0.33 秒，接近 π0.5 的 0.32 秒）重新预测一次，结果在 `eval_exec10/`（`eval_run.py --execution-frames 10`）。
重跑提交脚本只补缺的部分：还在队列里的同名作业直接沿用，训完的不重训，没训完的接着训，评过的 checkpoint 跳过。

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

**评测慢的原因（2026-10-26 查明并修复）**：MuJoCo 的离屏渲染（EGL）不认 Slurm 分配的 GPU，同一节点上所有评测进程
都在 0 号卡上画相机图像。实测 pick_box 带三路相机只有 9 步/秒（不带相机 269 步/秒，模型每次预测约 0.5 秒，
默认设置下每约 82 步才预测一次，占比很小）。`eval_bigym2.py` 现在把 `MUJOCO_EGL_DEVICE_ID` 设成分配到的 GPU，
同样条件 97 步/秒。换卡渲染的图像与原来相差不超过 1 个亮度级、只涉及约 0.007% 的像素，和同型号卡在不同节点上
重复渲染的差异（0.0024%）同一量级，已有结果仍可比。另外两套结果的 Slurm 作业名现在带输出目录名，
避免同名任务互相当成依赖。

## 6. 第一套结果（和 π0.5 同口径，2026-10-26 出齐默认设置）

最后 5 个 checkpoint（26k–30k）各 100 回合合并的成功率（%），默认每 1.67 秒重新规划；论文数字取自 Table II。
所有结果都有官方排行榜记录，协议违规项为 0。给合作者的说明：https://claude.ai/artifact/U9xGXFHTGDgZf3fqc87uFn

| 任务 | 三路（主结果） | 只用头部 | 头部 + pt_419h | π0.5 | ACT | DP | CQN-AS | DEAS |
|---|---|---|---|---|---|---|---|---|
| pick_box | 10 | 5 | 15 | 19 | 61 | 10 | 48 | 14 |
| reach_target_multi_modal | 54 | 60 | 62 | 90 | 81 | 84 | 19 | 79 |
| reach_target_dual | 22 | 20 | 16 | 58 | 30 | 27 | 3 | 34 |
| drawer_top_close | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| drawer_top_open | 87 | 89 | 86 | 99 | 92 | 98 | 89 | 79 |
| move_plate | 51 | 35 | 29 | 65 | 60 | 62 | 48 | 27 |
| move_two_plates | 6 | 1 | 9 | 68 | 39 | 18 | 21 | 2 |
| dishwasher_close | 1 | 0 | 0 | 79 | 93 | 51 | 78 | 27 |
| dishwasher_load_cups | 17 | 10 | 25 | 92 | 56 | 51 | 47 | 6 |
| **9 个任务平均** | **39** | **36** | **38** | **75** | **68** | **56** | **50** | **41** |

- 主结果低于 π0.5、ACT、两个编程智能体（61、65）、DP、CQN-AS 和 DEAS，只高于 DrQ-v2+（11）。
- dishwasher_close 只有 0–1%，而动作接口用完美预测能到 87%：问题在模型学到的动作。
- 学习曲线（三路）：两个够取任务和关抽屉在 5k–15k 步就平了；开抽屉、端一个盘子、装杯子和 pick_box
  到 25k–30k 步还在涨或波动，更长的训练可能有帮助（第二套会回答）。
- 只用头部的 pick_box 有 0.2% 的回合摔倒（0.33 秒时 0.4%），其余各组为 0。
- 每 0.33 秒重新规划的列见 `log/bigym2/main9/results.md`，pick_box 和 dishwasher_close 的部分还在评测。

## 7. 失败原因诊断（2026-10-26）

三路相机、第一套第 30k 步。脚本在 A800 的 `outputs/diag/`（`diag_vision.py`、`diag_trainscenes.py`），录像在 Mac 的 `outputs/videos/`。

**图像：模型确实在用，而且要实时画面。** 官方种子 100 回合，把图像换成全黑、整回合只给第一帧、或别的场景的第一帧：

| 任务 | 正常 | 全黑 | 只给第一帧 | 别的场景 |
|---|---|---|---|---|
| reach_target_multi_modal | 52 | 0 | 4 | 3 |
| reach_target_dual | 25 | 0 | 0 | 0 |
| drawer_top_close | 100 | 0 | 0 | 0 |
| drawer_top_open | 90 | 0 | 0 | 0 |
| move_plate | 39 | 0 | 0 | 0 |
| dishwasher_load_cups | 24 | 0 | 0 | 0 |

**不是死记演示。** 从 60 条演示各自的初始状态闭环执行，成功率和官方种子上差不多：pick_box 10 / 18、multi_modal 55 / 52、dual 35 / 25、
关抽屉 100 / 100、开抽屉 88 / 90、move_plate 55 / 39、两个盘子 8 / 9、关洗碗机 0 / 0、装杯子 22 / 24。

**录像里的失败方式**：手停在目标旁几厘米（dual 进不了 5 cm 判定、盘子放不进槽）；在两个子步骤之间卡住不动
（pick_box 放下箱子后不松手、关洗碗机来回扫不推托盘）；转身过头或底盘漂走（装杯子）。另外模型看不到实测的夹爪状态：
手部条件是它自己下达过的开合指令，适配器也丢掉了手指关节，所以空抓后不会察觉、也不会重抓（演示里也没有重抓的例子）。

**第二套后期崩溃不是训练/测试场景差距**：move_plate 第 60k 步在训练场景上 44/60（73%），第 100k 步只有 7/60（12%），
和官方种子上的 68% → 10% 一致。后期模型连训练场景都做不好，更像对动作拟合过细、执行中出一点偏差就纠正不了。
对策先试权重滑动平均（`configs/bigym2_recipe_ema.yaml`，衰减 0.999 和 0.9999），在 700000 起的验证种子上挑权重，
再用官方种子评 80k–100k。

**EMA 没有解决问题，真正的原因是训练不稳定（2026-10-27）。** 三个崩得最厉害的任务用同样设置重训、同时存权重滑动平均（0.999、0.9999），
在 700000 起的验证种子上各 50 回合（80k / 90k / 100k）：开抽屉 原始 92/86/50、0.999 90/84/42、0.9999 84/84/0；
move_plate 和 reach_target_dual 三种权重都几乎是 0。原因是这些训练本身发散了：move_plate 新跑的这次运动损失从 5k 步起飙到 3–40
（正常约 0.001）；reach_target_dual 两次训练都在 63k–67k 步发散，正好对应它从 65k 起成功率归零；原第二套的 dishwasher_load_cups
从 15k 步起发散，所以每个 checkpoint 都是 0。没发散的 move_plate、开抽屉，90k 之后的运动损失也比 40–60k 时高 7–15 倍，对应后期掉分。
第一套（单卡、每步 48、30k 步、无增强）只有偶发的单点尖峰、都能恢复。排除了评测代码的问题：同一个旧 checkpoint 用新代码在验证种子上 64%、
在官方前 20 个种子上 14/20（旧代码 13/20，17 个回合结果相同）；新训练的 80k checkpoint 用旧代码评也是 0%。

## 8. 待办

- 找出训练发散的原因再修（学习率、精度、状态遮挡、梯度裁剪），而不是靠权重平均。

- 第二套（和 ACT/DP 同口径）结果出来后补进这里和给合作者的说明。
- 检验模型是否在用图像（遮掉或替换图像，看成功率变化）。
