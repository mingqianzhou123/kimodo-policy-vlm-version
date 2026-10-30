# HumanoidToolBench

代码：`kimodo_vla/benchmarks/toolbench/`（`adapter.py`、`demos.py`、`policy.py`）、`controllers/decoupled_wbc.py`；
脚本：`serve_toolbench.py`、`prepare_toolbench.py`、`check_toolbench_roundtrip.py`。通用规则见 [ARCHITECTURE](../ARCHITECTURE.md)。
历史接口来源：SNU-PI/HumanoidToolBench（3fec9bd）的 `docs/PUBLIC_EVALUATION.md`、`data/README.md`、
`src/humanoidtoolbench/actions/g1.py`，ToolBook 数据集（`snupilab/humanoidtoolbench-teleop`，revision 8b2cd31e）的元数据，
和官方 ACT 模型（`snupilab/humanoidtoolbench-act-sim-3003`）的 `action_contract.json`。
本机实际训练/评估使用官方提交 `2cd4f006f6f2ebc536a3fb45aff8ce2eaf1382d7`，完整结果见[最终报告](../../reports/toolbench_scratch40k_20261027.md)。

## 1. 接口

| | 内容 |
|---|---|
| 运行方式 | HTTP：它们的评测器连我们（`scripts/serve_toolbench.py`）。`POST /act`（带 NumPy 编码的 JSON），可选 `GET /info`、`POST /release`；一个服务只接一个评测器，另一个会话在 60 秒内请求会得到 409。原因：它锁定 transformers 4.57.1，我们要 5.1.0，装不进同一个环境 |
| 每次请求给什么 | 头部相机 `rgb_head_stereo_left` 360×640；`state` [1, 32]：两手各 7、两臂各 7、腰 roll/pitch/yaw 测量值、上一次发出的骨盆高度指令；指令文本（6 句固定）；回合第一次请求带 `history.reset`。**没有骨盆位置和朝向，没有腿** |
| 要我们回什么 | 一整段 36 维指令 [T, 36]，和 SIMPLE 的 36 维布局相同：两只 Dex3 手 14 + 手臂 14 + 躯干 roll/pitch/yaw + 骨盆高度 + vx、vy + 转向开关 + 目标朝向。评测器把整段执行完才再请求 |
| 底层控制器 | NVIDIA decoupled WBC（GR00T Balance/Walk）：vx、vy 在机器人自身坐标系下；转向开关绝对值 ≥ 0.1 时，以「误差 / 0.5 秒」、最快 1 rad/s 转向目标朝向（绝对角）；vx、vy、开关的大小 ≥ 0.1 就用行走策略，全为 0 时站立 |
| 手 | Dex3 三指手，关节目标连续；我们发「闭合度 × 闭合姿态」 |
| 相机 | 头部一路（手腕相机只在数据里有，评测时不给） |
| 控制频率 | 50Hz；每次预测 50 个模型帧；按预测末帧限制实际执行 81/82 个控制步 |

## 2. 已定的事

- **没有骨盆朝向，用我们发出的朝向推算**（2026-10-25 定）：请求里确实没有骨盆位姿（评测文档的请求表和
  `policies/remote_humanoid.py` 只发 32 维状态）。适配器拿上一次发出的目标朝向当作当前朝向。
- **每段只给一次状态，中间用发出的目标补齐**：评测器在两次请求之间执行完整段动作，适配器把中间这些步用我们
  发出的目标补上，这样历史仍然每个控制步一条。训练数据按同样方式构造（第 3 节）。
- **转向开关照操作员的用法**（`turn="operator"`，默认）：操作员只在转身时打开开关（ToolBook 里 0–5% 的帧），
  值就是转身角速度，其余时间为 0，朝向不被跟踪、会自然漂移约 ±0.3 rad。所以规划出的朝向在转、且角速度
  ≥ 0.1 rad/s 时才开。一直开等于朝向闭环，BiGym 上证明有害，回放里也偏得多 3 倍以上（第 4 节）。
  开关取「下一步」的朝向变化：操作员第 t 行的开关值等于目标朝向从 t 到 t+1 的变化率。
- **目标朝向是世界系的绝对角**：三个场景都让机器人以 -90° 出生（`SPAWN_YAW`），适配器加上这个固定偏移。
- **手的闭合姿态用 ToolBench 自己的**（`adapter.CLOSED_HANDS`）：操作员的手指令基本是开、合两种，中间有很短的过渡。
- **图像**：训练数据和评测时都先缩到 224×224（同一个函数 `adapter.resize_image`），所以 DINOv3 看到的像素相同。
- **训练目标是操作员指令**（和 BiGym 一样）：手臂、躯干、手来自 `action`，骨盆高度来自 `teleop.base_height_command`，
  行走来自 `teleop.navigate_command`（vx、vy、转向开关、目标朝向）。
  录制的第 t 行是第 t+1 步的目标。官方 ACT 模型的数据说明也是这么转换的。

## 3. 训练数据（`prepare_toolbench.py`）

读 ToolBook 的 `raw/` 录制（LeRobot 格式，50fps），生成和 BiGym 同格式的训练文件。默认只用成功的演示
（`episodes.jsonl` 里的 `episode_success`），`--all` 连失败的一起用。改过名的旧场景名（`G1StickMove` → `G1BallMove`、
`G1HookRetrieve` → `G1BallRetrieve`）按评测器的名字归类，指令用评测器那 6 句（L0 和 L1/L2 不同）。

- 历史：按 `KimodoPlanner.control_offsets` 的实际分块时间网格，每 81/82 步一个实测状态，中间用指令补齐，朝向一律用指令朝向——和评测时适配器报告的
  逐步相同（`tests/test_toolbench_demos.py`）。
- 手：闭合度按连续值存（`hand_mode="continuous"`）。
- 图像：头部相机缩到 224×224。一条 21 秒的演示约 97MB，全部 ToolBook 估计 350–400GB。

把演示的指令经适配器换回 36 维指令（不跑仿真）：速度、朝向、手臂与记录的差 < 3e-6；转向开关在 13,778 个
控制步上和操作员完全一致（改成取「下一步」变化之前，14,258 行里错 21 行）。

## 4. 仿真回放检查（`check_toolbench_roundtrip.py`，2026-10-26）

在 A800 上 ToolBench 自己的环境里跑（第 5 节）。

**测什么**：ToolBench 公开版的工具是程序生成的、台面布局也和录制用的内部工具不同，录制时的场景复原不了，
所以不看任务成功率，只看机器人动作对不对。机器人放在录制时的出生位置，用评测器自己的 agent、稳定流程和
decoupled WBC 执行整条演示，每条演示跑五种方式：照录制的指令（参照）、我们的 50Hz 换算、经过 Kimodo 30Hz
（评测时的路径，含规划器时间修正）、转向开关一直开、录制指令加 3e-6 m/s 噪声（这个检查的下限）。

**结果**（和「照录制指令」那次比的终点位置差、朝向最大差；只算所有方式都跑成的演示。一部分运行因登录节点上
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
2. 转向开关照操作员用法是对的；一直开偏得多 3 倍以上。
3. 开关改成取「下一步」变化后，位置差从 0.072 降到 0.063 m、朝向差从 0.069 降到 0.051 rad（76 条的均值）。
4. 走路控制器只走出指令速度的约 1/3（原始录制里也是这样），所以实际路线比「指令积分出的路线」短。
   训练目标按指令、评测按同一方式换回去，和 BiGym 一样不受影响。
5. Kimodo 30Hz 表示的额外损失：位置中位数约 2.5 cm、朝向约 0.02 rad。

## 5. 环境（A800，没有 GitHub）

在 Mac 上 clone HumanoidToolBench（3fec9bd）和它锁定版本的 gear_sonic、decoupled_wbc，拷到 A800 的
`third_party/HumanoidToolBench-3fec9bd`；用系统 Python 3.10 建 `.venv`，按 `uv.lock` 导出的 `requirements-a800.txt`
从阿里云 PyPI 镜像和 download.pytorch.org 安装，另加 pyarrow（读录制用）；场景网格经 hf-mirror.com 下载
（`fetch_assets_mirror.py`，需要 `SSL_CERT_FILE` 指向 certifi）。ToolBook 的文件列表要在能连 huggingface.co 的机器上
生成（经 hf-mirror 列目录时分页链接会跳回 huggingface.co），再按文件名经 hf-mirror 下载。
运行时设 `LD_LIBRARY_PATH=<Humanoid_Vla>/third_party/glvnd/lib MUJOCO_GL=egl PYTHONPATH=<本仓库>`。

## 6. 2026-10-27：本机 scratch 联合训练适配

完整审计和可复核证据见 [前置验证报告](../../reports/toolbench_preflight_20261027.md)。

- 固定仿真数据清单：`/root/workspace/ToolBook/manifest.json`，3,003 条成功演示，18 条件，2,322,445 个 50Hz 帧；不包含 hardware 数据。所有原始 Parquet 和头部视频逐文件校验 SHA256。
- 转换后：1,394,672 个 30Hz 帧、1,247,525 个完整未来窗口。全部图像逐帧解码，特征、闭合度和文件形状逐条检查。
- 修正原先固定 83 步的观测补齐时刻：模型受最后预测帧限制，实际是 81/82 步。新增回归测试逐项对比实际 planner 时间网格。
- 连续手部历史改为适配器产生的实测/补齐闭合度，与推理一致；未来目标仍是操作员命令。二值手的历史约定保持原实现。
- 六句 LLM2Vec embedding 复用已验证的冻结编码缓存，记录来源和 SHA256；不使用任务策略预训练权重。
- 本机官方资产校验通过，54 次仿真回放无错误、无摔倒。46 个项目测试通过，4 个依赖额外安装或旧仓库路径的测试跳过。
- 双卡 20 步、四卡完整数据 100 步及八卡 20 步训练通过；18 条件各 1 回合、200 步的 HTTP 诊断完成。诊断分数不代表正式成功率。

### 训练与实时监督

```bash
# 本机已完成的 run；以下只查看结果
cat log/toolbench/joint18_scratch40k_20261027_8gpu_supervision/status.json
tail -f log/toolbench/joint18_scratch40k_20261027_8gpu/train.log

# 在新目录复现实验：先运行 prepare/audit/roundtrip 并通过 supervisor 的全部 gate
PYTHONPATH=. python scripts/supervise_toolbench.py \
  --run-dir log/toolbench/<new-run> --gpus 0,1,2,3,4,5,6,7 \
  --diagnostic-gpus 1,2,3 --eval-gpus 0,1,2,3,4,5,6,7
```

配置是 `configs/toolbench.yaml`，复用通用 `scripts/train.py`，不引入另一个 benchmark 专用模型或训练实现。
18 条件混合，按全部 eligible windows 均匀采样。参考官方 ACT 发布记录的 40k optimizer steps、global batch 128、
seed 1000、BF16、1k warmup、cosine、AdamW betas (0.95, 0.999) 和 weight decay 1e-6。
Kimodo 自身保留 100 帧历史、50 帧预测、ControlNet 四层、连续手头，以及 backbone/adapter 的分组学习率。
这不是 ACT 的 100×50Hz 动作表示，不能称为逐项复现 ACT。

Scratch 的明确含义：没有 policy checkpoint 初始化，特别是不加载 419h 权重；冻结 Kimodo-G1 运动先验、
DINOv3、LLM2Vec；任务 ControlNet 从运动先验复制层并初始化新适配层，手头随机初始化，与该项目完整策略的 scratch 设计一致。
`initialization.json` 记录来源及全部可训练参数；`require_scratch` 拒绝策略 checkpoint 初始化。

本次八卡 DDP 在 0–7 卡训练，micro batch 16、梯度累积 1，global batch 128。已有其他作业时共存，
不终止其他实验；1–3 卡用于在线诊断。每 10 步记录 loss、梯度、LR、吞吐、显存；每分钟记录所有 GPU 使用率；
每 2k 保存 checkpoint 和优化器状态；10k/20k/30k 做 18 条件、单回合、完整长度的诊断。
出现非有限数值、进程失败或 20 分钟无进度会停止流水线并记录原因。
`*_supervision/progress.html` 是每 30 秒更新的本地进度页。

### 正式评测

训练结束后自动加载 `checkpoint_40000`，调用未修改的官方 evaluator：18 条件、各 100 回合、
test split、seeds 10000–10099、每回合最多 3000 个 50Hz 控制步。
最多 18 个独立服务并行，分布在 8 张 GPU 上、每卡最多 3 个服务；每个服务只服务一个 evaluator；保留完整图像、日志、官方校验 receipt 和 SHA256。
最终汇总由官方 `write_summary` 生成，要求所有结果可报告且模型/配置/源码/资产指纹一致。
文件位于 `*_supervision/evaluation_18x100/summary.{json,md}`；另生成含 Wilson 区间的 `RESULTS.md`。
训练 loss 下降和适配验证通过本身不能证明任务有效性，结论须依据正式的 18 条件成功率。

## 7. 表示限制与待研究项

- 历史 A800 环境缺 `libOpenGL.so.0`；本次 5090 主机的 NVIDIA EGL 和正式四路视频核验已通过。
- `root_boundary`（每段第一帧位移从哪里算起）：回放用的是完美预测，比不出差别，要等有训练好的 ToolBench 模型再比。
  现在按 SIMPLE 用外推。
- 手的闭合姿态：录制里右手有两种闭合姿态，各占约一半；模型只输出闭合度，评测时只能选一种（现在用 ToolBench 自己那种）。
- 绝对骨盆朝向没有实测输入，目前仍由历史命令推算；这是公开 32 维状态的观测限制。

### 本次训练过程诊断

`checkpoint_4000` 已执行 18 条件、各 1 回合、完整 3000 步诊断，全部官方校验通过，单种子成功数为 0/18。
这不是正式 100 回合结果，也尚不能支持工具使用有效性。视频显示策略已有动作输出，但未完成抓取。
该轮同时验证了每卡 2–3 个独立策略/仿真服务的并发编排，18 条件约两分钟完成；正式评测继续使用同一隔离方式。
证据：运行监督目录下 `diagnostic_4000/summary.json` 和对应四路相机视频。

### 本次 40k 最终结果（已完成）

10k/20k/30k 的单种子诊断均为 0/18。随后完成 40k 训练及官方 18×100 回合评估，全部回执
`validated`、`reportable: true`；7,200 段视频、21,071,564 帧通过官方逐帧检查。
最终宏平均成功率 **3.94%（71/1800）**：L0 65/600（10.83%），L1 6/600（1.00%），L2 0/600。

| 场景 | L0-S | L0-R | L1-S | L1-R | L2-S | L2-R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BallMove | 11% | 8% | 3% | 2% | 0% | 0% |
| BallRetrieve | 4% | 7% | 0% | 0% | 0% | 0% |
| IceBreak | 15% | 20% | 1% | 0% | 0% | 0% |

实际 L1 成功视频包含推球和持锤破冰，但总成功率仍低，不能据此宣称完整架构已在此基准上稳健有效。
单训练种子、无模块消融或同协议基线，不支持架构优势的因果结论。
完整置信区间、失败诊断、源码/权重哈希、复现信息见[最终报告](../../reports/toolbench_scratch40k_20261027.md)。

独立重评最终 checkpoint（会使用官方已验证结果缓存；需要重新仿真可换新输出目录）：

```bash
PYTHONPATH=. /root/workspace/miniforge3/envs/kimodo/bin/python scripts/eval_toolbench.py \
  --checkpoint log/toolbench/joint18_scratch40k_20261027_8gpu/checkpoint_40000 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output log/toolbench/joint18_scratch40k_20261027_8gpu_supervision/evaluation_18x100
```

运行中可另开 `scripts/monitor_toolbench_eval.py --supervision <supervision-dir>`，
保存逐任务回合数、临时成功数和 GPU 实况；最终评分仍以官方 summary 为准。

本轮固定容量/损失的系统重训方案见 [调参实验登记](../../reports/toolbench_tuning_plan_20261027.md)。
编排入口为 `scripts/tune_toolbench.py --root log/toolbench/tuning_20261027`，
先核验已保存的前置检查，再执行开发集筛选、scratch 重训、预算比较和锁定后的独立测试。
运行状态、视频和结果位置见该登记；未完成阶段不会提前生成最终结果。
