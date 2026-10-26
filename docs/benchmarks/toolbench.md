# HumanoidToolBench

代码：`kimodo_vla/benchmarks/toolbench/`（`adapter.py`、`demos.py`、`policy.py`）、`controllers/decoupled_wbc.py`；
脚本：`serve_toolbench.py`、`prepare_toolbench.py`、`check_toolbench_roundtrip.py`。通用规则见 [ARCHITECTURE](../ARCHITECTURE.md)。
来源：SNU-PI/HumanoidToolBench（3fec9bd）的 `docs/PUBLIC_EVALUATION.md`、`data/README.md`、
`src/humanoidtoolbench/actions/g1.py`，ToolBook 数据集（`snupilab/humanoidtoolbench-teleop`，revision 8b2cd31e）的元数据，
和官方 ACT 模型（`snupilab/humanoidtoolbench-act-sim-3003`）的 `action_contract.json`。

## 1. 接口

| | 内容 |
|---|---|
| 运行方式 | HTTP：它们的评测器连我们（`scripts/serve_toolbench.py`）。`POST /act`（带 NumPy 编码的 JSON），可选 `GET /info`、`POST /release`；一个服务只接一个评测器，另一个会话在 60 秒内请求会得到 409。原因：它锁定 transformers 4.57.1，我们要 5.1.0，装不进同一个环境 |
| 每次请求给什么 | 头部相机 `rgb_head_stereo_left` 360×640；`state` [1, 32]：两手各 7、两臂各 7、腰 roll/pitch/yaw 测量值、上一次发出的骨盆高度指令；指令文本（6 句固定）；回合第一次请求带 `history.reset`。**没有骨盆位置和朝向，没有腿** |
| 要我们回什么 | 一整段 36 维指令 [T, 36]，和 SIMPLE 的 36 维布局相同：两只 Dex3 手 14 + 手臂 14 + 躯干 roll/pitch/yaw + 骨盆高度 + vx、vy + 转向开关 + 目标朝向。评测器把整段执行完才再请求 |
| 底层控制器 | NVIDIA decoupled WBC（GR00T Balance/Walk）：vx、vy 在机器人自身坐标系下；转向开关绝对值 ≥ 0.1 时，以「误差 / 0.5 秒」、最快 1 rad/s 转向目标朝向（绝对角）；vx、vy、开关的大小 ≥ 0.1 就用行走策略，全为 0 时站立 |
| 手 | Dex3 三指手，关节目标连续；我们发「闭合度 × 闭合姿态」 |
| 相机 | 头部一路（手腕相机只在数据里有，评测时不给） |
| 控制频率 | 50Hz；每次执行 50 个模型帧（83 个控制步） |

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

- 历史：每 83 步（50 个模型帧）一个实测状态，中间用指令补齐，朝向一律用指令朝向——和评测时适配器报告的
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

## 6. 待办

- 正式评测要渲染头部相机，A800 上还缺 `libOpenGL.so.0`；回放不渲染图像，所以没受影响。
- `root_boundary`（每段第一帧位移从哪里算起）：回放用的是完美预测，比不出差别，要等有训练好的 ToolBench 模型再比。
  现在按 SIMPLE 用外推。
- 手的闭合姿态：录制里右手有两种闭合姿态，各占约一半；模型只输出闭合度，评测时只能选一种（现在用 ToolBench 自己那种）。
- 训练：`train.py` 还没有 ToolBench 的配置，也还没给 6 句指令算文本 embedding。
