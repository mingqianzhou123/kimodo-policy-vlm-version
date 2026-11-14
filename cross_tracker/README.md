# Cross Tracker

人形跨 tracker 混训的独立工作目录。数据准备、朴素 baseline、CCT 对照和后续方法均在这里管理。

**当前轮次**：箱子搬放单任务，SONIC / TWIST2 / SONIC+TWIST2，各seed42，SenseCore 三个独立作业，各8×A800，可并行执行；见 [本轮协议](docs/box-screen.md)。下文30k/60k与12次多任务矩阵是已被替代的历史方案，不继续提交。

服务器位置：`humanoidVLA/cross_tracker/`（百度为 `/root/workspace/humanoidVLA/cross_tracker/`）。本地位于同一代码库的 `kimodo-vla version/cross_tracker/`；服务器同步状态见 [当前进度](docs/status.md)。现有 BiGym 调参、Kimodo 复现和其他实验不迁入本目录。

```text
cross_tracker/
  configs/             实验协议；本机路径放 paths.local.json（不入版本控制）
  scripts/             少量数据准备、运行登记入口
  data/manifests/       冻结的数据清单，纳入版本控制
  data/cache/           下载、转换后的大文件，不入版本控制
  runs/<run_id>/        单次运行的配置、代码快照、日志、权重、评测
  docs/                代码梳理、状态和必要的实验结论
  tests/               数据隔离和追溯规则的检查
```

当前已实现名单冻结、输入打包、固定版本资源下载、修复数据转 Kimodo 特征、独立检查和运行登记。共用训练器保持原样，任务/来源均衡采样放在 `src/sampling.py`；已完成诊断训练，2026-11-14 已启动首个正式 baseline，当前运行见 [进度](docs/status.md)。

## 数据清单

依赖：Python 3.11+、NumPy、PyArrow。将 `configs/paths.example.json` 复制为 `configs/paths.local.json`，填写已有审计和修复产物的位置，然后运行：

```sh
python scripts/freeze_data.py --paths configs/paths.local.json
```

按任务、来源固定抽取 54 / 100 / 94 条，按整条示教划分 80% 训练和 20% 验证。实际每个来源的训练数是 43 / 80 / 75，验证数是 11 / 20 / 19；这些总数不是全部投入训练。名单不随训练 seed 改变，17 条隔离轨迹不放回。清单是数值候选集，不代表示教成功已获认证，也不代表视频和训练加载器已就绪。

原始数据共用已有只读缓存；只保存来源路径、固定 revision、内容哈希和原始行号。修正数据使用 `candidate.action` 和 `numeric_eligible`，不能退回原始 `action`。未来归一化只拟合对应来源组合的训练集。

## 运行追溯

`scripts/register_run.py` 只登记、不执行命令。指定目的、阶段、训练 seed、来源、依赖仓库和命令；它将配置、清单与本项目代码复制到新的 `runs/<run_id>/`，归档依赖仓库的已提交代码和 tracked 修改。重名报错，避免覆盖。依赖仓库的 untracked 文件只列出、不复制；若实际运行依赖其中的文件，必须先纳入版本管理再登记，不能把这份登记当作完整运行证据。运行环境、实际命令及恢复过程在真正执行时补记。

运行目录中统一使用 `train.log`、`metrics.jsonl`、`checkpoints/`、`eval/`。改变数据、方法或超参数就新建 run；续训沿用原目录并记录恢复 checkpoint。诊断、短程验证、正式 baseline 分别登记为 `diagnostic`、`pilot`、`baseline`，不混为正式结果。

第一轮只做 3 个单源和 3 个双源组合，每组 2 个 seed。双源训练步数为单源的两倍。用户已确定沿用 Kimodo-VLA，从官方 pt_419h checkpoint_1000000 初始化；资源固定在 `weights-v1.json`。`kimodo_pilot.yaml` 仅用于短程检查，正式预算现已冻结为单源 30k / 双源 60k，见 [baseline 协议](docs/baseline-protocol.md)。

## Kimodo 数据接入

`scripts/package_data.py` 将 9 份数值源、修复层及元数据打包，共 45 个文件；`assets-v1.json` 列出被选中轨迹实际用到的 24 个视频分片。`scripts/fetch_assets.py --kind videos|weights` 校验完整内容后才接纳下载文件。已有资源只有哈希一致才复用。

`scripts/prepare_kimodo.py` 只读冻结名单；每段先校验原始行号、可用帧掩码、修复标签和视频偏移，再调用公共 `training_episode`。输出在 `data/cache/prepared-v1/{train,validation}/{task}/{tracker}/`。每条有 `.pt`、`.frames.npy` 和包含哈希/源帧坐标的 `.json` 完成记录，中断半成品报错，不当作成功。

有效片段被视为新的虚拟回合：状态与目标共用起始朝向和 30 Hz 网格，图像沿用原视频偏移；不会把被删掉的启动帧放回历史。Kimodo 使用 100 帧历史（起始不足时按原模型规则 masked padding）、50 帧完整预测。旧冻结清单中的 50 帧窗口是 50 Hz 数值预检窗口，不直接作为 Kimodo 的训练窗口；实际窗口数以转换完成记录为准。

运动归一化沿用预训练 Kimodo 的统计量，不从本次验证集或留出 tracker 拟合。训练中均匀选择任务、来源、轨迹和轨迹内有效窗口，避免长轨迹取得更高权重。

## 开发原则

共用仓库只作为显式依赖，不复制模型、仿真器、环境或权重。新增代码优先薄适配；一个入口负责一件事，文件变长时按职责拆分。新文件不散落回 `新方案/`、BiGym 日志目录或其他调参目录。历史审计保留原位，索引见 [代码梳理](docs/codebase.md)。

## 短程入口与检查

`scripts/check_data.py` 检查全部冻结数值片段；指定 `--prepared` 与 `--videos` 后，另外逐像素核对首、中、末原视频图像，检查实际加载器的历史掩码和预测边界。输出报告区分数值覆盖数与真实转换覆盖数。

`scripts/train_pilot.py <run_id>` 仅允许已登记、尚未启动的 diagnostic/pilot，使用同一 Kimodo 训练循环和本目录的均衡采样；拒绝不完整预训练权重、名单/代码变动和正式 baseline 阶段。最多 2000 步、单 GPU；训练产物位于 `runs/<run_id>/training/`。该入口已在百度完成 SONIC 单源三任务 200 步 GPU 短程并保存 checkpoint；这只验证运行链路，不代表单源已学会任务。

百度补充的 PyAV 安装在 `data/cache/python`，运行视频准备/检查时使用 `PYTHONPATH=cross_tracker/data/cache/python`，不改公共虚拟环境。资源下载支持断点续传；仍以完整文件的固定哈希为准。

仿真环境与缺失依赖见 [仿真状态](docs/simulation.md)。基础物理/RGB 及 SONIC/TWIST2 三任务的六项短闭环已通过；MimicLite 原始控制器仍缺失。真实仿真使用 `scripts/prepare_eval_text.py` 生成派生任务名缓存，保留训练文本 embedding 不变。

正式入口为 `scripts/train_baseline.py <run_id> --gpu <物理编号>`；先登记 baseline，再以 `--check-only` 核对配置、证据和空闲 GPU。诊断入口继续限制为最多 2000 步。正式训练与评测 seed、计分 checkpoint 见 [冻结协议](docs/baseline-protocol.md)。
