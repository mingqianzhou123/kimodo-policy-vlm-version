# HumanoidToolBench：scratch 40k 联合训练最终报告

已完成适配、前置验证、八张 RTX 5090 上的 40,000 步联合训练，以及官方 18 条件 × 100 回合测试。**宏平均成功率 3.9444%（71/1,800）**；18 份官方回执均为 `validated`、`reportable: true`。所有条件使用同一策略权重、推理配置、适配源码、仿真源码、资产和运行时指纹。

结论：完整策略链路能够训练、部署，并产生少量真实的推球和破冰成功行为；本次训练的总体成功率偏低，尤其 L2 为零，**不足以验证完整架构在该 benchmark 上具有稳健有效性或架构优势**。本实验只有一个训练种子，未做消融或同环境基线比较，不能据此归因某个模块，也不能把适配跑通或 loss 下降当作任务有效性的证据。

## 18 个条件成功率

每格为成功率，每个条件 100 个独立环境种子；S/R 沿用官方环境名称。

| 场景 | L0-S | L0-R | L1-S | L1-R | L2-S | L2-R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BallMove | 11% | 8% | 3% | 2% | 0% | 0% |
| BallRetrieve | 4% | 7% | 0% | 0% | 0% | 0% |
| IceBreak | 15% | 20% | 1% | 0% | 0% | 0% |

按难度聚合：L0 为 65/600（10.83%）；L1 为 6/600（1.00%）；L2 为 0/600（0%）。各任务 95% Wilson 区间见运行目录的 `RESULTS.md` 和 `results_18.csv`；0/100 的区间上限约 3.70%，不能解释成真实成功概率严格为零。

官方记录中 71 回合为 `success`，1,729 回合为 `timeout`，没有 `agent_exhausted`。逐任务诊断保存在 `final_audit.json`：例如 BallRetrieve-L1-S/R 各有 4/5 个回合曾抓起正确工具，但最终成功均为零；BallMove-L1-S/R 有 12/7 个回合发生正确工具与目标接触，最终仅 3/2 次成功。这些指标说明接触或短暂抓取不足以保证完成目标，不能替换官方成功判据。

## Scratch 和训练配置

- 没有加载任何任务策略 checkpoint，明确 `policy_checkpoint: null`、`policy_419h_loaded: false`；40k run 从头启动，没有从此前 smoke、4k 或旧实验继续训练。
- 保留项目完整架构要求的冻结 Kimodo-G1 运动先验、DINOv3 和 LLM2Vec。ControlNet 从运动先验复制骨干层并初始化新适配层，连续手部头随机初始化；这不代表所有基础模型权重随机初始化。冻结文本编码的六句输出通过校验缓存复用。
- 四层 ControlNet、连续手部头；91,715,990 个可训练参数。100 个历史模型帧、50 个未来模型帧，30Hz 表示，50Hz 控制；每次执行 81/82 控制步。
- 配置 `configs/toolbench.yaml`；通用 `scripts/train.py`。18 条件混合，按 eligible windows 均匀采样，未改为任务均衡采样。global batch 128、每 GPU micro batch 16、八卡 DDP、BF16、seed 1000。
- 40,000 次 optimizer 更新；1,000 步 warmup、cosine，AdamW betas (0.95, 0.999)、weight decay 1e-6；backbone/adapter 分组 LR 使用项目配方。预算与若干优化设置参考官方 ACT benchmark；动作表示和模型结构不同，不声称复现 ACT。
- 每 10 步记录指标，每 2k 保存模型和优化器，每分钟记录 GPU；10k/20k/30k 各执行 18 条件的单种子完整长度诊断。这三次均为 0/18，均通过官方核验，未用于替换最终 100 种子评估或挑选最好 checkpoint。
- 全部 4,000 条训练指标有限，最终权重有限；40k 末次记录 loss 0.03320，末 100 更新的记录均值 0.02868。训练阶段约 2 小时 9 分钟；包括四路视频完整核验的正式评估约 2 小时 20 分钟。与其他实验共享 GPU，未停止其他作业。

## 数据和接口验证

固定 ToolBook simulation revision `8b2cd31e107b64cb13f812ea217a63a20845c78a`；3,003 条成功演示、2,322,445 个 50Hz 帧。全部所选原始文件 SHA256 校验，全部训练图像实际解码；转换后 1,394,672 个模型帧、1,247,525 个窗口。未混入 hardware 数据。

完整记录见 [前置验证报告](toolbench_preflight_20261027.md)。本次修正了固定 83 步的历史采样假设，改为实际 81/82 步网格，并使连续手部的训练历史与在线观测一致。43 维原始录制、32 维公开策略状态、36 维动作之间逐字段检查了左右手顺序、手臂顺序、腰部 RPY、上次高度指令、身体坐标系速度、世界系目标朝向和 t→t+1 目标语义。

46 个项目测试通过、4 个跳过；双卡/四卡/八卡 smoke 训练完成；18 条件实际 HTTP 图像交互通过。18 条件 × 三种路径的 54 次完整演示运动回放均无错误、无摔倒。另核对多回合 reset、会话隔离和噪声种子序列，确认回合间清空策略历史。固定噪声的条件敏感性检查中，重复推理一致，改变图像或文本会改变动作；此项只说明条件路径生效，不代表任务成功。

## 官方评估及可复核证据

使用未修改的本机 HumanoidToolBench `2cd4f006f6f2ebc536a3fb45aff8ce2eaf1382d7`，独立环境，test split、seeds 10000–10099，每回合最多 3,000 个 50Hz 控制步，官方 decoupled WBC、成功判据及 `write_summary`。18 个 evaluator 各有独立策略服务，并行分布于八张卡。推理 BF16、10 个 diffusion steps、50 个模型帧执行长度、turn=operator、root_boundary=extrapolate。

官方逐帧解码核验了 **7,200 段四路相机视频、21,071,564 帧**，检查时间戳、H.264/640×360/50Hz、非黑帧、相机不重复、帧数、种子覆盖、源码和资产身份。`reportable: true` 表示符合官方评估协议，不代表达到有效策略性能。

运行根目录：`log/toolbench/joint18_scratch40k_20261027_8gpu`。
监督目录：`log/toolbench/joint18_scratch40k_20261027_8gpu_supervision`（稳定入口 `log/toolbench/current_supervision`）。

| 内容 | 相对监督目录的路径 |
| --- | --- |
| 官方 18 任务汇总 | `evaluation_18x100/summary.json`、`summary.md` |
| 官方回执及完整视频 | `evaluation_18x100/<condition>/run-*/benchmark_result.json`、`videos/` |
| 成功率及 95% 区间 | `RESULTS.md`、`results_18.csv` |
| 最终种子/哈希/帧数审计及失败诊断 | `final_audit.json` |
| 全量最终训练数值审计 | `final_training_audit.json` |
| 初始化和训练配置 | run 目录 `initialization.json`、`config.json`、`dataset.json` |
| 监督状态和 GPU 记录 | `status.json`、`progress.html`、`gpu.jsonl`、`evaluation_progress.json` |
| 完整训练/诊断日志 | `training.log`、`diagnostic_{10000,20000,30000}/` |
| 训练/评估启动时源码 | `source_snapshot.tar.gz`、`evaluation_source_snapshot.tar.gz`、对应 SHA256 记录 |
| 最终交付源码快照 | `delivery_source_snapshot.tar.gz`、`delivery_source_sha256.json` |
| 多回合 reset 核对 | `episode_reset_audit.json` |
| 可视语言条件检查 | `conditioning_4000.json` |
| 成功 L1 演示关键帧 | `success_L1_contactsheet.jpg`、`success_L1_examples.json` |

两段实际成功示例均来自种子 10003：BallMove-L1-R 完成推球目标；IceBreak-L1-S 用锤触发破冰。对应四路原始成功视频保留在各自 `videos/episode_10003/`，关键帧不是合成图。

最终策略：`log/toolbench/joint18_scratch40k_20261027_8gpu/checkpoint_40000/training_state.pt`。
SHA256：`f729af6a69c74a41c1c084907969b3682aa27dbe707b0c9e3ec05e76222692d7`。

## 解释限制及后续研究方向

每手单个闭合度不能保留录制中的两种拇指姿态，原始关节残差最大 0.5rad；公开策略状态没有实测骨盆位置/朝向，采用命令推算并补齐动作分块的历史；公开工具与台面布局不能精确还原录制场景。因此运动回放验证的是控制一致性，不能作为任务成功上界。这些是已量化或确认的限制，尚未通过消融证明它们是本次低成功率的原因。

若继续研究，应优先分别检验手部表示、公开场景分布差异和更短执行段的闭环控制，并用相同种子及官方判据比较；增加同协议基线/模块消融后才能讨论架构贡献。本次保留完整架构和既定训练预算，未为提高分数改写官方评估或追加未经报告的调参训练。
