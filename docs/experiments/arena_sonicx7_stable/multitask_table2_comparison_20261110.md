# N1.7 VLM 注入模型与 HumanoidArena Table 2 scratch 基线对比

日期：2026-11-10  
分支：`model_improvement_lct1`  
训练运行：`scratch_stable_20261105_exclusive4`  
最终 checkpoint：`checkpoint_500000`

## 结论摘要

本实验的七任务总体成功率为 **65.7%（276/420）**，论文 Table 2 中 Kimodo-Policy 的 scratch 行为 **67.6%**，差值为 **−1.9 个百分点**。因此，当前 N1.7 VLM 视觉注入方案已经达到接近原 Kimodo scratch 基线的总体水平，但尚未实现七任务总体提升。

改进集中在 HOI 和部分 HSI 任务：HOI 平均成功率为 **53.9%**，高于基线 **51.7%**；Boxing 和 VisNavi 分别高出 **10.0** 和 **8.3** 个百分点。总体下降主要来自 OpenDoor 和 SitSofa，分别低于基线 **20.0** 和 **18.4** 个百分点。

## 实验协议

- 训练：七任务联合 scratch，未加载 419h 训练 checkpoint；最终训练步数 500,000。
- 模型：Kimodo Root、Body、文本条件、动作历史、扩散去噪和 HandNet 主流程保留；视觉 ControlNet 替换为 N1.7 VLM、共享四层图文适配器和八个 Cross-Attention 残差注入模块。
- 评估：每个任务 3 个种子（0、1、2），每个种子 20 回合，共 60 回合。
- 推理设置：原 SONIC 环境，FP32，10 步扩散，执行 15 帧，RTC 关闭，控制频率 50 Hz，预测窗口 50 帧。
- 对照：只引用论文 Table 2 的 `Ours`（scratch、无 VLA 预训练初始化）结果；按用户要求没有重新运行原 scratch 基线。
- 有效性检查：七项 `validation.json` 均为 `valid_inference=true`，每项均为 60 回合且存在真实策略推理记录；没有 HTTP 5xx 或空推理结果。

论文来源：`/root/workspace/kimodo-policy-paper/main_arxiv_codex.md`，Table 2。论文 scratch 行为：Football 35.0%、DoubleDesk 40.0%、P&P Box 80.0%、OpenDoor 93.3%、SitSofa 96.7%、Boxing 66.7%、VisNavi 61.7%、Overall 67.6%。

## 逐任务结果

| 分组 | 任务 | 当前模型 | 成功/回合 | 当前种子 SD | Table 2 scratch | 差值 | 当前 95% Wilson 区间 |
|---|---|---:|---:|---:|---:|---:|---:|
| HOI | Football | 38.3% | 23/60 | 7.6 pp | 35.0±7.1% | **+3.3 pp** | 27.1–51.0% |
| HOI | DoubleDesk | 43.3% | 26/60 | 2.9 pp | 40.0±0.0% | **+3.3 pp** | 31.6–55.9% |
| HOI | P&P Box | 80.0% | 48/60 | 5.0 pp | 80.0±4.1% | **0.0 pp** | 68.2–88.2% |
| HSI | OpenDoor | 73.3% | 44/60 | 2.9 pp | 93.3±6.2% | **−20.0 pp** | 61.0–82.9% |
| HSI | SitSofa | 78.3% | 47/60 | 10.4 pp | 96.7±4.7% | **−18.4 pp** | 66.4–86.9% |
| HSI | Boxing | 76.7% | 46/60 | 7.6 pp | 66.7±4.7% | **+10.0 pp** | 64.6–85.6% |
| HSI | VisNavi | 70.0% | 42/60 | 18.0 pp | 61.7±6.2% | **+8.3 pp** | 57.5–80.1% |

`pp` 表示 percentage points。Wilson 区间只描述当前模型在 60 回合下的二项比例不确定性；论文基线是论文报告的实验统计值，不是本实验重新采样得到的同一批回合，因此不能把两者区间直接当作严格的同分布显著性检验。

## 分组和总体统计

| 统计范围 | 当前模型 | 回合数 | Table 2 scratch | 差值 | 当前 95% Wilson 区间 |
|---|---:|---:|---:|---:|---:|
| HOI 平均（Football、DoubleDesk、P&P Box） | 53.9%（97/180） | 180 | 51.7% | **+2.2 pp** | 46.6–61.0% |
| HSI 平均（OpenDoor、SitSofa、Boxing、VisNavi） | 74.6%（179/240） | 240 | 79.6% | **−5.0 pp** | 68.7–79.7% |
| Overall（七任务） | **65.7%（276/420）** | 420 | **67.6%** | **−1.9 pp** | 61.1–70.1% |

三个种子的总体成功率分别为：seed 0 为 **67.1%（94/140）**，seed 1 为 **60.7%（85/140）**，seed 2 为 **69.3%（97/140）**；总体种子样本标准差约 **4.5 个百分点**。HOI 和 HSI 的种子聚合成功率分别为：

- HOI：58.3%、48.3%、55.0%，均值 53.9%，样本标准差 5.1 个百分点。
- HSI：73.8%、70.0%、80.0%，均值 74.6%，样本标准差 5.1 个百分点。

## 失败类型

从 420 个 episode 文件的 `failure_reason` 统计：

| 结果 | 数量 | 比例 |
|---|---:|---:|
| 成功 | 276 | 65.7% |
| Timeout | 123 | 29.3% |
| Fall | 21 | 5.0% |
| 失败合计 | 144 | 34.3% |

Fall rate 按正式评估 episode 记录为 **5.0%**，与论文 Table 2 scratch 行报告的 AFR **5.0%** 一致。失败主要是超时而非摔倒，说明当前模型的稳定性已经接近基线，但部分任务还不能在规定时限内完成目标。

任务级失败分布如下：

- Football：37 次 timeout，无 fall。
- DoubleDesk：22 次 timeout、12 次 fall。
- P&P Box：12 次 timeout，无 fall。
- OpenDoor：16 次 timeout，无 fall。
- SitSofa：13 次 timeout，无 fall。
- Boxing：14 次 timeout，无 fall。
- VisNavi：9 次 timeout、9 次 fall。

## 分析

### 1. HOI 视觉注入基本有效

Football 和 DoubleDesk 各提升 3.3 个百分点，P&P Box 保持 80.0%。这说明 N1.7 VLM 特征经过共享四层适配器后，能够为 Root/Body 的动作 Query 提供有用的物体和交互场景条件。HOI 平均提升 2.2 个百分点，且三项任务的种子波动不大到中等。

### 2. HSI 结果出现任务分化

Boxing 和 VisNavi 分别提升 10.0 和 8.3 个百分点，说明新视觉路径对目标定位、空间导航或局部场景识别具有收益。但 OpenDoor 和 SitSofa 大幅下降，说明当前共享视觉条件与这两类任务的动作时序、姿态终态或场景语义对齐仍不足。仅凭成功率不能断言具体原因，需要结合失败轨迹、视觉注意区域和动作完成时间进一步定位。

### 3. Overall 接近基线，但收益被两个 HSI 任务抵消

HOI 的提升和 Boxing/VisNavi 的提升不足以抵消 OpenDoor/SitSofa 的下降，最终 Overall 为 65.7%，比论文 scratch 低 1.9 个百分点。这个差距小于单个任务的典型 60 回合抽样波动，但任务级方向具有清晰结构：当前方案不是所有任务均匀退化，而是对特定场景任务存在适配问题。

### 4. 结果尚不支持直接解冻整个 VLM

当前 VLM 已冻结，新增注入模块和 HandNet 适配部分训练完成；模型在七任务上已经获得接近基线的结果。下一步应先针对 OpenDoor/SitSofa 做低风险诊断和小规模适配器/注入层微调，确认文本映射、场景配置、任务采样和成功判定均正确，再考虑解冻四层适配器或视觉编码器。直接解冻整个 VLM 会增加显存、训练不稳定和灾难性遗忘风险。

## 可复现文件

- 最终汇总：`/root/workspace/kimodo-policy-vlm-version/log/arena_sonicx7_groot/scratch_stable_20261105_exclusive4/experiment_summary.json`
- 训练配置：`/root/workspace/kimodo-policy-vlm-version/log/arena_sonicx7_groot/scratch_stable_20261105_exclusive4/config.json`
- 最终 checkpoint：`/root/workspace/kimodo-policy-vlm-version/log/arena_sonicx7_groot/scratch_stable_20261105_exclusive4/checkpoint_500000`
- 七任务监控状态：`/root/workspace/kimodo-policy-vlm-version/outputs/groot_scratch_stable_20261105_exclusive4_status.json`
- 评估输出目录：`/root/workspace/kimodo-policy-vlm-version/log/arena_sonicx7_groot/scratch_stable_20261105_exclusive4/groot_<task>_formal/`
- 实时面板：<https://console.bce.baidu.com/api/aihcbox/proxy/v1/webide/dv-7eacb69a00/su/vscode/proxy/8765/monitor/dashboard.html>

