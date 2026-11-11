# Fullbody v1：最终正式结果

最终checkpoint：600000；18条件×100回合；test seeds10000–10099；每回合最多3000控制步；D10/E20/math/RTC=0。

总成功 **293/1800＝16.28%**。L0：243/600＝40.50%；L1：41/600＝6.83%；L2：9/600＝1.50%。18条件均完成官方验证，执行失败条件数0。

| 条件 | 成功 /100 |
|---|---:|
| G1BallMove-L0-R | 47 |
| G1BallMove-L0-S | 50 |
| G1BallMove-L1-R | 6 |
| G1BallMove-L1-S | 15 |
| G1BallMove-L2-R | 0 |
| G1BallMove-L2-S | 0 |
| G1BallRetrieve-L0-R | 17 |
| G1BallRetrieve-L0-S | 14 |
| G1BallRetrieve-L1-R | 1 |
| G1BallRetrieve-L1-S | 4 |
| G1BallRetrieve-L2-R | 0 |
| G1BallRetrieve-L2-S | 0 |
| G1IceBreak-L0-R | 65 |
| G1IceBreak-L0-S | 50 |
| G1IceBreak-L1-R | 9 |
| G1IceBreak-L1-S | 6 |
| G1IceBreak-L2-R | 9 |
| G1IceBreak-L2-S | 0 |

原始基线为314/1800＝17.44%；本版少21次成功，没有达到提高原始成功率的目标。多项修改同时生效，不能将结果归因于某一个修复。单训练种子也不足以证明架构优势。

L1/L2共1150次失败：631次未达到正确工具抬升阈值，165次抬升后未接触目标，338次抬升且接触目标但目标条件从未成立，16次目标条件成立但未连续保持50控制步。这是按事件优先级划分的统计，不是物理根因标注。

[版本说明](REPORT.md) · [机器可读结果](results.json) · [逐条件CSV](results.csv)
