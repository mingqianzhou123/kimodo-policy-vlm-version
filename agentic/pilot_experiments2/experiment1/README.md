# 实验 1：历史日志两段误差拆解

状态：做完（2026-11-26）。没有新跑仿真，只读历史日志。结论见下面“结论”。

## 问的是什么

11 月 23–25 日的管线失败，原因在生成侧还是执行侧。SONIC 在 HITCH 的测试里只有 27.9%，HEFT 是 94.2%，之前“Kimodo 不行”的结论可能被 SONIC 污染。

## 做法

取全部历史回合，成功和失败都要。每个回合算两段误差，对手腕位置、手腕朝向、根部位置、躯干朝向分别统计，再比较成功和失败回合在两段上的分布。

- 第一段：目标位姿到 Kimodo 参考。
- 第二段：Kimodo 参考到 SONIC 实际执行。

## 输出和判读

一张两段误差对成败的表，和一句结论。

- 第二段区分成败：之前的结论被 SONIC 污染，Kimodo 要在好 tracker 上重新评估。
- 第一段区分成败：生成本身有问题。
- 日志缺字段：记下缺什么，补进实验 2 之后的日志格式，不为此重跑。

## 盘点时确认的（2026-11-26）

- 两段都算得出来。日志里其实有三层：Kimodo 原始输出、经 IK 投影和根部修正后发出去的参考、SONIC 实际执行，第一段可以再拆成两步。
- 躯干朝向没有直接记录，要用根部朝向加腰关节算。
- 这些回合的目标位姿来自仿真器，没有真的 VLM，所以这里量不到 VLM 的误差。
- 接触力有逐步记录，可以把误差只量在第一次碰到物体之前，并单独量从第一次碰到到合手之间物体被推了多远（Codex 复核留下的未决问题）。这两个量正好对应实验 3 的第 3 类和第 4 类。

## 事先定死的口径（2026-11-26，跑真实数据、看成败之前写下）

脚本 `stage_split.py`，测试 `test_exp1_stage_split.py`。

- **用哪些回合**：运行目录里所有 `exp1-*`（箱子，11 月 24–25 日）、`exp4-*`（箱子，延迟实验）、`exp1b-*`（锤子）的抓取回合，调试回合（`dbg-*`）不用。走路和伸手类（`exp2*`、`wa-*`）没有“目标—成败”这一对，不算。只看每个回合的第一轮（第一轮失败后场面多半已经被破坏）；成败用 `grasp-result.json` 的 `first_attempt_success`（仿真里的物理判据：抬高 10 cm 且夹住 0.5 s）。每轮的失败标签只看合手时箱子的速度，不用。
- **分组**：家族（箱子 `exp1`、箱子延迟 `exp4`、锤子 `exp1b`）× 条件（物体静止 / 往复，站着 / 先走过去；锤子的注入错位单独一组）。不跨家族合并。
- **五层**：T 任务目标（箱子：合手前双手要到的 `pre` 位姿，就是合手把关检查的那个；锤子：右手的 `grip` 位姿；根部：本轮站定时的根部；躯干：站定时的躯干朝向）→ K Kimodo 原始输出（这一步的 29 个关节，XML 正运动学）→ S 发出的参考（同一个计划经 IK 投影、根部固定之后）→ R SONIC 这一步实际在跟的参考（`execution.jsonl`）→ A 机器人实际。第一段 = T→R，拆成 1a（T→K）、1b（K→S）和交接（S→R，应接近 0）；第二段 = R→A。
- **四个量**：手腕位置（m，两手平均；锤子只看右手）、手腕朝向（度）、根部水平位置（m）、躯干朝向（度，根部朝向加腰部三个关节做正运动学）。
- **接触阈值**：任务用的手对物体的接触力 ≥ 0.05 N 算“碰到”（等于代码里判定夹持和撤驱动用的 `hand_force_n`）；1 N 作为敏感性检查另报。
- **时刻**：P = 第一次碰到和合手两者中较早那个的前一步（两者都没有：第一轮最后一个伸手步）；C = 第一次合手；W = 截止到 P 的 0.5 s 平均。物体：第一次碰到到合手之间物体水平移动了多少（先合手后碰到记 0）；目标过时：锁存时的物体位置和 P / C 时实际位置之差。
- **主对比（最多三个）**：P 时手腕位置的第一段误差、P 时手腕位置的第二段误差、碰到到合手之间物体被推的距离。其余都是描述。
- **“区分成败”的标准**：成功和失败各至少 5 个；失败回合比成功回合误差大的概率（AUC）≥ 0.70 且 95% 自助法区间下限 > 0.5（或 AUC ≤ 0.30 且上限 < 0.5），并且双侧 Mann–Whitney p < 0.01。达不到“各 5 个”的组记“不能下结论”。
- **判读**（照计划，不改）：只有第二段区分成败 → 之前“Kimodo 不行”的结论被 SONIC 污染；只有第一段区分 → 生成本身有问题；两段都区分或都不区分照实写；字段缺失记下来。

## 结论（2026-11-26）

**一句话**：2,131 个历史抓取回合里，SONIC 收到的手腕参考和规划器要的位置、朝向一致（第一段中位数 0；Kimodo 自己 2–3 cm 的手腕误差在 IK 投影这一步就被盖掉了，从没到过 SONIC）。能区分成败的只有第二段：静止箱子里，失败回合在碰到或合手前手腕离参考 16.5–20.9 cm（成功 4.7–5.4 cm），骨盆离参考 7.5–16.7 cm（成功 3.0–3.6 cm）；锤子里手腕朝向差 24°（成功 6.5°）。物体在动的组里两段都不区分，区分成败的是目标过时和碰到后物体被推走。按计划的判读：**之前“Kimodo 不行”的结论被 SONIC 污染**（动态条件下还混着目标过时），Kimodo 要在更好的 tracker 上重新评估；而且这批日志根本量不到 Kimodo 的手腕精度，因为它的手腕输出没有被执行过。

按事先的规则逐组的读数（主对比是 P 时手腕位置的第一段和第二段）：

| 组 | 回合 | 第一轮成功 [95% 区间] | 规则读数 | 实际怎么读 |
|---|---|---|---|---|
| 箱子 · 静止 · 站着 | 220 | 137 [0.56, 0.68] | 第二段区分 | 同左；失败主要是合手把关一直没过（58 / 83），来自工作区地图的远格和“站到位再抓”那组 |
| 箱子 · 静止 · 先走过去 | 120 | 86 [0.63, 0.79] | 第二段区分 | 同左；走路本身每个计划的误差不区分，最后站位离目标 4.6 vs 7.0 cm 区分 |
| 箱子 · 往复 · 站着 | 180 | 141 [0.72, 0.84] | 第一段区分 | **规则的漏洞**：第一段在成败两边中位数都是 0.00 cm（全组中位数 0.001 mm、95 分位 0.002 mm，最大 7.7 mm），区分靠的是 0.01 mm 以下的数值残差。物理上两段都不区分；区分的是合手时目标过时（2.7 vs 6.8 cm）、碰到到合手物体被推（1.7 vs 5.0 cm）和手腕朝向第二段（5.1° vs 11.9°） |
| 箱子 · 往复 · 先走过去 | 180 | 6 [0.02, 0.07] | 都不区分 | 只有 6 个成功；区分的是目标过时（8.0 vs 40.0 cm），走路太久，箱子早走了 |
| 箱子延迟 · 往复 · 站着 | 694 | 621 [0.87, 0.92] | 都不区分 | 区分的是物体被推（1.8 vs 3.0 cm）和目标过时（2.2 vs 3.7 cm） |
| 箱子延迟 · 静止 · 站着 | 299 | 273 [0.88, 0.94] | 都不区分 | 区分的是物体被推（0.6 vs 2.2 cm）和合手时的目标过时（0.7 vs 2.3 cm，静止箱子的“过时”就是被推走的距离）；手腕朝向第二段也区分（3.9° vs 6.6°） |
| 锤子 · 静止 · 站着 | 118 | 83 [0.62, 0.78] | 都不区分（按手腕位置） | 手腕**朝向**第二段强区分（6.5° vs 24.3°，AUC 0.94）；35 个失败里 29 个是合手把关没过（把关的朝向容差 14°） |
| 锤子 · 往复 · 站着 | 80 | 17 [0.14, 0.31] | 都不区分 | 成功全在一格；区分的是目标过时（11.3 vs 39.4 cm） |
| 锤子 · 注入错位 | 240 | 181 [0.70, 0.80] | 都不区分 | 目标是故意给错的；任务目标到实际手腕的总误差区分（2.0 vs 2.8 cm） |

## 各段误差 × 成败（P 时，中位数 成功 / 失败，括号里是 AUC）

P = 第一次碰到物体和合手两者中较早那个的前一步。AUC = 失败回合误差大于成功回合的概率；0.5 是分不开，“是”按事先的标准（各 ≥ 5 个、AUC ≥ 0.70 且区间下限 > 0.5 或对称、p < 0.01）。完整的表（每个量的四分位、区间、p，合手时和 P 前 0.5 s 的版本）在下面折叠的部分和输出目录的 `table.md`。

| 组 | 手腕位置 1a：Kimodo→指令 (cm) | 第一段：指令→SONIC 参考 (cm) | 第二段：参考→实际 (cm) | 手腕朝向 第二段 (°) | 根部 第二段 (cm) | 合手时目标过时 (cm) | 碰到→合手 物体被推 (cm) |
|---|---|---|---|---|---|---|---|
| 箱子 · 静止 · 站着 | 1.88 / 2.61 (0.83 是) | 0.00 / 0.00 (0.42) | **4.74 / 20.86 (0.87 是)** | 5.30 / 10.32 (0.79 是) | 2.96 / 16.69 (0.84 是) | 0.42 / 2.28 (0.82 是) | 0.20 / 0.00 (0.44) |
| 箱子 · 静止 · 先走过去 | 1.89 / 3.06 (0.87 是) | 0.00 / 0.00 (0.58) | **5.37 / 16.50 (0.83 是)** | 4.75 / 12.20 (0.76 是) | 3.60 / 7.48 (0.83 是) | 0.89 / 2.24 (0.70) | 0.81 / 0.00 (0.39) |
| 箱子 · 往复 · 站着 | 1.86 / 1.97 (0.59) | 0.00 / 0.00 (0.80，见上，数值残差) | 5.07 / 4.90 (0.41) | 5.13 / 11.90 (0.86 是) | 3.34 / 3.55 (0.56) | 2.73 / 6.81 (0.87 是) | 1.65 / 4.99 (0.86 是) |
| 箱子 · 往复 · 先走过去 | 2.00 / 2.29 (0.62) | 0.00 / 0.00 (0.42) | 6.69 / 6.53 (0.44) | 9.51 / 7.19 (0.37) | 4.09 / 2.81 (0.36) | 8.02 / 39.97 (0.95 是) | 7.98 / 0.05 (0.31) |
| 箱子延迟 · 往复 · 站着 | 2.01 / 2.15 (0.58) | 0.00 / 0.00 (0.59) | 5.00 / 4.38 (0.38) | 6.74 / 8.10 (0.65) | 3.33 / 3.16 (0.41) | 2.18 / 3.68 (0.76 是) | 1.77 / 3.01 (0.72 是) |
| 箱子延迟 · 静止 · 站着 | 1.80 / 2.20 (0.74 是) | 0.00 / 0.00 (0.69) | 4.94 / 4.46 (0.46) | 3.93 / 6.60 (0.76 是) | 3.19 / 2.95 (0.44) | 0.67 / 2.27 (0.85 是) | 0.64 / 2.24 (0.86 是) |
| 锤子 · 静止 · 站着 | 2.22 / 2.47 (0.60) | 0.00 / 0.00 (0.49) | 1.99 / 2.31 (0.64) | **6.52 / 24.28 (0.94 是)** | 1.27 / 1.89 (0.74 是) | 0.00 / 0.00 (0.51) | 0.00 / 0.00 (0.65) |
| 锤子 · 往复 · 站着 | 2.05 / 2.30 (0.57) | 0.00 / 0.00 (0.60) | 1.78 / 2.20 (0.70) | 4.82 / 4.55 (0.50) | 1.39 / 1.52 (0.66) | 11.33 / 39.43 (0.96 是) | 0.01 / 0.00 (0.16) |
| 锤子 · 注入错位 | 2.14 / 2.40 (0.57) | 0.00 / 0.00 (0.44) | 1.89 / 2.08 (0.64) | 5.12 / 5.87 (0.58) | 1.46 / 1.40 (0.49) | 1.98 / 4.72 (0.47) | 0.00 / 0.10 (0.68) |

第一轮失败是怎么碰到物体的（合手把关一直没过 / 合手前先碰到 / 先合手）：箱子静止站着 58 / 24 / 1，静止走过去 22 / 8 / 4，往复站着 0 / 39 / 0，往复走过去 23 / 80 / 71，延迟往复 2 / 65 / 6，延迟静止 0 / 26 / 0；锤子静止 29 / 2 / 4，往复 0 / 9 / 54，注入错位 3 / 52 / 4。“合手把关没过”对应实验 3 的第 3 类（目标对、身体没到），“合手前先碰到、把物体推走”对应第 4 类。

走路那一段（箱子 · 静止 · 先走过去，按计划逐个比位移）：Kimodo 每个计划的根部误差 7.39 / 7.70 cm（AUC 0.57），SONIC 每个计划的位移误差 12.20 / 12.60 cm（0.58），机器人实际走的距离是参考的 0.80 / 0.71（0.36，p = 0.016，不算区分），最后站位离走路目标 4.61 / 6.98 cm（0.70，区分）。走路两段都不单独区分，站位误差是累计出来的。

几处要知道的细节：

- **静止站着这组的失败大多来自两格设计**：工作区地图（`exp1-map`，100 个，故意包含够不着的远格）50 个、“站到位再抓”（`exp1-stance`）8 个，都是合手把关没过，失败时手腕第二段 22.7 / 22.9 cm、骨盆第二段 18.2 / 18.1 cm——SONIC 在远伸手时骨盆往后退，手到不了。区域大小的静止格（`exp1-region`、`exp1-long-static`、`exp1-xlong-static`）每格只有 3–4 个失败，全是合手前先碰到，物体被推约 2 cm，第二段和成功回合相近（失败 4.4–6.8 cm，成功 4.7–5.3 cm）。逐格数字见下面折叠的第二张表。
- **躯干朝向没有任务目标**：这里的“第一段”是参考相对站定时躯干的前倾，静止站着的失败回合里参考倾 29°（成功 13°），SONIC 只做到一部分（第二段 25° vs 11°）。远伸手要更大的前倾，这是够不够得着的问题，不是 Kimodo 的误差。
- Kimodo 自己的手腕误差（1a）在三个静止箱子组也区分成败（1.8–1.9 vs 2.2–3.1 cm），但它被 IK 投影盖掉、没有传到 SONIC，只能说明难的摆放对 Kimodo 和 SONIC 都难，不能说明它造成了失败。

## 方法补充（定死口径之外，按时间顺序）

- **第一段比的是“指令”，不是最终抓取位姿**：伸手过程中，规划的手腕路径本来就还在路上，合手前还会叠加一个闭环修正（量到手腕差多少就往前多给多少，每只手最多 12 cm），参考是故意越过目标的。所以第一段的起点是“规划路径在这一步的位置 + 当时的修正量”。路径按 `rounds.jsonl` 里的路点和时刻重建，修正量按策略的规则（增益 0.7、上限 12 cm，两个配置文件里从没改过）从每次规划时量到的手腕重放。重放和日志里合手时记下的修正量对得上：各组中位数 0.0 mm，最大 0.7 mm。另报两项：“路径还没到”（任务位姿到规划路径）和“修正量”。这一处是看了 4 个样例回合、发现参考故意越过目标之后改的，那时还没有任何成败比较。
- **走路按计划逐个比位移**：走路时参考的根部在绝对坐标上会漂开（普通走路片段里最多一米多），SONIC 跟的是参考根部的运动而不是绝对位置，所以走路段比的是每个计划期间的位移。第一段用 Kimodo 自己记的根部误差（普通走路片段：终点误差 `end_error_m`；官方方式走路：路径误差 `root_path_error_m`，两者定义不同）。
- **一次修正**：第一次完整汇总后发现修正量重放多算了一次（离开伸手阶段那一次调用策略不做修正），改了以后重跑，各组的规则读数和改之前完全一样。
- **跳过的回合**：2,133 个运行目录，2 个没跑完的延迟实验回合（`exp4-fixed-jgem-v010` 的种子 6005、6006）没有可用的伸手步，跳过；分析 2,131 个。
- **数值核对**：发出的参考和 SONIC 实际在跟的参考之间（交接）各组最大 0.0 mm。第一段手腕位置的中位数各组都是 0.001 mm 以下；95 分位在站着的组里 ≤ 15 mm，在先走过去的组里 54–104 mm（约 5% 的回合碰到物体时参考还没到指令位置，原因没有查）。

## 局限

- **没有 VLM**：这些回合的目标位姿来自仿真器真值，VLM 的误差（实验 3 的第 2 类）量不到。唯一的“目标误差”是物体在动时锁存的位置过时，以及锤子组里故意注入的错位。
- **这批日志量不到 Kimodo 的手腕精度**：伸手阶段的手腕参考由 IK 投影决定，根部固定在站定位置，Kimodo 只提供其他关节的姿态和走路。所以“第二段区分成败”说明的是 SONIC 执行 IK 给的参考时失败，不能反过来证明 Kimodo 的手腕输出够好。
- **规则没有物理下限**：事先的标准只看 AUC 和 p，没有写“差别至少多少厘米”，于是箱子往复站着组 0.01 mm 以下的数值残差也判成了“第一段区分”。以后的实验要加物理下限（比如 1 cm 或 5°），这次按原规则照实报，并在上表注明。
- **合手时刻有选择效应**：合手把关要求手腕在目标 5 cm、14° 以内，合手时的误差被截断，所以主对比用 P 而不是合手时刻。
- **组内混了不同设计**：静止站着组的失败大多来自故意包含远格的地图；先走过去的往复组只有 6 个成功；锤子往复组成功都在一格。这几组的区分结论要连同逐格表一起看。
- **往复的箱子在第一次被碰到时撤掉驱动**，之后靠惯性滑，所以“碰到到合手被推的距离”包含滑行（`episodes.jsonl` 里有碰到时物体的速度 `object_speed_at_contact_mps`）。
- 次要指标很多，没有做多重比较校正；判读只用两个主对比。

## 日志缺的字段（补进实验 2 之后的日志格式）

1. 每一步发出的“指令目标”：规划路径上的手腕位姿和当时的闭环修正量。这次是从 `rounds.jsonl` 重建、重放出来的。
2. 走路时每个计划要求的根部位移，以及参考根部和实际根部之间的偏移；现在参考根部在绝对坐标上漂移，只能比位移。
3. 躯干朝向和骨盆高度的目标：现在的规划器没有这两项指令，实验 3 的指令接口里有，要逐步记录。
4. 普通走路片段（非官方方式）没有逐步的 Kimodo 原始 / 发出指令，只有整段的 `plan-*.npz`。
5. 接触只有左手、右手、机器人其他部位三个合力，没有接触点，看不出物体是被拳头、手指还是前臂推走的。
6. VLM 的原始输出和它给的目标（这批本来没有 VLM）。
7. 每轮的失败标签只看合手时箱子的速度（Codex 复核指出），以后由逐步日志判定，不再存这种标签。

## 对实验 3 选规划器的含义

伸手阶段现有管线实际上已经是“IK 定手腕 + SONIC 执行”，Kimodo 的手腕输出在 IK 之前就被替换掉了。所以实验 3 用运动学（IK）规划器出手部目标，不会比现在差；Kimodo 现在真正起作用的地方是走路和全身姿态。决定成败的是执行（第二段）和动态条件下的目标过时，tracker 选哪个（实验 2）比规划器选哪个更关键。要不要在更好的 tracker 上重新评估 Kimodo，由用户决定。

## 数据、备份和输出

- 读的数据：开发机 `KVLA_AGENTIC_RUNS` 下 `exp1-*`、`exp1b-*`、`exp4-*` 的运行目录（2,133 个，11 月 24–25 日）。
- 备份（只读，非视频文件：json、jsonl、npz，75,337 个文件，未压缩 41.4 GB）：开发机持久盘 `/mnt/cfs/d5bvk6/exp1-stage-split-runs-20261126.tgz`（和 `exp2-runs-20261124.tgz` 放在一起），压缩后 16.4 GB。核对：归档列出的文件清单和打包清单逐行一致（75,337 个），记录在输出目录的 `backup-log.txt`。
- 输出（不进 git）：`/mnt/cfs/d5bvk6/personal/mingqianzhou/exp1-stage-split-20261126/`：`episodes.jsonl`（每回合一行，所有量）、`summary.json`、`table.md`、`cells.md`、`DONE.txt`（代码版本 `05cce20`）。本文所有数字都从这次输出里抄出。

## 怎么复现

在开发机上，用策略环境的 Python（不需要显卡，48 个进程约 1 分钟，汇总约 2 分钟）：

```bash
source agentic/env.local.sh
P=$KVLA_SIM_CACHE/envs/policy/bin/python
S=agentic/pilot_experiments2/experiment1/stage_split.py
OUT=<输出目录>
$P $S episodes --out $OUT --workers 48
$P $S summary --out $OUT
$P $S table --out $OUT > $OUT/table.md
$P $S cells --out $OUT > $OUT/cells.md
$P -m pytest agentic/pilot_experiments2/experiment1 -q
```

从备份恢复时，把 `.tgz` 解到一个目录，`episodes` 加 `--runs <那个目录>`。

<details><summary>完整的表：每组每个量（stage_split.py table 的输出，原样）</summary>

#### box · moving-stand

180 episodes (exp1-long-move003, exp1-long-move010, exp1-move003, exp1-move010, exp1-xlong-move003, exp1-xlong-move010); first-attempt success 141/180 [0.72, 0.84]; verdict: **stage 1 separates (generation side)**

first attempt, success / failure: never closed 0 / 0, touched before closing 110 / 39, closed first 31 / 0

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 0.002, 7.692], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 141 / 39 | 1.86 | 1.97 | 0.59 [0.48, 0.68] | 0.094 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 141 / 39 | 0.00 | 0.00 | 0.80 [0.72, 0.86] | 1.7e-08 | yes |
| wrist position, stage 2 (reference -> robot) (cm) | 141 / 39 | 5.07 | 4.90 | 0.41 [0.31, 0.50] | 0.073 | no |
| wrist position, closed-loop correction in force (cm) | 141 / 39 | 2.68 | 0.00 | 0.23 [0.18, 0.28] | 1.2e-08 | yes |
| wrist position, task target -> robot (cm) | 141 / 39 | 3.49 | 7.16 | 0.86 [0.80, 0.91] | 9.8e-12 | yes |
| wrist orientation, 1a (deg) | 141 / 39 | 10.25 | 10.59 | 0.52 [0.43, 0.61] | 0.763 | no |
| wrist orientation, stage 1 (deg) | 141 / 39 | 0.00 | 0.00 | 0.83 [0.76, 0.89] | 1.9e-10 | yes |
| wrist orientation, stage 2 (deg) | 141 / 39 | 5.13 | 11.90 | 0.86 [0.79, 0.92] | 7.7e-12 | yes |
| root position, 1a (cm) | 141 / 39 | 1.26 | 1.12 | 0.34 [0.26, 0.43] | 0.002 | no |
| root position, stage 1 (cm) | 141 / 39 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 141 / 39 | 3.34 | 3.55 | 0.56 [0.46, 0.66] | 0.253 | no |
| torso orientation, 1a (deg) | 141 / 39 | 8.32 | 7.23 | 0.34 [0.26, 0.43] | 0.003 | no |
| torso orientation, stage 1 (deg) | 141 / 39 | 12.66 | 5.99 | 0.19 [0.12, 0.26] | 2.5e-09 | yes |
| torso orientation, stage 2 (deg) | 141 / 39 | 10.06 | 5.11 | 0.21 [0.15, 0.28] | 3.5e-08 | yes |
| object pushed, first contact -> close (cm) | 141 / 39 | 1.65 | 4.99 | 0.86 [0.80, 0.92] | 4.0e-12 | yes |
| target staleness at P (cm) | 141 / 39 | 2.51 | 6.88 | 0.83 [0.75, 0.89] | 3.2e-10 | yes |
| target staleness at close (cm) | 141 / 39 | 2.73 | 6.81 | 0.87 [0.80, 0.92] | 2.7e-12 | yes |
| wrist position, stage 2 at close (cm) | 141 / 39 | 4.50 | 4.27 | 0.44 [0.33, 0.54] | 0.216 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 141 / 39 | 5.67 | 5.58 | 0.40 [0.32, 0.49] | 0.066 | no |

#### box · moving-walk

180 episodes (exp1-wide-long-move003, exp1-wide-long-move010, exp1-wide-move003, exp1-wide-move010, exp1-wide-xlong-move003, exp1-wide-xlong-move010); first-attempt success 6/180 [0.02, 0.07]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 23, touched before closing 6 / 80, closed first 0 / 71

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 104.366, 223.395], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 5 / 160 | 2.00 | 2.29 | 0.62 [0.36, 0.87] | 0.369 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 6 / 174 | 0.00 | 0.00 | 0.42 [0.23, 0.59] | 0.486 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 6 / 174 | 6.69 | 6.53 | 0.44 [0.31, 0.57] | 0.635 | no |
| wrist position, closed-loop correction in force (cm) | 6 / 174 | 0.00 | 2.32 | 0.71 [0.56, 0.80] | 0.072 | no |
| wrist position, task target -> robot (cm) | 6 / 174 | 14.09 | 9.23 | 0.37 [0.21, 0.52] | 0.270 | no |
| wrist orientation, 1a (deg) | 5 / 160 | 10.38 | 10.70 | 0.54 [0.43, 0.64] | 0.772 | no |
| wrist orientation, stage 1 (deg) | 6 / 174 | 0.00 | 0.00 | 0.38 [0.19, 0.58] | 0.313 | no |
| wrist orientation, stage 2 (deg) | 6 / 174 | 9.51 | 7.19 | 0.37 [0.24, 0.50] | 0.280 | no |
| root position, 1a (cm) | 5 / 160 | 0.85 | 1.17 | 0.64 [0.45, 0.84] | 0.289 | no |
| root position, stage 1 (cm) | 6 / 174 | 0.00 | 0.00 | 0.45 [0.28, 0.55] | 0.439 | no |
| root position, stage 2 (cm) | 6 / 174 | 4.09 | 2.81 | 0.36 [0.20, 0.58] | 0.243 | no |
| torso orientation, 1a (deg) | 5 / 160 | 6.72 | 8.59 | 0.80 [0.67, 0.91] | 0.023 | no |
| torso orientation, stage 1 (deg) | 6 / 174 | 7.62 | 13.05 | 0.83 [0.73, 0.91] | 0.006 | yes |
| torso orientation, stage 2 (deg) | 6 / 174 | 5.43 | 10.42 | 0.88 [0.74, 0.98] | 0.002 | yes |
| walk, Kimodo root error per plan (stage 1) (cm) | 6 / 174 | 7.45 | 7.30 | 0.43 [0.22, 0.65] | 0.547 | no |
| walk, displacement error per plan (stage 2) (cm) | 6 / 174 | 10.25 | 11.73 | 0.58 [0.27, 0.86] | 0.513 | no |
| walk, robot travel / reference travel | 6 / 174 | 0.80 | 0.81 | 0.56 [0.37, 0.76] | 0.599 | no |
| walk, stance vs walk goal (cm) | 6 / 174 | 4.48 | 4.79 | 0.52 [0.31, 0.72] | 0.867 | no |
| object pushed, first contact -> close (cm) | 6 / 174 | 7.98 | 0.05 | 0.31 [0.22, 0.39] | 0.093 | no |
| target staleness at P (cm) | 6 / 174 | 11.21 | 40.37 | 0.94 [0.89, 0.98] | 2.6e-04 | yes |
| target staleness at close (cm) | 6 / 151 | 8.02 | 39.97 | 0.95 [0.90, 0.99] | 1.8e-04 | yes |
| wrist position, stage 2 at close (cm) | 6 / 151 | 4.99 | 4.55 | 0.46 [0.28, 0.64] | 0.724 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 6 / 174 | 5.84 | 5.45 | 0.43 [0.32, 0.56] | 0.580 | no |

#### box · static-stand

220 episodes (exp1-long-static, exp1-map, exp1-region, exp1-stance, exp1-xlong-static); first-attempt success 137/220 [0.56, 0.68]; verdict: **stage 2 separates (tracker side)**

first attempt, success / failure: never closed 0 / 58, touched before closing 79 / 24, closed first 58 / 1

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 7.457, 112.68], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 137 / 83 | 1.88 | 2.61 | 0.83 [0.77, 0.89] | 1.8e-16 | yes |
| wrist position, stage 1 (command -> consumed reference) (cm) | 137 / 83 | 0.00 | 0.00 | 0.42 [0.33, 0.50] | 0.037 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 137 / 83 | 4.74 | 20.86 | 0.87 [0.81, 0.92] | 6.1e-20 | yes |
| wrist position, closed-loop correction in force (cm) | 137 / 83 | 2.72 | 12.00 | 0.77 [0.69, 0.84] | 1.5e-11 | yes |
| wrist position, task target -> robot (cm) | 137 / 83 | 2.44 | 9.88 | 0.94 [0.90, 0.97] | 5.3e-28 | yes |
| wrist orientation, 1a (deg) | 137 / 83 | 11.07 | 16.81 | 0.78 [0.69, 0.85] | 7.8e-12 | yes |
| wrist orientation, stage 1 (deg) | 137 / 83 | 0.00 | 0.00 | 0.44 [0.35, 0.53] | 0.139 | no |
| wrist orientation, stage 2 (deg) | 137 / 83 | 5.30 | 10.32 | 0.79 [0.72, 0.84] | 1.4e-12 | yes |
| root position, 1a (cm) | 137 / 83 | 1.30 | 1.38 | 0.56 [0.48, 0.64] | 0.136 | no |
| root position, stage 1 (cm) | 137 / 83 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 137 / 83 | 2.96 | 16.69 | 0.84 [0.77, 0.90] | 5.3e-17 | yes |
| torso orientation, 1a (deg) | 137 / 83 | 8.52 | 15.30 | 0.84 [0.78, 0.90] | 1.7e-17 | yes |
| torso orientation, stage 1 (deg) | 137 / 83 | 13.35 | 29.38 | 0.80 [0.73, 0.87] | 5.4e-14 | yes |
| torso orientation, stage 2 (deg) | 137 / 83 | 11.09 | 24.71 | 0.79 [0.71, 0.86] | 5.7e-13 | yes |
| object pushed, first contact -> close (cm) | 137 / 83 | 0.20 | 0.00 | 0.44 [0.37, 0.51] | 0.105 | no |
| target staleness at P (cm) | 137 / 83 | 0.00 | 0.00 | 0.50 [0.42, 0.58] | 0.976 | no |
| target staleness at close (cm) | 137 / 25 | 0.42 | 2.28 | 0.82 [0.74, 0.89] | 4.7e-07 | yes |
| wrist position, stage 2 at close (cm) | 137 / 25 | 3.97 | 4.44 | 0.56 [0.42, 0.70] | 0.383 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 137 / 83 | 5.19 | 20.87 | 0.83 [0.77, 0.89] | 9.4e-17 | yes |

#### box · static-walk

120 episodes (exp1-static, exp1-wide-long-static, exp1-wide-static, exp1-wide-xlong-static); first-attempt success 86/120 [0.63, 0.79]; verdict: **stage 2 separates (tracker side)**

first attempt, success / failure: never closed 0 / 22, touched before closing 55 / 8, closed first 31 / 4

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.0, 54.166, 197.264], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 84 / 33 | 1.89 | 3.06 | 0.87 [0.78, 0.94] | 4.3e-10 | yes |
| wrist position, stage 1 (command -> consumed reference) (cm) | 86 / 34 | 0.00 | 0.00 | 0.58 [0.44, 0.72] | 0.153 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 86 / 34 | 5.37 | 16.50 | 0.83 [0.72, 0.92] | 2.1e-08 | yes |
| wrist position, closed-loop correction in force (cm) | 86 / 34 | 3.19 | 12.00 | 0.69 [0.54, 0.82] | 0.001 | no |
| wrist position, task target -> robot (cm) | 86 / 34 | 2.74 | 8.91 | 0.88 [0.80, 0.94] | 1.4e-10 | yes |
| wrist orientation, 1a (deg) | 84 / 33 | 11.69 | 14.09 | 0.75 [0.62, 0.85] | 3.4e-05 | yes |
| wrist orientation, stage 1 (deg) | 86 / 34 | 0.00 | 0.00 | 0.58 [0.45, 0.71] | 0.170 | no |
| wrist orientation, stage 2 (deg) | 86 / 34 | 4.75 | 12.20 | 0.76 [0.65, 0.87] | 8.7e-06 | yes |
| root position, 1a (cm) | 84 / 33 | 1.51 | 1.45 | 0.49 [0.37, 0.63] | 0.920 | no |
| root position, stage 1 (cm) | 86 / 34 | 0.00 | 0.00 | 0.50 [0.48, 0.54] | 0.838 | no |
| root position, stage 2 (cm) | 86 / 34 | 3.60 | 7.48 | 0.83 [0.72, 0.91] | 2.8e-08 | yes |
| torso orientation, 1a (deg) | 84 / 33 | 8.72 | 13.09 | 0.75 [0.63, 0.87] | 2.7e-05 | yes |
| torso orientation, stage 1 (deg) | 86 / 34 | 13.80 | 20.57 | 0.68 [0.54, 0.82] | 0.002 | no |
| torso orientation, stage 2 (deg) | 86 / 34 | 11.92 | 21.06 | 0.75 [0.63, 0.86] | 2.8e-05 | yes |
| walk, Kimodo root error per plan (stage 1) (cm) | 86 / 34 | 7.39 | 7.70 | 0.57 [0.45, 0.69] | 0.231 | no |
| walk, displacement error per plan (stage 2) (cm) | 86 / 34 | 12.20 | 12.60 | 0.58 [0.45, 0.70] | 0.185 | no |
| walk, robot travel / reference travel | 86 / 34 | 0.80 | 0.71 | 0.36 [0.24, 0.48] | 0.016 | no |
| walk, stance vs walk goal (cm) | 86 / 34 | 4.61 | 6.98 | 0.70 [0.58, 0.81] | 6.4e-04 | yes |
| object pushed, first contact -> close (cm) | 86 / 34 | 0.81 | 0.00 | 0.39 [0.28, 0.51] | 0.058 | no |
| target staleness at P (cm) | 86 / 34 | 0.00 | 0.00 | 0.65 [0.53, 0.76] | 0.011 | no |
| target staleness at close (cm) | 86 / 12 | 0.89 | 2.24 | 0.70 [0.49, 0.87] | 0.029 | no |
| wrist position, stage 2 at close (cm) | 86 / 12 | 4.50 | 5.11 | 0.59 [0.40, 0.78] | 0.296 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 86 / 34 | 6.03 | 17.77 | 0.87 [0.78, 0.94] | 4.2e-10 | yes |

#### box-latency · moving-stand

694 episodes (exp4-anchored-d025-v003, exp4-anchored-d025-v010, exp4-anchored-d075-v003, exp4-anchored-d075-v010, exp4-anchored-d194-v003, exp4-anchored-d194-v010, exp4-anchored-d383-v003, exp4-anchored-d383-v010, exp4-anchored-d620-v003, exp4-anchored-d620-v010, exp4-anchored-jgem-v010, exp4-blocking-d620-v003, exp4-blocking-d620-v010, exp4-fixed-d025-v003, exp4-fixed-d025-v010, exp4-fixed-d075-v003, exp4-fixed-d075-v010, exp4-fixed-d194-v003, exp4-fixed-d194-v010, exp4-fixed-d383-v003, exp4-fixed-d383-v010, exp4-fixed-d620-v003, exp4-fixed-d620-v010, exp4-fixed-jgem-v010); first-attempt success 621/694 [0.87, 0.92]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 2, touched before closing 528 / 65, closed first 93 / 6

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 14.999, 50.28], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.688]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 621 / 73 | 2.01 | 2.15 | 0.58 [0.51, 0.65] | 0.022 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 621 / 73 | 0.00 | 0.00 | 0.59 [0.52, 0.65] | 0.012 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 621 / 73 | 5.00 | 4.38 | 0.38 [0.31, 0.45] | 8.7e-04 | no |
| wrist position, closed-loop correction in force (cm) | 621 / 73 | 2.57 | 0.00 | 0.34 [0.29, 0.40] | 3.6e-06 | no |
| wrist position, task target -> robot (cm) | 621 / 73 | 3.79 | 4.38 | 0.60 [0.53, 0.67] | 0.006 | no |
| wrist orientation, 1a (deg) | 621 / 73 | 10.36 | 10.11 | 0.45 [0.38, 0.53] | 0.201 | no |
| wrist orientation, stage 1 (deg) | 621 / 73 | 0.00 | 0.00 | 0.63 [0.57, 0.70] | 1.7e-04 | no |
| wrist orientation, stage 2 (deg) | 621 / 73 | 6.74 | 8.10 | 0.65 [0.59, 0.72] | 1.7e-05 | no |
| root position, 1a (cm) | 621 / 73 | 1.25 | 1.01 | 0.34 [0.28, 0.41] | 1.0e-05 | no |
| root position, stage 1 (cm) | 621 / 73 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 621 / 73 | 3.33 | 3.16 | 0.41 [0.34, 0.47] | 0.008 | no |
| torso orientation, 1a (deg) | 621 / 73 | 7.96 | 7.53 | 0.43 [0.37, 0.50] | 0.067 | no |
| torso orientation, stage 1 (deg) | 621 / 73 | 11.52 | 6.85 | 0.33 [0.26, 0.39] | 1.2e-06 | no |
| torso orientation, stage 2 (deg) | 621 / 73 | 9.66 | 5.72 | 0.32 [0.26, 0.39] | 4.7e-07 | no |
| object pushed, first contact -> close (cm) | 621 / 73 | 1.77 | 3.01 | 0.72 [0.66, 0.79] | 3.0e-10 | yes |
| target staleness at P (cm) | 621 / 73 | 1.93 | 2.80 | 0.59 [0.51, 0.66] | 0.015 | no |
| target staleness at close (cm) | 621 / 71 | 2.18 | 3.68 | 0.76 [0.70, 0.81] | 1.3e-12 | yes |
| wrist position, stage 2 at close (cm) | 621 / 71 | 4.36 | 3.80 | 0.39 [0.32, 0.47] | 0.003 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 621 / 73 | 5.62 | 5.30 | 0.36 [0.29, 0.42] | 5.0e-05 | no |

#### box-latency · static-stand

299 episodes (exp4-anchored-d025-v000, exp4-anchored-d075-v000, exp4-anchored-d194-v000, exp4-anchored-d383-v000, exp4-anchored-d620-v000, exp4-blocking-d620-v000, exp4-fixed-d075-v000, exp4-fixed-d194-v000, exp4-fixed-d383-v000, exp4-fixed-d620-v000); first-attempt success 273/299 [0.88, 0.94]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 0, touched before closing 170 / 26, closed first 103 / 0

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 0.15, 7.586], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.004, 0.723]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 273 / 26 | 1.80 | 2.20 | 0.74 [0.62, 0.83] | 7.0e-05 | yes |
| wrist position, stage 1 (command -> consumed reference) (cm) | 273 / 26 | 0.00 | 0.00 | 0.69 [0.57, 0.81] | 0.001 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 273 / 26 | 4.94 | 4.46 | 0.46 [0.33, 0.61] | 0.552 | no |
| wrist position, closed-loop correction in force (cm) | 273 / 26 | 2.93 | 0.00 | 0.29 [0.18, 0.42] | 4.1e-04 | yes |
| wrist position, task target -> robot (cm) | 273 / 26 | 2.52 | 4.35 | 0.83 [0.77, 0.88] | 2.6e-08 | yes |
| wrist orientation, 1a (deg) | 273 / 26 | 10.93 | 9.76 | 0.36 [0.23, 0.50] | 0.018 | no |
| wrist orientation, stage 1 (deg) | 273 / 26 | 0.00 | 0.00 | 0.72 [0.60, 0.84] | 2.0e-04 | yes |
| wrist orientation, stage 2 (deg) | 273 / 26 | 3.93 | 6.60 | 0.76 [0.67, 0.84] | 1.1e-05 | yes |
| root position, 1a (cm) | 273 / 26 | 1.35 | 0.95 | 0.22 [0.13, 0.32] | 2.6e-06 | yes |
| root position, stage 1 (cm) | 273 / 26 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 273 / 26 | 3.19 | 2.95 | 0.44 [0.33, 0.56] | 0.327 | no |
| torso orientation, 1a (deg) | 273 / 26 | 8.25 | 8.54 | 0.46 [0.36, 0.56] | 0.502 | no |
| torso orientation, stage 1 (deg) | 273 / 26 | 13.72 | 7.24 | 0.27 [0.17, 0.39] | 1.4e-04 | yes |
| torso orientation, stage 2 (deg) | 273 / 26 | 11.49 | 6.20 | 0.28 [0.17, 0.40] | 1.9e-04 | yes |
| object pushed, first contact -> close (cm) | 273 / 26 | 0.64 | 2.24 | 0.86 [0.79, 0.92] | 8.0e-10 | yes |
| target staleness at P (cm) | 273 / 26 | 0.00 | 0.00 | 0.65 [0.52, 0.77] | 0.013 | no |
| target staleness at close (cm) | 273 / 26 | 0.67 | 2.27 | 0.85 [0.78, 0.91] | 4.9e-09 | yes |
| wrist position, stage 2 at close (cm) | 273 / 26 | 4.18 | 3.98 | 0.43 [0.32, 0.56] | 0.256 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 273 / 26 | 5.45 | 5.00 | 0.42 [0.31, 0.53] | 0.160 | no |

#### hammer · moving-stand

80 episodes (exp1b-move003-d194, exp1b-move003-d75, exp1b-move010-d194, exp1b-move010-d75); first-attempt success 17/80 [0.14, 0.31]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 0, touched before closing 14 / 9, closed first 3 / 54

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 0.002, 0.002], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 17 / 63 | 2.05 | 2.30 | 0.57 [0.43, 0.70] | 0.410 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 17 / 63 | 0.00 | 0.00 | 0.60 [0.46, 0.74] | 0.217 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 17 / 63 | 1.78 | 2.20 | 0.70 [0.55, 0.83] | 0.014 | no |
| wrist position, closed-loop correction in force (cm) | 17 / 63 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| wrist position, task target -> robot (cm) | 17 / 63 | 1.89 | 2.28 | 0.63 [0.49, 0.77] | 0.091 | no |
| wrist orientation, 1a (deg) | 17 / 63 | 5.63 | 6.83 | 0.75 [0.62, 0.87] | 0.001 | yes |
| wrist orientation, stage 1 (deg) | 17 / 63 | 0.00 | 0.00 | 0.63 [0.48, 0.76] | 0.112 | no |
| wrist orientation, stage 2 (deg) | 17 / 63 | 4.82 | 4.55 | 0.50 [0.35, 0.65] | 0.986 | no |
| root position, 1a (cm) | 17 / 63 | 0.78 | 0.75 | 0.43 [0.26, 0.62] | 0.397 | no |
| root position, stage 1 (cm) | 17 / 63 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 17 / 63 | 1.39 | 1.52 | 0.66 [0.53, 0.77] | 0.049 | no |
| torso orientation, 1a (deg) | 17 / 63 | 7.57 | 7.44 | 0.41 [0.27, 0.56] | 0.274 | no |
| torso orientation, stage 1 (deg) | 17 / 63 | 12.00 | 12.17 | 0.58 [0.42, 0.74] | 0.317 | no |
| torso orientation, stage 2 (deg) | 17 / 63 | 7.87 | 8.73 | 0.55 [0.38, 0.71] | 0.506 | no |
| object pushed, first contact -> close (cm) | 17 / 63 | 0.01 | 0.00 | 0.16 [0.07, 0.27] | 1.3e-07 | yes |
| target staleness at P (cm) | 17 / 63 | 11.27 | 39.23 | 0.97 [0.92, 1.00] | 4.4e-09 | yes |
| target staleness at close (cm) | 17 / 63 | 11.33 | 39.43 | 0.96 [0.91, 1.00] | 7.7e-09 | yes |
| wrist position, stage 2 at close (cm) | 17 / 63 | 1.88 | 2.24 | 0.65 [0.50, 0.78] | 0.064 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 17 / 63 | 2.13 | 2.15 | 0.49 [0.35, 0.63] | 0.874 | no |

#### hammer · static-stand

118 episodes (exp1b-map, exp1b-region); first-attempt success 83/118 [0.62, 0.78]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 29, touched before closing 31 / 2, closed first 52 / 4

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 0.002, 0.004], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 83 / 35 | 2.22 | 2.47 | 0.60 [0.47, 0.72] | 0.091 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 83 / 35 | 0.00 | 0.00 | 0.49 [0.35, 0.63] | 0.869 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 83 / 35 | 1.99 | 2.31 | 0.64 [0.52, 0.75] | 0.018 | no |
| wrist position, closed-loop correction in force (cm) | 83 / 35 | 0.00 | 0.00 | 0.50 [0.45, 0.56] | 0.981 | no |
| wrist position, task target -> robot (cm) | 83 / 35 | 1.96 | 3.16 | 0.79 [0.68, 0.88] | 7.9e-07 | yes |
| wrist orientation, 1a (deg) | 83 / 35 | 6.75 | 10.68 | 0.86 [0.77, 0.94] | 5.7e-10 | yes |
| wrist orientation, stage 1 (deg) | 83 / 35 | 0.00 | 0.00 | 0.57 [0.43, 0.70] | 0.251 | no |
| wrist orientation, stage 2 (deg) | 83 / 35 | 6.52 | 24.28 | 0.94 [0.87, 0.99] | 6.1e-14 | yes |
| root position, 1a (cm) | 83 / 35 | 0.75 | 0.81 | 0.62 [0.48, 0.75] | 0.036 | no |
| root position, stage 1 (cm) | 83 / 35 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 83 / 35 | 1.27 | 1.89 | 0.74 [0.62, 0.85] | 3.4e-05 | yes |
| torso orientation, 1a (deg) | 83 / 35 | 7.58 | 8.67 | 0.77 [0.67, 0.85] | 4.4e-06 | yes |
| torso orientation, stage 1 (deg) | 83 / 35 | 11.99 | 11.28 | 0.43 [0.30, 0.57] | 0.251 | no |
| torso orientation, stage 2 (deg) | 83 / 35 | 8.51 | 5.85 | 0.36 [0.24, 0.50] | 0.021 | no |
| object pushed, first contact -> close (cm) | 83 / 35 | 0.00 | 0.00 | 0.65 [0.56, 0.75] | 0.004 | no |
| target staleness at P (cm) | 83 / 35 | 0.00 | 0.00 | 0.54 [0.44, 0.64] | 0.486 | no |
| target staleness at close (cm) | 83 / 6 | 0.00 | 0.00 | 0.51 [0.27, 0.77] | 0.919 | no |
| wrist position, stage 2 at close (cm) | 83 / 6 | 2.03 | 3.22 | 0.86 [0.74, 0.95] | 0.004 | yes |
| wrist position, stage 2, 0.5 s before P (cm) | 83 / 35 | 2.06 | 3.24 | 0.78 [0.68, 0.87] | 1.7e-06 | yes |

#### hammer · static-stand-injected

240 episodes (exp1b-inj-across+02, exp1b-inj-across+05, exp1b-inj-across-02, exp1b-inj-across-05, exp1b-inj-along+05, exp1b-inj-along+10, exp1b-inj-along-05, exp1b-inj-along-10, exp1b-inj-turn+20, exp1b-inj-turn+40, exp1b-inj-turn-20, exp1b-inj-turn-40); first-attempt success 181/240 [0.70, 0.80]; verdict: **neither stage separates**

first attempt, success / failure: never closed 0 / 3, touched before closing 97 / 52, closed first 84 / 4

checks (P, mm, median / 95th percentile / max): stage 1 wrist position [0.001, 0.002, 0.002], hand-off [0.0, 0.0, 0.0], replayed vs logged correction [0.0, 0.0, 0.0]

| quantity | n succ / fail | median succ | median fail | AUC [95% CI] | p | separates |
|---|---|---|---|---|---|---|
| wrist position, Kimodo vs command (1a) (cm) | 181 / 59 | 2.14 | 2.40 | 0.57 [0.48, 0.67] | 0.084 | no |
| wrist position, stage 1 (command -> consumed reference) (cm) | 181 / 59 | 0.00 | 0.00 | 0.44 [0.35, 0.53] | 0.146 | no |
| wrist position, stage 2 (reference -> robot) (cm) | 181 / 59 | 1.89 | 2.08 | 0.64 [0.55, 0.72] | 0.002 | no |
| wrist position, closed-loop correction in force (cm) | 181 / 59 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| wrist position, task target -> robot (cm) | 181 / 59 | 1.96 | 2.76 | 0.83 [0.77, 0.88] | 6.7e-14 | yes |
| wrist orientation, 1a (deg) | 181 / 59 | 6.34 | 5.56 | 0.45 [0.36, 0.54] | 0.242 | no |
| wrist orientation, stage 1 (deg) | 181 / 59 | 0.00 | 0.00 | 0.47 [0.39, 0.55] | 0.500 | no |
| wrist orientation, stage 2 (deg) | 181 / 59 | 5.12 | 5.87 | 0.58 [0.49, 0.67] | 0.053 | no |
| root position, 1a (cm) | 181 / 59 | 0.76 | 0.72 | 0.43 [0.32, 0.53] | 0.090 | no |
| root position, stage 1 (cm) | 181 / 59 | 0.00 | 0.00 | 0.50 [0.50, 0.50] | 1.000 | no |
| root position, stage 2 (cm) | 181 / 59 | 1.46 | 1.40 | 0.49 [0.39, 0.59] | 0.822 | no |
| torso orientation, 1a (deg) | 181 / 59 | 7.56 | 7.47 | 0.43 [0.35, 0.52] | 0.103 | no |
| torso orientation, stage 1 (deg) | 181 / 59 | 12.03 | 11.59 | 0.41 [0.33, 0.49] | 0.038 | no |
| torso orientation, stage 2 (deg) | 181 / 59 | 8.40 | 8.66 | 0.47 [0.38, 0.56] | 0.563 | no |
| object pushed, first contact -> close (cm) | 181 / 59 | 0.00 | 0.10 | 0.68 [0.62, 0.75] | 1.2e-05 | no |
| target staleness at P (cm) | 181 / 59 | 1.98 | 4.97 | 0.55 [0.47, 0.64] | 0.244 | no |
| target staleness at close (cm) | 181 / 56 | 1.98 | 4.72 | 0.47 [0.38, 0.56] | 0.535 | no |
| wrist position, stage 2 at close (cm) | 181 / 56 | 1.91 | 2.42 | 0.69 [0.60, 0.75] | 2.8e-05 | no |
| wrist position, stage 2, 0.5 s before P (cm) | 181 / 59 | 2.10 | 2.47 | 0.62 [0.53, 0.71] | 0.006 | no |


</details>

<details><summary>逐格的表（stage_split.py cells 的输出，原样；中位数 成功 / 失败）</summary>

| family · condition | cell | n | 1st success | failures: never closed / touched first / closed first | wrist stage 2 (cm) | root stage 2 (cm) | wrist orient. stage 2 (deg) | staleness at close (cm) | pushed (cm) |
|---|---|---|---|---|---|---|---|---|---|
| box · moving-stand | exp1-long-move003 | 30 | 27 | 0 / 3 / 0 | 5.0 / 3.9 | 3.4 / 3.4 | 4.2 / 9.8 | 2.3 / 3.1 | 1.1 / 2.6 |
| box · moving-stand | exp1-long-move010 | 30 | 26 | 0 / 4 / 0 | 5.0 / 4.3 | 3.3 / 3.6 | 4.3 / 8.1 | 1.3 / 3.2 | 1.1 / 2.7 |
| box · moving-stand | exp1-move003 | 30 | 20 | 0 / 10 / 0 | 5.3 / 5.1 | 3.3 / 3.5 | 7.4 / 12.2 | 5.0 / 7.8 | 3.3 / 5.4 |
| box · moving-stand | exp1-move010 | 30 | 23 | 0 / 7 / 0 | 5.0 / 4.9 | 3.3 / 3.2 | 5.4 / 12.9 | 3.4 / 6.9 | 2.3 / 5.3 |
| box · moving-stand | exp1-xlong-move003 | 30 | 23 | 0 / 7 / 0 | 5.4 / 5.0 | 3.5 / 3.7 | 6.4 / 12.7 | 5.1 / 7.8 | 3.3 / 5.0 |
| box · moving-stand | exp1-xlong-move010 | 30 | 22 | 0 / 8 / 0 | 5.0 / 5.2 | 3.3 / 3.6 | 4.0 / 11.9 | 2.1 / 6.5 | 0.9 / 5.5 |
| box · moving-walk | exp1-wide-long-move003 | 30 | 1 | 4 / 19 / 6 | 6.5 / 7.2 | 3.6 / 3.1 | 8.2 / 11.6 | 2.2 / 37.6 | 2.0 / 3.5 |
| box · moving-walk | exp1-wide-long-move010 | 30 | 1 | 3 / 10 / 16 | 6.6 / 6.0 | 4.6 / 2.6 | 13.1 / 6.3 | 6.5 / 67.2 | 6.4 / 0.0 |
| box · moving-walk | exp1-wide-move003 | 30 | 2 | 3 / 22 / 3 | 7.0 / 6.9 | 5.5 / 2.3 | 8.0 / 10.7 | 17.6 / 34.1 | 13.0 / 18.0 |
| box · moving-walk | exp1-wide-move010 | 30 | 1 | 3 / 11 / 15 | 6.8 / 4.5 | 4.9 / 2.4 | 9.2 / 6.2 | 5.7 / 47.4 | 2.8 / 0.0 |
| box · moving-walk | exp1-wide-xlong-move003 | 30 | 1 | 5 / 9 / 15 | 7.3 / 4.9 | 3.3 / 3.0 | 19.1 / 5.3 | 9.6 / 49.0 | 9.5 / 0.0 |
| box · moving-walk | exp1-wide-xlong-move010 | 30 | 0 | 5 / 9 / 16 | — / 6.1 | — / 3.2 | — / 5.4 | — / 78.3 | — / 0.0 |
| box · static-stand | exp1-long-static | 30 | 27 | 0 / 3 / 0 | 5.3 / 6.8 | 3.2 / 3.3 | 3.5 / 4.7 | 0.8 / 2.1 | 0.8 / 2.1 |
| box · static-stand | exp1-map | 100 | 36 | 50 / 13 / 1 | 4.2 / 22.7 | 2.8 / 18.2 | 7.1 / 15.2 | 0.3 / 2.4 | 0.0 / 0.0 |
| box · static-stand | exp1-region | 30 | 26 | 0 / 4 / 0 | 4.7 / 4.9 | 3.0 / 3.0 | 4.4 / 6.2 | 0.6 / 2.2 | 0.6 / 2.2 |
| box · static-stand | exp1-stance | 30 | 21 | 8 / 1 / 0 | 3.5 / 22.9 | 2.0 / 18.1 | 6.2 / 4.8 | 0.8 / 1.7 | 0.8 / 0.0 |
| box · static-stand | exp1-xlong-static | 30 | 27 | 0 / 3 / 0 | 5.1 / 4.4 | 3.3 / 3.0 | 3.6 / 7.5 | 0.7 / 2.3 | 0.6 / 2.3 |
| box · static-walk | exp1-static | 30 | 16 | 10 / 2 / 2 | 7.5 / 16.8 | 4.8 / 7.1 | 6.2 / 13.8 | 0.2 / 0.9 | 0.0 / 0.0 |
| box · static-walk | exp1-wide-long-static | 30 | 25 | 3 / 1 / 1 | 5.1 / 20.1 | 3.6 / 16.0 | 4.7 / 12.2 | 1.0 / 1.1 | 1.0 / 0.0 |
| box · static-walk | exp1-wide-static | 30 | 25 | 4 / 1 / 0 | 5.0 / 12.7 | 3.3 / 8.5 | 3.7 / 20.0 | 1.0 / 4.7 | 1.0 / 0.1 |
| box · static-walk | exp1-wide-xlong-static | 30 | 20 | 5 / 4 / 1 | 5.3 / 11.7 | 3.5 / 6.9 | 4.4 / 6.5 | 0.9 / 2.5 | 0.8 / 0.1 |
| box-latency · moving-stand | exp4-anchored-d025-v003 | 30 | 27 | 0 / 3 / 0 | 4.5 / 4.2 | 2.9 / 3.2 | 6.0 / 7.1 | 1.2 / 2.1 | 0.7 / 2.3 |
| box-latency · moving-stand | exp4-anchored-d025-v010 | 30 | 30 | 0 / 0 / 0 | 4.6 / — | 3.4 / — | 8.1 / — | 1.2 / — | 1.6 / — |
| box-latency · moving-stand | exp4-anchored-d075-v003 | 30 | 28 | 0 / 1 / 1 | 4.8 / 3.7 | 3.0 / 2.8 | 5.6 / 6.1 | 1.2 / 1.6 | 1.0 / 1.1 |
| box-latency · moving-stand | exp4-anchored-d075-v010 | 30 | 27 | 0 / 3 / 0 | 4.6 / 3.5 | 3.2 / 2.9 | 7.4 / 9.1 | 1.6 / 2.8 | 1.3 / 2.9 |
| box-latency · moving-stand | exp4-anchored-d194-v003 | 30 | 29 | 0 / 1 / 0 | 5.3 / 4.3 | 3.3 / 3.4 | 4.4 / 5.4 | 1.3 / 2.8 | 1.0 / 2.8 |
| box-latency · moving-stand | exp4-anchored-d194-v010 | 30 | 28 | 0 / 2 / 0 | 5.0 / 5.1 | 3.6 / 3.1 | 8.0 / 7.6 | 1.3 / 2.3 | 1.9 / 2.6 |
| box-latency · moving-stand | exp4-anchored-d383-v003 | 30 | 28 | 0 / 2 / 0 | 5.0 / 5.2 | 3.2 / 3.0 | 4.8 / 8.7 | 1.2 / 2.5 | 0.8 / 2.8 |
| box-latency · moving-stand | exp4-anchored-d383-v010 | 30 | 29 | 0 / 1 / 0 | 4.8 / 5.1 | 3.4 / 2.5 | 8.3 / 6.9 | 1.6 / 2.8 | 1.9 / 3.1 |
| box-latency · moving-stand | exp4-anchored-d620-v003 | 30 | 25 | 0 / 4 / 1 | 5.0 / 5.3 | 3.3 / 2.8 | 4.6 / 6.8 | 1.2 / 3.1 | 0.7 / 3.0 |
| box-latency · moving-stand | exp4-anchored-d620-v010 | 30 | 29 | 0 / 1 / 0 | 5.1 / 3.2 | 3.7 / 2.8 | 6.6 / 10.2 | 1.8 / 2.0 | 1.4 / 2.4 |
| box-latency · moving-stand | exp4-anchored-jgem-v010 | 29 | 28 | 0 / 1 / 0 | 4.7 / 3.4 | 3.6 / 2.8 | 7.9 / 7.5 | 1.8 / 2.7 | 2.4 / 2.8 |
| box-latency · moving-stand | exp4-blocking-d620-v003 | 30 | 29 | 0 / 1 / 0 | 4.8 / 8.2 | 3.4 / 3.8 | 5.3 / 3.6 | 2.8 / 3.4 | 1.5 / 3.0 |
| box-latency · moving-stand | exp4-blocking-d620-v010 | 30 | 29 | 0 / 1 / 0 | 5.5 / 3.8 | 3.5 / 3.5 | 4.8 / 7.9 | 3.2 / 3.7 | 2.1 / 3.4 |
| box-latency · moving-stand | exp4-fixed-d025-v003 | 30 | 24 | 0 / 5 / 1 | 4.4 / 3.8 | 3.1 / 3.0 | 6.8 / 7.4 | 2.2 / 2.5 | 1.0 / 2.3 |
| box-latency · moving-stand | exp4-fixed-d025-v010 | 30 | 24 | 0 / 6 / 0 | 5.6 / 4.5 | 3.2 / 3.3 | 9.8 / 10.1 | 2.9 / 4.8 | 3.4 / 4.9 |
| box-latency · moving-stand | exp4-fixed-d075-v003 | 30 | 25 | 0 / 4 / 1 | 5.0 / 4.4 | 3.3 / 3.3 | 7.4 / 11.1 | 4.4 / 4.5 | 1.9 / 3.4 |
| box-latency · moving-stand | exp4-fixed-d075-v010 | 30 | 26 | 0 / 4 / 0 | 6.2 / 4.9 | 3.2 / 3.1 | 7.8 / 11.1 | 3.4 / 5.0 | 4.3 / 6.3 |
| box-latency · moving-stand | exp4-fixed-d194-v003 | 30 | 21 | 0 / 9 / 0 | 5.6 / 4.9 | 3.4 / 3.4 | 4.6 / 10.4 | 2.9 / 7.8 | 1.8 / 4.4 |
| box-latency · moving-stand | exp4-fixed-d194-v010 | 30 | 26 | 0 / 4 / 0 | 4.7 / 4.9 | 3.5 / 3.6 | 8.2 / 6.9 | 3.1 / 4.4 | 2.4 / 3.7 |
| box-latency · moving-stand | exp4-fixed-d383-v003 | 30 | 25 | 0 / 3 / 2 | 5.2 / 3.7 | 3.6 / 3.2 | 6.4 / 6.2 | 3.4 / 3.8 | 1.6 / 2.5 |
| box-latency · moving-stand | exp4-fixed-d383-v010 | 30 | 26 | 0 / 4 / 0 | 5.9 / 6.5 | 3.4 / 3.3 | 6.7 / 7.7 | 2.8 / 3.9 | 2.9 / 4.9 |
| box-latency · moving-stand | exp4-fixed-d620-v003 | 30 | 30 | 0 / 0 / 0 | 4.7 / — | 3.3 / — | 4.7 / — | 2.3 / — | 1.8 / — |
| box-latency · moving-stand | exp4-fixed-d620-v010 | 30 | 27 | 0 / 3 / 0 | 5.4 / 5.2 | 3.7 / 3.5 | 8.2 / 13.2 | 3.5 / 5.3 | 3.1 / 4.5 |
| box-latency · moving-stand | exp4-fixed-jgem-v010 | 5 | 1 | 2 / 2 / 0 | 5.5 / 5.0 | 4.0 / 3.3 | 7.9 / 11.0 | 3.3 / 3.1 | 3.6 / 0.0 |
| box-latency · static-stand | exp4-anchored-d025-v000 | 30 | 26 | 0 / 4 / 0 | 4.4 / 3.7 | 2.8 / 3.0 | 5.5 / 6.6 | 0.6 / 2.0 | 0.6 / 2.0 |
| box-latency · static-stand | exp4-anchored-d075-v000 | 30 | 26 | 0 / 4 / 0 | 4.7 / 4.8 | 3.0 / 3.1 | 4.4 / 7.0 | 0.3 / 2.4 | 0.5 / 2.8 |
| box-latency · static-stand | exp4-anchored-d194-v000 | 30 | 26 | 0 / 4 / 0 | 5.1 / 5.6 | 3.3 / 3.2 | 3.4 / 6.2 | 0.2 / 2.3 | 0.3 / 2.3 |
| box-latency · static-stand | exp4-anchored-d383-v000 | 30 | 27 | 0 / 3 / 0 | 5.0 / 5.3 | 3.3 / 2.8 | 3.6 / 5.9 | 0.4 / 2.4 | 0.6 / 2.3 |
| box-latency · static-stand | exp4-anchored-d620-v000 | 30 | 29 | 0 / 1 / 0 | 5.1 / 3.9 | 3.3 / 2.7 | 3.8 / 10.4 | 0.6 / 2.6 | 0.6 / 3.4 |
| box-latency · static-stand | exp4-blocking-d620-v000 | 30 | 29 | 0 / 1 / 0 | 5.2 / 4.0 | 3.3 / 2.8 | 3.8 / 10.3 | 0.9 / 2.6 | 0.9 / 2.5 |
| box-latency · static-stand | exp4-fixed-d075-v000 | 30 | 26 | 0 / 4 / 0 | 4.7 / 4.9 | 3.0 / 3.0 | 4.4 / 6.2 | 0.6 / 2.2 | 0.6 / 2.2 |
| box-latency · static-stand | exp4-fixed-d194-v000 | 30 | 28 | 0 / 2 / 0 | 5.3 / 5.4 | 3.3 / 3.0 | 3.5 / 7.6 | 0.9 / 1.7 | 0.9 / 1.7 |
| box-latency · static-stand | exp4-fixed-d383-v000 | 30 | 27 | 0 / 3 / 0 | 5.2 / 4.4 | 3.3 / 3.0 | 3.6 / 7.5 | 0.7 / 2.3 | 0.6 / 2.3 |
| box-latency · static-stand | exp4-fixed-d620-v000 | 29 | 29 | 0 / 0 / 0 | 5.0 / — | 3.2 / — | 4.1 / — | 1.0 / — | 1.0 / — |
| hammer · moving-stand | exp1b-move003-d194 | 20 | 0 | 0 / 6 / 14 | — / 2.2 | — / 1.5 | — / 4.5 | — / 18.9 | — / 0.0 |
| hammer · moving-stand | exp1b-move003-d75 | 20 | 17 | 0 / 3 / 0 | 1.8 / 2.7 | 1.4 / 1.6 | 4.8 / 4.4 | 11.3 / 11.2 | 0.0 / 0.1 |
| hammer · moving-stand | exp1b-move010-d194 | 20 | 0 | 0 / 0 / 20 | — / 2.4 | — / 1.5 | — / 4.8 | — / 50.0 | — / 0.0 |
| hammer · moving-stand | exp1b-move010-d75 | 20 | 0 | 0 / 0 / 20 | — / 2.1 | — / 1.5 | — / 5.1 | — / 39.6 | — / 0.0 |
| hammer · static-stand | exp1b-map | 88 | 54 | 29 / 2 / 3 | 1.9 / 2.3 | 1.3 / 1.9 | 7.7 / 24.3 | 0.0 / 0.0 | 0.0 / 0.0 |
| hammer · static-stand | exp1b-region | 30 | 29 | 0 / 0 / 1 | 2.1 / 3.3 | 1.3 / 1.7 | 5.4 / 10.9 | 0.0 / 0.0 | 0.0 / 0.0 |
| hammer · static-stand-injected | exp1b-inj-across+02 | 20 | 18 | 0 / 2 / 0 | 1.7 / 3.2 | 1.3 / 1.3 | 5.4 / 9.1 | 1.7 / 1.7 | 0.2 / 0.3 |
| hammer · static-stand-injected | exp1b-inj-across+05 | 20 | 0 | 1 / 19 / 0 | — / 2.0 | — / 1.4 | — / 5.5 | — / 5.0 | — / 0.0 |
| hammer · static-stand-injected | exp1b-inj-across-02 | 20 | 19 | 0 / 0 / 1 | 2.1 / 3.3 | 1.5 / 1.7 | 5.1 / 10.9 | 2.0 / 2.0 | 0.0 / 0.0 |
| hammer · static-stand-injected | exp1b-inj-across-05 | 20 | 20 | 0 / 0 / 0 | 2.1 / — | 1.5 / — | 5.1 / — | 5.0 / — | 0.0 / — |
| hammer · static-stand-injected | exp1b-inj-along+05 | 20 | 20 | 0 / 0 / 0 | 2.0 / — | 1.5 / — | 5.3 / — | 5.0 / — | 0.0 / — |
| hammer · static-stand-injected | exp1b-inj-along+10 | 20 | 20 | 0 / 0 / 0 | 1.9 / — | 1.5 / — | 5.4 / — | 10.0 / — | 0.0 / — |
| hammer · static-stand-injected | exp1b-inj-along-05 | 20 | 19 | 0 / 0 / 1 | 1.9 / 3.3 | 1.5 / 1.7 | 5.1 / 10.9 | 5.0 / 5.0 | 0.0 / 0.0 |
| hammer · static-stand-injected | exp1b-inj-along-10 | 20 | 10 | 0 / 10 / 0 | 1.7 / 2.1 | 1.5 / 1.1 | 6.4 / 4.2 | 10.0 / 10.0 | 0.1 / 0.1 |
| hammer · static-stand-injected | exp1b-inj-turn+20 | 20 | 19 | 0 / 0 / 1 | 1.8 / 3.4 | 1.4 / 1.7 | 4.9 / 11.1 | 0.1 / 0.2 | 0.0 / 0.0 |
| hammer · static-stand-injected | exp1b-inj-turn+40 | 20 | 14 | 1 / 4 / 1 | 1.7 / 2.4 | 1.4 / 1.4 | 5.8 / 6.1 | 0.3 / 0.3 | 0.2 / 0.1 |
| hammer · static-stand-injected | exp1b-inj-turn-20 | 20 | 17 | 0 / 3 / 0 | 1.8 / 1.9 | 1.4 / 1.1 | 5.5 / 12.2 | 0.2 / 0.2 | 0.2 / 0.2 |
| hammer · static-stand-injected | exp1b-inj-turn-40 | 20 | 5 | 1 / 14 / 0 | 1.7 / 1.9 | 1.1 / 1.5 | 3.0 / 6.3 | 0.2 / 0.1 | 0.2 / 0.1 |

</details>
