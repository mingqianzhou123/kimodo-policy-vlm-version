# 前期实验

正式方法定下来之前做的诊断实验。计划是 `新方案/agentic vla 初步实验计划.md`（2026-11-23），判定线在开跑前写进计划，跑完不改。

一个实验一个文件夹，里面是它的记录（`README.md`）、只有它用的脚本和配置、以及这些脚本的测试。

| 实验 | 问的是什么 | 结果 | 按计划的判定 |
|---|---|---|---|
| [实验一](experiment1_blocking_grasp/README.md) | “跑完、停稳、再观察”的阻塞式做法，目标在动时是做不成还是只是慢。先过静态抓取门槛 | 走过去再抓 17 / 30 没过线；改成站着不挪步、箱子在 10 cm 区域内，静态 28 / 30 过线。箱子往复时要么第一轮抓到要么抓不到，合手时箱子偏离锁存位置超过约 3 cm 成功率开始掉，超过 5 cm 不到一半（[续](experiment1_blocking_grasp/stance-dynamic-20261125.md)） | 短延迟 0.03 m/s 问题 A 成立（勉强越线）；其余五个移动条件主要损失速度。只对厘米级位移成立 |
| [实验一（锤子）](experiment1b_hammer_grasp/README.md) | 抓箱子的失败机理哪些是通用的，哪些只是箱子的特性：换成 DoubleDesk 上右手单手握锤柄 | 进行中 | 静态区域验收 26 / 30 才过线 |
| [实验二](experiment2_reach_coverage/README.md) | 冻结 Kimodo 加冻结 SONIC 这个底座够不够用；Kimodo 的预测和实际差多少 | 到达 0/18，加手臂 IK 1/18；向前走到位 2/5，侧步和后退 0/10；走动时 1.0 s 预测偏 32 cm | 底座不可用；走动时不能对着预测推理 |
| [实验三](experiment3_pointing/README.md) | 没微调的 VLM 能否指准接触点和预抓取点，调用一次多久 | Qwen3-VL-4B 78.75%、1.37 s；GPT-6.2 Astra 100%、3.88 s；Gemini 4 Argon 92.5%、7.66 s | 线是 50%，三个都过，指点不是瓶颈 |

实验四、五还没开始，开始时各自新建文件夹。`experiments.json` 是进度页读的计划登记表（`agentic/scripts/overview.py`）。

## 什么放这里，什么不放

放这里：只为某一个前期实验写的入口脚本、分析脚本、它的目标和种子配置、它的记录、它的测试。

不放这里，留在共用位置：

- 策略和算法，在 `kimodo_vla/`（箱子抓取 `oracle/grasp_policy.py`，走动与到达 `oracle/locomotion.py`，指点 `vlm/pointing.py`、`backends.py`、`pointing_eval.py`）。
- 仿真进程里的钩子，在 `agentic/arena/`。
- 启动器 `agentic/scripts/run_oracle.py` 和它直接用到的文件，包括箱子抓取的参数 `agentic/configs/grasp-loop.json`（启动器和仿真钩子都读它）。

这里的脚本开头有一行把 `agentic/scripts` 加进搜索路径，用来找 `paths.py`。

## 怎么跑

```bash
# 在仓库根目录
pytest agentic/tests agentic/pilot_experiments -q
```

每个实验怎么复现写在它自己的记录末尾。
