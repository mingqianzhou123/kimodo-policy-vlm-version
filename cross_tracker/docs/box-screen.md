# 箱子搬放：三组初筛（2026-11-14）

用户授权：一个任务，SONIC、TWIST2、SONIC+TWIST2，seed 42；分别提交 SenseCore 的三个单节点 8×A800 作业。此前三任务、两 seed 的矩阵暂停扩展，已有运行保留为历史预实验。

| run_id | 来源 | 训练示教 | 更新步数 | 计分 checkpoint |
| --- | --- | ---: | ---: | ---: |
| box-sonic-s42-20261114 | SONIC | 43 | 10,000 | 10,000 |
| box-twist2-s42-20261114 | TWIST2 | 43 | 10,000 | 10,000 |
| box-sonic-twist2-s42-20261114 | SONIC+TWIST2 | 86（每来源43） | 20,000 | 20,000 |

- 三个独立 ACP 作业，每个只运行表中一组、各8×A800；相互无依赖，可并行调度。每个作业失败即停、不自动重启，不影响另外两组。原串行作业已在训练开始前停止并归档；没有扩展到其他任务或 seed。
- 从同一官方 pt_419h checkpoint_1000000 微调 ControlNet 和手部头。冻结 Kimodo/DINO/语言模型，不复用其他任务的调参配置。
- 全局 batch=8，8卡各 micro=1，不累积；学习率不按卡数放大。BF16，头部相机，历史100帧、预测50帧。其余架构/损失沿用 arena.yaml。
- 单源每来源期望80,000个采样窗口；双源均匀采样两个来源，总160,000个窗口，每来源仍期望80,000。双源使用两倍计算量，这不是固定总数据量/计算量的来源多样性消融。
- 单源 warmup=333、双源666，按各自预算 cosine 衰减；单源5k/10k、双源10k/20k保存，初筛仅评最后一个，不按成功率挑 checkpoint。
- 数据池仍引用原始 balanced-v1.json，不重新抽样。configs/box-screen.json 的 active_tasks 只有 HOI_pp_box；tasks 等原字段仅描述冻结母清单。每组 selected-data.json 明确列出实际训练/验证名单。
- 每来源11条验证示教不参与训练；MimicLite、拳击和导航均不进入本轮训练。17条隔离异常不回流。原示教成功仍未获日志认证。
- 评测两个目标 tracker SONIC/TWIST2，共3×2=6项。每项固定场景1000–1019，每场景1回合；相同场景/模型重置seed，最大1450控制步，50Hz；模型FP32、10步扩散、执行15个30Hz动作帧、RTC=0。
- 评测在已核验的百度 Isaac 环境另行执行；SenseCore 本作业只训练，不声称其已配置 Isaac。一次seed与20场景只用于方向判断，尚不证明收敛或统计稳定性。

SenseCore 工作目录：`/data/NTU_slab/mingqianzhou/humanoid_vla/kimodo-policy-vlm-version/cross_tracker`；`humanoid_vla/cross_tracker` 为同一目录的入口链接，数据/权重以只读引用复用，源码不复制。配置、快照、转换数据、核验报告、日志和训练产物都在 cross_tracker 下。

提交前 CPU 核验：共享源码哈希一致；两来源108条（86训练+22验证）修正数据重新转换并逐条检查原视频首/中/末帧、历史/预测窗口与文件哈希；完整预训练权重/骨干哈希核验；训练命令参数及八个rank的采样独立性测试。真实8卡启动以排队后的运行日志为准。

提交状态与作业ID见 `runs/box-screen-setup-20261114/submission.json`。没有该回执或状态仍为preparing，不能视为已提交。
