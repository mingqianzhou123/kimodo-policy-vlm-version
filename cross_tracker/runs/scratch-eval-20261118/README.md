# Full-budget scratch final-checkpoint evaluation

最新：21:54发现r2已因GPU资源忙在16:42退出；5个有效结果已核验，独立r3已续跑。六组样本与CCT136750步状态见[最新核查](progress-2154.md)。以下为16:21启动时历史记录。

2026-11-18：用户要求评测 SenseCore 已完成的三组 scratch。SONIC 单源、TWIST2 单源、朴素双源各训练 200,000 步，内部 global_step 与最终权重张量均实际校验；CCT 尚在训练，不混入此批。

`verify_sensecore.py` 通过已授权 CCI 在 CPU 上读取三组 checkpoint，检查全部张量有限值和每个文件 SHA256。`sensecore-checkpoints.json` 保存原始训练启动、scratch 初始化、配置、代码和提交记录；`local-verification.json` 记录本机六文件哈希复验。三个权重各 366,949,667 字节，模型不上传 Git。

统一 3 个 group seed（0、1、2），每组 20 个 repeat。episode seed 使用官方拼写 `Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby` 的 SHA256 尾 4 字节正 31 位。三个模型分别接 SONIC、TWIST2，总计 360 回合；只用预定最终 200k，不挑 checkpoint 或场景。按场景轮换六组合，一项运行错误停止对应组合并保存错误，其他组合继续。

SONIC 沿用之前完整官方复现协议：官方 base_test YAML、m2 USD、每回合新仿真进程、FP32、10 diffusion、执行15模型帧、RTC0、original timing、确定性 Torch、实时历史、每步刷新录像、终止后录像10步，最多1450步。TWIST2 沿用已核验记录兼容协议：官方 TWIST2 YAML、默认机器人资产、CPU物理、原生38ms物理间隔、标称50Hz标签/策略接口、first_prediction 根锚点。TWIST2 同源闭环仍未通过，不能称作论文 TWIST2 baseline 复现；真实模型评测无真值历史或动作注入。三模型在同一目标 tracker 的配置完全一致。

开发机仅 T2M-PLUS `dv-e21ddf3a87`。GPU6 仿真、GPU7 策略。四组微调和 GPU2 原有任务继续运行。评测代码复制到独立 `code/`，资产复用现有缓存并记录哈希，不覆盖训练快照。正式结果保存于该快照的 `cross_tracker/runs/scratch200k-*/`，队列记录 `batch.json`；每回合保存命令、权重哈希、seed、HTTP和仿真日志、结果及录像。

当前状态：三个最终 checkpoint 已在 SenseCore CPU 实际读取并检查有限值、在本机与百度复验哈希、在 AFS 独立目录完成复制及 SHA256 核验。T2M-PLUS 修复后批次 `scratch-eval-20261118-r2` 已启动，PID 1343246；首个 SONIC→SONIC 正式回合完成：1450 步 timeout、success=false，1461 帧视频解码核验通过，`valid_for_success_rate=true`。队列自动进入 TWIST2→TWIST2，第二模型加载、重置及连续真实 infer HTTP 200 已核实。360 回合仍在运行，当前仅 1 个有效回合，尚无完整成功率。

首轮 `scratch-eval-20261118` 因隔离快照遗漏 `kimodo_vla/skeleton/assets` 中 `.p` 和 XML 资产，模型服务在加载前失败；六组合的启动错误保留，不计为任务失败或成功率分母。`prepare_r2.py` 保留原目录，在新目录复制并校验 142 个文件，补齐资产；模型、权重、场景、协议不变。R2 的真实推理 HTTP 200 和步数日志证明已越过该启动故障。

CCT 最后 CPU 查询实际为 97,100/200,000 步，只有至 80k 的完整阶段 checkpoint，未把中途权重混入本批。开发机四组微调继续运行。

原始远端启动回执已通过 Jupyter 下载为 `remote-startup-receipts.json`（29,803 字节，SHA256 `580035084da77fa5fff61dd89b1e0d9604db4e98c15c9a5663ac258d2a97bdaf`），包含实际配置、142文件快照清单、三个权重清单、运行中的 batch 快照和首回合完整 run.json。该回执时点 completed=1、failed=0；failed 指执行错误，不是任务成功率中的失败。

16:21 UI 最后核验：第二模型 TWIST2 连续 infer HTTP 200；四组微调 SONIC/TWIST2/mix/CCT 继续到17,850/18,200/17,450/12,250步。截图与结构化抄录保存于 `evaluation-running.png`、`running-ui-verification.json`，与下载的原始回执区分标记。
