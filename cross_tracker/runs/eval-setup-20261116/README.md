# T2M-PLUS 评测恢复（2026-11-16）

用户授权：把昨天 SenseCore 已完成的 checkpoint 移到 T2M-PLUS 评测，不启动新训练。

## 已核验

- 正确实例 `dv-e21ddf3a87` 可通过 WebIDE 登录，8×RTX5090。初次检查 GPU 0 有其他进程，未干预；GPU 1–7 当时空闲，启动评测前还需再查。
- 原 `/root/workspace/humanoidVLA/cross_tracker` 不存在；指定范围的 workspace/mnt 搜索未找到旧项目。现有目录是旧的其他 Kimodo 实验，保持原状，不能据此断言历史数据彻底丢失。
- 从 GitHub 新建独立 `/root/workspace/humanoidVLA`，实际 HEAD `e42bbdaee1cf1e633279743005222edbf5809ff6`。命令级清除失效的 localhost:7890 代理，未改全局代理配置。
- 独立 Python 3.11.15 环境已创建。Isaac Sim 5.0 固定依赖正在下载；启动回执见 `sim-install-start.json`，不能等同安装通过。
- 现有 IsaacLab HEAD 是固定的 `46dff135f44683f031edf346e544fcfd8456b2bb`；现有 Arena 版本不同，所需 `44a7222` 已尝试独立获取，但首次 fetch 因 GnuTLS/early EOF 失败，未 checkout。场景资产未在原预期目录找到，仍待恢复。
- 约14:12 UTC+8，SenseCore 控制台显示三个微调 baseline 成功，三个 scratch 和 CCT 正式作业仍为“启动中”。CCI 网页终端也只发现三个微调 baseline 的 checkpoint。
- 约14:15，CCI 网页终端重新计算三个最终 AFS 备份的权重和 config SHA256；六个哈希均与旧备份记录相同，见 `evaluation-plan.json`。

## 当前阻塞与后续

- CCI 容器运行正常，localhost:22 可连接，公网映射仍为原地址；本机 SSH 超时，T2M-PLUS 的 TCP 检查也未连通。EIP 白名单开启，尚未验证当前出口是否在名单内，未改任何白名单规则。
- 自动审批拒绝向 api.ipify.org 查询出口，尚未发送查询。已请用户选择恢复 NTU VPN 或授权查询出口。
- 待传输三个最终权重、共享模型依赖和文本缓存，并在 T2M-PLUS 逐文件校验。
- 待完成仿真安装、场景与控制器资产恢复、实际 GPU/短闭环检查；之后执行固定20场景、两个目标tracker的评测。正式评测入口仍需完成并验证。
- 新增四个作业只在确认完整最终 checkpoint 后加入。当前评测回合数为0，没有成功率结果。

本次远端操作仅恢复代码/依赖和读取状态，未启动、重启或取消训练。

15:23 UTC+8补查：WebIDE连接不稳定，改用同一T2M-PLUS的Notebook终端成功执行检查。依赖安装PID 110180仍在，日志显示多项依赖已下载，但安装未完成；不能视作环境检查通过。网络处理选项仍待用户回复。

## 15:58 UTC+8 更新（取代上文网络阻塞状态）

用户要求重试后，SenseCore SSH 已恢复，未查询公网出口或修改白名单。三个最终 checkpoint 的六个文件已下载并逐文件 SHA256 验证；见 `local-checkpoints.json`。三个传输包已上传 T2M-PLUS，SONIC 包哈希已核验，另两包正在核验。固定 Arena 源码包已核验并解压，失败的 git fetch 目录独立保留；Isaac Lab 从现有干净仓库的固定提交归档至独立缓存。场景资产上传仍在进行。

开发机直连 Hugging Face 超时，恢复脚本退出；共享运动/视觉权重改从 SenseCore 已有副本下载中。仿真依赖安装仍未完成，后续 Lab/DDS 安装脚本只等待该安装退出并核对版本，不会启动仿真或训练。策略环境复用系统 Torch 2.10.0+cu129；首个 uv 解析尝试要另装新 Torch，已仅终止该安装进程，改为在隔离环境安装指定 transformer/peft 版本。GPU 0 有其他工作，GPU 1–7 当时空闲。正式评测仍为 0 回合。

16:04 UTC+8：三个 checkpoint 解包后六个文件在 T2M-PLUS 逐文件哈希核验通过，回执已收回 `t2m-checkpoints.json`。策略导入通过（Torch 2.10.0+cu129 / transformers 5.1.0 / peft 0.18.1）。新增 `evaluate_box.py` 固定最终 checkpoint、20个共同种子及同一推理参数；`--diagnostic` 单独运行50步，不计入成功率。历史桥接补充 HTTP 故障传播，防止上游吞异常后把基础设施错误计成模型失败；3个相关测试通过。真正 GPU 推理和闭环仍未运行。

运动/视觉权重通过本机官方 HF 固定 revision 下载、12个文件完整核验，传输包 SHA256见 `policy-transfer.json`；冗余慢速 SCP 已停止，其半成品不作为已验证副本。官方 PyTorch wheel 也已在本地完成校验，见 `torch-wheel-source.json`，均在上传中。

16:13 UTC+8：T2M-PLUS 已实际读取三个 checkpoint，global_step 分别为10k/10k/20k；共享代码已同步到 `9c02c47c4cb5488d49534bcd5323c60b8530d7d1`，相关3项测试在开发机再次通过。官方场景、SONIC、表面材质已导入；共享策略依赖包仍在上传。SenseCore 只读复查没有新增 scratch/CCT training_state.pt；一次审批服务 Bad Request 后，同路径重试成功，未修改任何训练。

16:17–16:23 UTC+8 定期备份检查：本次仅备份与同步，未启动评测或训练。T2M-PLUS 新增26份文本回执已回收；AFS 7个完整checkpoint重新逐文件校验，独立inode通过，没有新权重需要复制。原慢速基础安装PID110180和仅等待它的适配安装PID141702在心跳前已停止；本地包安装R2为PID173648，完成状态仍待检查。共享权重上传/导入任务仍在进行。接下来须先核验R2退出状态和153项基础包，再手动接续 `install_sim_adapters.py`（新的启动日志名），完成GPU短闭环后才跑 `evaluate_box.py` 正式120回合。
