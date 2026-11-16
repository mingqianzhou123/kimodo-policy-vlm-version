# T2M-PLUS 评测恢复（2026-11-16）

用户授权：把昨天 SenseCore 已完成的 checkpoint 移到 T2M-PLUS 评测，不启动新训练。

## 已核验

- 正确实例 `dv-e21ddf3a87` 可通过 WebIDE 登录，8×RTX5090。初次检查 GPU 0 有其他进程，未干预；GPU 1–7 当时空闲，启动评测前还需再查。
- 原 `/root/workspace/humanoidVLA/cross_tracker` 不存在；指定范围的 workspace/mnt 搜索未找到旧项目。现有目录是旧的其他 Kimodo 实验，保持原状，不能据此断言历史数据彻底丢失。
- 从 GitHub 新建独立 `/root/workspace/humanoidVLA`，实际 HEAD `e42bbdaee1cf1e633279743005222edbf5809ff6`。命令级清除失效的 localhost:7890 代理，未改全局代理配置。
- 独立 Python 3.11.15 环境已创建。Isaac Sim 5.0 固定依赖正在下载；启动回执见 `sim-install-start.json`，不能等同安装通过。
- 现有 IsaacLab HEAD 是固定的 `46dff135f44683f031edf346e544fcfd8456b2bb`；现有 Arena 版本不同，所需 `44a7222` 已开始独立获取。场景资产未在原预期目录找到，仍待恢复。
- 约14:12 UTC+8，SenseCore 控制台显示三个微调 baseline 成功，三个 scratch 和 CCT 正式作业仍为“启动中”。CCI 网页终端也只发现三个微调 baseline 的 checkpoint。
- 约14:15，CCI 网页终端重新计算三个最终 AFS 备份的权重和 config SHA256；六个哈希均与旧备份记录相同，见 `evaluation-plan.json`。

## 当前阻塞与后续

- CCI 容器运行正常，localhost:22 可连接，公网映射仍为原地址；本机 SSH 超时，T2M-PLUS 的 TCP 检查也未连通。EIP 白名单开启，尚未验证当前出口是否在名单内，未改任何白名单规则。
- 自动审批拒绝向 api.ipify.org 查询出口，尚未发送查询。已请用户选择恢复 NTU VPN 或授权查询出口。
- 待传输三个最终权重、共享模型依赖和文本缓存，并在 T2M-PLUS 逐文件校验。
- 待完成仿真安装、场景与控制器资产恢复、实际 GPU/短闭环检查；之后执行固定20场景、两个目标tracker的评测。正式评测入口仍需完成并验证。
- 新增四个作业只在确认完整最终 checkpoint 后加入。当前评测回合数为0，没有成功率结果。

本次远端操作仅恢复代码/依赖和读取状态，未启动、重启或取消训练。
