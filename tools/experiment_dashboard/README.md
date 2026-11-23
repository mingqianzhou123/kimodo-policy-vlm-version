# humanoidVLA 实验研究台

服务器入口：开发机 WebIDE 的端口代理，端口 8765（需要控制台登录权限）。

从项目根目录运行：

```sh
. .setup/env.sh
python scripts/experiment_dashboard.py --serve tools/experiment_dashboard --port 8765
```

服务仅监听 127.0.0.1，借助现有 WebIDE 的认证代理访问。后台进程日志是 `.setup/dashboard.console.log`。每 30 秒重新读取原始记录，不启动或停止训练；网页相同频率获取数据。页面显示采集时间、指标修改时间、进程状态，进程退出时不继续标成运行中。

- 实验概览：队列、曲线、故障恢复事件、后续判断规则。
- 实验台账：搜索/筛选、最多三实验比较、参数字典、CSV 导出和原始证据路径。
- 验证评测：checkpoint、成功率、回合数、种子与执行帧。未评测不记成 0。
- 历史基线：九任务两套配置及头部变体、对照模型、视觉诊断和 EMA 结果；保留来源与口径。
- 研究观察：事实、假设、下一实验计划。实时页通过本机服务保存到 `docs/experiments/research_notes.json`；快照版仅存当前浏览器，可导出/导入 JSON 备份。

## 数据

`docs/experiments/bigym2_ledger.json` 保存实验目录与历史摘要；实时采集覆盖当前配置、状态和曲线，不改原始训练日志。输出 `tools/experiment_dashboard/data.json`。历史补充在 `history.json`，已标注为文档证据。

导出快照：

```sh
python scripts/experiment_dashboard.py --export docs/experiments/dashboard.json
```

网页支持导入该文件。快照会明确显示“非实时连接”，刷新只读取托管快照，不伪装与开发机连接。

2026-11-12：用户选择只使用开发机实时网页；不发布 Sites。服务器版不依赖外部托管。未提交 GitHub。

## 恢复记录

首轮装杯子在最后记录 29,850 步后 NCCL ALLREDUCE 超时，25k checkpoint 可读取且包含优化器/调度器；按既定判据未检测到损失发散。恢复前保留完整指标、日志和状态到 `.setup/recovery-20261112`。当前恢复脚本 `.setup/recover_queue.py` 只尝试一次，失败即停止，不覆盖首轮 console；成功后评 25k/30k，再运行 move_plate 30k，再按既定规则续训。
