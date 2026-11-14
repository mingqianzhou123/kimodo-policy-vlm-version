# 备份和同步

用户于2026-11-14授权定期推送跨tracker进展、单独备份权重，并同步百度与SenseCore。GitHub目标：`mingqianzhou123/kimodo-policy-vlm-version`。仅提交本项目 `cross_tracker/` 的代码、配置、冻结清单、小型实验记录；不夹带其他实验的工作区修改。

- 每个实现/核验/提交作业里程碑完成后commit并push；另设每2小时补备检查。定时任务需要本机及Codex可用，不等同服务器端全天候备份。
- 大文件、视频、模型、预训练权重、转换数据、缓存、环境、归档及原始运行日志不进入Git。GitHub只保存结构化实验记录；提交前检查暂存名单、单文件大小（不超过2MiB）和凭据，不复制或上传含凭据的原始日志。
- 权重备份位于SenseCore AFS的 `humanoid_vla/cross_tracker_backups/weights/`，在训练运行目录之外，逐文件SHA256校验，不用硬链接冒充独立副本。已存在且哈希一致的权重不重复拷贝。AFS副本是独立目录备份，不宣称跨存储容灾。
- 只备份完整checkpoint；持续写入的resume文件先生成稳定快照，校验失败不视为备份完成。百度不可达时，其独有checkpoint必须列为待补备，不能声称已有副本。预训练权重已在本机和SenseCore的校验副本可复用。
- SenseCore开发/正式数据路径：`/data/NTU_slab/mingqianzhou/humanoid_vla/kimodo-policy-vlm-version/cross_tracker`。当前在SenseCore CPU CCI开发CCT，GPU核验仍需实际分配GPU。
- 百度只允许T2M-PLUS（`dv-e21ddf3a87`），项目为 `/root/workspace/humanoidVLA/cross_tracker`。不可达时保留待同步状态，禁止改连T2M、Rethink-humanoid或退役A800。
- 每次同步先对照远端与上次清单，收回远端新增实验记录/源码，保留双方冲突副本，不能直接覆盖远端更新。同步代码与配置后逐文件校验，并记录Git提交和实际部署文件哈希；不同环境路径单独配置，不覆盖数据、权重、日志或正在运行作业快照。
- GitHub是代码与小型记录的恢复点；运行目录的代码快照和记录保持不可覆盖。排队/运行中的三个朴素baseline入口不因CCT开发而改动。

增量权重入口：scripts/backup_weights.py --destination <AFS独立备份目录>。只扫描 runs/*/training/checkpoint_* 中完整的权重和配置，比较源、目标SHA256；不自动复制持续变化的resume.pt。当前已核验的代码同步范围为58个cross_tracker活跃文件及72个共享依赖，报告在 runs/backup-sync-20261114/code-sync.json；百度仍待恢复连接。
