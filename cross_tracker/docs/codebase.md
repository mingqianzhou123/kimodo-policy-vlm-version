# 代码梳理（2026-11-13）

## 边界

| 现有目录 | 角色 | 本实验如何使用 |
| --- | --- | --- |
| `kimodo-vla version/` | 本地 Kimodo-VLA 仓库，含 BiGym 调参与 Arena 适配 | 可引用公共模型和接口；不把混训日志、配置塞进其 BiGym 目录 |
| `kimodo-policy-main/` | 原 Kimodo 与 HumanoidArena/LeRobot 实现 | 接口与实现参考；不是本实验的运行输出目录 |
| `reproduction/` | 旧 A800 复现及服务器脚本 | 历史记录；不直接启动或复用绝对路径 |
| `humanoid_robot_suite/` | 机器人、仿真及历史备份 | 按需只读引用，不复制进混训目录 |
| `新方案/` | 方案、已完成审计、修复和视频核验 | 历史证据冻结，后续新增工作归 `cross_tracker/` |

本地 `kimodo-vla version/` 与百度 `/root/workspace/humanoidVLA/` 的 HEAD 均为 `d61b5cf99ce865523ad9746f8d060881d279cc16`，两边都有其他实验的未提交修改，文件内容不能仅凭 HEAD 视为完全相同。本次只新增各自的 `cross_tracker/`。远端父目录未发现 AGENTS.md，且创建前没有 cross_tracker 目录。

## 训练前要解决的真实接入问题

1. `kimodo_vla/benchmarks/arena/demos.py::read_episode` 直接读原始 `action`，未读取修复层或可用帧标记。必须经过修复数据适配后才能用于本实验。
2. `kimodo_vla/benchmarks/arena/policy.py::ROOT_BOUNDARY` 和 `serve_arena.py` 的控制器选项只有 SONIC、TWIST2。MimicLite 服务接入与参考窗口还需单独验证，不能仅添加名称就宣称可评测。
3. `configs/arena.yaml` 继承 BiGym 配置，默认 200k 步、全局 batch 64，输出路径可落到旧 `log/arena`。它是旧复现配置，不是当前小模型混训方案。
4. 原 HumanoidArena/LeRobot 的 `datasets/factory.py` 直接拒绝多数据集输入；`lerobot_train.py` 默认按帧打乱或用 EpisodeAwareSampler，均未实现这里约定的任务/来源/轨迹均衡抽样。
5. 若选择 SmolVLA，本地默认 state/action 容量是 32/32；当前接口是 64/40，必须显式配置并检查权重加载。模型尚未选定，本次不擅自切换骨干。
6. 原视频分片覆盖多条示教，修复后的有效片段仍需沿用原始帧号与视频偏移。冻结名单不等于已完成可训练视频包。

## 历史证据入口

以下路径相对本地工作区 `/Users/zmqsfiles/Research/人形 vla`，远端需另行同步或配置：

- `新方案/训练前数据与接口审计-2026-11-13.md`
- `新方案/修复结果-2026-11-13.md`
- `新方案/17条异常与视频成功性复核-2026-11-13.md`
- `新方案/repair-2026-11-13/candidate-v2/`：当前修复层；candidate-v1 已弃用。
- `新方案/validation-2026-11-13/status.json`：17 条异常继续隔离，成功状态待核验。

旧路径暂不搬动，避免破坏报告链接和修复产物中的路径。后续下载及派生数据写入 `cross_tracker/data/cache/`，清单存 `data/manifests/`。迁移历史数据时先核对哈希，再更新本机路径配置，不能靠复制多份来管理版本。

评测文本：训练用 `data/cache/text/` 保持不动。`scripts/prepare_eval_text.py` 将同一向量写入 `data/cache/text-eval/`，只把 `task_name` 改成仿真的完整任务 ID，供评测服务器使用。每个源文件/派生文件 SHA256 和向量逐元素一致性保存在派生缓存的 `manifest.json`。原因是上游仿真发送 `Isaac-Move-…`，原始缓存键是 `HOI_pp_box` 等数据集名。

`src/arena_history.py` 为新版 Arena 的 HTTP 客户端补齐真实状态历史：每个 50 Hz 控制步采集一次 64 维观测，请求成功后清空已发送部分，episode reset 后清空历史。拒绝采样间隔缺口和末帧与当前状态不一致；不重复当前帧或生成虚构历史。`tests/test_arena_history.py` 覆盖跨请求的 26 个实测样本、reset 边界和漏帧拒绝。

同一桥接模块修正 Isaac Sim 5 下 TWIST2 的 render=True 步进语义：物理单步和纯渲染分别调用，避免图像刷新额外推进时间。该兼容修复对所有后续实验一致使用，运行代码快照保存在各 run 中；不把此项混入 CCT 或条件化方法。
