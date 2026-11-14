# 百度仿真环境

状态：独立环境已安装；基础物理、RGB 相机与正常退出已通过。SONIC/TWIST2 在三个任务上的六项短闭环均通过；MimicLite 闭环仍缺控制器权重；2026-11-14 已启动首个 SONIC 正式 baseline。

所有新增依赖和资产保存在 `cross_tracker/data/cache/`；共享 HumanoidArena 和训练 `.venv` 保持独立。

| 项目 | 固定位置或版本 |
| --- | --- |
| 仿真 Python | `data/cache/envs/isaaclab-2.2/bin/python`，3.11.15 |
| 仿真 Torch | 已安装 2.7.0 + CUDA 12.8；训练环境仍为 2.10.0 + CUDA 12.9 |
| Isaac Sim | 已安装 5.0.0，含扩展缓存 |
| Isaac Lab | 从已核验共享仓库归档到 `data/cache/sources/IsaacLab-2.2.0`；v2.2.0，`46dff135f44683f031edf346e544fcfd8456b2bb`，不在共享源码中产生安装文件 |
| HumanoidArena | 独立固定到 `44a72229f11bab47a145e12db39ed891b677e783`，位于 `data/cache/sources/`；共享旧版 `6847928` 缺 MimicLite 评测入口，保持不动 |
| 场景资产 | `data/cache/sim-assets/arena/assets/{objects,robots}`，官方 20260703 release_clean 包 |
| SONIC | `data/cache/sim-assets/sonic`，官方 `nvidia/GEAR-SONIC` revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f` 的根目录 encoder/decoder/config；不选低延迟或 v1.1 子目录 |
| TWIST2 | 固定使用独立 Arena 源码中的 `TWIST2/assets/ckpts/twist2_1017_20k.onnx`（官方评测脚本默认值）；SHA256 `2d1fb3a31e4e967f70ecfefc3ad1e7b2ac491677068b89f60b565a94e7735061`，ONNX Runtime CUDA 加载及三任务短闭环均已核验 |
| MimicLite | 所需 `policy-xua2csee-4000.onnx` 缺失，本地只有同名 YAML；其他公开模型不能直接替代 |

开发机已确认 Ubuntu 22.04.5、GLIBC 2.35、RTX 5090、驱动 580.159.04。GPU 2 有其他任务，诊断使用前重新检查空闲 GPU。

运行记录在 `runs/sim-setup-20261113/`：环境快照、实际安装命令、下载源测试、退出码、资产校验与失败日志。初次代理下载 PyTorch 无进展，已保留日志并改用验证可用的官方 HTTPS 直连。没有关闭 TLS 校验。

官方场景包 2,683,806,071 字节，SHA256 `d0a0c6be2e47ac9a3b161ec9ca11a9d67908ef755fe0fd96445404938376447b`，本地 ZIP 全部 CRC 检查通过；1406 个条目，解压约 3.93 GB。它不含 MimicLite checkpoint。SONIC 3 个文件已按官方仓库 SHA256 / Git blob ID 核验。远端 `assets-import.json` 已确认场景完整导入、SONIC 三文件校验通过；基础仿真与两个 tracker 的三任务短闭环已通过。

完成标准：安装包与依赖检查 → Isaac Lab 导入 → 无界面物理及相机渲染 → 三任务资产解析 → 三 tracker 接口及同 tracker 回放 → 固定场景种子闭环检查。不能用包安装成功、CUDA 运算或 HTTP 测试替代仿真运行。

参考：[Isaac Lab 2.2 官方安装指南](https://docs.nvidia.com/isaaclab/2.2/isaac-lab/local-installation/pip-installation-recommended)、共享项目 `docs/04_environment_setup.md`、[官方场景包](https://drive.google.com/file/d/1TCa_aVRmFrZs_l4wlxkqanNebvDtChNk/view)。

## MimicLite 控制器缺失核查

`policy-xua2csee-4000.onnx` 是执行参考动作的底层控制模型，不是示教数据或 Kimodo 训练 checkpoint。原始数据仍可用于离线训练；缺少匹配控制器会阻止 MimicLite 闭环评测。现有同名 YAML 只有接口/参数配置，不能代替 ONNX。

已检查 GitHub 固定版本完整文件树与 release、HF raw/refpose/models 全目录、ModelScope raw 全部 14,086 条目，以及 ModelScope MimicLite 专用归档全部 280 条目，均未找到目标 ONNX。官方场景包也不含控制器。ModelScope 的两个 `.zip` 实际使用 XZ 压缩；2026-11-14 已完成约 29 GB 总归档完整 SHA256、全部 512 条目及 XZ EOF 校验，没有候选权重、嵌套归档或 Mimic 路径；16 份 `.bak` 是统计 JSON 备份。补充证据在 `runs/archive-audit-20261114/`。完整检索证据在 `runs/sim-setup-20261113/public-weight-search.json`。

[官方 MimicLite 评测脚本](https://github.com/William-wAng618/HumanoidArena/blob/44a72229f11bab47a145e12db39ed891b677e783/isaaclab_twist2_g1/script/eval_scripts/mimic_lite/run_vla_eval.sh) 明确依赖这份 ONNX。未找到之前保留为外部依赖，不用其他公开 MimicLite 版本冒充采集时控制器，也未向作者发送消息。

安装网络记录：R4 的 NVIDIA 索引查找失败；R5 已解析 153 个固定依赖，但下载 `isaacsim-asset` 时 TLS 连接提前关闭。26 个官方 NVIDIA wheel（4,663,571,949 字节）已在本地按官方 SHA256 全部核验，已完整传入开发机并逐文件校验；R6 已从验证后的本地 wheel 安装成功，普通 Python 依赖使用清华镜像。失败尝试保留原始日志，不修改训练环境。

场景接口补充：上游导航任务区分 `inference`（`small_warehouse_digital_twin_validation.usd`）与 `replay_compat`（采集时的 `small_warehouse_digital_twin.usd`）。官方资产包两者均有。正式对比必须给所有 tracker 固定同一评测场景；示教回放则使用采集兼容场景，避免将几何差异误判成示教失败。R9/R10 已在该 inference 场景完成两个 tracker 的短闭环；采集兼容场景的完整示教回放仍待核验。


## 已通过的运行检查

- R7：GPU 0 上运行 180 步，方块高度 0.997956 → 0.125，RGB 为 240×320×3、标准差 10.331；进程正常退出 0。证据 `smoke-result-r7.json`、`smoke-launch-r7.json`、`smoke-rgb-r7.png`。
- Warp 固定 1.9.0 并在 AppLauncher 前导入，避免 Kit 优先加载旧 1.7.1；最新 1.19.0 与此版 Isaac 的接口不兼容。
- setuptools 固定 80.9.0，packaging 23.0、wheel 0.44.0、click 8.1.7、idna 3.10；完整依赖检查通过。GLU/OpenGL/Vulkan 官方 Ubuntu 包解压到独立 `data/cache/system-libs`。
- Isaac Lab 的停止回调会在无界面退出时继续等待渲染。诊断包装器只在关闭前设置 `_disable_app_control_on_stop_handle=True`，未修改共享源码或任务物理逻辑。
- Unitree SDK revision `814556d15970dd2ecf1c9984e845ca02ab07e206` 与 CycloneDDS 0.10.2 revision `9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd` 在隔离环境安装，完整依赖检查通过。导入 Arena 前把 DDS 绑定 `lo`。
- SONIC encoder 输入 1762、输出 64；decoder 输入 994、输出 29。TWIST2 输入 1432、输出 29；均已用 ONNX Runtime 创建 session。
- 实际仿真发送完整 `Isaac-Move-…` 任务名。`scripts/prepare_eval_text.py` 创建派生 `text-eval` 缓存，仅修改任务名，逐张量确认 embedding 不变；真实 GPU HTTP R3 已通过。

完整场景诊断记录在 `runs/sim-closed-loop-20261113*/`，始终最多 50 步，不是任务成功率评测。R1 缺 Unitree SDK，已补齐。R2 停在首帧场景渲染同步，保存了 Python/native stack，正常终止信号未生效后只对本次 simulator PID 强制结束。资产中 `M_TrafficCone.mdl` 调用了不存在的 `OmniUe4Base_1` 材质函数；导入模块实际导出 `OmniUe4Base`，已修复四份隔离资产，原文与修复前后 SHA256 留存。R3 因前次进程尚未释放显存而被空闲检查拦截；未启动新模型。R4 材质函数错误已消失，但外部材质加载仍使渲染等待；R5 本地化依赖后约 29 秒内完成完整场景创建与初始重置，随后因 CPU 版 ONNX Runtime 缺 CUDA provider 停止，尚未执行机器人控制循环。已开始替换为固定 GPU 版 1.22.0。

已进一步定位外部材质依赖：4 个 USDA 场景文件共 64 处 MDL 引用仍指向 HTTP 远程地址，官方场景 ZIP 不含这 4 个 MDL 与 12 张贴图。已通过相同官方 S3 的 HTTPS 地址下载原始内容，保留 URL/字节数/SHA256，将这些引用改成相对本地路径；未修改贴图像素、几何或物理。`table-material-localization.json` 保存修改前后哈希，原 USD 位于 `data/cache/setup/table-usd-originals`。材质包 45,101,306 字节，SHA256 `514af6314b7f5f581eb6da4dcaa95a30407fa4d786bff198250a356991ca3ab5`。

GPU 运行时补齐：官方 `onnxruntime-gpu==1.22.0` wheel SHA256 `d30c1512f22b1f01bacb4f177d49cbefd23e0f4bef56066f1282992d133e6ff8`。开发机直下载超时后，本地完整下载并按同一哈希传入安装；CPU 版已卸载，避免两个发行包覆盖同名模块。启动适配层去掉没有库支持的 TensorRT provider，保留 CUDA 和 CPU provider 的真实加载检查。R6 的 TensorRT 回退失败保留。

R7 已运行 50 步并生成视频，但模型请求全部因缺少 `observation.state_history` 返回 HTTP 500；上游 provider 捕获错误继续走步，episode 本身未记录该异常。诊断脚本要求实际成功的 `/infer` 请求，并拒绝 HTTP 500，因此正确判失败。新增真实 50 Hz 状态历史桥接后再试；不能把这次 50 步写成闭环通过。

R8：SONIC 箱子完成 50 步、2 次真实 `/infer` 200，并保存视频；首个 encoder GPU 推理约 54 秒，后续低层推理较快，不能据首轮平均值推断稳态吞吐。TWIST2 在第 25 步被时间检查阻止：0.48 → 0.52 秒。源码核对确认 `step(render=True)` 在 Isaac Sim 5 中调用 app.update，按渲染间隔推进物理，而上游 TWIST2 在物理子步循环使用它。桥接层仅对 TWIST2 改为 `step(render=False)` 后单独 `render()`，保留原始 physics dt/decimation 与图像刷新条件；新增测试确认渲染不额外走物理。R9 使用统一桥接版本复核六个任务/controller 组合。


## 最终核验结果

SONIC 箱子采用 R9，其余五项采用 R10，六项桥接源码逐字节一致；R9 连续启动下一项时因 GPU 利用率采样尚未回落而被保护检查拦截，R10 在最多 15 秒内等待确认空闲，未放宽阈值。六项均为 50 步、2 次真实模型推理、51 帧视频、退出码 0。独立审计通过真实状态 0.02 秒间隔、历史包边界、完整视频解码和相同物体重置 seed。审计：`closed-loop-audit.json`。这些结果仅确认运行与接口，不是成功率实验；运行实际结束原因均为短诊断步数上限。

全部修复与依赖仅在 `cross_tracker`。训练虚拟环境和共享源码未修改；首个 SONIC 正式训练已于 2026-11-14 启动，SenseCore 作业未提交。完整 MimicLite 评测仍受原始 ONNX 缺失阻塞。
