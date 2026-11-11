# 商汤 SenseCore 上的脚本

这些文件复制到 SenseCore 共享盘上的工作目录（下面记作 `KVLA_HOME`）里用，脚本按自己所在的文件夹找到 `KVLA_HOME`，
所以代码里不写服务器路径。代码在 `KVLA_HOME/kimodo-policy-vlm-version`，环境、权重、数据、日志和训练输出都在 `KVLA_HOME` 下。

| 文件 | 作用 |
|---|---|
| `env.sh` | 每个开发会话和作业开头 source：Python 环境、镜像站、权重和数据位置 |
| `setup_env.sh` | 在开发机上建 Python 环境（可重复跑） |
| `setup_egl_loader.sh` | 给评测用的 EGL 加载器（作业镜像没有 libEGL，见 `docs/benchmarks/bigym2.md` 第 10 节） |
| `download.sh`、`fetch.py` | 下载权重、Arena 数据和 VLM（hf-mirror，按清单下载） |
| `acp_job.sh` | ACP 作业的启动命令外壳：`bash <KVLA_HOME>/acp_job.sh <名字> <命令...>`，日志写到 `KVLA_HOME/logs/acp/` |
| `bigym_run.sh` | 一个 BiGym 训练（作业里的所有卡）加验证种子上的评测；同一条命令重跑会接着训、跳过已评的 |
| `render_probe.py` | 检查 MuJoCo 是否拿到 GPU 的 EGL；`bigym_run.sh` 训练前调用 |
| `gpu_selfcheck.py` | 新环境的 GPU 自检：CUDA、文字特征、Arena 推理、VLM |
| `arena_smoke.sh`、`arena_serve_check.py` | Arena 训练的短检查 |

提交作业（控制台 → 高性能算力池 ACP → 创建任务）：AI 文件存储卷挂到 `/data`；评测要渲染的作业在「高级配置」里加环境变量
`NVIDIA_DRIVER_CAPABILITIES=all`。例：

```bash
bash <KVLA_HOME>/acp_job.sh bg_stable_plate_s1000 bash <KVLA_HOME>/bigym_run.sh stability/stable_combo \
    configs/bigym2_recipe_lr05.yaml move_plate 1000 30000 25000,30000 \
    training.optimizer.betas=[0.9,0.95] training.spike_skip.factor=5
```

改了这里的脚本要复制回 `KVLA_HOME`，两边保持一样。
