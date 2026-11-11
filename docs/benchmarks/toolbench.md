# HumanoidToolBench — Fullbody v1 E20

本分支整理的版本为600k fullbody v1，正式成功率 **16.28%（293/1800）**。实现与限制见[版本说明](../../reports/toolbench_fullbody_v1_20261108/REPORT.md)，各条件成绩见[最终结果](../../reports/toolbench_fullbody_v1_20261108/RESULTS.md)。

训练/推理均使用完整实测历史、29身体关节、直接缩放224×224、切点前100帧历史，无当前姿态硬锚定。每次预测50个30Hz帧，执行前20帧，50Hz控制步按33/34/33循环。RTC关闭。

## 数据准备与检查

先按仓库README安装依赖及基础权重，并准备ToolBook原始数据、固定成功演示manifest和六句指令的LLM2Vec文本缓存。缓存不能沿用旧接口生成的数据；使用新目录。

```bash
PYTHONPATH=. python scripts/prepare_toolbench.py \
  --data-dir /path/to/ToolBook/raw --manifest /path/to/ToolBook/manifest.json \
  --out data/toolbench/prepared_fullbody_v1_e20_20261108 --workers 8
PYTHONPATH=. python scripts/audit_toolbench_fullbody.py \
  --toolbook /path/to/ToolBook \
  --prepared data/toolbench/prepared_fullbody_v1_e20_20261108 \
  --out log/toolbench/fullbody_data_audit.json
```

`audit_toolbench_fullbody.py --help` 列出额外审计参数。单元验证可运行 `python -m pytest tests/`。

## 四卡训练与最终评测

以下命令是复现入口，不会由整理流程自动执行。训练从零开始，选择空闲四卡；不要复用旧优化器或旧缓存。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  scripts/train.py --config configs/toolbench_fullbody_v1_e20.yaml \
  --run-dir log/toolbench/fullbody_v1_reproduction

PYTHONPATH=. python scripts/eval_toolbench.py \
  --checkpoint log/toolbench/fullbody_v1_reproduction/checkpoint_600000 \
  --benchmark /path/to/HumanoidToolBench --gpus 0,1,2,3 \
  --workers-per-gpu 1 --output log/toolbench/fullbody_v1_reproduction_eval \
  --episodes 100 --seed-start 10000 --max-steps 3000 \
  --execution-frames 20 --diffusion-steps 10 --attention-backend math
```

模拟器使用独立Python环境；对应运行的HumanoidToolBench commit为`2cd4f006f6f2ebc536a3fb45aff8ce2eaf1382d7`。先按[模拟器接口补丁说明](../../integrations/toolbench/README.md)应用实测历史补丁；该补丁负责逐控制步采集和HTTP传输。不能直接使用只发最新32维状态的旧客户端。`serve_toolbench.py`读取checkpoint中保存的当前帧条件开关；本版必须为off。

`check_toolbench_roundtrip.py`用于原始命令、姿态及30Hz转换的运动回放检查，不会完整恢复录制的工具布局，不能用其结果代替任务成功率。

训练权重、原始数据、缓存、日志和测评视频不在Git中；报告仅保留小型数值摘要。旧v2和后续诊断资料保留在本地忽略目录。
