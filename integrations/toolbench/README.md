# ToolBench 实测历史接口补丁

这是16.28%版本必需的模拟器客户端改动。仅改策略观测传输：每控制步记录真实32维状态、29身体关节和base pose，成功请求后清理；不修改物理参数、工具、任务布局或成功判据。

对应上游提交：`2cd4f006f6f2ebc536a3fb45aff8ce2eaf1382d7`。在干净的对应版本中应用：

```bash
git -C /path/to/HumanoidToolBench apply --check /path/to/kimodo-policy-vlm-version/integrations/toolbench/measured_fullbody_history.patch
git -C /path/to/HumanoidToolBench apply /path/to/kimodo-policy-vlm-version/integrations/toolbench/measured_fullbody_history.patch
```

已经应用的工作副本不要重复应用。使用装有pytest和NumPy的Python环境运行传输回归测试，并通过PYTHONPATH指向应用补丁后的模拟器源码（模拟器专用环境可能未安装pytest）：

```bash
PYTHONPATH=/path/to/HumanoidToolBench/src python -m pytest /path/to/kimodo-policy-vlm-version/integrations/toolbench/test_toolbench_state_history.py -q
```

覆盖真实增量采集、数组复制、reset、失败重试和完整29关节。基础策略仓库测试不能替代这项跨仓库传输检查。
