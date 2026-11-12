# AdaLN-Zero 条件注入改进版：实现说明与实验记录

本文记录 `model_improvement_lct1` 基础上的 AdaLN-Zero 改进版本，包括设计动机、代码路径、参数冻结边界、训练配置、验证方法、已启动的 Arena 训练和后续对比方式。

当前实现位于本地分支 `adaln_zero_adapter`。本工作只修改本地仓库，未上传 GitHub。

## 1. 改动目标

原有 VLM 改进版把 GR00T N1.7 的视觉语言特征送入外部 CrossAttention，再通过零初始化输出投影将残差注入冻结的 Kimodo action backbone。这个方案保留了冻结主干的优点，但条件信息必须通过 query-key/value 注意力路径进入，注入能力受限于注意力投影和序列交互。

本版本的目标是：

1. 保留原 CrossAttention 版本，保证后续可以进行匹配对比。
2. 增加一个外置 AdaLN-Zero adapter，不改写 Kimodo Transformer layer 的内部实现。
3. 把 VLM 条件和扩散 timestep 一起编码为调制参数。
4. 保持 Kimodo action backbone 冻结，避免把实验变成重新训练 action backbone。
5. 维持原有 Root/Body 分支、future-only 注入、HandNet 条件路径和损失函数。
6. 用 4 卡进行七任务 HumanoidArena 联合训练，并把学习率从原来偏保守的 VLM recipe 提高到适合零初始化 adapter 的范围。

设计依据来自 [DecoupledActionExpert](https://github.com/jianzhou0420/DecoupledActionExpert) 的分阶段训练和条件适配思路，以及论文 [Freeze, Share, Shrink: Rethinking the Action Backbone in Diffusion Policies](https://arxiv.org/abs/2511.12101) 对冻结 action backbone 时调制类条件注入的讨论。核心启发是：冻结 action backbone 后，条件 adapter 必须具备稳定、直接的调制能力；仅依赖注意力路径时，条件可能难以有效打开。

## 2. 原始版本的注入路径

原有路径可以概括为：

```text
图像 + 指令
    ↓
GR00T N1.7 VLM
    ↓
共享 VLM LayerNorm + 4 层 SelfAttention adapter
    ↓
VLCondition(tokens, valid-token mask)，宽度 2048
    ↓
Root/Body 各自 4 个 CrossAttentionResidual
    ↓
Kimodo denoiser 的第 3、7、11、15 层之后
    ↓
只加到 future 50 个 motion token
```

CrossAttention residual 的形式是：

```text
q = LayerNorm(action_hidden)
u = MultiHeadAttention(q, VLM_tokens, VLM_tokens, mask=valid_token_mask)
residual = ZeroLinear(u)
action_hidden ← action_hidden + residual
```

`ZeroLinear` 的 weight 和 bias 均为零，所以新模块初始时不改变原有 denoiser 输出。原模块仍由 [`GrootInjection`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/groot_conditioning.py:395) 创建，默认模式仍然是 `cross_attention`。

## 3. AdaLN-Zero 的新路径

新路径只替换外部 fuser，Kimodo 的 16 层 Transformer 和每个 Transformer layer 的代码不变：

```text
GR00T N1.7 VLM tokens [B, L, 2048]
        + valid-token mask
             │
             ├── mask-aware mean pooling → [B, 2048]
             │                              │
             │                              ├── LayerNorm + Linear → [B, 512]
             │                              │
             │                              └── Root/Body 各自条件 stem
             │
 diffusion timestep t [B]
             │
             └── PositionalEncoding + TimestepEmbedder → [B, 512]
                              │
                              └── concat([visual, time]) → MLP → [B, 512]
                                                        │
                                                        ├── block 3
                                                        ├── block 7
                                                        ├── block 11
                                                        └── block 15
```

每个 AdaLN-Zero residual 对 future motion hidden `h` 做：

```text
(shift, scale, gate) = Linear(condition)
z = LayerNorm(h, elementwise_affine=False)
residual = gate ⊙ (z ⊙ (1 + scale) + shift)
h ← h + residual
```

最后一个 `Linear` 在初始化时全部为零，因此 `shift=scale=gate=0`，`residual=0`。这使得 AdaLN 路径在初始化时是严格恒等映射；训练开始后，首先通过 modulation head 的梯度打开 gate，然后视觉条件和 timestep embedding 才会收到有效梯度。这也是单独设置 AdaLN optimizer group 的原因。

具体实现位于 [`AdaLNZeroResidual`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/groot_conditioning.py:260) 和 [`AdaLNZeroInjectionStack`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/groot_conditioning.py:293)。

## 4. 为什么使用 masked mean pooling

GR00T N1.7 的图像 token 和文本 token 拼在一起，样本之间的有效 token 长度可能不同。直接对 padding 后的 token 做普通平均会让 padding 数量影响条件向量，导致同一条有效 VLM 内容因为 batch 中其他样本长度不同而产生不同条件。

因此实现了 mask-aware pooling：

```python
weights = valid_mask[..., None].to(tokens.dtype)
pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
```

实现位置是 [`masked_mean_tokens`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/groot_conditioning.py:40)。它支持每个样本不同的 VLM token 长度，padding token 不参与统计，并且在有效 token 不变时，额外添加任意 padding 不会改变 pooled condition。

原有 VLM adapter 仍然保留完整的 `VLCondition(tokens, mask)`，所以 HandNet 仍可使用原始 token 序列和 mask；AdaLN 只在 action backbone 外部使用 pooled condition。

## 5. timestep 条件的接入方式

训练时 `KimodoPolicy` 已经为每个样本采样扩散步 `t`，并把它用于 motion 加噪和 denoiser。以前 CrossAttention fuser 不需要这个参数，因此 backbone 对旧 fuser 仍保持三参数调用兼容。

现在 [`TransformerEncoderBlock`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/backbone.py:255) 通过能力检测：

```python
if getattr(hint_fuser, "accepts_timesteps", False):
    future_hint = hint_fuser(layer, condition, future_motion_tokens, timesteps)
else:
    future_hint = hint_fuser(layer, condition, future_motion_tokens)
```

`AdaLNZeroInjectionStack.accepts_timesteps = True`，原 `InjectionStack` 没有这个属性，因此 AdaLN 版本获得 timestep 条件，而 CrossAttention 版本的调用约定和已有测试不变。

`max_diffusion_steps` 默认是 1000，与当前 Kimodo diffusion schedule 对齐；配置项是 `model.adaln_max_diffusion_steps`。

## 6. Root/Body 和 future-only 约束

Root 与 Body 使用两个独立的 `AdaLNZeroInjectionStack`：

```text
root VLM condition → root visual projection → root condition MLP → 4 heads
body VLM condition → body visual projection → body condition MLP → 4 heads
```

两个分支不共享 AdaLN 参数，保留原有 Root/Body 条件解耦。注入位置仍为 `(3, 7, 11, 15)`，每次只处理 `future_start` 之后的 50 个 motion token。历史观测 token 不接收 VLM residual，历史部分仍由条件运动约束处理。

## 7. 冻结边界和可训练参数

### 固定参数

以下部分在 AdaLN 训练中保持冻结：

- Kimodo denoiser 的 action backbone，包括 Root/Body 的 16 层 Transformer。
- GR00T N1.7 VLM backbone。
- 训练初始阶段的 GR00T VLM adapter。
- 原始文本 encoder；本实验使用预计算文本 embedding。

### 可训练参数

训练开始时可训练：

- Kimodo ControlNet 的复制/适配部分。
- Root/Body AdaLN-Zero 条件 adapter。
- HandNet head。

第 10,000 步后，GR00T VLM adapter 加入训练；VLM backbone 仍然冻结。这个阶段由 `training.stages.adapter_start` 控制。

`set_training_stage()` 会先把全部模型设为冻结，再按阶段打开对应模块；`checkpoint_parameters()` 同时保存 ControlNet、HandNet 和 VLM adapter，避免 VLM adapter 在阶段切换或恢复训练时丢失。

相关代码：

- [`KimodoPolicy.set_training_stage`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/kimodo_policy.py:249)
- [`KimodoPolicy.checkpoint_parameters`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/kimodo_policy.py:265)
- [`training_stage`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/train.py:204)

## 8. 当前 AdaLN 配置中的 optimizer 分组和学习率

原 VLM 改进版的学习率较保守，主要使用 `2e-5` 和 `5e-6`。对于零初始化 AdaLN，过小的 modulation 学习率会导致 gate 打开过慢，因此单独增加了 `adaln` optimizer group：

| 参数组 | 当前学习率 | 作用 |
|---|---:|---|
| AdaLN adapter | `5e-5` | 训练 VLM+timestep 到 Kimodo hidden 的 shift/scale/gate 调制 |
| HandNet | `3e-5` | 训练手部动作头 |
| VLM adapter（4层vl selfattention） | `1e-5` | 第 10,000 步后加入训练 |
| Kimodo backbone | `0` | 始终冻结 |
| VLM backbone | `0` | 始终冻结 |

当前生产配置使用 `visual_conditioning=groot_n1d7` 和 `groot_injection_mode=adaln_zero`。在这个配置下，`KimodoPolicy` 中虽然仍使用历史成员名 `self.controlnet`，实际创建的是 `GrootInjection`，并没有额外的传统 ControlNet branch。真正训练的视觉注入参数是：

```text
controlnet.root_hint_fusion.*
controlnet.body_hint_fusion.*
```

它们包括 `visual_projection`、timestep embedding、condition MLP，以及第 3/7/11/15 层的 AdaLN modulation heads。`param_groups()` 会把这些参数识别为 `adaln` group，并使用 `adaln_lr=5e-5`。

当前 AdaLN 配置中的 `control_adapter_lr=5e-5` 只是保留的兼容字段，不对应一个额外的活动参数组；当前训练实际生效的是 `adaln_lr`。

调度方式是全局 cosine decay、5000 步 warmup，VLM adapter 进入训练后单独使用 5000 步 adapter warmup。生产配置保持 bf16 autocast、fused AdamW、global batch 128。

## 9. 生产训练配置

配置文件是 [`arena_sonicx7_groot_adaln.yaml`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/configs/arena_sonicx7_groot_adaln.yaml:1)，继承 [`arena_sonicx7_groot_stable.yaml`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/configs/arena_sonicx7_groot_stable.yaml:1)。

关键配置：

```yaml
model:
  groot_injection_mode: adaln_zero
  adaln_condition_dim: 512
  adaln_pool: masked_mean
  adaln_max_diffusion_steps: 1000
  compile: true
  compile_scope: denoiser
  compile_mode: reduce-overhead

main:
  batch_size: 128
  micro_batch_size: 32
  num_workers: 8
  max_steps: 500000

training:
  optimizer:
    adaln_lr: 5.0e-5
    # 兼容旧配置；当前 AdaLN VLM 路径实际使用 adaln_lr。
    control_adapter_lr: 5.0e-5
    hand_lr: 3.0e-5
    vl_adapter_lr: 1.0e-5
  scheduler:
    warmup_steps: 5000
    adapter_warmup_steps: 5000
```

四卡分配为 GPU 4、5、6、7，每卡 micro-batch 32，梯度累积后 global batch 为 128。生产阶段使用每卡 8 个 DataLoader worker；短验证阶段临时把 worker 设为 0，避免同一监督进程中连续启动多个 DDP DataLoader 时发生 worker shutdown 问题。

## 10. 验证项目

### 10.1 单元测试

`tests/test_groot_conditioning.py` 覆盖：CrossAttention zero projection 初始恒等、AdaLN-Zero 初始 residual 为零、modulation head 和 VLM 条件梯度、timestep 条件、padding 不变性、Root/Body head 独立性、future-only 注入、stage freeze/unfreeze、checkpoint round-trip 和 HandNet detach 边界。

当前结果：

```text
11 passed
```

### 10.2 完整回归测试

忽略受 sandbox socket 权限影响的 server/toolbench 网络测试后：

```text
64 passed, 4 skipped
```

完整测试中剩余失败来自环境禁止创建 socket，并非模型或 AdaLN 逻辑失败。

### 10.3 GR00T/Arena 真实预检

预检报告：

```text
outputs/sonicx7_adaln_preflight/groot_preflight.json
```

已检查 700 个 episode、293,864 个训练窗口、7 个 Arena 任务、真实 VLM 输出 `[B, L, 2048]`、可变 valid-token 长度、AdaLN 零注入、Root/Body freeze 和 detach、stage 1/2 梯度、checkpoint round-trip 以及推理期间每个 plan 一次 VLM 调用。

### 10.4 四卡连续/断点续训验证

验证程序分别运行连续训练 4 步、训练 2 步保存 checkpoint、恢复到第 4 步，并比较模型参数、Adam 状态、scheduler、每卡 RNG 和 DDP 参数哈希。

结果文件：

```text
outputs/adaln_zero_ddp4_20261112h/resume_equivalence.json
```

结果：

```text
passed: true
max_parameter_error: 3.64e-12
max_optimizer_error: 5.82e-11
optimizer_scheduler_rng_checked: true
```

早期使用 48 步验证时观察到多卡 bf16 归约导致约 `1e-4` 量级的有界浮点差异。进一步定位确认训练状态、数据顺序和 RNG 没有漂移；最终短验证在相同 eager kernel 路径下达到近机器精度一致。生产阶段仍使用 denoiser compile 以提高吞吐。

## 11. 当前生产训练和评估流程

当前生产运行目录：

```text
/root/workspace/kimodo-policy-vlm-version/log/arena_sonicx7_groot/adaln_zero_ddp4_20261112h
```

训练命令等价于：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
.venv-groot/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 scripts/train.py \
  --config configs/arena_sonicx7_groot_adaln.yaml \
  --run-dir log/arena_sonicx7_groot/adaln_zero_ddp4_20261112h \
  --audit-distributed
```

训练结束后，watcher 会自动检查 `checkpoint_500000/training_state.pt`，运行 AdaLN checkpoint 的 `football`、`doubledesk`、`pp_box` 评估，再使用同一组 evaluation seeds 评估原 CrossAttention baseline，最后保存每个任务的成功率、每 seed 成功率、样本标准差、Wilson 95% 区间和相对 baseline 的绝对成功率差。

watcher 脚本：[`eval_arena_adaln_after_train.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/eval_arena_adaln_after_train.py:1)。在正式 500k checkpoint 和评估完成前，不把当前 loss 解释成 Arena 成功率，也不把论文中的 MimicGen/LIBERO 数字当作本地 Arena 结果。

## 12. 如何切换两个版本

AdaLN-Zero：

```yaml
model:
  groot_injection_mode: adaln_zero
```

原 CrossAttention：

```yaml
model:
  groot_injection_mode: cross_attention
```

两种模式共享数据集、VLM encoder、diffusion schedule、future-only 位置和评估脚本，但 adapter 参数不同，不能直接混用 checkpoint。公平对比时应保持 episode 清单、global batch、micro-batch、training seed、checkpoint step、DDIM step 数、执行 horizon、controller 和 evaluation seeds 一致。

## 13. 后续建议

当前先等待 AdaLN 500k checkpoint 和三任务正式评估。拿到结果后建议：

1. 对比 AdaLN 和 CrossAttention 的三任务成功率及 Wilson 区间。
2. 检查 AdaLN gate 的平均绝对值，以及 Root/Body 各层 gate 是否集中在浅层或深层。
3. 分析 timestep embedding 在不同 diffusion step 的 gate 强度。
4. 检查 VLM adapter 在第 10,000 步解冻后是否出现 loss 或成功率突变。
5. 如 gate 长时间接近零，再考虑提高 AdaLN 学习率；如收敛过快但泛化不足，再降低 AdaLN 学习率。
6. 完成当前匹配实验后，再考虑共享 Root/Body condition stem、减少 AdaLN head 数量或引入 layer-specific timestep gate。

## 14. 主要文件索引

| 文件 | 作用 |
|---|---|
| [`groot_conditioning.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/groot_conditioning.py) | VLM condition、原 CrossAttention、AdaLN-Zero、模式切换 |
| [`backbone.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/modules/backbone.py) | 在指定 Transformer layer 后注入 future hint，并传递 timestep |
| [`kimodo_policy.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/kimodo_vla/model/kimodo_policy.py) | 配置字段、模型组装、冻结和 forward 数据流 |
| [`train.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/train.py) | optimizer 分组、学习率调度、checkpoint 和指标记录 |
| [`check_groot.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/check_groot.py) | 真实 VLM/Arena 预检和 freeze/gradient audit |
| [`run_arena_stable.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/run_arena_stable.py) | DDP 连续训练、断点恢复验证和生产启动流程 |
| [`eval_arena_adaln_after_train.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/scripts/eval_arena_adaln_after_train.py) | 训练完成后的 AdaLN/CrossAttention 匹配评估 |
| [`test_groot_conditioning.py`](/root/workspace/kimodo-policy-vlm-version-model_improvement_lct1/tests/test_groot_conditioning.py) | AdaLN、padding、梯度、冻结和恢复测试 |
