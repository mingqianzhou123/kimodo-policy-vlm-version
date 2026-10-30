# 本次训练配置、视频及 Table 3 格式结果

全部结论读取本次实际 run 的 config.json、dataset.json 和官方 summary，不只依赖 YAML 默认值。

## 视频

全量视频目录：`log/toolbench/joint18_scratch40k_20261027_8gpu_supervision/evaluation_18x100/`。

目录结构：`<condition>/run-<id>/videos/episode_<seed>/<camera>_<success|failed>.mp4`。
共7,200段，覆盖1,800回合，每回合四路：`head_stereo_left`、`head_stereo_right`、`wrist_left`、`wrist_right`；640×360、50fps。成功与失败回合均保留。监督目录下 `video_index.csv` 为全部文件索引。

成功示例：BallMove-L1-R / run-0rajjw9g / episode_10003（推球）；IceBreak-L1-S / run-b1g3axqt / episode_10003（破冰）。

## Table 3 口径

论文 Table III：[原文](https://arxiv.org/html/2610.02089v1#S4.T3)；六格中数字顺序是 BallMove / BallRetrieve / IceBreak，D 对应本地环境名 R。单位为成功率百分比。

| 策略 | S-L0 | S-L1 | S-L2 | D-L0 | D-L1 | D-L2 | L1/L2 均值 | 全18均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kimodo-policy scratch 40k | 11 / 4 / 15 | 3 / 0 / 1 | 0 / 0 / 0 | 8 / 7 / 20 | 2 / 0 / 0 | 0 / 0 / 0 | 0.50 | 3.94 |
| 同策略：三场景均值 | 10.00 | 1.33 | 0.00 | 11.67 | 0.67 | 0.00 | 0.50 | 3.94 |

L1/L2 均值为12个条件的宏平均，6/1200=0.50%；全18均值为71/1800=3.9444%。每条件100回合、seeds10000–10099、最多3000个50Hz控制步。各条件均通过官方核验。第二行是同策略六组的三场景均值，不是第二次训练。

## 实际训练参数

| 参数 | 本次取值 |
| --- | --- |
| 数据 | 3,003条仿真演示；18条件；1,247,525个完整窗口 |
| 数据采样 | 全部 eligible windows 均匀采样，DistributedSampler；不是每任务等概率 |
| 步数 | 40,000次optimizer更新 |
| 硬件与批量 | 8×RTX5090；DDP；global batch128；每卡micro batch16；梯度累积1 |
| 精度与随机种子 | BF16 autocast；训练seed1000；hand_init_seed3407 |
| 输入 | 单路头部左相机；RGB224×224；冻结DINOv3 ViT-L/16；LLM2Vec缓存特征4096维 |
| 时间表示 | 模型30Hz；history100帧；future/action chunk50帧；控制50Hz |
| 可训练模块 | ControlNet四层+连续手头；91,715,990参数 |
| ControlNet设置 | motion_mask_mode=concat；control_fusion_mode=both；scale=1；detach_root_control_for_body=true |
| 手头 | hidden256；4层；4heads；FFN1024；每手一个连续闭合度 |
| 初始化 | policy_checkpoint=null；policy_419h_loaded=false；冻结Kimodo-G1运动先验、DINOv3、LLM2Vec；ControlNet复制运动先验层并初始化适配层，手头随机初始化 |
| 优化器 | fused AdamW；betas(0.95,0.999)；eps1e-8 |
| 初始峰值LR | ControlNet骨干3e-5；适配层1e-4；手头1e-4 |
| Weight decay | 1e-6；bias和一维参数0decay |
| LR调度 | 1000步线性warmup，之后cosine，最低为峰值0.1，即3e-6/1e-5/1e-5 |
| 损失 | 4×root MSE + body MSE + hand-state MSE + 0.2×hand-transition L1；按有效帧mask计算 |
| 梯度裁剪 | ControlNet与手头分别clip norm1.0 |
| EMA | 不使用（ema_decays=[]） |
| 数据增强 | image_shift=0；proprio_mask_prob=0 |
| 数据加载 | 每个rank4个workers（8卡共32个） |
| 日志/保存 | 每10步记录；每2000步保存；最终评估固定checkpoint_40000 |
| 实时监督 | GPU遥测；非有限值/进程退出/停滞检测；10k/20k/30k各18条件1回合诊断 |

评估推理独立设置：BF16、10 diffusion steps、execution_frames50、rtc_overlap_frames0、turn=operator、root_boundary=extrapolate；每回合从1000起递增的噪声seed，每条件重置。50个模型预测帧对应81/82个实际控制步，并非50个控制步。

本实验的scratch指任务策略不加载419h等策略权重；仍使用完整架构的冻结基础模型，不是全模型随机初始化。

实际配置：`log/toolbench/joint18_scratch40k_20261027_8gpu/config.json`。初始化记录：同目录`initialization.json`。启动配置：`configs/toolbench.yaml`（继承bigym2.yaml）。本次只有一个训练种子，不包含多训练种子的均值或标准差。
