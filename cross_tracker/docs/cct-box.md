# 箱子搬放：CCT 适配 Kimodo

用户已选择保留控制器文字元数据的版本。实现直接在 SenseCore CPU CCI 编写，新增文件全部位于 cross_tracker。这是按论文结构实现并适配 Kimodo 的对照，不称为官方代码完整复现。正式作业尚未提交；GPU 过拟合和实际评测接口仍需通过。

## 公平对照

对比 box-sonic-twist2-s42-20261114：同一 Kimodo-VLA、官方 pt_419h 初始化、修正标签和训练名单；SONIC/TWIST2 各43条训练、11条验证。seed42，20,000次更新，全局batch8，8卡各micro1，warmup666，只评20k checkpoint。评测沿用SONIC/TWIST2、场景1000–1019和冻结闭环参数。CCT独立作业，不修改前三个作业。

上下文取同任务、同tracker的另外三条训练示教，排除整个目标示教，绝不从验证集或评测回放取上下文。每条预选43个图像时刻，组合时取43/43/42，共128个时刻，覆盖三条示教的完整有效时间跨度。没有增加独立示教数量；本轮只支持已见tracker、同任务结论，不声称跨任务或未见tracker零样本能力。

configs/cct-eval-context.json 在评测前固定每个tracker的三条上下文示教，不得根据成功率更换。

## 数据与实现

- 图像取已核验的30Hz转换缓存，按原转换规则回查50Hz原始行。转移为实际状态35[t]、修正指令40[t]、实际状态35[t+1]，共110维。状态含根朝向6D和29个关节位置，不把指令当实际状态，不跨排除帧或示教边界。
- 下一状态是原始相邻行，间隔20ms，不是稀疏上下文中的下一张图。保存转换帧、原始行、下一行和哈希。上下文归一化只拟合86条训练示教的采样转移；Kimodo原有归一化不变。
- 冻结DINOv3提取patch。交互编码器宽512、256个查询、6层latent self-attention，投影到4096维；额外增加示教槽位embedding，区分三条独立轨迹。
- 控制器描述经冻结LLM2Vec编码，与256个交互token、原任务文本token一起输入Kimodo。解除原模型仅保留一个文本token的截断；普通baseline不受影响。
- src/cct_policy.py 薄适配模型，scripts/train_cct_worker.py 复用共享DDP训练器。新增参数进入优化器和checkpoint。src/cct_eval.py 显式加载CCT权重和固定上下文，拒绝把普通初始化权重当CCT评测。
- 暂不支持精确恢复上下文采样器，worker拒绝 --resume；新起跑不能称为原运行续训。

## 与论文的差异

[原论文](https://arxiv.org/html/2610.39280)使用文字元数据、图像及指令—实际响应上下文，经Perceiver式编码器提供给策略。本适配保留这些输入及瓶颈结构，但使用Kimodo、DINOv3、LLM2Vec和扩散MSE目标，未替换成原论文的π0.5、DINOv2与flow matching。只缓存上下文编码结果，不声称实现π0.5的KV缓存。

描述包含已知控制器名称和40维接口；采集对应增益未独立核验，明确写为未指定。没有伪造增益，也未对黑箱tracker使用未经验证的力矩等价合成重标注。新增参数、显存及耗时应单独报告，相同步数不等于相同计算量。官网代码链接此前返回404，这里是论文结构复现而非官方代码移植。

## 执行顺序

1. prepare_cct.py：校验权重、真实行号和RGB，生成86条训练上下文及文字，全部完成才写manifest.json。CCI共享LLM2Vec配置曾只改路径；已隔离保存与发布清单一致的原始配置，派生路径不修改共享文件。
2. tests/test_cct.py：隔离、采样对应、梯度、文字前缀及保存读回。
3. check_cct_model.py <新的diagnostic运行名> --device cpu：真实前后向、checkpoint读回、一次预测，无优化更新。GPU再用 --device cuda --steps 200 做固定窗口过拟合。
4. 核验CCT专用评测入口、8卡短程和配置快照后，登记并提交独立8×A800正式run。检查未通过不启动20k训练。

CPU通过不等于GPU或任务成功率通过。备份规则见 [备份与同步](backup-sync.md)。
