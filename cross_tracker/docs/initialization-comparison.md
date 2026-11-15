# 两套初始化对照（2026-11-15）

用户要求保留预训练微调，并增加直接用示教训练的from-scratch对照。箱子搬放、SONIC/TWIST2/双源三组，沿用每来源43训练/11验证及seed42、场景1000–1019。

预训练三组已经完成10k/10k/20k；第二套拟用相同预算做配对初筛，不能把相同步数下的比较解释为各自充分收敛。若从头训练尚未学会，是否延长训练须统一预先定义规则，不能只按某来源的成功率挑预算。

用户已确认采用第一种初始化定义；第二种仅保留为区别说明：

- Kimodo默认不加载pt_419h：不继承419小时示教预训练的策略权重，但保留冻结Kimodo运动先验；ControlNet部分层复制自该先验，新适配层和手部头新初始化，DINO和文本编码器冻结。应准确标注为“without demonstration-policy pretraining”，不能宣称所有动作模型参数随机初始化。
- 完全随机动作策略：Kimodo运动网络也重置并参与训练，需新增初始化、优化参数分组和checkpoint加载逻辑；当前入口会冻结该网络，直接省略init参数不能实现。

两套各3个朴素baseline，共6个；CCT当前先准备与已完成双源微调匹配的正式20k组。CCT的scratch版本未提交，可在初始化定义确定后纳入对应比较，不能拿不同初始化的CCT和朴素混训直接归因于方法收益。

CCT GPU过拟合、8卡短程、HTTP检查已通过。正式入口scripts/run_cct_baseline.py已准备，要求SONIC和TWIST2真实仿真报告匹配同一checkpoint和上下文银行，检查通过前拒绝启动。正式run：box-cct-sonic-twist2-s42-20261115，状态planned，未提交。T2M-PLUS仍排队；SenseCore sgresearch三页规格均为A800/CPU，无RTX规格。当前不能真实完成RGB物理闭环。
