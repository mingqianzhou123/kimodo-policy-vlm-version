"""Evidence and decision rules for the staged ToolBench experiment report."""
from __future__ import annotations

import json
import math
from pathlib import Path


def counts(row):
    return (f"{row['primary_l12']:.2%}（{row['primary_successes']}/{row['primary_episodes']}）",
            f"{row['auxiliary_all18']:.2%}（{row['total_successes']}/{row['total_episodes']}）")


def identity(row):
    return tuple(row.get(k) for k in ('checkpoint_sha256','checkpoint_config_sha256','stage',
        'policy_seed','seed_start','episodes','max_steps','execution','diffusion','adapter_sha256'))


def unique(rows):
    seen=set();result=[]
    for row in rows:
        key=identity(row)
        if key not in seen:result.append(row);seen.add(key)
    return result


def wilson(successes,total):
    p=successes/total;z=1.96;d=1+z*z/total
    middle=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return f'{middle-half:.2%}–{middle+half:.2%}'


def near_tie(a,b):
    se=math.sqrt(sum(max(r['primary_l12']*(1-r['primary_l12']),1/r['primary_episodes'])/r['primary_episodes'] for r in (a,b)))
    return abs(a['primary_l12']-b['primary_l12'])<=max(.005,1.96*se)+1e-12


def evidence_table(rows):
    result=['| 策略/评估记录 | 训练步数 | 回合/条件 | L1/L2 成功率（成功/回合） | 全18成功率（成功/回合） |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        p,a=counts(r)
        result.append(f"| [{r['name']}](evaluation/{r['name']}/summary.json) | {r['training_step']:,} | {r['episodes']} | {p} | {a} |")
    return result+['']


def detailed_analysis(root,trials):
    root=Path(root);rows=unique(trials)
    out=['## 统计口径与可比较范围', '',
         '本报告按“已验证数据 → 比较依据 → 阶段决定 → 尚未证明的结论”记录实验判断。'
         '训练 loss 只用于检查数值稳定性和收敛趋势，不替代仿真成功率。', '',
         '- 主指标：12个 L1/L2 条件成功率的等权平均；辅助指标：全部18条件的等权平均。'
         '每个条件回合数相同时，分别等于成功次数/（12×回合数）及成功次数/（18×回合数）。',
         '- screen：环境种子20000–20049，每条件50回合，主指标分母600，全18分母900；'
         'confirm：22000–22099，每条件100回合，分母分别1200、1800。'
         '两组独立场景集合应分别比较，不能合并分母或择优当作正式成绩。',
         '- 六列 Table 3 对应 S-L0/L1/L2、D-L0/L1/L2；D使用官方R环境。'
         '三场景为 BallMove、BallRetrieve、IceBreak，原始分场景率见 stage_table3.md 附表。',
         '- 同一权重、推理设置、环境种子和已验证凭据的缓存别名只算一次证据。'
         '尤其200k的 extension_confirm 与 coarse_budget_confirm 是复用关系，不是两次独立成功率验证。',
         '- 同 seed 1000 的多个预算不是多个独立训练种子。正式结果须在配置锁定后单独报告；'
         '当前开发种子成绩不作为官方标准测试成绩。', '']
    out+=['## 阶段1：50k采样初筛与临时固定', '']
    sample=[r for r in rows if r['stage']=='screen' and r['training_step']==50000
            and r['name'].split('_simple1')[0] in ('window_uniform','task_uniform','level_balanced')]
    out+=evidence_table(sample)
    out+=['三个采样候选的 L1/L2 都只有1/600次成功（0.17%），主指标无法区分。'
          '全18的差别主要来自L0，不能据此声称某种采样改善了执行能力。'
          'window_uniform 的辅助指标最高，因此原流程让它与 task_uniform 进入确认；'
          '这只是筛选顺序，不是采样优越性的结论。', '']
    confirmations=[r for r in rows if r['name'].startswith('sampling_confirm_')]
    out+=evidence_table(confirmations)
    out+=['确认中 task_uniform 为4/1200，window_uniform 为3/1200，仅相差一次成功，'
          '即0.083个百分点。如此稀疏的成功事件不足以支持可靠排序。'
          '用户要求优先提高L1/L2，因此停止继续区分采样，临时固定 task_uniform，'
          '并保持SIMPLE峰值学习率1e-5/3e-5/3e-5、容量、损失权重不变。'
          '两个追加200回合采样评估都只完成38回合即取消，其部分结果明确排除。'
          '采样选择属于节省预算的暂定方案，不能写成“最优采样已找到”。', '']
    out+=['## 阶段2：固定200k cosine horizon的连续预算摸底', '']
    coarse=sorted([r for r in rows if r['name'].startswith('task_uniform_simple1_seed1000_e50_s')
                   and r['stage']=='screen'],key=lambda r:r['training_step'])
    out+=evidence_table(coarse)
    out+=['该阶段用同一scratch轨迹的50k间隔checkpoint比较，E50/D10、数据、采样、峰值学习率一致。'
          'cosine horizon始终200k，200k后学习率维持峰值的0.1倍。'
          '因此250k/300k摸底是“已衰减到底后继续训练”，不是horizon分别250k/300k的完整训练方案。', '',
          '50k→100k→150k时主指标为1→4→13次成功，说明更长训练在这组开发场景上有收益；'
          '150k→200k仅13→14次成功，出现平台迹象。延长到250k仍14次、300k到18次，'
          '改善远小于早期，且L2仍很低。全18从2.00%升至约10.7%–10.9%，'
          '其中相当部分来自L0；这不能替代主指标的提升。', '']
    out+=['## 阶段3：150k/200k确认与是否延长预算', '']
    late=[r for r in rows if r['name'].startswith('extension_confirm_')]
    out+=evidence_table(sorted(late,key=lambda r:r['training_step']))
    out+=['150k与200k确认的L1/L2分别19/1200、20/1200，增益只有一次成功；'
          '辅助指标反而从7.89%降至6.56%。因此既不能宣称200k显著优于150k，'
          '也不能只看下降的辅助指标就否定主指标。', '',
          '预先约定的延长规则为：200k确认主指标高于150k，或晚期筛选波动≥0.5个百分点、'
          '确认差绝对值<0.5个百分点且满足保守近似打平规则时，继续250k/300k摸底。'
          '本次确认增益0.083个百分点，满足第一条，故扩展至300k。'
          '这是执行已登记探索规则，不等于统计上已证明延长预算有效。', '']
    out+=['## 阶段4：250k/300k确认与预算候选选择', '']
    top=sorted([r for r in trials if r['name'].startswith('coarse_budget_confirm_')],key=lambda r:r['training_step'])
    out+=evidence_table(top)
    if len(top)>=2:
        ordered=sorted(top,key=lambda r:(r['primary_l12'],r['auxiliary_all18']),reverse=True)
        best,second=ordered[:2]
        gap=100*(best['primary_l12']-second['primary_l12'])
        out += [f"确认点估计第一为{best['training_step']//1000}k，第二为{second['training_step']//1000}k，"
                f"主指标相差{gap:.3f}个百分点；按预定保守近似打平规则：{'接近' if near_tie(best,second) else '不接近'}。"
                '该规则是安排追加训练种子的资源分配启发式，不是配对显著性检验。', '']
    out+=['300k初筛为3.00%（18/600），换确认种子后为1.50%（18/1200）；'
          '250k初筛2.33%、确认2.17%，确认排名第一。'
          '这说明300k的初筛优势未跨开发场景集合保持，不能选取初筛最高值当作最终性能。'
          '排名按“确认L1/L2优先、全18辅助”得出250k、200k两个候选；'
          '它们相邻50k，故没有触发再加入相邻预算。', '',
          '确认中250k比200k多6次L1/L2成功。作为稀疏计数的量级参考：', '',
          '| 预算 | L1/L2成功数 | 汇总二项Wilson约95%区间（描述用途） |',
          '| --- | ---: | ---: |']
    for r in top:
        out.append(f"| {r['training_step']//1000}k | {r['primary_successes']}/{r['primary_episodes']} | {wilson(r['primary_successes'],r['primary_episodes'])} |")
    out+=['', '这些区间有明显重叠；各条件成功概率并不相同，区间仅帮助理解计数量级，'
          '不是正式的分层或配对检验，更不度量训练随机种子方差。'
          '目前可说250k是已测配置中的确认点估计最好，不能声称已找到稳定的全局最优。', '']
    out+=['## 阶段5：预算匹配重训、200k等价复用与资源清理', '',
          '预算匹配的目的是让max_steps和cosine horizon同时等于候选预算，'
          '而不是为每个候选机械地重复同seed训练。250k的horizon由200k改成250k，'
          '整个学习率曲线改变，因此需要新scratch训练；200k的horizon本来就是200k，'
          '只改变停止上限不会改变前200k的训练定义，所以应复用原200k权重及对应评估。', '',
          '修订9纠正了此前调度遗漏：对比全部配置，只在两边均显式指定decay_steps、'
          '其它字段完全一致且源运行步数上限足够时，允许忽略max_steps这个停止字段。'
          '复用前核对checkpoint实际步数、权重/config SHA和scratch初始化；'
          '保留原config和官方receipt，不改文件来伪造缓存身份。', '']
    reuse_file=root/'training_reuse.json'
    if reuse_file.exists():
        for name,r in json.loads(reuse_file.read_text()).items():
            out += [f"- `{name}` → `{r['checkpoint']}`。原因：{r['reason']}。"]
        out.append('')
    cleanup=root/'duplicate200k_cleanup.json'
    if cleanup.exists():
        c=json.loads(cleanup.read_text())
        out += [f"重复训练取消在step {c['last_recorded_step']:,}；已回收{c['removed_bytes']/1024**3:.2f} GiB冗余权重/优化器文件。"
                '保留配置、训练日志、metrics和取消记录；该部分运行不参与选参，也不作为独立seed。', '']
    matched=[r for r in rows if r['name'].startswith('budget_fit_')]
    if matched:out+=evidence_table(matched)
    else:out+=['目前尚无新250k horizon匹配模型的完整18条件成绩，不能拿摸底250k的9.89%当作该重训模型的成绩。', '']
    out+=['修订9原定（修订10起暂停）：候选匹配训练完成后先初筛再确认；若前两名主指标满足近似打平规则，'
          '各补训练seed1001，再按两训练种子的主指标均值、辅助指标均值选择。'
          '同seed重复训练不替代这一稳定性验证。', '']
    out+=['## 修订9后续阶段的判断标准（修订10起暂停）与证据边界', '']
    future=[r for r in rows if not (r['name'].startswith(('window_uniform_','task_uniform_simple1_',
            'level_balanced_','sampling_confirm_','extension_confirm_','coarse_budget_confirm_','budget_fit_')))]
    if future:out+=evidence_table(future)
    out += ['1. 训练预算确定后，在选定权重上比较执行帧数E50/30/20/10，再比较扩散步数D10/20/30；'
            '候选必须使用相同开发种子和回合数。若有提前利用空闲GPU的评估，明确标为候选预取，'
            '只有对应预算最终入选时才可复用，不反向改变预算选择。',
            '2. 较短执行段可能更频繁接收图像和状态反馈，但也改变历史分布和每秒推理成本；'
            '它是否提高成功率必须由实测判断。若采用新的E，需要对应历史的scratch重训，'
            '并检查选定预算及±50k邻居，再与同扩散设置的E50参考确认比较。',
            '3. 更多扩散步数增加采样计算，但不保证提高成功率。只用开发确认结果选参，'
            '按既定打平规则处理成本，不用最终测试反复挑配置。',
            '4. 最终锁定配置后完成训练seed1000/1001/1002的正式18×100测试，'
            '环境种子10000–10099，报告各seed和均值±样本标准差；另做30000–30099留出评估。',
            '5. 当前结论仅适用于固定完整模型及固定适配器。尚未完成的阶段没有成功率结论；'
            '低成功率不能单凭loss诊断为学习率、容量或接口中的某一项原因。'
            '已验证接口并不证明表达能力与闭环反馈足够，后续须结合实际rollout视频定位失败环节。', '']
    locked=root/'locked_configuration.json'
    out += [f"当前配置锁定状态：{'已锁定，详见locked_configuration.json' if locked.exists() else '尚未锁定'}。"
            f'本报告基于{len(trials)}条已核验记录，其中{len(rows)}条为去除相同凭据别名后的证据；'
            '同权重跨不同环境种子的结果仍分别保留。', '']
    return '\n'.join(out)
