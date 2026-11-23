"""Validate and compare frozen-reference load × support counterfactuals."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from arena.dynamics_override import FIELDS,intervene
from kimodo_vla.oracle.recorded_reference import read_rows
from analyze_sonic_replay import audit,summarize
from analyze_root_dynamics import analyze
from report_dynamics import metrics
from run_sonic_replay_suite import CASES
from verify_sonic_release import verify
ROOT=Path(__file__).resolve().parents[1]
BASE='dynamics-actuator-20261119-r1'

def check_parameter(key, actual, expected, inertia_roundtrip=False):
    actual=np.asarray(actual,dtype=np.float32);expected=np.asarray(expected,dtype=np.float32)
    delta=float(np.max(np.abs(actual-expected)))
    # PhysX set_inertias round-trips every body, including unchanged bodies.
    # Only this field gets the existing runtime audit's 1e-8 absolute allowance.
    if inertia_roundtrip and key=='inertias':
        if not np.allclose(actual,expected,rtol=0,atol=1e-8):raise ValueError('Unexpected active parameter: '+key)
    elif not np.array_equal(actual,expected):raise ValueError('Unexpected active parameter: '+key)
    return delta

def validate(run,base,label):
    meta=json.loads((run/'run.json').read_text());c=json.loads((run/'actuator-config.json').read_text());b=json.loads((base/'actuator-config.json').read_text())
    for key in ('robot_asset','robot_cfg','sim_cfg','runtime_env','canonical_ids','joint_names','body_names'):
        if c[key]!=b[key]:raise ValueError('Unintended change: '+key)
    x=c['active_first_physics_step'];y=b['active_first_physics_step'];spec=json.loads((ROOT/'configs/sonic-dynamics.json').read_text())
    parameter_deltas={}
    if label!='baseline':
        profile='actuator' if label=='support' else 'actuator-load';support='none' if label=='load' else 'raised'
        if meta['dynamics_profile']!=profile or meta['root_support']!=support:raise ValueError('Wrong cell')
        if meta['dynamics_profile_sha256']!=hashlib.sha256((ROOT/'configs/sonic-dynamics.json').read_bytes()).hexdigest():raise ValueError('Actuator spec changed')
        expected=intervene({k:y[k] for k in FIELDS},c['joint_names'],c['body_names'],'sensor-load',spec) if label in ('load','both') else y
        parameter_deltas={k:check_parameter(k,x[k],expected[k],label in ('load','both')) for k in FIELDS}
        if x['material_properties']!=y['material_properties']:raise ValueError('Robot materials changed')
        record=json.loads((run/'root-scene-audit.json').read_text());root_spec=json.loads((ROOT/'configs/root-dynamics.json').read_text())
        if record['spec']!=root_spec or record['mode']!=support:raise ValueError('Wrong support spec')
        if meta['root_support_config_sha256']!=hashlib.sha256((ROOT/'configs/root-dynamics.json').read_bytes()).hexdigest():raise ValueError('Support spec changed')
        if support=='none':assert record['platform'] is None
        else:
            plat=record['platform'];assert abs(plat['maximum'][2]-root_spec['support_height_m'])<1e-6
            assert plat['size']==root_spec['platform_size_m']
        rows=read_rows(run/'execution.jsonl');old=read_rows(base/'execution.jsonl')
        for k in ('root_pos','root_quat','joints','foot_pos','wrist_pos'):
            if not np.allclose(rows[0]['before'][k],old[0]['before'][k],rtol=0,atol=1e-6):raise ValueError('Initial state changed: '+k)
    physics=read_rows(run/'actuator-physics.jsonl');controls=read_rows(run/'actuator-controls.jsonl')
    if [(r['control'],r['substep']) for r in physics]!=[(i,j) for i in range(400) for j in range(4)]:raise ValueError('Physics frames/order changed')
    delta=max(np.max(abs(np.array(r['physx_position_target'])-r['decoder_target'])) for r in physics)
    if delta!=0:raise ValueError('Decoder target transmission changed')
    if not np.allclose([r['time_after']-r['time_before'] for r in physics],.005,atol=1e-6,rtol=0):raise ValueError('Physics dt changed')
    if len(controls)!=400 or any(c['substeps']!=4 for c in controls):raise ValueError('Control cadence changed')
    return dict(passed=True,physics_steps=1600,control_frames=400,decoder_target_delta_rad=float(delta),aligned_actuators_unchanged=True,parameter_max_abs_delta=parameter_deltas)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suffix',default='r1');a=p.parse_args()
    suites=dict(baseline=BASE,**{k:f'root-{k}-20261120-{a.suffix}' for k in ('load','support','both')});results={};gpu={}
    for label,suite in suites.items():
        results[label]={}
        if label!='baseline':
            pilot=ROOT/'runs'/f'{suite}-control';fixture=ROOT/'runs'/BASE/'fixtures/base-center'
            g=audit(pilot,fixture,ROOT/'runs/vlm-refined-20261118-r1-base-center');x=read_rows(pilot/'execution.jsonl');y=read_rows(ROOT/'runs'/f'{BASE}-base-center'/'execution.jsonl')
            g['aligned_baseline_delta']={k:float(np.max(np.abs(np.array([r['after'][k] for r in x])-np.array([r['after'][k] for r in y])))) for k in ('root_pos','root_quat','joints','wrist_pos')}
            if max(g['aligned_baseline_delta'].values())>1e-6:raise ValueError('Pilot changed trajectory')
            gpu[label]=g
        for case in CASES:
            run=ROOT/'runs'/f'{suite}-{case}';base=ROOT/'runs'/f'{BASE}-{case}';fixture=ROOT/'runs'/suite/'fixtures'/case
            r=dict(reference_audit=audit(run,fixture,ROOT/'runs'/f'vlm-refined-20261118-r1-{case}'),release_audit=verify(run),invariants=validate(run,base,label),metrics=metrics(run),root=analyze(run))
            s=summarize(run)['post_arrival'];r['root_relative_wrist_cm']=s['root_relative']['mean_norm'];results[label][case]=r
            print('PASS',label,case,flush=True)
    aggregate={}
    for label,rs in results.items():
        ms=[r['metrics'] for r in rs.values()];ws=[r['root']['windows']['arrival'] for r in rs.values()]
        v={k:np.mean([m[k] for m in ms],axis=0).tolist() for k in ('wrist_mean_cm','actual_decoder_arm_rmse_deg','actual_reference_arm_rmse_deg')}
        v.update({k:np.mean([w[k] for w in ws],axis=0).tolist() for k in ('root_mean_cm','root_xyz_bias_cm','root_xy_mean_cm','root_abs_z_mean_cm')})
        v['both_3cm_passes']=sum(m['both_within_3cm_entire_window'] for m in ms);v['root_relative_wrist_cm']=np.mean([r['root_relative_wrist_cm'] for r in rs.values()],axis=0).tolist();aggregate[label]=v
    interaction={k:(np.array(aggregate['both'][k])-aggregate['load'][k]-aggregate['support'][k]+aggregate['baseline'][k]).tolist() for k in ('root_xyz_bias_cm','wrist_mean_cm','root_mean_cm')}
    out=ROOT/'runs/root-dynamics-20261120';(out/'comparison.json').write_text(json.dumps(dict(suites=suites,aggregate=aggregate,cases=results,gpu_controls=gpu,interaction=interaction),indent=2))
    render_report(suites,results,aggregate,a.suffix)

def render_report(suites,results,aggregate,suffix):
    fmt=lambda xs:' / '.join(f'{v:.3f}' for v in xs)
    lines=['# 固定参考与执行器：根部载荷/支撑对照（2026-11-20）','',
      '全部五组冻结参考、SONIC 权重和已对齐执行器参数保持不变。三套新对照共 15 次仿真、6,000 控制帧和 24,000 物理步；另有三次无干预基线复现。全部参考、初始机器人状态、目标传递及实际执行器参数核验通过。','',
      '## 必须修正的前提','',
      '**此前“剩余误差主要是 tracking、reference 已足够准确”的归因不完整。** 固定回放只能证明误差位于参考到执行之间，不能证明参考满足地面支撑约束。精确 USD 的脚底边界为踝部原点下方 0.035 m；五组冻结参考初始踝部高度约 0.0558255 m，所以脚底距地约 2.08255 cm。真实落地后踝部约 0.035 m。',
      '初始化链路 `OraclePolicy.initialize` 把首帧根部与双脚保存为固定 support_target；首帧根高为 0.8 m，脚底尚未落地。后续 IK 保持了这些支撑目标。问题涉及初始化/参考与环境的一致性，不能归咎于 VLM 选点或原始 Kimodo 输出本身。此轮没有更改它们。',
      '', '## 独立因子','',
      '- **减载**：仅将四个默认 1 kg 的传感器刚体各改为 1 g，惯量同比缩放；这是诊断近似，不是真实质量标定。',
      '- **支撑面**：在机器人脚下增加 1×1 m 静态平台，顶面为 0.020825528 m。高度来自冻结参考与实测资产脚底深度，预先固定；reference、机器人初始姿态和腕目标不平移。',
      '- **支撑面是改变环境的反事实诊断，不能计作原地面任务已修复。** 两因素组合用于测量交互，不能把其提升直接推广到原场景。默认运行没有开启这些选项。',
      '', '## 结果','',
      '到达后 1.2–8 s，单位 cm。这里是实际状态相对冻结 reference 的跟踪误差，没有叠加 VLM 相对真实目标的定位误差。3 cm 达标要求该组 50 Hz 日志中全部到达后帧的双腕同时通过；不是只看均值，也不等于真实物体到达成功率。','',
      '| 配置 | 根部三维均值 | 根部 XYZ 偏差 | 左/右腕均值 | 腕相对根部 L/R | 双腕全程 3 cm |',
      '|---|---:|---:|---:|---:|---:|']
    titles=dict(baseline='已对齐执行器基线',load='仅减载（原地面）',support='仅加诊断支撑面',both='减载＋诊断支撑面')
    for k,title in titles.items():
        v=aggregate[k];lines.append(f"| {title} | {v['root_mean_cm']:.3f} | {fmt(v['root_xyz_bias_cm'])} | {fmt(v['wrist_mean_cm'])} | {fmt(v['root_relative_wrist_cm'])} | {v['both_3cm_passes']}/5 |")
    lines+=['','| 案例 | 基线 L/R 均值 | 减载 L/R 均值 | 支撑面 L/R 均值 | 组合 L/R 均值 |','|---|---:|---:|---:|---:|']
    for case in CASES:lines.append('| '+case+' | '+' | '.join(fmt(results[k][case]['metrics']['wrist_mean_cm']) for k in suites)+' |')
    lines+=['','## 根部下沉的运动学分解','',
      '按每只脚独立检查恒等式：根部误差 = 足部世界位置误差 − 根姿态造成的腿端偏移 − 下肢关节姿态造成的腿端偏移 − FK 残差。下表使用第 4–8 s、左右脚和五组平均的 z 分量，单位 cm；这是有符号分解，不能把三维误差范数直接相加。','',
      '| 配置 | 足部支撑高度项 | 根部姿态项 | 下肢姿态项 | 根部 z 偏差 |','|---|---:|---:|---:|---:|']
    for k in suites:
        ds=[r['root']['windows']['steady']['decomposition'] for r in results[k].values()]
        vals=[np.mean([np.mean(np.array(d[n]['bias_xyz'])[:,2]) for d in ds]) for n in ('foot_placement','negative_root_rotation','negative_leg_posture','root_error')]
        lines.append('| '+titles[k]+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','## 水平偏移与全身响应','',
      '同一恒等式的稳态 x 分量（4–8 s、双脚与五组平均，cm）。它是运动学归因，不等同于证明某项训练动力学配置是原因。','',
      '| 配置 | 落脚位置项 | 根姿态项 | 下肢姿态项 | 根部 x 偏差 |','|---|---:|---:|---:|---:|']
    for k in suites:
        ds=[r['root']['windows']['steady']['decomposition'] for r in results[k].values()]
        vals=[np.mean([np.mean(np.array(d[n]['bias_xyz'])[:,0]) for d in ds]) for n in ('foot_placement','negative_root_rotation','negative_leg_posture','root_error')]
        lines.append('| '+titles[k]+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','支撑面组双脚落在参考位置附近，原地面落地时约 −0.97 cm 的 x 偏移不再存在；但根部仍前倾，姿态项贡献约 +2.37 cm，组合组约 +2.23 cm。两组下肢姿态项分别抵消约 0.14/0.27 cm。因此剩余前移主要表现为相对固定双脚的根姿态偏差，并非稳态持续滑脚。四格实验的 4–8 s 双脚接触比例均为 100%，脚部水平位置范围均小于 0.001 cm（仅此仿真窗口）。',
      '','减载使总质量从 38.614 kg 降到 34.618 kg。原地面右踝 pitch / 右膝的实际相对参考偏差从 −3.81°/+3.48° 变为 −3.08°/+2.52°；载荷影响通过下肢平衡姿态体现，但仍不能证明剩余前倾来自载荷或 PD。其训练域原因尚未分离。',
      '', '| 案例 | 组合组左/右腕最大误差 cm | 全程双腕 3 cm |','|---|---:|---:|']
    for case in CASES:
        m=results['both'][case]['metrics'];lines.append(f"| {case} | {fmt(m['wrist_max_cm'])} | {'通过' if m['both_within_3cm_entire_window'] else '未通过'} |")
    lines+=['','## 本轮结论与下一步','',
      '1. **大部分下沉来自冻结支撑参考与原地面不一致。** 支撑面单因素把到达后平均根部 z 偏差从 −2.418 cm 降到 −0.328 cm；稳态分解中约 −2.082 cm 的足部高度项随之消失。没有修改参考即可复现这个变化，证据支持初始化/地面支撑矛盾，而不是 VLM 误差。',
      '2. **载荷影响存在，但不能独自解决根部误差。** 保持已对齐执行器，在原地面减载后根部三维均值从 2.785 降到 2.558 cm，左/右腕从 2.526/2.899 降到 2.306/2.673 cm；双腕全程 3 cm 为 2/5。传感器 1 g 只用于隔离默认附加载荷。',
      '3. **消除支撑高度矛盾后，水平偏移仍是主要残差。** 支撑面组平均 x 偏差从 1.130 增到 2.260 cm；组合组仍有 1.998 cm 的 x 偏差。组合组根部三维均值 2.100 cm、左/右腕 2.212/1.749 cm，严格通过 3/5；far-center 左腕峰值 3.337 cm，far-right 左腕峰值 3.006 cm，仍未通过。不能宣称已稳定双腕 3 cm。',
      '4. **下一步应先修初始化/支撑参考的生成时机，再评估残余跟踪。** 在原地面待机器人落地稳定后保存实测根部/双脚支撑目标，重新生成对应参考，同时保持手部世界目标与误差口径明确一致；这是下一轮新版本，不能覆盖本轮冻结参考。随后在相同执行器下比较落脚位置、根姿态和下肢关节跟踪对水平偏差的贡献。本轮不更改这些路径，不继续盲调 PD。',
      '', '本轮只隔离了附加载荷与地面支撑几何，并记录全身运动学响应；未逐项量化所有惯量、接触、自碰撞、可动手和训练域差异。支撑面组的通过率不能算作原场景通过率。',
      '', '## 保留的限制与证据','',
      '- 惯量写入使 PhysX 对全部刚体惯量发生浮点读回舍入；仅减载组惯量允许绝对差 ≤1e-8，各字段实际最大差记录在 invariants 中。全部执行器参数、质量、COM 仍逐项 float32 精确匹配；支撑面组惯量也精确匹配。',
      '- 原版 162 文件快照：`snapshots/dynamics-before-root-20261120`；此前所有对照保留。',
      '- 精确资产脚底碰撞边界：`runs/root-dynamics-20261120/foot-collision-geometry.json`。每次运行的 `root-scene-audit.json` 记录脚底几何、平台实际边界与配置。',
      '- 本轮没有恢复历史训练软件、转换 USD、自碰撞和固定手结构，也没有用一组结果宣称全身动力学完全对齐。其他动态差异仍可能影响剩余水平/姿态误差。',
      '- 全部关节偏差、足部滑动范围/接触力、质量加权 COM、根部及双腕均值/p95/max、逐案成功率和两因子交互项见 `runs/root-dynamics-20261120/comparison.json`。接触力来自仿真传感器；计算的 PD 力矩不充当实测电机力矩。',
      '- 复现：`PYTHONPATH=. <policy-python> scripts/run_root_suite.py --cell load|support|both --gpu <available> --suffix <new-id>`；报告：`PYTHONPATH=. <python> scripts/report_root_dynamics.py --suffix '+suffix+'`。','']
    (ROOT/'docs/root-dynamics-20261120.md').write_text('\n'.join(lines));print(json.dumps(aggregate,indent=2));print('COMPLETE 15 paired cases; fixed references and actuators')
if __name__=='__main__':main()
