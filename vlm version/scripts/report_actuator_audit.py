"""Reproduce the five-case actuator audit and its Chinese evidence report."""
import argparse
import ast
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from kimodo_vla.core.types import G1_JOINT_NAMES
from analyze_actuator_audit import summarize
from analyze_sonic_replay import audit, summarize as replay_summary
from run_sonic_replay_suite import CASES

ROOT=Path(__file__).resolve().parents[1]
OFFICIAL=ROOT/'runs/actuator-audit-20261119/official'
COMMIT='1e1ec4f5f27e6c3e03048f693bd7c60cce04afda'


def configuration(config):
    state=config['active_first_physics_step'];ids=config['canonical_ids']
    links={l.attrib['name']:l for l in ET.parse(OFFICIAL/'training-main.urdf').getroot().findall('link')}
    comparisons=[];extras=[]
    for name,m,com,inertia in zip(config['body_names'],state['masses'],state['coms'],state['inertias']):
        link=links.get(name);el=link.find('inertial') if link is not None else None
        if el is None:
            extras.append(dict(name=name,mass_kg=m,urdf_status='missing_link' if link is None else 'no_inertial'))
            continue
        pos=el.find('origin');xyz=np.array([float(v) for v in pos.attrib.get('xyz','0 0 0').split()])
        R=Rotation.from_euler('xyz',[float(v) for v in pos.attrib.get('rpy','0 0 0').split()]).as_matrix()
        v=el.find('inertia').attrib
        I=np.array([[float(v.get('i'+a+b,v.get('i'+b+a))) for b in 'xyz'] for a in 'xyz'])
        # PhysX get_inertias returns the full matrix expressed in the link frame,
        # not just diagonal principal moments; do not rotate it by COM quaternion again.
        comparisons.append(dict(name=name,mass_abs_delta_kg=abs(m-float(el.find('mass').attrib['value'])),
            com_max_abs_delta_m=float(abs(np.asarray(com[:3])-xyz).max()),
            inertia_max_abs_delta_kg_m2=float(abs(np.asarray(inertia).reshape(3,3)-R@I@R.T).max())))
    params={};omega=20*np.pi
    for name,i in zip(G1_JOINT_NAMES,ids):
        if any(x in name for x in ('hip_pitch','hip_roll','knee')):armature=.025101925;effort=139.;speed=20.
        elif 'hip_yaw' in name or name=='waist_yaw_joint':armature=.010177520;effort=88.;speed=32.
        elif any(x in name for x in ('wrist_pitch','wrist_yaw')):armature=.00425;effort=5.;speed=22.
        elif 'ankle' in name or name in ('waist_roll_joint','waist_pitch_joint'):armature=2*.003609725;effort=50.;speed=37.
        else:armature=.003609725;effort=25.;speed=37.
        params[name]=dict(runtime_kp=state['dof_stiffnesses'][i],official_current_kp=armature*omega**2,
            runtime_kd=state['dof_dampings'][i],official_current_kd=4*armature*omega,
            runtime_effort=state['dof_max_forces'][i],official_current_effort=effort,
            runtime_armature=state['dof_armatures'][i],official_current_armature=armature,
            runtime_speed=state['dof_max_velocities'][i],official_current_speed=speed)
    # Read literal provider arrays without importing or executing its simulator code.
    source=ROOT/'runs/sonic-source-audit-20261117/action_provider/action_provider_sonic.py'
    literals={}
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('G1_ACTION_SCALE_ISAACLAB','SONIC_DEFAULT_POS'):
                literals[name]=np.asarray(ast.literal_eval(node.value.args[0]),dtype=np.float32)
    order=[config['sonic_ids'].index(i) for i in ids]
    scales=np.array([.25*v['official_current_effort']/v['official_current_kp'] for v in params.values()])
    defaults=[]
    for name in G1_JOINT_NAMES:
        value=0.
        if 'hip_pitch' in name:value=-.312
        elif 'knee' in name:value=.669
        elif 'ankle_pitch' in name:value=-.363
        elif 'shoulder_pitch' in name:value=.2
        elif 'shoulder_roll' in name:value=.2 if name.startswith('left_') else -.2
        elif 'elbow' in name:value=.6
        defaults.append(value)
    return dict(total_runtime_mass_kg=float(sum(state['masses'])),
        urdf_explicit_mass_kg=sum(float(l.find('inertial/mass').attrib['value']) for l in links.values() if l.find('inertial/mass') is not None),
        official_training_runtime_mass_unknown=True,extra_or_unspecified_links=extras,common_links=comparisons,
        provider_scale_vs_official_current_max_abs=float(abs(literals['G1_ACTION_SCALE_ISAACLAB'][order]-scales).max()),
        provider_default_vs_official_current_max_abs_rad=float(abs(literals['SONIC_DEFAULT_POS'][order]-defaults).max()),
        parameters=params,official_comparison_scope='Current pinned official source, not proven exact configuration used to train archived root SONIC checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',default='actuator-audit-20261119-r2');a=p.parse_args()
    suite=ROOT/'runs'/a.suite;results=[];configs=[];replays=[]
    for case in CASES:
        run=ROOT/'runs'/f'{a.suite}-{case}';source=ROOT/'runs'/f'vlm-refined-20261118-r1-{case}'
        check=audit(run,suite/'fixtures'/case,source)
        if max(check['actual_wrist_replay_vs_original_cm']['max_norm'])>0:
            raise ValueError('Read-only audit changed actual wrist trajectory')
        replay=replay_summary(run);replay['replay_audit']=check;replays.append(replay)
        (suite/f'{case}-replay-analysis.json').write_text(json.dumps(replay,indent=2))
        result=summarize(run);results.append(result)
        if result['control_frames']!=400:raise ValueError('Expected 400 controls')
        for key in ('decoder_to_physx_target_max_abs_rad','provider_to_physx_kp_max_abs',
                    'provider_to_physx_kd_max_abs','feedforward_max_abs_nm','velocity_target_max_abs',
                    'external_force_max_abs','external_torque_max_abs','q_limit_violation_max_rad',
                    'target_limit_violation_max_rad'):
            if result[key]!=0:raise ValueError('Expected zero: '+key)
        for name,force in result['contact_max_n'].items():
            if 'ankle_roll' not in name and force!=0:raise ValueError('Unexpected non-foot contact: '+name)
        (run/'actuator-analysis.json').write_text(json.dumps(result,indent=2))
        configs.append(json.loads((run/'actuator-config.json').read_text()))
        if configs[-1]['provider']['use_effort_control'] or configs[-1]['provider']['output_delay_steps']!=0:
            raise ValueError('Position mode/delay changed')
    keys=('dof_stiffnesses','dof_dampings','dof_max_forces','dof_max_velocities','dof_armatures','dof_friction_coefficients','masses','coms','inertias')
    for config in configs[1:]:
        if config['robot_asset']!=configs[0]['robot_asset']:raise ValueError('Robot asset differs')
        for k in keys:
            if config['active_first_physics_step'][k]!=configs[0]['active_first_physics_step'][k]:raise ValueError('Configuration differs: '+k)
    provider_source=ROOT/'runs/sonic-source-audit-20261117/action_provider/action_provider_sonic.py'
    run_meta=json.loads((ROOT/'runs'/f'{a.suite}-{CASES[0]}'/'run.json').read_text())
    if hashlib.sha256(provider_source.read_bytes()).hexdigest()!=run_meta['arena_sha256']['action_provider/action_provider_sonic.py']:
        raise ValueError('Provider source archive differs from executed code')
    compare=configuration(configs[0])
    asset=json.loads((ROOT/'runs/actuator-audit-20261119/asset-body-properties-verified.json').read_text())
    if asset['robot_asset']!=configs[0]['robot_asset']:raise ValueError('Static asset inspection differs')
    compare['asset_unspecified_sensor_masses']={b['name']:b['attributes']['physics:mass'] for b in asset['bodies']
        if b['name'] in ('imu_in_pelvis','imu_in_torso','d435_link','mid360_link')}
    if len(compare['asset_unspecified_sensor_masses'])!=4 or any(v['authored'] or float(v['value'])!=0 for v in compare['asset_unspecified_sensor_masses'].values()):
        raise ValueError('Sensor mass declaration differs')
    (suite/'configuration-comparison.json').write_text(json.dumps(compare,indent=2))
    (suite/'actuator-summary.json').write_text(json.dumps(results,indent=2));(suite/'summary.json').write_text(json.dumps(replays,indent=2))
    lines=['# SONIC → 执行器静差审计（2026-11-19）','',
        '结论：五组固定参考的执行器目标传递无误；肩部静差与实际重力负载 / 有限位置刚度的平衡偏差高度一致。当前集成混用了 SONIC 增益与 Arena 的执行器限幅、armature 和机器人载荷配置。这轮只诊断，没有调 PD、修改动力学、VLM 或夹爪。','',
        '## 对照与测量边界','',
        '- 保留原版快照 `snapshots/sonic-before-actuator-audit-20261119`，147 个原文件已按原 manifest 校验。',
        '- 原始五组消费参考、目标、场景、tracker 权重和原始桥接逻辑均通过逐项核验。共 2,000 控制帧 / 8,000 个 5 ms 物理步；控制周期 20 ms。五组实际双腕轨迹相对原版的最大差均为 0。',
        '- SONIC 解码目标 → PhysX 位置目标最大差 0 rad；provider Kp/Kd → PhysX Kp/Kd 最大差 0。未加载 VLM、Kimodo 或在线 IK。',
        '- 统计静差窗口为每组第 4–8 秒。下表重力对照采用每个控制周期最后一个物理步，与 link 位姿严格同步；完整 200 Hz 统计另存 JSON。关节静差定义为实际关节角减 SONIC 解码目标；腕跟踪误差定义为实际腕位置减冻结执行参考，不是 VLM 选点误差。',
        '- r1 首次采样因 Arena 重置后缓存 PhysX view 失效而停止；失败目录保留。r2 每步获取当前 view，完整五组均通过，控制路径不变。','',
        '## 五组结果','',
        '| 组别 | 右腕跟踪均值（到达后，cm） | 右肩 pitch 静差 / 重力预测（°） | 右肩 roll 静差 / 重力预测（°） |',
        '|---|---:|---:|---:|']
    for case,result,replay in zip(CASES,results,replays):
        p=result['joints']['right_shoulder_pitch_joint'];r=result['joints']['right_shoulder_roll_joint']
        lines.append(f"| {case} | {replay['post_arrival']['tracking']['mean_norm'][1]:.3f} | {p['actual_target_control_end_bias_deg']:.3f} / {p['gravity_predicted_bias_deg']:.3f} | {r['actual_target_control_end_bias_deg']:.3f} / {r['gravity_predicted_bias_deg']:.3f} |")
    max_res=max(abs(r['joints'][n]['gravity_bias_residual_mean_deg']) for r in results for n in ('right_shoulder_pitch_joint','right_shoulder_roll_joint'))
    gravity_diff=max(r['gravity_api_vs_link_geometry_max_abs_nm'] for r in results)
    example=replays[0]['steady_4_8s']['right_joints'];roll='right_shoulder_roll_joint'
    lines+=['',f'右肩 pitch/roll 的平均静差与重力预测之差最大 **{max_res:.3f}°**。重力矩分别通过 PhysX API 和实际质量、COM、link 位姿、USD 关节轴独立计算，两者全关节最大差 **{gravity_diff:.6f} N·m**。',
        '', '`q_actual − q_target ≈ −τ_gravity_comp / Kp`。位置控制必须保持一定角度偏差才能提供托住手臂的力矩。SONIC 解码目标本身可以学到提前量，所以该偏差不等于相对原始参考的净误差；最后几厘米仍包含 tracker 目标补偿残差和根部误差，提前量可能过大，也可能过小。',
        '', f"例如 base-center 右肩 roll：SONIC 目标相对参考 {example['decoder_reference'][roll]['bias']:.2f}°，实际相对 SONIC 目标 {example['actual_decoder'][roll]['bias']:+.2f}°，相加才是实际相对参考 {example['actual_reference'][roll]['bias']:.2f}°。让实际角度完全贴住 SONIC 输出不等于消除参考跟踪误差，直接加重力补偿可能重复补偿。",
        '', '## 实际生效的配置','',
        '| 右臂关节 | 实际 Kp / 官方当前 Kp | 实际 Kd / 官方当前 Kd | 实际力矩上限 / 官方当前上限（N·m） | 实际 / 官方当前 armature（kg·m²） |',
        '|---|---:|---:|---:|---:|']
    for name,ps in compare['parameters'].items():
        if name.startswith('right_') and any(k in name for k in ('shoulder','elbow','wrist')):
            lines.append(f"| {name.removeprefix('right_').removesuffix('_joint')} | {ps['runtime_kp']:.3f} / {ps['official_current_kp']:.3f} | {ps['runtime_kd']:.3f} / {ps['official_current_kd']:.3f} | {ps['runtime_effort']:.0f} / {ps['official_current_effort']:.0f} | {ps['runtime_armature']:.6f} / {ps['official_current_armature']:.6f} |")
    max_arm_fraction=max(j['pd_estimate_full_run_max_limit_fraction'] for r in results for n,j in r['joints'].items() if any(k in n for k in ('shoulder','elbow','wrist')))
    max_common_mass=max(j['mass_abs_delta_kg'] for j in compare['common_links']);max_com=max(j['com_max_abs_delta_m'] for j in compare['common_links']);max_inertia=max(j['inertia_max_abs_delta_kg_m2'] for j in compare['common_links'])
    lines+=['',
        '- SONIC 在位置控制模式只改写 PhysX Kp/Kd，没有同步替换 Arena 的力矩上限和 armature。肩 roll/yaw、elbow 与 ankle pitch 的 Kp 还有本地手动加值；不是一整套一致的官方配置。',
        f"- Provider 的动作缩放与默认角度也按关节名重排核对：相对当前官方定义最大差分别为 {compare['provider_scale_vs_official_current_max_abs']:.2e} 和 {compare['provider_default_vs_official_current_max_abs_rad']:.2e} rad，未发现此处顺序或尺度错误。",
        '- 全身 29 个控制关节的附加转子惯量（armature）为 0，不代表机器人 link 本体惯量为 0；关节摩擦系数为 0。运行时没有输出延迟、外加力或力矩，速度目标及前馈力矩均为 0。重力启用，自碰撞关闭，位置/速度求解迭代为 8/4。',
        f'- 五组实际角度及位置目标均未超限；全程双臂使用实际 Kp/Kd 估算的 PD 力矩最大仅达实际上限的 {100*max_arm_fraction:.1f}%。接触传感器在手臂/手部记录为 0，未抓箱，排除本轮持物或桌面接触解释。该估算不等于隔离测得的电机力矩。',
        '- `IsaacLab applied_torque` 在隐式执行器中按旧缓存增益估算，不能当实际驱动力矩。其 arms 缓存仍为 40/5 或 20/1，而 PhysX 已采用上表增益。报告保留旧估算、实际增益 PD 估算和 PhysX projected joint force，明确区分三者。','',
        '## 载荷与惯量','',
        f"运行时整机 **{compare['total_runtime_mass_kg']:.5f} kg**；官方 URDF 显式 inertial 质量合计 **{compare['urdf_explicit_mass_kg']:.5f} kg**。共同有惯量声明的 link，其质量最大差 {max_common_mass:.2e} kg、COM 最大差 {max_com:.2e} m、惯量矩阵最大差 {max_inertia:.2e} kg·m²，未发现主体惯量抄错。",'',
        '| link | 当前质量（kg） | 官方 URDF 状态 |','|---|---:|---|']
    for x in compare['extra_or_unspecified_links']:
        lines.append(f"| {x['name']} | {x['mass_kg']:.2f} | {'未包含此 link' if x['urdf_status']=='missing_link' else '存在但未声明 inertial'} |")
    lines+=['',
        '对运行中相同 SHA256 的 USD 做了独立只读检查：这四个 1 kg sensor body 的 `physics:mass` 均 **没有 authored value**，USD 读取的是默认值 0；它们进入 PhysX 后的实际质量均为 1 kg。不是源码明确指定了四个 1 kg 传感器，问题出在未明确声明质量后的导入结果。证据为 `runs/actuator-audit-20261119/asset-body-properties-verified.json`。', '',
        '**不能将 4.22 kg 全部断言为相对原始训练环境的多余载荷。** 官方 URDF 也有三个无 inertial 的 sensor link，尚无该 checkpoint 原始转换后 USD 的质量快照；它们是否在训练导入时成为默认质量仍需核对。可以确定当前新增的 D435 与双手相机合计 1.22 kg；四个 1 kg sensor body 都确实在当前 PhysX 中存在。',
        '', '双手相机各 0.11 kg 对肩部的重力矩贡献已单独离线计算，见各 `actuator-analysis.json` 的 `hand_camera_gravity_mean_nm`。躯干/骨盆传感器不直接属于肩部关节的子树，影响整机支撑与根部动力学，不能用它们直接解释全部手臂静差。',
        '', '## 下一步（本轮未执行）','',
        '1. 先建立与当前 SONIC 权重对应的可追溯动力学基线：核对原始导入选项、无惯量传感器默认质量和 checkpoint 版本。当前官方源码只作为有版本记录的参照，不冒充该权重精确训练配置。',
        '2. 保留本轮五组冻结参考，分别做“载荷一致化”和“完整执行器配置一致化”的单因素对照；不能同时改质量、增益、armature 后把改善归于某一项。自碰撞若需调整也应单独测。',
        '3. 若配置核对后仍有净腕偏差，再比较重力前馈或目标补偿。SONIC 可能已经学到部分重力补偿，直接再叠加会重复补偿；本轮重力平衡证据不是直接加力矩的授权结论，也不证明某组新 PD 能稳定双腕 3 cm。',
        '', '## 证据与复现','',
        f'- 结果：`runs/{a.suite}/actuator-summary.json`、`configuration-comparison.json`。',
        f'- 每组：`runs/{a.suite}-<case>/actuator-config.json`、`actuator-physics.jsonl`、`actuator-controls.jsonl`、`actuator-usd-joints.json`、`actuator-analysis.json`。',
        '- 源码与 API 证据：`runs/actuator-audit-20261119/environment-source`、`api-source`、`official`；官方文件均保存 URL、固定 revision 和 SHA256。',
        '- 执行：`PYTHONPATH=. <policy-python> scripts/run_sonic_replay_suite.py --suite <new-id> --actuator-audit --sim-gpu 7 --gpu-min-free-mib 16384`。',
        f'- 重建报告：`PYTHONPATH=. <policy-python> scripts/report_actuator_audit.py --suite {a.suite}`。','',
        f"官方对照：[当前固定版本机器人配置](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/{COMMIT}/gear_sonic/envs/manager_env/robots/g1.py)、[部署参数](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/{COMMIT}/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp)、[URDF](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/{COMMIT}/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf)。",
        '', '[当前 root SONIC 权重的发布配置](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_release/config.yaml) 保存了质量随机化、初始关节偏移和地面材料随机化等训练设置，因此以上名义配置差异不能单独证明因果。','']
    (ROOT/'docs/actuator-audit-20261119.md').write_text('\n'.join(lines))
    print('PASS five cases, 8000 physics steps; report generated')


if __name__=='__main__':main()
