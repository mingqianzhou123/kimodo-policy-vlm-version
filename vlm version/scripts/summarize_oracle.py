"""Summarize stage errors without treating completion as grasp success."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(rows, key):
    values = np.asarray([r[key] for r in rows if key in r],dtype=float)
    if not values.size:
        return None
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite metric {key}')
    return {'count':len(values),'mean':values.mean(axis=0).tolist(),
            'p95':np.quantile(values,.95,axis=0).tolist(),'max':values.max(axis=0).tolist(),
            'last':values[-1].tolist()}


def summarize(run):
    def read(name):
        path=run/name
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    plans,execution=read('plans.jsonl'),read('execution.jsonl')
    after=[x for x in execution if x.get('after_arrival')]
    result={'run':run.name,'plans':len(plans),'control_steps':len(execution),
        'after_arrival_steps':len(after),'grasp_attempted':False,
        'units':{'position':'metres','angle':'degrees','time':'seconds'},
        'plan_metrics':{key:stats(plans,key) for key in ('coordinate_roundtrip_m','feature_target_m',
            'kimodo_fk_target_m','projected_wrist_target_m','projection_wrist_delta_m',
            'projected_wrist_angle_deg','inference_s')},
        'execution_metrics':{key:stats(execution,key) for key in ('control_dt','tracking_wrist_m',
            'measured_xml_vs_sim_wrist_m','measured_target_m','consumed_reference_target_m')},
        'after_arrival_metrics':{key:stats(after,key) for key in ('tracking_wrist_m','measured_target_m',
            'consumed_reference_target_m','measured_target_angle_deg')}}
    if after:
        errors=np.array([r['measured_target_m'] for r in after])
        result['both_wrists_within_3cm_fraction']=float(np.mean(np.all(errors<=.03,axis=-1)))
    if plans and (run/'target.json').exists():
        target=np.array(json.loads((run/'target.json').read_text())['targets_world'])[:,0]
        constraints=[]
        for plan in plans:
            data=np.load(run/f"plan-{plan['plan']:03d}.npz")
            for k in plan['keyframes']:
                if k<=0: continue  # separate future ability from an already-missed current target
                raw=data['raw_positions_world'][k,:,0]
                fk=data['kimodo_fk_world'][k,:,0]
                projected=data['projected_wrist_world'][k]
                constraints.append({'feature_wrist_target_m':np.linalg.norm(raw-target,axis=-1).tolist(),
                    'fk_wrist_target_m':np.linalg.norm(fk-target,axis=-1).tolist(),
                    'projected_wrist_target_m':np.linalg.norm(projected-target,axis=-1).tolist(),
                    'projection_delta_m':np.linalg.norm(projected-fk,axis=-1).tolist()})
        result['positive_horizon_constraint_metrics']={k:stats(constraints,k) for k in
            ('feature_wrist_target_m','fk_wrist_target_m','projected_wrist_target_m','projection_delta_m')}
        result['first_plan_metrics']=plans[0]
    for name in ('run.json','target.json','episode.json'):
        if (run/name).exists():
            value=json.loads((run/name).read_text())
            if name=='run.json':value={k:value.get(k) for k in ('status','mode','sim_exit_code','error')}
            result[name.removesuffix('.json')]=value
    (run/'summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.run),indent=2))
