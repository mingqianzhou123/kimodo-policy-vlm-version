"""Audit frozen inputs and export one row per condition/hand, without inflating repeats."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from kimodo_vla.vlm.stability import review_suite,stats

ROOT=Path(__file__).resolve().parents[1]


def audit(folder):
    result=review_suite(folder)
    suite=json.loads((folder/'suite.json').read_text())
    baseline=json.loads((ROOT/'runs/vlm-reach-20261117-r1/run.json').read_text())
    fixed_keys=('checkpoint_sha256','tracker_sha256','arena_sha256','variant','handoff_seconds',
                'joint_speed_rad_s','support_constraints','hand_constraints','trajectory_projection','reference_source')
    for name,sha in suite['frozen_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    for c in result['cases']:
        meta=json.loads((ROOT/'runs'/c['run']/'run.json').read_text())
        for key in fixed_keys:assert meta[key]==baseline[key],(c['run'],key)
        for name,sha in suite['frozen_sha256'].items():
            if name.endswith('.py') and name.startswith(('kimodo_vla/','scripts/','arena/')):
                assert meta['source_sha256'][name]==sha,(c['run'],name)
        # Inference parameters are frozen inside infer_vlm_points.py; Kimodo flags checked separately.
        command=meta['policy_command'];old=baseline['policy_command']
        for flag in ('--diffusion-steps','--constraint-cfg','--text-mode','--root-anchor','--mode'):
            assert command[command.index(flag)+1]==old[old.index(flag)+1],flag
    groups={}
    for field in ('reach_rad','camera_lateral_m'):
        groups[field]={}
        for value in sorted({c[field] for c in result['cases']}):
            cases=[c for c in result['cases'] if c[field]==value]
            first=[c['intents'][0] for c in cases if c['intents']]
            completed=[c for c in cases if c['complete']]
            groups[field][str(value)]={'conditions':len(cases),'accepted':sum(x['accepted'] for x in first),
                'visual_error_cm':stats([x['world_error_cm'] for x in first if 'world_error_cm' in x]),
                'episode_mean_tracking_cm':stats([c['error_cm']['tracking']['mean'] for c in completed]),
                'episode_mean_total_cm':stats([c['error_cm']['total']['mean'] for c in completed])}
    spreads={}
    for reach in sorted({c['reach_rad'] for c in result['cases']}):
        cases=[c for c in result['cases'] if c['reach_rad']==reach and c['intents']]
        if not cases:continue
        assert np.allclose([c['truth_world'] for c in cases],cases[0]['truth_world'],atol=1e-5)
        hand_spreads={}
        for side in ('left','right'):
            points=np.array([h['world'] for c in cases for h in c['intents'][0].get('hands',[])
                             if h['hand']==side and 'world' in h])
            hand_spreads[side]={'views':len(points),'max_pairwise_world_distance_cm':
                float(np.linalg.norm(points[:,None]-points[None,:],axis=-1).max()*100) if len(points) else None}
        spreads[str(reach)]=hand_spreads
    rows=[]
    for c in result['cases']:
        first=c['intents'][0] if c['intents'] else {}
        for hand,side in enumerate(('left','right')):
            row={'run':c['run'],'reach_rad':c['reach_rad'],'camera_lateral_m':c['camera_lateral_m'],
                 'hand':side,'accepted':first.get('accepted',False),'completed':c['complete'],
                 'frames':c['executed_frames'],'rejection':first.get('rejection',c.get('error'))}
            for src,dst in [('world_error_cm','visual_cm'),('pixel_error','pixel_error')]:
                row[dst]=first[src][hand] if src in first else (first.get('hands',[{},{}])[hand].get('world_error_cm') if src=='world_error_cm' else None)
            for component in ('reference','tracking','total'):
                for metric in ('mean','p95','max'):
                    row[f'{component}_{metric}_cm']=c.get('error_cm',{}).get(component,{}).get(metric,[None,None])[hand]
            rows.append(row)
    with (folder/'per-condition.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    evidence={'frozen_source_files':len(suite['frozen_sha256']),'weights_tracker_arena_match_previous':True,
              'grouped':groups,'same_goal_cross_view_spread':spreads}
    (folder/'audit.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path)
    audit(p.parse_args().suite)
