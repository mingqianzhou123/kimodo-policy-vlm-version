"""Audit saved evaluation receipts, episode events and per-step failure evidence.

No policy inference, simulator or model/training changes. Uses one CPU thread.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import io
import json
import os
from pathlib import Path
import statistics
import time


def read(path):return json.loads(path.read_text())
def ever(row,key):return bool(row['ever_true'].get(key,False))
def goal_key(condition):
    if '-L0-' in condition:return 'picked_correct_tool'
    return 'on_spot' if 'BallMove' in condition else 'in_target_area' if 'BallRetrieve' in condition else 'ice_broken'


def stats(rows):
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
        outcome=dict(Counter(r['outcome'] for r in rows)),
        first_touch_roles=dict(Counter(r['final_metrics']['first_touch_role'] for r in rows)),
        events=dict(Counter(k for r in rows for k,v in r['ever_true'].items() if v)),
        failures=dict(Counter(r['failure_bucket'] for r in rows if not r['success'])),
        lifted_and_target=sum(ever(r,'picked_correct_tool') and ever(r,'tool_touched_target') for r in rows),
        target_without_lift=sum(not ever(r,'picked_correct_tool') and ever(r,'tool_touched_target') for r in rows),
        lifted50=sum(r['max_consecutive'].get('picked_correct_tool',0)>=50 for r in rows),
        first_move_directions=dict(Counter(r['final_metrics']['first_move_toward_target'] for r in rows if 'first_move_toward_target' in r['final_metrics'])),
        median_max_lift_m=statistics.median(r['numeric_max']['tool_lift'] for r in rows))


def analyze(root):
    output=root/'failure_analysis';output.mkdir(exist_ok=True)
    summary=read(root/'summary.json');settings=read(root/'evaluation_settings.json');assert summary['failed']==0 and summary['completed']==18
    groups={};allrows=[];t=time.monotonic()
    for c in summary['conditions']:
        run=Path(c['result']).parent;receipt=read(run/'benchmark_result.json');assert receipt['status']=='validated'
        assert receipt['seeds']==list(range(settings['seed_start'],settings['seed_start']+settings['episodes']))
        rows=[];name=c['env_id'].split('/')[-1];key=goal_key(name)
        for path in sorted((run/'summaries').glob('episode_*.json')):
            row=read(path);row.update(condition=name,seed=int(path.stem.split('_')[-1]),summary_path=str(path),video_directory=str(run/'videos'/path.stem))
            duration=Counter();current=Counter();maximum=Counter();minimum_distance=100;first_distance=None;laststep=0;first_seen={};bool_ever={};lastmetrics=None
            metric_keys=['picked_correct_tool',key]
            for line in (run/'metrics'/f'{path.stem}.jsonl').open():
                record=json.loads(line);m=record['metrics'];assert record['step']==laststep+1;laststep=record['step'];lastmetrics=m
                for k in set(metric_keys):
                    value=bool(m.get(k,False));duration[k]+=value;current[k]=current[k]+1 if value else 0;maximum[k]=max(maximum[k],current[k])
                for k,v in m.items():
                    if isinstance(v,bool):
                        bool_ever[k]=bool_ever.get(k,False) or v
                        if v and k not in first_seen:first_seen[k]=laststep
                if 'dist_to_target_bench' in m:
                    if first_distance is None:first_distance=m['dist_to_target_bench']
                    minimum_distance=min(minimum_distance,m['dist_to_target_bench'])
            assert laststep==row['step'] and lastmetrics==row['final_metrics']
            assert all(bool_ever[k]==v for k,v in row['ever_true'].items())
            assert all(first_seen.get(k)==v for k,v in row['first_true_step'].items())
            assert row['success']==(maximum[key]>=50)
            row.update(max_consecutive=dict(maximum),true_steps=dict(duration),initial_target_distance_m=first_distance,
                       min_target_distance_m=minimum_distance if first_distance is not None else None)
            if row['success']:bucket='success'
            elif '-L0-' in name:bucket='no_lift_threshold' if not ever(row,'picked_correct_tool') else 'lift_threshold_but_not_held50'
            elif not ever(row,'picked_correct_tool'):bucket='no_lift_threshold'
            elif not ever(row,'tool_touched_target'):bucket='lifted_but_no_correct_tool_target_contact'
            elif not ever(row,key):bucket='correct_tool_target_contact_but_no_goal'
            else:bucket='goal_reached_but_not_held50'
            row['failure_bucket']=bucket;rows.append(row);allrows.append(row)
        assert len(rows)==c['episodes'] and sum(r['success'] for r in rows)==c['successes']
        groups[name]=stats(rows);print(name,'verified',len(rows),'elapsed',round(time.monotonic()-t,1),flush=True)
    result=dict(scope='Saved900episodes and all per-step metrics; official receipts and goal hold durations checked; no new rollout',
        source=str(root),settings=settings,aggregate=stats(allrows),conditions=groups,
        levels={f'L{i}':stats([r for r in allrows if f'-L{i}-' in r['condition']]) for i in range(3)},
        l12=stats([r for r in allrows if '-L0-' not in r['condition']]),
        scenarios={s:stats([r for r in allrows if s in r['condition'] and '-L0-' not in r['condition']]) for s in ['BallMove','BallRetrieve','IceBreak']})
    reached=[r for r in allrows if '-L2-' in r['condition'] and ever(r,'reached_target_bench')]
    result['l2_reach_audit']=dict(reached=len(reached),at_first_step=sum(r['first_true_step']['reached_target_bench']==1 for r in reached),
        median_first_reach_step=statistics.median(r['first_true_step']['reached_target_bench'] for r in reached),
        never_lifted=sum(not ever(r,'picked_correct_tool') for r in reached),
        lifted_before_first_reach=sum(ever(r,'picked_correct_tool') and r['first_true_step']['picked_correct_tool']<r['first_true_step']['reached_target_bench'] for r in reached),
        final_distance_gt04=sum(r['final_metrics']['dist_to_target_bench']>.4 for r in reached),
        final_distance_gt08=sum(r['final_metrics']['dist_to_target_bench']>.8 for r in reached))
    result['hold_audit']={}
    for s in ['BallMove','BallRetrieve','IceBreak']:
        chosen=[r for r in allrows if s in r['condition'] and '-L0-' not in r['condition']]
        key=goal_key(chosen[0]['condition']);contact=[r for r in chosen if ever(r,'tool_touched_target')]
        result['hold_audit'][s]=dict(goal_ever=sum(ever(r,key) for r in chosen),success=sum(r['success'] for r in chosen),
            failed_goal_holds=[dict(condition=r['condition'],seed=r['seed'],max_hold=r['max_consecutive'][key]) for r in chosen if ever(r,key) and not r['success']],
            touched_no_lift=sum(not ever(r,'picked_correct_tool') for r in contact),
            one_ice_only=sum(ever(r,'ice_broke_1') and not ever(r,'ice_broke_2') for r in chosen),
            correct_tool_contact_no_one_ice=sum(ever(r,'tool_touched_target') and not ever(r,'ice_broke_1') for r in chosen) if s=='IceBreak' else None)
    (output/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'episodes.json').write_text(json.dumps(allrows,indent=2)+'\n')
    keys=['condition','seed','success','step','outcome','failure_bucket','max_lift_m','first_touch_role','ever_lift','lift_max_hold_steps','lift_total_steps','ever_correct_target_contact','goal_ever','goal_max_hold_steps','ever_reached','first_reach_step','first_lift_step','final_distance_m','summary_path','video_directory']
    with (output/'episode_failure_details.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in allrows:
            key=goal_key(r['condition']);writer.writerow(dict(condition=r['condition'],seed=r['seed'],success=r['success'],step=r['step'],outcome=r['outcome'],failure_bucket=r['failure_bucket'],max_lift_m=r['numeric_max']['tool_lift'],first_touch_role=r['final_metrics']['first_touch_role'],ever_lift=ever(r,'picked_correct_tool'),lift_max_hold_steps=r['max_consecutive']['picked_correct_tool'],lift_total_steps=r['true_steps']['picked_correct_tool'],ever_correct_target_contact=ever(r,'tool_touched_target'),goal_ever=ever(r,key),goal_max_hold_steps=r['max_consecutive'][key],ever_reached=ever(r,'reached_target_bench'),first_reach_step=r['first_true_step'].get('reached_target_bench'),first_lift_step=r['first_true_step']['picked_correct_tool'],final_distance_m=r['final_metrics'].get('dist_to_target_bench'),summary_path=r['summary_path'],video_directory=r['video_directory']))
    print(json.dumps({k:result[k] for k in ['levels','l12','scenarios','l2_reach_audit','hold_audit']},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--evaluation',type=Path,required=True);args=parser.parse_args();os.nice(10);analyze(args.evaluation.resolve())
