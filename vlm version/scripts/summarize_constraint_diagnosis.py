"""Summarize paired input sensitivity and an explicitly non-executed overwrite probe."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.model.features import KimodoFeatures


def summarize(directory):
    rep=KimodoFeatures(50).representation
    ids=[rep.skeleton.bone_order_names.index(f'{s}_wrist_yaw_skel') for s in ('left','right')]
    records=[json.loads(x) for x in (directory/'results.jsonl').read_text().splitlines()]
    pairs=[]
    for p in sorted(directory.glob('*-baseline.npz')):
        d=np.load(p);mask=torch.from_numpy(d['mask']);ks=mask.any(-1).nonzero().flatten();ks=ks[ks>0]
        row={'baseline':p.name}
        shift=Path(str(p).replace('baseline','shift_x_5cm'))
        if shift.exists():
            shifted=np.load(shift)
            row['target_shift_model_xyz_m']=[.05,0.,0.]
            for name in ('raw_wrist','fk_wrist'):
                row[f'{name}_response_m']=(shifted[name][ks]-d[name][ks]).mean(0).tolist()
        out=torch.from_numpy(d['motion_features']);values=torch.from_numpy(d['conditions'])
        delta=out-values
        row['root_xz_residual_m']=delta[ks][:,[0,2]].norm(dim=-1).mean().item()
        row['local_wrist_residual_m']=delta[:,rep.slice_dict['local_joints_positions']].reshape(-1,34,3)[ks][:,ids].norm(dim=-1).mean(0).tolist()
        # This shows why output-channel overwriting is not an IK solution.
        overwritten=out.clone();overwritten[mask]=values[mask]
        for mode in ('positions','rotations'):
            decoded=rep.inverse(overwritten[None],is_normalized=False,posed_joints_from=mode)
            wrist=decoded['posed_joints'][0,ks][:,ids].numpy()
            row[f'overwrite_counterfactual_{mode}_error_m']=np.linalg.norm(wrist-d['target'],axis=-1).mean(0).tolist()
        pairs.append(row)
    result={'inferences':len(records),'all_checks_passed_guidance_calls':sum(r['checked_guidance_calls'] for r in records),
        'all_checks_passed_root_inputs':sum(r['checked_root_inputs'] for r in records),
        'baseline_reproduction_max':max(r['output_vs_saved_max'] for r in records if r['variant']=='baseline'),
        'paired_sensitivity':pairs,'records':records,
        'counterfactual_executed':False,'simulator_runs_added':0}
    (directory/'diagnosis-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    print(json.dumps(summarize(p.parse_args().directory),indent=2))
