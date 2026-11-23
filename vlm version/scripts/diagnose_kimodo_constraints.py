"""Paired frozen-weight ablations of saved oracle inputs; never execute actions."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
from kimodo_vla.model.native_policy import NativeKimodoPolicy
from kimodo_vla.model.features import KimodoFeatures
from kimodo_vla.model.planner import _episode_noise_seed
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import Frame, XmlFK, BASIS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=2);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('../checkpoints/Kimodo-G1-RP-v1'))
    p.add_argument('--runs',nargs='+',default=['oracle-hold-20261117-r4','oracle-reach-20261117-r1'])
    p.add_argument('--plans',nargs='+',type=int,default=[0,3])
    p.add_argument('--variants',nargs='+',choices=['baseline','no_future','shift_x_5cm','cfg1','cfg4','compact','steps200','full_pose','history_last1','history_last5','history_pose_only','no_current_target','fixed_history'],default=['baseline','no_future','shift_x_5cm','cfg1','cfg4','compact','steps200','full_pose'])
    a=p.parse_args()
    state=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    uuid,mem,util=map(str.strip,state.split(','))
    if int(mem)>100 or int(util)!=0:raise RuntimeError(f'GPU occupied: {state}')
    os.environ['CUDA_VISIBLE_DEVICES']=uuid
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    model=NativeKimodoPolicy(a.checkpoint,constraint_cfg=2.).cuda().eval()
    features=KimodoFeatures(50);rep=model.representation
    ids=[rep.skeleton.bone_order_names.index(f'{s}_wrist_yaw_skel') for s in ('left','right')]
    handids=[rep.skeleton.bone_order_names.index(f'{s}_{j}_skel') for s in ('left','right') for j in ('wrist_yaw','hand_roll')]
    metadata={'gpu':state.strip(),'checkpoint':str(a.checkpoint.resolve()),'variants':a.variants,
        'checkpoint_sha256':{str(f.relative_to(a.checkpoint)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.checkpoint.rglob('*') if f.is_file()}}
    (a.output/'run.json').write_text(json.dumps(metadata,indent=2))
    for runname in a.runs:
        run=Path('runs')/runname
        target=json.loads((run/'target.json').read_text());frame=Frame(target['origin'],np.array(target['heading']))
        initial=json.loads((run/'execution.jsonl').read_text().splitlines()[0])['before']
        goalq=target.get('witness_joint_pos') or initial['joints']
        from kimodo_vla.oracle.geometry import rotation_wxyz
        pose=features.codec.decode_joint_configuration_pose(np.array([goalq]*50),
            np.array([frame.points(initial['root_pos'])@BASIS]*50),
            root_rotation_matrices=np.array([frame.heading.T@rotation_wxyz(initial['root_quat'])]*50),
            joint_names=G1_JOINT_NAMES)
        fullgoal=features.features(pose['local_rot_mats'],pose['root_positions'])
        for plan in a.plans:
            data=np.load(run/f'plan-{plan:03d}.npz');history=torch.from_numpy(data['history'])
            original_values=torch.from_numpy(data['conditions']);original_mask=torch.from_numpy(data['mask'])
            keyframes=original_mask.any(-1).nonzero().flatten().tolist()
            positive=[k for k in keyframes if k>0]
            goal=frame.points(np.array(target['targets_world'])[:,0])
            seed=_episode_noise_seed(1000,plan)
            base_noise=torch.randn((1,150,417),device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
            baseline=None
            for variant in a.variants:
                values,mask=original_values.clone(),original_mask.clone()
                hist=history.clone();histmask=torch.ones_like(hist,dtype=torch.bool)
                weight=1. if variant=='cfg1' else 4. if variant=='cfg4' else 2.
                model.denoiser.weight=weight
                model.config.action_history=len(history) if variant=='compact' else 100
                h=model.config.action_history;steps=200 if variant=='steps200' else 50
                variant_goal=goal.copy()
                if variant=='no_future':mask.zero_()
                if variant=='shift_x_5cm':
                    for j in handids:
                        col=rep.slice_dict['local_joints_positions'].start+3*j
                        values[keyframes,col]+=.05
                    variant_goal[:,0]+=.05
                if variant=='full_pose':
                    values[keyframes]=fullgoal[keyframes];mask[keyframes]=True
                if variant in ('history_last1','history_last5'):
                    keep=1 if variant=='history_last1' else 5
                    histmask[:-keep]=False
                if variant=='history_pose_only':
                    for block in ('velocities','foot_contacts'):histmask[:,rep.slice_dict[block]]=False
                if variant=='no_current_target':mask[0]=False
                if variant=='fixed_history':
                    initial_pose=features.codec.decode_joint_configuration_pose(np.array([initial['joints']]*5),
                        np.array([frame.points(initial['root_pos'])@BASIS]*5),
                        root_rotation_matrices=np.array([frame.heading.T@rotation_wxyz(initial['root_quat'])]*5),
                        joint_names=G1_JOINT_NAMES)
                    initial_feature=features.features(initial_pose['local_rot_mats'],initial_pose['root_positions'])[0]
                    hist[:]=initial_feature
                future_norm=rep.normalize(values.cuda()[None]).masked_fill(~mask.cuda()[None],0)
                calls=[];input_checks=[]
                def before_guidance(module,args,kw):
                    if not calls:kw['x'].copy_(base_noise[:,-(h+50):])
                    torch.testing.assert_close(kw['motion_mask'][:,h:].bool(),mask.cuda()[None])
                    torch.testing.assert_close(kw['observed_motion'][:,h:],future_norm)
                    assert kw['x_pad_mask'][:,h:].all()
                    calls.append(int(kw['timesteps'][0]))
                    # Capture what the real root Transformer must receive.
                    expected=kw['x']*(1-kw['motion_mask'])+kw['observed_motion']*kw['motion_mask']
                    input_checks.append((expected.detach().clone(),kw['motion_mask'].detach().clone()))
                def root_input(module,args):
                    expected,m=input_checks[-1]
                    if len(root_seen)% (1 if weight==1 else 2)==0:
                        torch.testing.assert_close(args[0],torch.cat((expected,m),-1))
                    root_seen.append(True)
                root_seen=[]
                hook=model.denoiser.register_forward_pre_hook(before_guidance,with_kwargs=True)
                rh=model.denoiser.model.root_model.register_forward_pre_hook(root_input)
                begin=time.perf_counter()
                try:
                    with torch.inference_mode():
                        out=model.predict_future('',torch.zeros(1,3,1,1),hist,history_feature_mask=histmask,
                            text_feat=torch.zeros(1,4096,device='cuda'),future_condition=values,
                            future_condition_mask=mask,diffusion_steps=steps,
                            generator=torch.Generator(device='cuda').manual_seed(seed))
                finally:hook.remove();rh.remove()
                motion=out['motion_features'].cpu()
                raw=rep.inverse(motion.cuda()[None],is_normalized=False,posed_joints_from='positions')['posed_joints'][0,:,ids].cpu().numpy()
                fk=out['posed_joints'][:,ids].cpu().numpy()
                root=out['root_positions'].cpu();local=out['local_rot_mats'].cpu()
                q=features.codec.encode(local,root,hand_binary=torch.zeros(50,2))[:,9:38].numpy()
                if variant=='baseline':baseline=motion.clone()
                record={'run':runname,'plan':plan,'variant':variant,'history_frames':len(history),'history_canvas':h,
                    'steps':steps,'cfg':weight,'seed':seed,'future_mask_cells':int(mask.sum()),'keyframes':keyframes,
                    'checked_guidance_calls':len(calls),'checked_root_inputs':len(root_seen),
                    'inference_s':time.perf_counter()-begin,
                    'feature_target_m':np.linalg.norm(raw[positive]-variant_goal,axis=-1).mean(axis=0).tolist(),
                    'fk_target_m':np.linalg.norm(fk[positive]-variant_goal,axis=-1).mean(axis=0).tolist(),
                    'masked_feature_mae':float((motion-values)[mask].abs().mean()) if mask.any() else None,
                    'output_vs_saved_max':float((motion-torch.from_numpy(data['motion_features'])).abs().max()) if variant=='baseline' else None,
                    'output_vs_baseline_rms':float((motion-baseline).square().mean().sqrt()) if baseline is not None else None}
                file=f'{runname}-p{plan}-{variant}'
                np.savez_compressed(a.output/f'{file}.npz',motion_features=motion.numpy(),raw_wrist=raw,
                    fk_wrist=fk,joints=q,conditions=values.numpy(),mask=mask.numpy(),target=variant_goal)
                with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
    metadata['status']='completed';(a.output/'run.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
