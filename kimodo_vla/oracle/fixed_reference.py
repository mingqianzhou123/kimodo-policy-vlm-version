"""Prescribed joint-limited references. No generator, IK, retargeting or learned policy."""
import json
from pathlib import Path
import numpy as np
from kimodo_vla.core.types import G1_JOINT_NAMES, yaw_matrix, yaw_of
from .geometry import XmlFK, rotation_wxyz


class FixedReference:
    def __init__(self, run, mode):
        if mode not in ('hold','reach'):
            raise ValueError('Fixed reference supports hold/reach only')
        self.run, self.mode = Path(run), mode
        xml=Path(__file__).resolve().parents[1]/'skeleton/assets/g1skel34/xml/g1.xml'
        self.fk=XmlFK(xml,G1_JOINT_NAMES)
        self.reset()

    def reset(self, seed=None):
        self.start=None
        self.plan=0
        return {'status':'ok'}

    def initialize(self, sample):
        self.start=sample['time']
        self.root=np.array(sample['root_pos'])
        self.rotation=rotation_wxyz(sample['root_quat'])
        self.heading=yaw_matrix(float(yaw_of(self.rotation)))
        self.initial=np.array(sample['joints'])
        self.goal=self.initial.copy()
        if self.mode=='reach':
            for side in ('left','right'):
                self.goal[G1_JOINT_NAMES.index(f'{side}_shoulder_pitch_joint')]-=.15
        for q in (self.initial,self.goal):
            if not np.all((q>=self.fk.joint_limits[:,0]) & (q<=self.fk.joint_limits[:,1])):
                raise ValueError('Prescribed pose violates joint limits')
        wrist,rot=self.fk(self.goal,self.root,self.rotation)
        self.target={'source':'fixed','mode':self.mode,'arrival_time':self.start+1.2,
            'targets_world':wrist[:,None,:].tolist(),'wrist_rotation_world':rot.tolist(),
            'witness_joint_pos':self.goal.tolist(),'root_pos':self.root.tolist(),
            'root_quat':sample['root_quat'],'virtual_endpoint':False,
            'contact_template_validated':False,'joint_limited':True,
            'dynamic_feasibility_claimed':False,'ramp':'quintic, 1.2 seconds',
            'goal_displacement_m':np.linalg.norm(wrist-np.array(sample['wrist_pos']),axis=-1).tolist()}
        (self.run/'target.json').write_text(json.dumps(self.target,indent=2))

    def joints_at(self, elapsed):
        u=np.clip(np.asarray(elapsed)/1.2,0.,1.)
        weight=u**3*(10.+u*(-15.+6.*u))
        return self.initial+weight[...,None]*(self.goal-self.initial)

    def __call__(self, request):
        samples=request['observation']['oracle_history']
        if not samples: raise ValueError('Measured history required for initial state and time')
        if self.start is None: self.initialize(samples[0])
        sample=samples[-1]
        # Explicit after-step times, same 50 Hz grid and 25-command HTTP chunk as A.
        times=sample['time']+np.arange(1,26)/50
        qs=self.joints_at(times-self.start)
        previous=np.array(sample['reference_root_pos'])
        actions=[]; commands=[]
        relative=self.heading.T@self.rotation
        for i,(time,q) in enumerate(zip(times,qs)):
            delta=self.rotation.T@(self.root-previous)
            actions.append(np.r_[delta[:2],self.root[2],relative[:,:2].ravel(),q,0.,0.].tolist())
            wrist,rot=self.fk(q,self.root,self.rotation)
            commands.append({'plan':self.plan,'action_index':i,'time':float(time),
                'root_pos':self.root.tolist(),'root_rotation':self.rotation.tolist(),
                'joints':q.tolist(),'wrist_pos':wrist.tolist(),'wrist_rot':rot.tolist()})
            previous=self.root
        with (self.run/'commands.jsonl').open('a') as f:
            f.write(''.join(json.dumps(x)+'\n' for x in commands))
        record={'plan':self.plan,'time':sample['time'],'source':'fixed','mode':self.mode,
                'model_loaded':False,'commands':25}
        with (self.run/'plans.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        self.plan+=1
        return {'action_chunk':actions,'oracle_target':self.target}
