"""One RGB-D intent per episode, on the unchanged IK-hold motion backend."""
import json
import subprocess
from pathlib import Path
import numpy as np
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.vlm.geometry import backproject,load_capture


class VisualTargetPolicy(OraclePolicy):
    def __init__(self,*args,vlm_model,vlm_python,vlm_localizer='baseline',**kwargs):
        if vlm_localizer not in ('baseline','refined'):raise ValueError('Unknown VLM localizer')
        self.vlm_localizer=vlm_localizer
        self.vlm_model=Path(vlm_model).resolve();self.vlm_python=vlm_python
        super().__init__(*args,**kwargs)

    def reset(self,seed=None):
        self.capture=None
        return super().reset(seed)

    def initialize(self,sample):
        # mode=hold initializes the measured standing frame only. No reach
        # witness or object pose is used to choose the visual goal.
        super().initialize(sample)
        if self.capture is None:raise ValueError('RGB-D capture is required')
        path=Path(self.capture['capture_path']).resolve()
        if path!=self.run.resolve()/'vlm-capture.npz':raise ValueError('Capture outside this run')
        data=load_capture(path)
        if abs(float(data['time'])-sample['time'])>1e-5:raise ValueError('Capture and initial pose timestamps differ')
        result_path=self.run/'vlm-intent.json'
        module='kimodo_vla.vlm.'+('infer_refined' if self.vlm_localizer=='refined' else 'infer_points')
        with (self.run/'vlm.log').open('w') as log:
            subprocess.run([self.vlm_python,'-m',module,'--model',str(self.vlm_model),
                '--capture',str(path),'--out',str(result_path)],check=True,timeout=600,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(result_path.read_text());intent=result['intent']
        world=[];points=[]
        for side in ('left','right'):
            goal,info=backproject(intent[side],data['depth'],data['K'],data['camera_position'],data['camera_rotation'])
            world.append(goal);points.append(info)
        world=np.asarray(world);initial=np.asarray(sample['wrist_pos'])
        if np.any(np.linalg.norm(world-initial,axis=-1)>.15):
            raise ValueError('Visual target exceeds the 15 cm standing smoke-test workspace; not clipped or replaced with oracle')
        self.initial_visual_targets=self.targets.copy()
        self.targets+=(world-self.targets[:,0])[:,None,:]
        self.target_info.update(mode='visual-markers',target_source='qwen_rgb_depth',
            targets_world=self.targets.tolist(),witness_joint_pos=None,
            hand_path='quintic Cartesian interpolation to visual targets; initial wrist orientation',
            goal_displacement_m=np.linalg.norm(world-initial,axis=-1).tolist(),
            visual_points=points,motion_text=intent['motion_text'],
            text_conditioning=self.text_mode,vision_capture_sha256=result['capture_sha256'],
            camera_name=str(data['camera_name']),camera_pose_source='Isaac ground-truth calibrated sensor pose',
            gripper_mode='open',intent_refresh='once at episode start; simulation paused during inference')
        if self.text_mode=='task' and intent['motion_text'] not in self.texts:
            raise ValueError('Exact VLM motion phrase has no LLM2Vec embedding')
        (self.run/'target.json').write_text(json.dumps(self.target_info,indent=2))

    def path_goals(self,elapsed):
        u=np.clip(np.asarray(elapsed)/1.2,0,1);w=u**3*(10-15*u+6*u*u)
        world=self.initial_visual_targets[None]+w[:,None,None,None]*(self.targets-self.initial_visual_targets)[None]
        return self.frame.points(world),np.broadcast_to(self.target_rot_model,(len(u),2,3,3)).copy()

    def __call__(self,request):
        if self.frame is None:self.capture=request['observation'].get('vision')
        # The selected motion phrase is the exact text-cache key, not the Arena task instruction.
        from kimodo_vla.vlm.geometry import MOTION_TEXT
        return super().__call__({**request,'task':MOTION_TEXT})
