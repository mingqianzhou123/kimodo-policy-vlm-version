"""Use a previously localized world goal with a newly measured support pose."""
import json
from pathlib import Path
import numpy as np
from kimodo_vla.oracle.policy import OraclePolicy
from kimodo_vla.vlm.policy import VisualTargetPolicy
from kimodo_vla.vlm.geometry import MOTION_TEXT


class ArchivedGoalPolicy(OraclePolicy):
    path_goals = VisualTargetPolicy.path_goals

    def __init__(self, *args, archived_goal, **kwargs):
        self.archived_goal = Path(archived_goal)
        super().__init__(*args, **kwargs)

    def initialize(self, sample):
        super().initialize(sample)
        old = json.loads(self.archived_goal.read_text())
        if old['target_source'] != 'qwen_rgb_depth' or old['motion_text'] != MOTION_TEXT:
            raise ValueError('Expected an archived visual reaching goal')
        self.initial_visual_targets = self.targets.copy()
        self.targets = np.asarray(old['targets_world'])
        self.target_rot_world = np.asarray(old['wrist_rotation_world'])
        self.target_rot_model = self.frame.rotations(self.target_rot_world)
        self.target_info.update(mode='visual-markers', target_source='archived_visual_world',
            source_target=str(self.archived_goal), targets_world=self.targets.tolist(),
            wrist_rotation_world=self.target_rot_world.tolist(), motion_text=MOTION_TEXT,
            goal_displacement_m=np.linalg.norm(self.targets[:,0]-sample['wrist_pos'],axis=1).tolist(),
            hand_path='quintic from settled hands to archived wrists and virtual endpoints; archived world orientation')
        (self.run/'target.json').write_text(json.dumps(self.target_info, indent=2))

    def __call__(self, request):
        return super().__call__({**request, 'task': MOTION_TEXT})
