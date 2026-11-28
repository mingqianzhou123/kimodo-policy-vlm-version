"""HITCH-planned references as Arena replay segments (port check: the same references in MuJoCo and in Arena).

    HITCH_ROOT=<HITCH checkout> <HITCH>/.venv/bin/python heft_segments.py --out <dir>

Plans every motion of ``heft_mujoco.REQUESTS`` with HITCH's planner from HITCH's standing start (rubber hands, the
hand the paper's tracking test used), exactly as ``heft_mujoco.py`` does, and writes ``<out>/hitch-<motion>.json`` in
the segment format of ``kimodo_vla/oracle/demo_walks.py`` (``actions`` [T, 40] in Arena's action layout, joints in
``G1_JOINT_NAMES`` order, root steps in the body frame; ``window`` = the planned motion; ``direction``). Replayed
with ``run_oracle.py --reference-source demo --demo-segment <file> --start-xy ... [--tracker heft]``: a 1 s blend
from the settled Arena stance into the first frame, the motion, then its last frame held, open loop.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[4]))
from heft_mujoco import REQUESTS  # noqa: E402

DIRECTION = {'forward': 'forward', 'sidestep': 'left', 'backward': 'backward', 'turn': 'turn'}
LEAD_S = 0.2   # standing frames before the motion, so the blend ends on a still pose


def encode(world, joint_names_from, joint_names_to):
    """World frames [T, 36] (root xyz, quat wxyz, joints) -> Arena actions [T, 40], heading of frame 0 = +x."""
    from scipy.spatial.transform import Rotation
    pos, quat = world[:, :3], world[:, 3:7]
    rot = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    yaw0 = np.arctan2(rot[0, 1, 0], rot[0, 0, 0])
    head = Rotation.from_euler('z', -yaw0).as_matrix()
    rot = np.einsum('ij,tjk->tik', head, rot)
    pos = (head @ (pos-pos[0]).T).T
    step = np.zeros_like(pos)
    step[1:] = pos[1:]-pos[:-1]
    out = np.zeros((len(world), 40))
    out[:, :2] = np.einsum('tji,tj->ti', rot, step)[:, :2]
    out[:, 2] = world[:, 2]
    out[:, 3:9] = rot[:, :, :2].reshape(len(world), 6)
    index = [list(joint_names_from).index(n) for n in joint_names_to]
    out[:, 9:38] = world[:, 7:][:, index]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    os.environ.setdefault('WBC_HAND', 'rubber')
    from kimodo_vla.core.types import G1_JOINT_NAMES
    from wbc.core.frames import SE2
    from wbc.core.layout import JOINTS
    from wbc.core.request import parse_request
    from wbc.core.types import StartState, posture_for_height
    from wbc.paths import INITIAL_REFERENCE
    from wbc.planner.request import plan
    from wbc.robot.hands import current_hand
    initial = current_hand().rest(np.load(INITIAL_REFERENCE)['qpos'][0])
    a.out.mkdir(parents=True, exist_ok=True)
    lead = int(round(LEAD_S*50))
    for name, request in REQUESTS.items():
        start = StartState(initial, posture_for_height(float(initial[2])))
        planned = plan(parse_request(request), start=start, frame=SE2(0., 0., 0.), horizon=8)
        if not planned.report.passed:
            raise RuntimeError(f'{name}: HITCH checks rejected the plan')
        world = np.vstack([np.repeat(planned.world[:1], lead, 0), planned.world])
        actions = encode(world, JOINTS, G1_JOINT_NAMES)
        segment = {'source': 'hitch_planner', 'request': request, 'hand': os.environ['WBC_HAND'],
                   'direction': DIRECTION.get(name, 'in_place'), 'motion': name,
                   'window': [lead, len(actions)-1], 'actions': actions.tolist()}
        (a.out/f'hitch-{name}.json').write_text(json.dumps(segment))
        print(name, len(actions), 'frames', segment['direction'])


if __name__ == '__main__':
    main()
