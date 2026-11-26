"""HITCH's whole-body planner as a line-oriented helper process (runs in HITCH's own Python environment).

    WBC_HAND=dex3 <HITCH>/.venv/bin/python -u kimodo_vla/agent/hitch_server.py

HITCH (github.com/Axellwppr/HITCH, MIT) needs MuJoCo for its kinematics and pins its own numpy and Python 3.10, so it
cannot live in the policy server's environment. ``kinematic.KinematicPlanner`` starts this file in HITCH's
environment and talks to it with one JSON object per line on stdin / stdout.

Request: ``{"request": <HITCH request>, "start_pose": [36], "frame": [x, y, yaw], "torso": [lean, turn],
"hands": <intents or null>, "wrist_targets": {side: {"position": [3], "rotation": [3][3]}}}`` in the world W.
``start_pose`` is root xyz, root quat wxyz and the 29 joints in HITCH's order (``wbc.core.layout.JOINTS``), the
reference pose the plan starts from; ``frame`` is the plan frame O in W, exactly as HITCH's own session builds them
(``wbc.session.session._plan``: the start is the reference tail expressed in O). ``wrist_targets`` (optional) are
wrist-link poses in W; they are turned here into HITCH's palm waypoints (its hand's palm offset, the same rotation
convention: x along the fingers, left palm = -y, right palm = +y) and added to ``request.hands``.

Reply: ``{"ok": true, "world": [T][36], "passed": bool, "failures": [...], "end_torso": [lean, turn],
"hands": <intents>, "duration_s": float}`` or ``{"ok": false, "error": "...", "code": "..."}``.
"""
import json
import sys

import numpy as np


def main():
    from wbc.core.errors import WbcError
    from wbc.core.frames import SE2
    from wbc.core.request import parse_request
    from wbc.core.types import StartState, posture_for_height
    from wbc.planner.request import plan
    from wbc.robot.hands import current_hand

    hand = current_hand()
    print(json.dumps({'ok': True, 'ready': True, 'hand': hand.name}), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            frame = SE2(*(float(v) for v in message['frame']))
            to_plan = frame.inverse()
            request = dict(message['request'])
            for side, target in (message.get('wrist_targets') or {}).items():
                rotation = np.asarray(target['rotation'], float)
                palm = np.asarray(target['position'], float)+rotation @ np.asarray(hand.palm_position(side))
                palm_o = to_plan.apply_points(palm)
                turn = to_plan.rotation.as_matrix()
                normal = turn @ rotation[:, 1]*(-1. if side == 'left' else 1.)
                fingers = turn @ rotation[:, 0]
                waypoint = {'position': palm_o.tolist(), 'duration_s': float(target['duration_s']),
                            'orientation': {'normal': normal.tolist(), 'fingers': fingers.tolist()}}
                hands = dict(request.get('hands') or {})
                hands[side] = {'waypoints': [waypoint]}
                request['hands'] = hands
            pose = to_plan.apply_poses(np.asarray(message['start_pose'], float))
            start = StartState(pose, posture_for_height(float(pose[2])), tuple(message.get('torso') or (0, 0)),
                               message.get('hands'))
            planned = plan(parse_request(request), start=start, frame=frame, horizon=8)
            motion = planned.motion
            reply = {'ok': True, 'world': np.asarray(planned.world).tolist(), 'passed': bool(planned.report.passed),
                     'failures': planned.report.findings(blocking=True),
                     'end_torso': list(motion.end_torso or start.torso), 'hands': planned.hands,
                     'duration_s': float(motion.duration), 'request': request}
        except WbcError as error:
            reply = {'ok': False, 'error': str(error), 'code': getattr(getattr(error, 'code', None), 'value', None),
                     'details': getattr(error, 'details', None)}
        except Exception as error:   # report, keep serving
            reply = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        print(json.dumps(reply, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)), flush=True)


if __name__ == '__main__':
    main()
