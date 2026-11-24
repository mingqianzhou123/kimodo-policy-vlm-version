import json

import numpy as np
from scipy.spatial.transform import Rotation

from kimodo_vla.oracle import demo_walks as dw


def straight(frames, velocity_body, yaw0=.3, yaw_rate=0., height=.78):
    """Actions [T, 40] of a root moving at a constant body-frame velocity, joints at zero."""
    a = np.zeros((frames, 40))
    yaw = yaw0+yaw_rate*np.arange(frames)/dw.RATE
    r = np.stack([Rotation.from_euler('z', y).as_matrix() for y in yaw])
    a[:, 3:9] = r[:, :, :2].reshape(frames, 6)
    a[1:, :2] = np.asarray(velocity_body)/dw.RATE
    a[:, 2] = height
    return a


def test_rot6d_and_integration_round_trip():
    a = straight(100, [.5, 0.], yaw0=np.pi/2)
    root = dw.integrate_root(a)
    np.testing.assert_allclose(root[-1, :2], [0., .5*99/dw.RATE], atol=1e-9)
    d, yaw = dw.heading_frame_motion(root, dw.rot6d_rows(a), 0, 99)
    np.testing.assert_allclose(d, [.5*99/dw.RATE, 0.], atol=1e-9)
    assert abs(yaw) < 1e-9


def test_windows_find_each_direction():
    for velocity, rate, want in (([.5, 0], 0, 'forward'), ([-.3, 0], 0, 'backward'), ([0, .3], 0, 'left'),
                                 ([0, -.3], 0, 'right'), ([0, 0], np.radians(60), 'turn')):
        a = straight(500, velocity, yaw_rate=rate)
        windows = dw.find_windows(a)
        assert windows and all(w['direction'] == want for w in windows), (want, windows)
        assert windows[0]['begin'] >= 170 and windows[0]['end']-windows[0]['begin'] <= 200


def test_low_root_is_not_a_walk():
    a = straight(500, [.5, 0], height=.5)
    assert dw.find_windows(a) == []


def test_stepping_tells_steps_from_sliding():
    t = np.arange(200)/dw.RATE
    feet = np.zeros((200, 2, 3))
    # alternate swings of 0.4 s at 1.25 Hz per foot; each swing moves the foot 0.3 m and lifts it 8 cm
    for side, phase in ((0, 0.), (1, .4)):
        x = 0.
        for i, ti in enumerate(t):
            u = ((ti-phase) % .8)/.4
            swinging = ti >= phase and u < 1
            feet[i, side, 0] = x+(.3*u if swinging else 0.)
            feet[i, side, 2] = .08*np.sin(np.pi*u) if swinging else 0.
            if swinging and i+1 < len(t) and ((t[i+1]-phase) % .8)/.4 >= 1:
                x += .3
    clean = dw.stepping(feet)
    assert clean['slide_m_s'] < .05 and clean['both_moving'] < .25  # 0.1 s speeds overlap the swing changes
    assert 8 <= clean['steps'] <= 10 and abs(clean['step_length_m']-.3) < .05
    slid = feet.copy()
    slid[..., 0] += .4*t[:, None]
    slid[..., 2] = 0.
    s = dw.stepping(slid)
    assert s['slide_m_s'] > .35 and s['both_moving'] > .9 and s['steps'] == 0


def sample(time=0., yaw=0., xy=(-4., -4.)):
    q = Rotation.from_euler('z', yaw).as_quat()[[3, 0, 1, 2]]
    return {'time': time, 'root_pos': [xy[0], xy[1], .78], 'root_quat': q.tolist(), 'joints': [0.1]*29,
            'wrist_pos': [[0, .2, 1], [0, -.2, 1]], 'wrist_rot': np.tile(np.eye(3), (2, 1, 1)).tolist()}


def test_timeline_blends_from_the_measured_pose_and_reanchors_heading():
    a = straight(300, [.4, 0], yaw0=1.2)
    a[:, 9:38] = .3
    segment = {'actions': a.tolist(), 'window': [170, 299], 'direction': 'forward'}
    tl = dw.Timeline(segment, sample(yaw=-.4))
    blend = tl.actions[:tl.segment_start]
    assert np.abs(blend[0, 9:38]-.1).max() < .01 and np.allclose(blend[-1, 9:38], .3)
    assert np.allclose(blend[:, :2], 0.)
    rot = dw.rot6d_rows(tl.actions)
    assert np.allclose(rot[0], np.eye(3), atol=1e-3)                    # starts at the measured heading (relative)
    assert np.allclose(rot[tl.segment_start:tl.segment_start+130], np.eye(3), atol=1e-6)  # segment re-headed to ahead
    root = dw.integrate_root(tl.actions)
    w0, w1 = tl.window
    np.testing.assert_allclose(root[w1, :2]-root[w0, :2], [.4*129/dw.RATE, 0.], atol=1e-6)
    assert len(tl.chunk(tl.length+10)) == 25


def test_demo_reference_serves_the_timeline_and_scores_a_perfect_follower(tmp_path):
    a = straight(300, [0, .3], yaw0=.0)
    path = tmp_path/'left.json'
    path.write_text(json.dumps({'name': 'left', 'direction': 'left', 'window': [170, 299], 'actions': a.tolist(),
                                'source': {}}))
    ref = dw.DemoReference(tmp_path, path)
    first = sample(yaw=.5)
    out = ref({'observation': {'oracle_history': [first]}})
    assert len(out['action_chunk']) == 25 and out['oracle_target']['direction'] == 'left'
    target = json.loads((tmp_path/'target.json').read_text())
    tl = ref.timeline
    # a body that follows the reference exactly, in the world frame the measured heading defines
    heading = Rotation.from_euler('z', .5).as_matrix()
    root = dw.integrate_root(tl.actions)
    rots = dw.rot6d_rows(tl.actions)
    steps = []
    for i in range(tl.length):
        p = np.r_[np.array(first['root_pos'][:2])+(heading[:2, :2] @ root[i, :2]), .78]
        q = Rotation.from_matrix(heading @ rots[i]).as_quat()[[3, 0, 1, 2]]
        steps.append({'after': {'time': (i+1)/dw.RATE, 'root_pos': p.tolist(), 'root_quat': q.tolist(),
                                'foot_pos': [[p[0], p[1]+.1, .05], [p[0], p[1]-.1, .05]]}})
    result = dw.score_run(steps, target, dw.load_segment(path))
    assert result['complete'] and not result['fell']
    assert abs(result['completion']-1) < 1e-6 and result['lateral_cm'] < 1e-6 and abs(result['heading_drift_deg']) < 1e-6


def test_lafan1_rows_become_actions_with_the_same_root_path_and_joint_order():
    from kimodo_vla.core.types import G1_JOINT_NAMES
    t = np.arange(90)/30.
    yaw = .4+.3*t
    rows = np.zeros((90, 36))
    rows[:, 0], rows[:, 1], rows[:, 2] = .5*t, -.2*t, .79
    rows[:, 3:7] = Rotation.from_euler('z', yaw[:, None]).as_quat()
    rows[:, 7:] = np.arange(29)/100.                         # joint k in LAFAN1 order holds k/100
    a = dw.lafan1_actions(rows)
    end = (len(a)-1)/dw.RATE
    assert len(a) == 1+int(t[-1]*dw.RATE)
    root = dw.integrate_root(a)
    np.testing.assert_allclose(root[-1, :2]-root[0, :2], [.5*end, -.2*end], atol=1e-6)
    knee = G1_JOINT_NAMES.index('right_knee_joint')
    assert abs(a[0, 9+knee]-dw.LAFAN1_G1_JOINTS.index('right_knee_joint')/100.) < 1e-12


def test_live_walk_is_scored_from_the_start_of_its_walk_phase():
    from walk_attribution import score_measured
    steps = []
    for i in range(400):
        x = -4.+max(0, i-100)*.3/dw.RATE                      # stands 2 s, then 0.3 m/s forward
        steps.append({'after': {'time': (i+1)/dw.RATE, 'root_pos': [x, -4., .78], 'root_quat': [1., 0., 0., 0.],
                                'foot_pos': [[x, -3.9, .05], [x, -4.1, .05]]},
                      'reference_root_world': [x+.1*i/dw.RATE, -4., .78],       # the reference asks for 0.1 m/s more
                      'reference_foot_world': [[-4., -3.9, .05], [-4., -4.1, .05]]})
    target = {'phases': [{'phase': 'walk', 'time': 101/dw.RATE}],
              'motion_goal': {'root': {'xy': [3., 0.], 'speed_m_s': .35}}}
    m = score_measured(steps, target, 4.)
    assert not m['fell'] and abs(m['progress']-.3/.35) < .01 and m['lateral_cm'] < 1e-6
    assert abs(m['reference_leg_progress']-.4/.35) < .01      # planted reference feet: the legs command it all
    assert score_measured(steps, target, 7.) is None          # the run ended before 7 s of walking


def test_leg_displacement_removes_what_planted_feet_skate():
    t = np.arange(100)/dw.RATE
    root = np.stack([.5*t, 0*t, .78+0*t], axis=1)
    planted = np.stack([np.stack([0*t, .1+0*t, 0*t], 1), np.stack([0*t, -.1+0*t, 0*t], 1)], axis=1)
    np.testing.assert_allclose(dw.leg_displacement(planted, root), [.5*t[-1], 0.], atol=1e-9)   # clean: the root's
    skating = planted.copy()
    skating[..., 0] += .2*t[:, None]                          # both planted feet slide forward at 0.2 m/s
    np.testing.assert_allclose(dw.leg_displacement(skating, root), [.3*t[-1], 0.], atol=1e-9)


def test_lead_through_window_ends_on_the_windows_last_pose(tmp_path):
    from rollout_kimodo import lead_through_window
    a = np.random.default_rng(0).normal(size=(300, 40))
    a[:, 3:9] = np.eye(3)[:, :2].reshape(6)
    src = tmp_path/'s.json'
    src.write_text(json.dumps({'name': 's', 'direction': 'left', 'window': [170, 299], 'actions': a.tolist(), 'source': {}}))
    seg = dw.load_segment(lead_through_window(src, tmp_path/'lead.json'))
    assert seg['window'] == [299, 300] and len(seg['actions']) == 301
    np.testing.assert_allclose(seg['actions'][300][2:], a[299, 2:]) and np.testing.assert_allclose(seg['actions'][300][:2], 0.)
