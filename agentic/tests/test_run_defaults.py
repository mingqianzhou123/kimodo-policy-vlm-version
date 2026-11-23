from run_oracle import parse_args

# The full command of the 2026-11-23 settled-start runs, as run_settled_suite.py spelled it then.
SETTLED_20261123 = ['--run-id', 'x', '--settled-goal', 'target.json', '--steps', '400', '--mode', 'hold',
    '--sim-gpu', '3', '--policy-gpu', '3', '--share-policy-sim-gpu',
    '--variant', 'bounded', '--support-constraints', 'root', '--joint-speed', '4',
    '--hand-constraints', 'continuous', '--trajectory-projection', 'ik-hold',
    '--visual-markers', '--marker-reach-rad', '0.15', '--camera-lateral-m', '0.06',
    '--actuator-audit', '--dynamics-profile', 'actuator']


def test_short_settled_command_equals_the_recorded_full_command():
    short = ['--run-id', 'x', '--settled-goal', 'target.json', '--sim-gpu', '3', '--policy-gpu', '3',
             '--share-policy-sim-gpu', '--marker-reach-rad', '0.15', '--camera-lateral-m', '0.06']
    assert vars(parse_args(short)) == vars(parse_args(SETTLED_20261123))


def test_plain_run_uses_the_frozen_backend_without_settled_extras():
    a = parse_args(['--run-id', 'x'])
    assert (a.variant, a.support_constraints, a.hand_constraints, a.trajectory_projection) == \
        ('bounded', 'root', 'continuous', 'ik-hold')
    assert (a.joint_speed, a.steps) == (4., 400)
    assert not a.actuator_audit and a.dynamics_profile is None and not a.visual_markers
