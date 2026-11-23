import json
from pathlib import Path
import numpy as np
import pytest
from arena.dynamics_override import intervene,FIELDS
from kimodo_vla.core.types import G1_JOINT_NAMES

from paths import AGENTIC as ROOT, RUNS


def fixture():
    source=RUNS/'actuator-audit-20261119-r2-base-center/actuator-config.json'
    if not source.exists():pytest.skip('Archived experiment unavailable')
    c=json.loads(source.read_text())
    state={k:np.array(c['active_first_physics_step'][k]) for k in FIELDS}
    spec=json.loads((ROOT/'configs/sonic-dynamics.json').read_text())
    return c,state,spec


def test_sensor_mass_ablation_does_not_change_actuators_or_authored_loads():
    c,s,spec=fixture();r=intervene(s,c['joint_names'],c['body_names'],'sensor-load',spec)
    changed=[k for k in FIELDS if not np.array_equal(r[k],s[k])]
    assert set(changed)=={'masses','inertias'}
    ids=[c['body_names'].index(n) for n in spec['sensor-load']['bodies']]
    other=[i for i in range(len(c['body_names'])) if i not in ids]
    np.testing.assert_array_equal(r['masses'][other],s['masses'][other])
    np.testing.assert_array_equal(r['inertias'][other],s['inertias'][other])
    np.testing.assert_allclose(r['masses'][ids],.001)
    np.testing.assert_allclose(r['inertias'][ids],s['inertias'][ids]*.001)
    np.testing.assert_allclose(s['masses'][ids],1.)


def test_actuator_profile_uses_joint_names_and_preserves_hand_and_load():
    c,s,spec=fixture();r=intervene(s,c['joint_names'],c['body_names'],'actuator',spec)
    ids=[c['joint_names'].index(n) for n in G1_JOINT_NAMES]
    hands=[i for i in range(len(c['joint_names'])) if i not in ids]
    for k in FIELDS:
        if k not in spec['actuator']:np.testing.assert_array_equal(r[k],s[k])
        else:
            np.testing.assert_array_equal(r[k][hands],s[k][hands])
            for n,v in spec['actuator'][k].items():assert r[k][c['joint_names'].index(n)]==v
    # Force the mapping to change while preserving the same named physical state.
    order=np.arange(len(c['joint_names']))[::-1];permuted={k:(v[order] if k.startswith('dof_') else v) for k,v in s.items()}
    rr=intervene(permuted,list(reversed(c['joint_names'])),c['body_names'],'actuator',spec)
    for k in spec['actuator']:np.testing.assert_array_equal(rr[k][order],r[k])


def test_rejects_unexpected_baseline_mass_or_incomplete_actuator_definition():
    c,s,spec=fixture();s['masses'][c['body_names'].index('d435_link')]=2
    with pytest.raises(ValueError,match='Unexpected sensor mass'):intervene(s,c['joint_names'],c['body_names'],'sensor-load',spec)
    spec['actuator']['dof_stiffnesses'].pop(G1_JOINT_NAMES[0])
    with pytest.raises(ValueError,match='Incomplete'):intervene(s,c['joint_names'],c['body_names'],'actuator',spec)
