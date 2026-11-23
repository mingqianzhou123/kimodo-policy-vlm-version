"""Interface checks, not a claim about trained-model or simulator performance."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from kimodo_vla.controllers.gripper_gate import ArrivalGripperGate, GripperEvent
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.model.modules.diffusion import Diffusion, DDIMSampler
from kimodo_vla.model.sparse_conditions import hand_keyframe, normalized_future_condition
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34


def test_hand_constraints_use_relative_horizontal_and_absolute_height():
    rep = KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=None)
    positions = torch.tensor([[1.3, 0.9, 2.4], [1.3, 0.9, 2.5]])
    values, mask = hand_keyframe(rep, frames=50, frame=30, smooth_root_xz=[1, 2],
        hand_positions={"right": positions}, wrist_rotations={"right": torch.eye(3)})
    joint = rep.skeleton.bone_order_names.index("right_wrist_yaw_skel")
    start = rep.slice_dict["local_joints_positions"].start + 3 * joint
    torch.testing.assert_close(values[30, start:start+3], torch.tensor([0.3, 0.9, 0.4]))
    assert mask.sum() == 14  # root XZ + 2 points * XYZ + wrist rot6d
    assert not mask[:30].any() and not mask[31:].any()
    assert not mask[:, rep.slice_dict["velocities"]].any()
    # Translating root AND targets moves only the global root, not hand offsets.
    moved, _ = hand_keyframe(rep, frames=50, frame=30, smooth_root_xz=[4, -2],
        hand_positions={"right": positions + torch.tensor([3, 0, -4])},
        wrist_rotations={"right": torch.eye(3)})
    torch.testing.assert_close(moved[:, rep.body_slice], values[:, rep.body_slice], atol=1e-6, rtol=1e-5)


def test_normalization_masks_unknown_nan_and_preserves_physical_zero():
    rep = SimpleNamespace(motion_rep_dim=3, normalize=lambda x: (x - 4) / 2)
    values = torch.tensor([[0., float("nan"), 6.]])
    mask = torch.tensor([[True, False, True]])
    result, _ = normalized_future_condition(rep, values, mask, batch_size=1, frames=1,
                                             device="cpu", dtype=torch.float32)
    torch.testing.assert_close(result, torch.tensor([[[-2., 0., 1.]]]))
    with pytest.raises(ValueError, match="non-finite"):
        normalized_future_condition(rep, values, torch.ones_like(mask), batch_size=1,
                                    frames=1, device="cpu", dtype=torch.float32)


class SpyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, **kw):
        self.calls.append({k: v.detach().clone() for k, v in kw.items() if torch.is_tensor(v)})
        # Intentionally does NOT honor conditions; the public method must not
        # manufacture a perfect endpoint by overwriting its final output.
        return torch.ones_like(kw["x"])


class StubControl(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))
        self.root_hint_fusion = self.body_hint_fusion = None

    def forward(self, *args):
        return {}, {}


def make_policy():
    model = KimodoPolicy.__new__(KimodoPolicy)
    nn.Module.__init__(model)
    model.config = KimodoPolicyConfig(action_history=3, action_chunk=4, enable_hand_head=False)
    model.denoiser = SpyDenoiser()
    model.controlnet = StubControl()
    model.image_encoder = nn.Identity()
    model.text_encoder = model.hand_head = None
    model.representation = SimpleNamespace(motion_rep_dim=8,
        slice_dict={"global_root_heading": slice(3, 5)},
        normalize=lambda x: (x - 4) / 2, unnormalize=lambda x: x * 2 + 4,
        inverse=lambda x, **_: {"root_positions": torch.zeros(*x.shape[:2], 3)})
    model.diffusion = Diffusion(num_base_steps=10)
    model.sampler = DDIMSampler(model.diffusion)
    return model


def infer(model, **kw):
    return model.predict_future(instruction="reach with the right hand",
        egoview=torch.zeros(1, 3, 2, 2), history_motion=torch.zeros(1, 2, 8),
        text_feat=torch.full((1, 1, 4096), 0.25), diffusion_steps=2,
        generator=torch.Generator().manual_seed(3), **kw)


def test_future_mask_and_dynamic_text_reach_every_denoiser_step():
    model = make_policy()
    values = torch.zeros(4, 8)
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask[2, 6] = True
    output = infer(model, future_condition=values, future_condition_mask=mask)
    assert len(model.denoiser.calls) == 2
    for call in model.denoiser.calls:
        assert call["motion_mask"][0, 5, 6] == 1  # H=3 + future frame 2
        assert call["observed_motion"][0, 5, 6] == -2  # normalization of physical 0
        assert call["motion_mask"][0, 3:].sum() == 1
        assert call["motion_mask"][0, 0].sum() == 0  # padded history remains padding
        assert call["text_feat"].shape == (1, 1, 4096)
        assert torch.all(call["text_feat"] == .25)
    assert output["motion_features"][2, 6] != 0  # no artificial endpoint guarantee


def test_no_constraints_and_empty_constraints_preserve_same_output():
    first = infer(make_policy())
    second = infer(make_policy(), future_condition=torch.zeros(4, 8),
                   future_condition_mask=torch.zeros(4, 8, dtype=torch.bool))
    torch.testing.assert_close(first["motion_features"], second["motion_features"], rtol=0, atol=0)


def test_incomplete_and_invalid_future_input_rejected():
    with pytest.raises(ValueError, match="supplied together"):
        infer(make_policy(), future_condition=torch.zeros(4, 8))
    with pytest.raises(ValueError, match="shape"):
        infer(make_policy(), future_condition=torch.zeros(5, 8),
              future_condition_mask=torch.zeros(5, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="bool"):
        infer(make_policy(), future_condition=torch.zeros(4, 8),
              future_condition_mask=torch.zeros(4, 8))


def event(name="grasp-1", close=True, deadline=2.):
    return GripperEvent(name, close, (0., 0., 1.), tuple(map(tuple, np.eye(3))), deadline)


def tick(gate, t, **kw):
    args = dict(now_s=t, observation_time_s=t, position=[0., 0., 1.],
                rotation=np.eye(3), speed_mps=0.)
    args.update(kw)
    return gate.update(**args)


def test_gate_waits_for_actual_arrival_and_survives_replans():
    gate = ArrivalGripperGate()
    gate.submit(event())
    assert not tick(gate, 0, position=[.2, 0, 1])
    for t in (.02, .04, .06, .08, .1):
        assert not gate.submit(event())  # same event cannot reset dwell
        assert not tick(gate, t)
    assert tick(gate, .12) and gate.status == "fired"
    assert tick(gate, .14, position=[1, 0, 1])  # keep holding after leaving the grasp site


@pytest.mark.parametrize("bad", [dict(speed_mps=1), dict(position=[1, 0, 1]),
    dict(rotation=np.diag([-1, -1, 1])), dict(position=[float("nan"), 0, 1])])
def test_gate_rejects_nonarrival_and_resets_dwell(bad):
    gate = ArrivalGripperGate(); gate.submit(event())
    tick(gate, 0); tick(gate, .02); tick(gate, .04, **bad)
    assert not tick(gate, .06) and not tick(gate, .08)


def test_stale_measurements_and_missing_ticks_cannot_complete_dwell():
    gate = ArrivalGripperGate(); gate.submit(event())
    tick(gate, 0)
    assert not tick(gate, .02, observation_time_s=0)
    assert not tick(gate, .2)  # fresh frame after a gap starts a new dwell
    assert not tick(gate, .22)


def test_timeout_cancel_and_reset_do_not_rearm_or_release_held_object():
    gate = ArrivalGripperGate(); gate.reset(closed=True)
    gate.submit(event("release-1", close=False, deadline=.05))
    tick(gate, 0)
    assert tick(gate, .06) and gate.status == "timed_out"
    assert not gate.submit(event("release-1", close=False, deadline=2))
    gate.cancel(); assert tick(gate, .08)
    gate.reset(); assert gate.submit(event("release-1", close=False))
    assert gate.status == "pending" and not gate.closed


def test_release_requires_arrival_as_well():
    gate = ArrivalGripperGate(); gate.reset(closed=True)
    gate.submit(event(close=False))
    for t in (0, .02, .04, .06, .08):
        assert tick(gate, t)
    assert not tick(gate, .1) and gate.status == "fired"
