"""Migration check: feed the same episode to the original HumanoidArena runtime and to KimodoPlanner.

A fake model records its inputs and returns input-dependent outputs, so the history
encoding is compared without checkpoints. Actions and the hand history differ on
purpose (KimodoPlanner samples predictions at the true control time, the original ran
one model frame behind; see tests/test_planner_timing.py), and so do the moments plans
run out; the new planner is therefore asked to plan at the steps the original did.
"""

import base64
from collections import deque
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

OLD_REPO = os.environ.get("KIMODO_ORIGINAL_REPO")
if not OLD_REPO:
    pytest.skip("set KIMODO_ORIGINAL_REPO to a kimodo-policy checkout to run", allow_module_level=True)
sys.path.insert(0, OLD_REPO)
import evaluation.humanoidarena_server as arena  # noqa: E402

from kimodo_vla.core.types import ActionChunk, Observation, yaw_matrix, G1_JOINT_NAMES  # noqa: E402
from kimodo_vla.model.planner import KimodoPlanner  # noqa: E402
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep  # noqa: E402
from kimodo_vla.skeleton.definitions import G1Skeleton34  # noqa: E402


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.controlnet = torch.nn.Linear(1, 1)
        self.fps = 30
        self.config = SimpleNamespace(action_chunk=50, action_history=100, hand_control_mode="binary")
        self.representation = KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=None)
        self.text_encoder = None
        self.calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def predict_future(self, *, egoview, history_motion, history_feature_mask, hand_history, **_):
        self.calls.append(
            dict(motion=history_motion.clone(), mask=history_feature_mask.clone(), hand=hand_history.clone(), image=egoview.clone())
        )
        s = float((history_motion * history_feature_mask).sum().remainder(1.0))
        n = len(self.calls)
        t = torch.arange(50, dtype=torch.float32) / 30
        rots = torch.eye(3).repeat(50, 34, 1, 1)
        yaw = 0.2 * t + 0.1 * s
        rots[:, 0, 0, 0] = torch.cos(yaw); rots[:, 0, 0, 2] = torch.sin(yaw)
        rots[:, 0, 2, 0] = -torch.sin(yaw); rots[:, 0, 2, 2] = torch.cos(yaw)
        bend = 0.3 * torch.sin(t + n) + 0.2 * s
        rots[:, 21, 1, 1] = torch.cos(bend); rots[:, 21, 1, 2] = -torch.sin(bend)
        rots[:, 21, 2, 1] = torch.sin(bend); rots[:, 21, 2, 2] = torch.cos(bend)
        root = torch.stack([0.3 * t + s, torch.full_like(t, 0.75), 0.1 * t], dim=1)
        hand = torch.stack([(t > 0.8).float(), (t > 1.2).float() * (n % 2)], dim=1)
        return dict(
            motion_features=torch.zeros(50, self.representation.motion_rep_dim),
            local_rot_mats=rots,
            root_positions=root,
            hand_binary=hand,
            history_last_root_position=torch.tensor([s, 0.75, 0.05]),
        )


def episode(steps=420):
    rng = np.random.default_rng(0)
    phase = rng.uniform(0, 2 * np.pi, 29)
    out = []
    for i in range(steps):
        t = i / 50
        yaw = 0.4 + 0.3 * np.sin(0.5 * t)
        joints = (0.2 * np.sin(0.7 * t + phase)).astype(np.float32)
        image = np.full((8, 8, 3), i % 255, np.uint8)
        out.append((yaw, joints, image))
    return out


def old_runtime(model):
    rt = object.__new__(arena.KimodoHumanoidArenaRuntime)
    rt.deterministic_eval = False
    rt.device = torch.device("cpu")
    rt.dtype = torch.float32
    rt.diffusion_steps = 10
    rt.control_fps = 50.0
    rt.execution_frames = 50
    rt.rtc_enabled, rt.rtc_overlap_frames, rt.rtc_frozen_frames, rt.rtc_ramp_power = False, 0, 0, 1.0
    rt.text_embeddings = {"t": torch.zeros(1, 8)}
    rt.task_instructions = {"t": "do it"}
    rt.task_aliases = {"t": "t"}
    rt.model = model
    rt.action_codec = arena.HumanoidArenaActionDecoder(arena.G1Skeleton34(), arena.G1_XML_PATH, fps=50.0)
    rt.state_motion_representation = arena.KimodoMotionRep(arena.G1Skeleton34(), fps=30, stats_path=None)
    rt.lock = threading.Lock()
    rt.episode_seed = None
    rt.inference_index = 0
    rt.reset()
    return rt


def test_planner_matches_original_runtime():
    data = episode()
    first = yaw_matrix(data[0][0])

    # Original runtime: the client sends every state since its previous request.
    old_model = FakeModel()
    rt = old_runtime(old_model)
    pending, cursor, plan, plan_steps = [], 0, None, []
    for step, (yaw, joints, image) in enumerate(data):
        relative = first.T @ yaw_matrix(yaw)
        pending.append(np.concatenate([relative[:, :2].reshape(6), joints, np.zeros(29, np.float32)]))
        if plan is None or cursor >= len(plan):
            payload = {
                "task": "t",
                "observation": {
                    "images": {"front": {"shape": list(image.shape), "dtype": "uint8",
                                         "data_b64": base64.b64encode(image.tobytes()).decode()}},
                    "state_history": np.stack(pending).tolist(),
                },
            }
            plan, cursor, pending = rt.infer(payload), 0, []
            plan_steps.append(step)
        cursor += 1

    # New code path, planning at the same steps.
    new_model = FakeModel()
    planner = KimodoPlanner(new_model, {"do it": torch.zeros(1, 8)}, control_fps=50.0, execution_frames=50)
    planner.reset()
    history = deque(maxlen=planner.history_steps)
    for step, (yaw, joints, image) in enumerate(data):
        history.append(Observation(
            images={"head": image}, root_pos=np.zeros(3, np.float32), root_rot=yaw_matrix(yaw),
            joint_pos=joints, joint_mask=np.ones(29, bool), hand=np.zeros(2, np.float32), instruction="do it", step=step,
        ))
        if step in plan_steps:
            planner.plan(list(history))

    assert len(old_model.calls) == len(new_model.calls), (len(old_model.calls), len(new_model.calls))
    worst = {}
    for i, (a, b) in enumerate(zip(old_model.calls, new_model.calls)):
        for key in ("motion", "mask", "image"):
            if a[key].shape != b[key].shape:
                raise AssertionError(f"call {i} {key}: shape {tuple(a[key].shape)} vs {tuple(b[key].shape)}")
            diff = float((a[key].float() - b[key].float()).abs().max()) if a[key].numel() else 0.0
            worst[key] = max(worst.get(key, 0.0), diff)
    print(f"plans: {len(new_model.calls)}  history length per plan: {[c['motion'].shape[1] for c in new_model.calls]}")
    print("max |old - new| model inputs:", {k: f"{v:.2e}" for k, v in worst.items()})
    assert all(v < 1e-4 for v in worst.values())
    assert all(a["hand"].shape == b["hand"].shape for a, b in zip(old_model.calls, new_model.calls))

