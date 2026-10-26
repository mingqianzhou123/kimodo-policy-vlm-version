"""The SIMPLE adapter answers the original SIMPLE runtime's requests identically (40-D and 36-D)."""

import numpy as np
import pytest
import torch

from original_runtime import FakeModel, arena_payload, compare_calls, episode, old_simple_runtime

from kimodo_vla.benchmarks.simple import hands
from kimodo_vla.benchmarks.simple.policy import command, make_policy, routes
from kimodo_vla.core.types import yaw_matrix
from kimodo_vla.model.planner import KimodoPlanner
from kimodo_vla.model.planner_original import OriginalTimingPlanner


def hand_measurement(step):
    """Fingers drifting between open and closed, sometimes still: exercises the debounce's settle checks."""
    progress = 0.5 + 0.5 * np.sin(step / 40.0)
    still = (step // 60) % 2 == 0
    left = hands.mujoco_to_natural(hands.BINARY_LEFT_CLOSE) * progress
    right = hands.mujoco_to_natural(hands.BINARY_RIGHT_CLOSE) * (1 - progress)
    velocity = np.full(7, 0.05 if still else 0.5, np.float32)
    return {"left_hand_q": left, "right_hand_q": right, "left_hand_dq": velocity, "right_hand_dq": velocity,
            "left_hand_tau_est": np.full(7, 0.1, np.float32), "right_hand_tau_est": np.full(7, 0.6, np.float32)}


def requests(data, hand_mode, control_frames):
    first = yaw_matrix(data[0][0])
    states, closures = [], []
    for step, (yaw, joints, image) in enumerate(data):
        relative = first.T @ yaw_matrix(yaw)
        states.append(np.concatenate([relative[:, :2].reshape(6), joints, np.zeros(29, np.float32)]))
        closures.append([0.5 + 0.4 * np.sin(step / 30.0), 0.5 + 0.4 * np.cos(step / 25.0)])
        if step % control_frames == 0:
            payload = arena_payload("t", image, states)
            if hand_mode == "continuous":
                payload["observation"]["hand_closure_history"] = np.asarray(closures, np.float32).tolist()
            else:
                payload["hand_observation"] = {k: v.tolist() for k, v in hand_measurement(step).items()}
            yield payload
            states, closures = [], []


@pytest.mark.parametrize("hand_mode", ["binary", "continuous"])
def test_simple_matches_original(hand_mode):
    data = episode(steps=500)
    old_model, new_model = FakeModel(hand_mode), FakeModel(hand_mode)
    rt, original = old_simple_runtime(old_model, 50, hand_mode)
    debounce = hands.HandDebounce(50.0) if hand_mode == "binary" else None
    planner = OriginalTimingPlanner(new_model, {"t": torch.zeros(1, 8)}, control_fps=50.0, execution_frames=50,
                            root_boundary="extrapolate", hand_filter=debounce)
    handlers = routes(make_policy(planner, hand_mode, debounce))
    rt.reset(3)
    handlers[("POST", "/reset")]({"seed": 3})

    worst_40 = worst_36 = 0.0
    switches = 0
    for payload in requests(data, hand_mode, control_frames=83):
        old = np.asarray(rt.infer(payload), np.float32)
        status, response = handlers[("POST", "/infer")](payload)
        new = np.asarray(response["action_chunk"], np.float32)
        assert status == 200 and new.shape == old.shape
        worst_40 = max(worst_40, float(np.abs(new - old).max()))
        old_36 = original.arena_action_to_simple(old, 50.0, hand_control_mode=hand_mode)
        worst_36 = max(worst_36, float(np.abs(command(new, hand_mode) - old_36).max()))
        switches += int(np.any(np.diff(old[:, 38:40], axis=0) != 0))
    worst = compare_calls(old_model.calls, new_model.calls)
    print(hand_mode, {k: f"{v:.1e}" for k, v in worst.items()}, f"40-D {worst_40:.1e}  36-D {worst_36:.1e}  chunks with a hand change {switches}")
    assert all(v < 1e-4 for v in worst.values()) and worst_40 < 1e-4 and worst_36 < 1e-4
    assert switches > 0  # the hands actually moved, so the debounce / closure paths were compared


def test_hand_debounce_matches_original_state_machine():
    """Random predictions and measurements through both state machines, 400 replans."""
    import evaluation.simple_server as original

    old = object.__new__(original._SimpleRuntimeMixin)
    old.simple_hand_control_mode, old.control_fps = "binary", 50.0
    old._init_simple_hand_fsm()
    new = hands.HandDebounce(50.0)
    rng = np.random.default_rng(0)
    changes, before = 0, new.state.copy()
    for replan in range(400):
        measurement = hand_measurement(int(rng.integers(0, 1000)))
        if rng.random() < 0.3:  # sometimes a hand barely moves, sometimes it is still moving
            measurement["left_hand_dq"] = np.zeros(7, np.float32)
        old._advance_simple_hand_phase(measurement)
        new.advance(measurement)
        # A per-hand intention that persists over a few replans, plus short noisy flips:
        # reaches closing, the two-replan opening rule, the tail-scheduled close and the noise rejection.
        level = np.where(rng.random(2) < 0.25, ~level, level) if replan else rng.random(2) < 0.5
        prediction = np.repeat(level[None], 50, 0).astype(np.float32)
        for side in range(2):
            if rng.random() < 0.4:
                start = int(rng.integers(0, 50))
                prediction[start:start + int(rng.integers(1, 4)), side] = 1 - prediction[start, side]
        prefix = torch.from_numpy(prediction[: int(rng.choice([15, 30, 50]))])
        expected, _ = old._resolve_simple_hand_prefix(prefix, detection_hand=torch.from_numpy(prediction))
        got = new(prefix, torch.from_numpy(prediction))
        assert torch.equal(expected, got), replan
        assert list(old.simple_hand_phase) == new.phase and np.array_equal(old.simple_hand_state, new.state), replan
        steps = int(round(len(prefix) * 50 / 30))
        old.simple_hand_last_chunk_steps = steps
        new.executed(steps)
        changes += int((new.state != before).any())
        before = new.state.copy()
    assert changes > 10, changes  # hands opened and closed many times
