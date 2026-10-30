"""Read the completed scratch40k evidence; optionally probe feedback timing on CPU.

PYTHONPATH=. /root/workspace/miniforge3/envs/kimodo/bin/python \
    scripts/analyze_toolbench_scratch40k.py --probe-feedback \
    --output reports/toolbench_scratch40k_analysis_evidence.json

No checkpoints, simulators, policy services or training jobs are started.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tarfile


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "log/toolbench/joint18_scratch40k_20261027_8gpu"
SUPERVISION = RUN.with_name(RUN.name + "_supervision")


def read(path):
    return json.loads(path.read_text())


def group_statistics(rows):
    return {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "ever_true": dict(Counter(k for r in rows for k, v in r["ever_true"].items() if v)),
        "first_touch_roles": dict(Counter(r["final_metrics"]["first_touch_role"] for r in rows)),
        "first_move_directions": dict(Counter(r["final_metrics"]["first_move_toward_target"]
                                             for r in rows if "first_move_toward_target" in r["final_metrics"])),
        "median_max_lift_m": statistics.median(r["numeric_max"]["tool_lift"] for r in rows),
    }


def probe_feedback():
    """Change only the newly received measured joints/hands, keeping history and image fixed.

    Tests the actual conditioning and decoding paths, with fixed synthetic model output.
    This is a structural sensitivity probe, not a trained-policy success experiment.
    """
    from types import SimpleNamespace
    import numpy as np
    import torch
    from kimodo_vla.benchmarks.toolbench.adapter import measured_observation
    from kimodo_vla.core.types import G1_JOINT_INDEX, yaw_matrix
    from kimodo_vla.model.planner import KimodoPlanner

    config = SimpleNamespace(hand_control_mode="continuous", action_chunk=50, action_history=100)
    model = SimpleNamespace(config=config, fps=30, text_encoder=None)
    planner = KimodoPlanner(model, {}, control_fps=50.0, root_boundary="extrapolate")
    planner._first_heading = yaw_matrix(0.0)
    result = []
    for step in (0, 81, 163, 245):
        history = []
        for index in range(step + 1):
            state = np.zeros(32, np.float32)
            state[31] = 0.74
            observation = measured_observation(state, 0.0, "test")
            observation.step = index
            history.append(observation)
        changed = copy.deepcopy(history)
        changed[-1].joint_pos[G1_JOINT_INDEX["left_elbow_joint"]] = 1.0
        changed[-1].hand[:] = 1.0
        baseline_motion, mask, _, _ = planner.condition_history(history)
        changed_motion, _, _, _ = planner.condition_history(changed)
        frames, cut = baseline_motion.shape[0], round(step * 0.6)
        baseline_hand = planner._condition_hand(history, cut, frames)
        changed_hand = planner._condition_hand(changed, cut, frames)
        _, _, local_rot, root = planner.kimodo.history(history, planner._first_heading)
        fixed_output = {
            "local_rot_mats": local_rot[-1:].repeat(50, 1, 1, 1),
            "root_positions": root[-1:].repeat(50, 1),
            "hand_closure": torch.full((50, 2), 0.25),
        }
        first = planner._decode(fixed_output, history[-1], cut, local_rot[-1], root[-1]).to_array40()
        second = planner._decode(fixed_output, changed[-1], cut, local_rot[-1], root[-1]).to_array40()
        result.append({
            "control_step": step,
            "history_model_frames": frames,
            "latest_elbow_perturbation_rad": 1.0,
            "latest_hand_perturbation": 1.0,
            "masked_motion_condition_max_change": float((baseline_motion - changed_motion).abs()[mask].max()) if mask.any() else 0.0,
            "hand_condition_max_change": float((baseline_hand - changed_hand).abs().max()) if baseline_hand.numel() else 0.0,
            "fixed_prediction_decoded_action_max_change": float(np.max(np.abs(first - second))),
        })
    # A prior request's measurement DOES enter a later plan. This is delay,
    # rather than permanent removal of all measured proprioception.
    history = history[:164]
    changed = copy.deepcopy(history)
    changed[81].joint_pos[G1_JOINT_INDEX["left_elbow_joint"]] = 1.0
    a, mask, _, _ = planner.condition_history(history)
    b, _, _, _ = planner.condition_history(changed)
    older_change = float((a - b).abs()[mask].max())
    return {"scope": "CPU structural probe; no trained model or simulated success measured",
            "current_measurement": result,
            "step_163_condition_change_from_step_81_measurement": older_change}


def analyze():
    evaluation = SUPERVISION / "evaluation_18x100"
    groups = {}
    for directory in sorted(evaluation.glob("G1*-L*-*")):
        if not directory.is_dir():
            continue
        receipts = [p for p in directory.glob("run-*/benchmark_result.json") if read(p).get("status") == "validated"]
        if len(receipts) != 1:
            raise ValueError(f"Expected one validated run for {directory}, found {len(receipts)}")
        receipt_path = receipts[0]
        receipt = read(receipt_path)
        rows = [read(p) for p in sorted(receipt_path.parent.glob("summaries/*.json"))]
        if len(rows) != receipt["episodes"] or sum(r["success"] for r in rows) != receipt["successes"]:
            raise ValueError(f"Summary/receipt disagreement: {directory}")
        groups[directory.name] = rows
    if len(groups) != 18:
        raise ValueError(f"Expected 18 conditions, found {len(groups)}")
    data = read(RUN / "dataset.json")
    total_windows = sum(data["windows"].values())
    raw = read(REPO / "outputs/toolbench_validation/raw_audit.json")
    metrics = [json.loads(line) for line in (RUN / "metrics.jsonl").read_text().splitlines()]
    training = {}
    for upper in (1000, 10000, 20000, 30000, 40000):
        rows = [r for r in metrics if upper - 1000 < r["step"] <= upper]
        training[str(upper)] = {k: statistics.mean(r[k] for r in rows) for k in (
            "loss", "root_loss", "body_loss", "hand_state_loss", "hand_transition_loss", "grad_control", "grad_hand")}
    paths = ["kimodo_vla/model/planner.py", "kimodo_vla/model/features.py",
             "kimodo_vla/benchmarks/toolbench/adapter.py", "kimodo_vla/data/episodes.py"]
    with tarfile.open(SUPERVISION / "source_snapshot.tar.gz") as archive:
        snapshot = {name: {"matches_training_snapshot": archive.extractfile(name).read() == (REPO / name).read_bytes(),
                           "current_sha256": hashlib.sha256((REPO / name).read_bytes()).hexdigest()}
                    for name in paths}
    trials = read(REPO / "log/toolbench/tuning_20261027/leaderboard.json")
    l0_metadata = []
    raw_root = REPO.parent / "ToolBook/raw"
    for path in sorted(raw_root.glob("*/*-L0-*/**/meta/episodes.jsonl")):
        l0_metadata.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    l2 = [r for name, rows in groups.items() if "-L2-" in name for r in rows]
    reached = [r for r in l2 if r["ever_true"]["reached_target_bench"]]
    reached_steps = [r["first_true_step"]["reached_target_bench"] for r in reached]
    return {
        "source_run": str(RUN.relative_to(REPO)),
        "scope": "Recomputed from all 1800 saved episode summaries and validated receipts; no new rollout",
        "aggregate": group_statistics([r for rows in groups.values() for r in rows]),
        "levels": {f"L{level}": group_statistics([r for name, rows in groups.items() if f"-L{level}-" in name for r in rows])
                   for level in range(3)},
        "conditions": {name: group_statistics(rows) for name, rows in groups.items()},
        "l2_reach_metric_limits": {
            "reached_at_step_1": sum(t == 1 for t in reached_steps),
            "median_first_reached_control_step": statistics.median(reached_steps),
            "reached_without_ever_lifting_correct_tool": sum(not r["ever_true"]["picked_correct_tool"] for r in reached),
            "lifted_before_first_reach": sum(r["first_true_step"].get("picked_correct_tool") is not None
                and r["first_true_step"]["picked_correct_tool"] < r["first_true_step"]["reached_target_bench"] for r in reached),
            "scope": "Reach is a latched 0.40m-radius metric, not evidence of carrying the tool or accurate manipulation",
        },
        "dataset": {
            "windows": total_windows,
            "sample_presentations": 40000 * 128,
            "window_equivalent_passes": 40000 * 128 / total_windows,
            "level_window_mass": {f"L{level}": sum(v for name, v in data["windows"].items() if f"-L{level}-" in name) / total_windows
                                  for level in range(3)},
            "episodes_with_hand_reconstruction_error_over_0_01_rad": sum(r["hand_max_error"] > 0.01 for r in raw["rows"]),
            "raw_episodes": raw["episodes"],
            "local_l0_metadata_episodes": len(l0_metadata),
            "local_l0_derived_prefix_episodes": sum(r.get("success_source") == "derived_l0_prefix" for r in l0_metadata),
            "local_l0_parent_outcomes": dict(Counter(r.get("source_termination_reason") for r in l0_metadata)),
            "hand_reconstruction_max_error_rad": raw["hand_max_error"],
            "frame_weighted_hand_mae_rad": sum(r["hand_mean_error"] * r["frames"] for r in raw["rows"]) / raw["frames"],
        },
        "training_1000_step_means": training,
        "source_identity": snapshot,
        "completed_tuning_trials_at_analysis": [{k: t[k] for k in (
            "name", "primary_successes", "primary_episodes", "total_successes", "total_episodes", "seed_start", "stage")}
            for t in trials],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-feedback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    if args.probe_feedback:
        result["feedback_probe"] = probe_feedback()
    content = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
        print(args.output)
    else:
        print(content)


if __name__ == "__main__":
    main()
