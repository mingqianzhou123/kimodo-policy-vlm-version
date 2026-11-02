"""Evaluate through the unchanged HumanoidArena SONIC evaluator (60 episodes).

Accepts both original ControlNet checkpoints and the new N1.7 checkpoints.
Use exactly the same command/environment for both; only --checkpoint changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys

from groot_experiment import atomic_json
from run_config import REPO

TASKS = {
    "football": {"task": "Isaac-Move-Football-Single-G129-Dex3-Wholebody", "yaml": "football_single_sonic_test.yaml", "max_steps": 2000, "paper_sr": .35},
    "doubledesk": {"task": "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody", "yaml": "doubledesk_sonic_test.yaml", "max_steps": 2000, "paper_sr": .40},
    "pp_box": {"task": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby", "yaml": "pp_box_sonic_test.yaml", "max_steps": 1450, "paper_sr": .80},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, default="doubledesk")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arena-root", type=Path, default=Path("/root/workspace/HumanoidArena"))
    parser.add_argument("--sim-python", default="/root/workspace/miniforge3/envs/unitree_sim_env/bin/python")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--port", type=int, default=19871)
    parser.add_argument("--smoke", action="store_true", help="one short episode; NEVER a success-rate measurement")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reference-manifest", type=Path,
                        help="Require identical simulator/controller assets and inference settings")
    args = parser.parse_args()
    task = TASKS[args.task]
    arena = args.arena_root.resolve()
    lab = arena / "isaaclab_twist2_g1"
    evaluator = lab / "script/eval_scripts/sonic/eval_vla_suite.py"
    environment = lab / "tasks/common_test_config/base_test" / task["yaml"]
    sonic = arena / "GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
    assets = [evaluator, environment, sonic / "model_encoder.onnx", sonic / "model_decoder.onnx"]
    for path in assets + [args.checkpoint / "training_state.pt", args.checkpoint / "config.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [args.sim_python, str(evaluator),
        "--task", task["task"],
        "--env_config_yaml", str(environment), "--model-path", str(args.checkpoint.resolve()),
        "--repeats_per_seed", "1" if args.smoke else "20", "--persistent_sim", "0",
        "--max_steps", "30" if args.smoke else str(task["max_steps"]), "--video_fps", "50",
        "--post_termination_record_steps", "10", "--record_video_every_n", "0", "--step_log_every_n", "100",
        "--robot_type", "unitree_g1_refpose_v3_1", "--sonic_encoder_path", str(sonic / "model_encoder.onnx"),
        "--sonic_decoder_path", str(sonic / "model_decoder.onnx"), "--sonic_vla_root_rot6d_layout", "row",
        "--sonic_vla_root_max_delta_deg", "26", "--results_dir", str(args.out.resolve()), "--headless",
        "--isaac_device", f"cuda:{args.gpu}", "--server_python", args.server_python,
        "--server_script", str(REPO / "scripts/serve_arena_original.py"), "--server_device", f"cuda:{args.gpu}",
        "--server_host", "127.0.0.1", "--server_port", str(args.port), "--server_scheme", "http",
        "--server_ready_timeout", "600", "--lerobot_server_timeout", "30"]
    for seed in ([0] if args.smoke else [0, 1, 2]):
        command.extend(["--seed", str(seed)])
    receipt = {"command": command, "smoke_only": args.smoke, "task_slug": args.task,
        "policy_checkpoint": str(args.checkpoint.resolve()),
        "policy_config_sha256": hashlib.sha256((args.checkpoint / 'config.json').read_bytes()).hexdigest(),
        "evaluation": {"dtype": "fp32", "diffusion_steps": 10, "execution_frames": 15,
                       "timing": "original", "rtc": False, "control_fps": 50, "prediction_frames": 50},
        "asset_sha256": {str(path.relative_to(arena)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in assets},
        "paper_reference_sr": task["paper_sr"], "paper_table": 2, "paper_row": "Ours (scratch)",
        "paper_reference_is_measured_result": False}
    if args.reference_manifest:
        reference = json.loads(args.reference_manifest.read_text())
        for key in ['asset_sha256', 'evaluation', 'task_slug']:
            if reference.get(key) != receipt[key]:
                raise RuntimeError(f'Evaluation environment/protocol drift: {key}')
        receipt['reference_manifest'] = str(args.reference_manifest.resolve())
    print(json.dumps(receipt, indent=2), flush=True)
    if args.dry_run:
        return
    if (args.out / "summary.json").exists():
        raise FileExistsError(f"Evaluation already exists: {args.out}; choose a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "evaluation_manifest.json", receipt)
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    sim_lib = str(Path(args.sim_python).resolve().parents[1] / "lib")
    env["LD_LIBRARY_PATH"] = sim_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(command, cwd=lab, env=env, check=True)
    # The upstream evaluator can exit zero after HTTP 500s and report timeout.
    # Require evidence of real policy inference, for smoke AND formal scoring.
    episode_files = list((args.out / "episodes").glob("*.json"))
    expected = 1 if args.smoke else 60
    if len(episode_files) != expected:
        raise RuntimeError(f"Expected {expected} episodes, found {len(episode_files)}")
    logs = list((args.out / "logs").glob("server__*.log"))
    successful_inferences = 0
    for path in logs:
        content = path.read_text(errors="replace")
        if re.search(r'POST /infer HTTP/[^\"]+\" 5\d\d', content) or "Traceback (most recent call last)" in content:
            raise RuntimeError(f"Model-server errors in {path}; this evaluation is invalid")
        successful_inferences += len(re.findall(r'POST /infer HTTP/[^\"]+\" 200', content))
    if successful_inferences == 0:
        raise RuntimeError("No successful model inference; simulator timeout is not a valid evaluation")
    atomic_json(args.out / "validation.json", {"valid_inference": True,
        "smoke_only": args.smoke, "episodes": len(episode_files),
        "successful_inferences": successful_inferences})


if __name__ == "__main__":
    main()
