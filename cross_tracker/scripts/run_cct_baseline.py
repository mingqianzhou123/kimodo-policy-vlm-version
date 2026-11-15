"""Launch a registered CCT baseline; explicitly record deferred simulation."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from backup_weights import backup
from run_box_screen import digest, read, write

ROOT = Path(__file__).resolve().parents[1]


def validate_simulation(report, controller, checkpoint_sha, context_sha):
    assert report["controller"] == controller and report["task"] == "box"
    assert report["status"] == "completed" and report["passed"]
    assert report["checkpoint_sha256"] == checkpoint_sha
    assert report["context_manifest_sha256"] == context_sha
    assert report["seed"] == 1000 and report["diffusion_steps"] == 10
    assert report["episode_steps"] == 50 and report["video_frames"] == 51
    assert report["successful_inference_requests"] >= 2
    assert report["history_packet_sizes"][:2] == [1, 25]
    assert report["exit_code"] == 0


def check_simulations(plan, checkpoint_sha):
    missing = []
    for controller, rel in plan["simulation_reports"].items():
        if not (ROOT / rel).is_file():
            missing.append(controller)
            continue
        validate_simulation(read(ROOT / rel), controller, checkpoint_sha,
                            plan["context_manifest_sha256"])
    if missing:
        approval = plan.get("simulation_deferral", {})
        if approval.get("user_authorized") is not True or not approval.get("request"):
            raise RuntimeError("Real CCT simulation checks still required: " + ", ".join(missing))
    return missing


def preflight(run):
    assert run.resolve().parent == ROOT / "runs"
    plan = read(run / "plan.json")
    assert plan["steps"] == 20000 and plan["seed"] == 42
    assert plan["sources"] == ["sonic", "twist2"]
    assert set(plan["simulation_reports"]) == {"sonic", "twist2"}
    assert not (run / "training").exists(), "Never overwrite or resume a formal run"
    for rel, sha in read(run / "source-files.json").items():
        assert digest(ROOT.parent / rel) == sha, rel
    for name, sha in plan["frozen_files"].items():
        assert digest(run / name) == sha, name
    checks = ROOT / "runs/cct-gpu-checks-20261114"
    result = read(checks / "result.json")
    assert result["passed"] and result["status"] == "completed"
    assert all(result["stages"][stage]["passed"] for stage in ("overfit", "ddp", "interface"))
    ddp = read(ROOT / "runs/cct-ddp-pilot-20261114/result.json")
    assert ddp["passed"] and ddp["world_size"] == 8 and ddp["steps"] == 200
    assets = read(ROOT / "runs/box-screen-setup-20261114/assets-verified.json")
    assert assets["passed"] and assets["prepared_episodes"] == 108
    for rel, sha in assets["files"].items():
        assert digest(ROOT / rel) == sha, rel
    bank_path = ROOT / "data/cache/cct-box-v1/manifest.json"
    assert digest(bank_path) == plan["context_manifest_sha256"]
    bank = read(bank_path)
    assert bank["complete"] and len(bank["episodes"]) == 86
    for rel, sha in bank["files"].items():
        assert digest(bank_path.parent / rel) == sha, rel
    check_simulations(plan, ddp["checkpoint_sha256"])
    return plan


def command(run):
    protocol = read(run / "config.json")
    initial = ROOT / "data/cache/weights" / protocol["initial_checkpoint"]["path"]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
            str(ROOT / "scripts/train_cct_worker.py"), "--config", str(ROOT / "configs/cct_box.yaml"),
            "--run-dir", str(run / "training"), "--init-checkpoint", str(initial),
            "--task", "HOI_pp_box/sonic", "HOI_pp_box/twist2", "--device", "cuda"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    plan = preflight(run)
    if args.check_only:
        print(json.dumps({"passed": True, "run_id": args.run_id, "training_started": False,
                          "simulation_deferral": plan.get("simulation_deferral")}))
        return
    import torch
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert len(gpus) == 8 and all("A800" in name for name in gpus), gpus
    record = read(run / "run.json")
    assert record["status"] in ("planned", "queued")
    argv = command(run)
    env = dict(os.environ, HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    env.pop("CCT_DIAGNOSTIC_DIR", None)
    record.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  actual_command=argv, gpus=gpus,
                  simulation_pending=check_simulations(plan, read(ROOT / "runs/cct-ddp-pilot-20261114/result.json")["checkpoint_sha256"]))
    write(run / "run.json", record)
    try:
        with (run / "launcher.log").open("x") as log:
            subprocess.run(argv, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (run / "training/metrics.jsonl").read_text().splitlines()]
        assert rows[-1]["step"] == plan["steps"] and all(math.isfinite(row["loss"]) for row in rows)
        checkpoint = run / "training/checkpoint_20000"
        payload = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        assert payload["global_step"] == plan["steps"]
        assert all(torch.isfinite(t).all() for t in payload["model"].values())
        del payload
        record["backup"] = backup(checkpoint, ROOT.parent.parent / "cross_tracker_backups/weights" / run.name / checkpoint.name)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        metrics = run / "training/metrics.jsonl"
        record.update(training_started=metrics.exists() and metrics.stat().st_size > 0,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        write(run / "run.json", record)


if __name__ == "__main__":
    main()
