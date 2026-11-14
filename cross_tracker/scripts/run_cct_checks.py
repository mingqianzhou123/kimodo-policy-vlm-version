"""Run GPU-only CCT gates in one allocation; never launch formal training."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from run_box_screen import digest, read, write
from backup_weights import backup

ROOT = Path(__file__).resolve().parents[1]


def preflight(setup):
    plan = read(setup / "plan.json")
    assert plan["formal_training"] is False and plan["steps"] == 200
    for relative, expected in read(setup / "source-files.json").items():
        assert digest(ROOT.parent / relative) == expected, relative
    assets = read(ROOT / "runs/box-screen-setup-20261114/assets-verified.json")
    assert assets["passed"] and assets["prepared_episodes"] == 108
    assert digest(ROOT / "data/manifests/balanced-v1.json") == assets["manifest_sha256"]
    for relative, expected in assets["files"].items():
        assert digest(ROOT / relative) == expected, relative
    manifest = ROOT / "data/cache/cct-box-v1/manifest.json"
    assert digest(manifest) == plan["context_manifest_sha256"]
    bank = read(manifest)
    assert bank["complete"] and len(bank["episodes"]) == 86
    for relative, expected in bank["files"].items():
        assert digest(manifest.parent / relative) == expected, relative
    for stage in ("overfit", "ddp", "interface"):
        run_id = plan[stage]
        assert (ROOT / "runs" / run_id).resolve().parent == ROOT / "runs"
        assert not (ROOT / "runs" / run_id).exists(), f"Refuse to overwrite {run_id}"
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup_id")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    setup = (ROOT / "runs" / args.setup_id).resolve()
    assert setup.parent == ROOT / "runs"
    plan = preflight(setup)
    if args.check_only:
        print(json.dumps(dict(passed=True, meaning="CPU preflight only; GPU gates not run")))
        return
    import torch
    assert torch.cuda.device_count() == 8, "Exactly eight allocated GPUs required"
    gpus = [torch.cuda.get_device_name(i) for i in range(8)]
    assert all("A800" in name for name in gpus), gpus
    os.chdir(ROOT.parent)
    report = dict(formal_training=False, passed=False, status="running", gpus=gpus,
                  started_utc=datetime.now(timezone.utc).isoformat(), source_commit=plan["source_commit"],
                  source_manifest_sha256=digest(setup / "source-files.json"),
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda, stages={})
    write(setup / "result.json", report)

    def run(stage, argv, extra_env=None):
        report["stages"][stage] = dict(status="running", argv=argv)
        write(setup / "result.json", report)
        with (setup / f"{stage}.log").open("x") as log:
            result = subprocess.run(argv, env=dict(os.environ, **(extra_env or {})), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=7200)
        report["stages"][stage].update(exit_code=result.returncode, status="exited")
        write(setup / "result.json", report)
        result.check_returncode()

    try:
        run("overfit", [sys.executable, str(ROOT / "scripts/check_cct_model.py"), plan["overfit"],
                        "--device", "cuda", "--steps", "200"])
        overfit = read(ROOT / "runs" / plan["overfit"] / "result.json")
        assert overfit["gpu_overfit_passed"] and overfit["optimization_steps"] == 200
        report["stages"]["overfit"]["passed"] = True
        backups = ROOT.parent.parent / "cross_tracker_backups/weights"
        report["overfit_backup"] = backup(ROOT / "runs" / plan["overfit"] / "checkpoint",
                                           backups / plan["overfit"] / "checkpoint")
        ddp = ROOT / "runs" / plan["ddp"]
        ddp.mkdir(exist_ok=False)
        write(ddp / "run.json", dict(stage="pilot", formal_training=False, status="running", seed=42))
        protocol = read(ROOT / "runs/box-sonic-twist2-s42-20261114/config.json")
        initial = ROOT / "data/cache/weights" / protocol["initial_checkpoint"]["path"]
        argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                str(ROOT / "scripts/train_cct_worker.py"), "--config", str(ROOT / "configs/cct_box.yaml"),
                "--run-dir", str(ddp / "training"), "--init-checkpoint", str(initial),
                "--task", "HOI_pp_box/sonic", "HOI_pp_box/twist2", "--device", "cuda",
                "main.stop_after=200", "main.checkpoint_every=200", "main.score_checkpoints=[200]",
                "main.log_every=10"]
        run("ddp", argv, {"CCT_DIAGNOSTIC_DIR": str(ddp)})
        ranks = [read(ddp / f"rank-{i}.json") for i in range(8)]
        assert all(r["passed"] and r["rank"] == i and r["world_size"] == 8 for i, r in enumerate(ranks))
        assert len({r["query_sha256"] for r in ranks}) == 1, "DDP parameters diverged"
        rows = [json.loads(line) for line in (ddp / "training/metrics.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == list(range(10, 201, 10))
        assert all(math.isfinite(r[k]) for r in rows for k in ("loss", "grad_control", "grad_hand"))
        checkpoint = ddp / "training/checkpoint_200"
        payload = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        assert payload["global_step"] == 200
        assert all(torch.isfinite(v).all() for v in payload["model"].values())
        del payload
        write(ddp / "result.json", dict(passed=True, steps=200, world_size=8, ranks=ranks,
              checkpoint_sha256=digest(checkpoint / "training_state.pt")))
        write(ddp / "run.json", dict(stage="pilot", formal_training=False, status="completed", seed=42))
        report["stages"]["ddp"]["passed"] = True
        report["ddp_backup"] = backup(checkpoint, backups / plan["ddp"] / "checkpoint_200")
        run("interface", [sys.executable, str(ROOT / "scripts/check_cct_interface.py"), plan["interface"],
                          "--checkpoint", str(checkpoint)])
        assert read(ROOT / "runs" / plan["interface"] / "result.json")["passed"]
        report["stages"]["interface"]["passed"] = True
        report.update(passed=True, status="completed", simulation="pending; not part of this A800 job")
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        write(setup / "result.json", report)


if __name__ == "__main__":
    main()
