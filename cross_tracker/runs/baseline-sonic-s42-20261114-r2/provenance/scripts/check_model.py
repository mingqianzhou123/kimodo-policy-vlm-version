"""Verify pretrained GPU forward/backward and overfit two fixed, real training windows."""
import argparse
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]
import torch
from torch.utils.data import default_collate
from fetch_assets import matches
from sampling import TrainingDataset
from train import build_model, check_initialization, initialize_from, param_groups
from run_config import load_yaml
from kimodo_vla.model.planner import load_text_embeddings


def forward(model, batch):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(batch["instruction"], batch["egoview"], batch["gt_motion"], batch["gt_mask"],
                     condition_motion=batch["condition_motion"],
                     condition_motion_mask=batch["condition_motion_mask"],
                     gt_hand=batch["gt_hand"], gt_hand_mask=batch["gt_hand_mask"],
                     text_feat=batch["text_embedding"], text_length=batch["text_length"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--data", type=Path, required=True, help="One prepared task/source training folder")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    if run.parent != ROOT / "runs" or not (run / "run.json").exists():
        parser.error("Register a new diagnostic run first")
    registered = json.loads((run / "run.json").read_text())
    if registered["stage"] != "diagnostic" or registered["status"] != "planned":
        parser.error("A fresh diagnostic registration is required")
    if not 1 <= args.steps <= 1000 or not args.data.resolve().is_relative_to(ROOT / "data/cache"):
        parser.error("Use 1..1000 diagnostic steps and this project's prepared cache")
    for folder in ("scripts", "src"):
        for path in (ROOT / folder).glob("*.py"):
            if path.read_bytes() != (run / "provenance" / folder / path.name).read_bytes():
                parser.error("Code changed after registration")
    protocol = json.loads((run / "config.json").read_text())
    checkpoint_rel = protocol["initial_checkpoint"]["path"]
    weights = json.loads((ROOT / "data/manifests/weights-v1.json").read_text())["files"]
    checkpoint = ROOT / "data/cache/weights" / checkpoint_rel
    for name in ("config.json", "training_state.pt"):
        if not (checkpoint / name).is_file() or not matches(checkpoint / name, weights[f"{checkpoint_rel}/{name}"]):
            parser.error("Pretrained checkpoint missing or not verified")
    shared = json.loads((ROOT / "data/manifests/shared-weights-v1.json").read_text())["files"]
    for name, expected in shared.items():
        if not (ROOT.parent / name).is_file() or not matches(ROOT.parent / name, expected):
            parser.error(f"Shared backbone asset differs from pinned release: {name}")
    config = load_yaml(ROOT / "configs/kimodo_pilot.yaml")
    config.initialization = {"checkpoint": str(checkpoint)}
    check_initialization(config)
    (run / "model-config.yaml").write_text(OmegaConf.to_yaml(config))
    (run / "diagnostic-config.json").write_text(json.dumps({
        "data": str(args.data.resolve()), "steps": args.steps, "scheduler": None,
        "gradient_clip": 1.0, "evaluation_seeds": [7001, 7002, 7003],
        "seed": registered["seed"], "torch": torch.__version__,
        "cuda": torch.version.cuda, "argv": sys.argv,
    }, indent=2)+"\n")
    text = load_text_embeddings(ROOT / "data/cache/text")
    dataset = TrainingDataset([args.data], text_embeddings=text)
    record = dataset.episodes[0]
    if record["instruction"] not in text:
        parser.error("The real task instruction has no cached LLM2Vec embedding")
    count = record["target_motion"].shape[0]-49
    indices = sorted({max(0, min(100, count-2)), count-1})
    batches = [{k: v.cuda() if torch.is_tensor(v) else v
                for k, v in default_collate([dataset[i]]).items()} for i in indices]
    torch.manual_seed(registered["seed"])
    model = build_model(config)
    initialize_from(model, config)
    model.cuda().train()
    groups = param_groups(model, config)
    opt = config.training.optimizer
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt.betas), eps=opt.eps, fused=True)
    parameters = [p for g in groups for p in g["params"]]
    registered.update(status="running", training_started=True, formal_training_started=False)
    (run / "run.json").write_text(json.dumps(registered, indent=2))

    def evaluate():
        model.eval()
        values = []
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            for seed in (7001, 7002, 7003):
                torch.manual_seed(seed)
                values.extend(float(forward(model, batch)["loss"]) for batch in batches)
        model.train()
        return sum(values)/len(values)

    initial = evaluate()
    if not torch.isfinite(torch.tensor(initial)):
        raise ValueError("Non-finite initial GPU loss")
    with (run / "metrics.jsonl").open("x") as log:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.manual_seed(registered["seed"]+step)
            total = 0.0
            for batch in batches:
                loss = forward(model, batch)["loss"]/len(batches)
                loss.backward(); total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            log.write(json.dumps({"step": step+1, "loss": total, "gradient_norm": float(norm)})+"\n")
            log.flush()
            if (step+1) % 20 == 0:
                print(f"STEP {step+1} LOSS {total:.5f}", flush=True)
    final = evaluate()
    result = {"episode": record["episode_id"], "window_indices": indices, "steps": args.steps,
              "initial_fixed_seed_loss": initial, "final_fixed_seed_loss": final,
              "ratio": final/initial, "passed": final < initial*0.5,
              "meaning": "fixed-window optimization diagnostic, not task success",
              "gpu": torch.cuda.get_device_name(0), "peak_memory_bytes": torch.cuda.max_memory_allocated()}
    (run / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    registered.update(status="completed" if result["passed"] else "check_failed", training_started=True)
    (run / "run.json").write_text(json.dumps(registered, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        run = (ROOT / "runs" / sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT
        record = run / "run.json"
        if run.parent == ROOT / "runs" and record.is_file():
            status = json.loads(record.read_text())
            if status["status"] == "running":
                metrics = run / "metrics.jsonl"
                status.update(status="failed", error_type=type(error).__name__,
                              training_started=metrics.exists() and metrics.stat().st_size > 0)
                record.write_text(json.dumps(status, indent=2)+"\n")
        raise
