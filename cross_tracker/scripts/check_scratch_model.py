"""Overfit fixed training windows using Kimodo defaults, without pt_419h."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]
import torch
from torch.utils.data import default_collate
from check_model import forward
from sampling import TrainingDataset
from train import build_model, param_groups
from run_config import load_yaml
from kimodo_vla.model.planner import load_text_embeddings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--sources", nargs="+", required=True, choices=["sonic", "twist2"])
    args = parser.parse_args()
    run = (ROOT / "runs" / args.run_id).resolve()
    assert run.parent == ROOT / "runs"
    run.mkdir(exist_ok=False)
    config = load_yaml(ROOT / "configs/box_screen.yaml")
    assert "initialization" not in config
    torch.manual_seed(42)
    dataset = TrainingDataset([ROOT / f"data/cache/prepared-v1/train/HOI_pp_box/{s}" for s in args.sources],
                              text_embeddings=load_text_embeddings(ROOT / "data/cache/text"))
    indices = []
    for source in args.sources:
        episode = next(i for i, row in enumerate(dataset.episodes) if row["tracker"] == source)
        windows = [i for i, (e, _) in enumerate(dataset.index) if e == episode]
        indices.extend([windows[min(100, len(windows)-1)], windows[-1]])
    batches = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in default_collate([dataset[i]]).items()}
               for i in indices]
    model = build_model(config).cuda().train()  # Deliberately no initialize_from / policy checkpoint.
    assert all(not p.requires_grad for p in model.denoiser.parameters())
    assert all(not p.requires_grad for p in model.image_encoder.parameters())
    groups = param_groups(model, config)
    opt = config.training.optimizer
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt.betas), eps=opt.eps, fused=True)
    parameters = [p for g in groups for p in g["params"]]
    record = dict(stage="diagnostic", status="running", seed=42, sources=args.sources,
                  initialization="kimodo_default_without_pt419h", policy_checkpoint_loaded=False)
    (run / "run.json").write_text(json.dumps(record, indent=2)+"\n")

    def evaluate():
        model.eval()
        values = []
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            for seed in (7001, 7002, 7003):
                torch.manual_seed(seed)
                values.extend(float(forward(model, batch)["loss"]) for batch in batches)
        model.train()
        return sum(values) / len(values)

    try:
        initial = evaluate()
        with (run / "metrics.jsonl").open("x") as log:
            for step in range(200):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(42+step)
                loss_sum = 0.0
                for batch in batches:
                    loss = forward(model, batch)["loss"] / len(batches)
                    loss.backward()
                    loss_sum += float(loss.detach())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                log.write(json.dumps(dict(step=step+1, loss=loss_sum, gradient_norm=float(norm)))+"\n")
                log.flush()
        final = evaluate()
        result = dict(passed=final < initial*.5, steps=200, initial_fixed_seed_loss=initial,
                      final_fixed_seed_loss=final, window_indices=indices,
                      policy_checkpoint_loaded=False, meaning="Optimization check, not task success")
        (run / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        assert result["passed"], result
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        (run / "run.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
