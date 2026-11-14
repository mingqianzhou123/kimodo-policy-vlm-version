"""Check real CCT data, gradient flow, checkpoint round-trip and prediction."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import torch
from torch.utils.data import default_collate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]
import train
from cct_data import CCTDataset
from cct_policy import CCTPolicy
from kimodo_vla.model.planner import load_text_embeddings
from repaired_data import checksum
from run_config import load_yaml


def forward(model, batch):
    return model(batch["instruction"], batch["egoview"], batch["gt_motion"], batch["gt_mask"],
                 condition_motion=batch["condition_motion"], condition_motion_mask=batch["condition_motion_mask"],
                 gt_hand=batch["gt_hand"], gt_hand_mask=batch["gt_hand_mask"],
                 text_feat=batch["text_embedding"], text_length=batch["text_length"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--steps", type=int, default=0, help="0 checks gradients without optimizing; GPU overfit uses 200")
    args = parser.parse_args()
    if not 0 <= args.steps <= 200 or (args.steps and args.device != "cuda"):
        parser.error("Optimization is a GPU diagnostic of at most 200 steps")
    run = (ROOT / "runs" / args.run_id).resolve()
    if run.parent != ROOT / "runs":
        parser.error("Invalid run ID")
    run.mkdir(exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    started = time.monotonic()
    paths = list((ROOT / "src").glob("cct_*.py")) + [Path(__file__), ROOT / "scripts/train_cct_worker.py",
                                                        ROOT / "configs/cct_box.yaml", ROOT / "configs/cct-metadata.json"]
    evidence = dict(device=args.device, optimization_steps=args.steps, passed=False,
                    source_sha256={str(p.relative_to(ROOT)): checksum(p) for p in paths},
                    context_cache_sha256=checksum(ROOT / "data/cache/cct-box-v1/manifest.json"))
    (run / "run.json").write_text(json.dumps(dict(stage="diagnostic", formal_training=False, **evidence), indent=2))
    config = load_yaml(ROOT / "configs/cct_box.yaml")
    protocol = json.loads((ROOT / "runs/box-sonic-twist2-s42-20261114/config.json").read_text())
    config.initialization = {"checkpoint": str(ROOT / "data/cache/weights" / protocol["initial_checkpoint"]["path"])}
    train.check_initialization(config)
    text = load_text_embeddings(ROOT / "data/cache/text")
    data = CCTDataset([ROOT / f"data/cache/prepared-v1/train/HOI_pp_box/{s}" for s in ["sonic", "twist2"]],
                      text_embeddings=text)
    indices = [next(i for i, (episode, cut) in enumerate(data.index)
                    if data.episodes[episode]["tracker"] == source and cut == 100)
               for source in ["sonic", "twist2"]]
    samples = [data[i] for i in indices]
    batches = [{k: v.to(args.device) if torch.is_tensor(v) else v
                for k, v in default_collate([sample]).items()} for sample in samples]
    train.KimodoPolicy = CCTPolicy
    model = train.build_model(config)
    train.initialize_from(model, config)
    model.to(args.device).train()
    optimizer = torch.optim.AdamW(train.param_groups(model, config), betas=tuple(config.training.optimizer.betas),
                                  eps=config.training.optimizer.eps)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else contextlib.nullcontext()

    def loss_values():
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[0] if args.device == "cuda" else []), amp():
            torch.manual_seed(7001)
            values = [float(forward(model, batch)["loss"]) for batch in batches]
        model.train()
        return sum(values) / len(values)

    initial = loss_values()
    for step in range(max(1, args.steps)):
        optimizer.zero_grad(set_to_none=True)
        with amp():
            loss = sum(forward(model, b)["loss"] for b in batches) / len(batches)
        loss.backward()
        if step == 0:
            gradients = {name: float(p.grad.float().norm()) for name, p in model.cct_encoder.named_parameters()
                         if p.grad is not None}
            if not gradients or not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in gradients.values()):
                raise ValueError("Missing, zero or non-finite CCT gradients")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
        if args.steps:
            optimizer.step()
        if step % 20 == 0:
            print(f"diagnostic {step+1}/{max(1,args.steps)} loss={float(loss):.6f}", flush=True)
    final = loss_values()
    checkpoint = run / "checkpoint"
    checkpoint.mkdir()
    state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"model": state, "global_step": args.steps}, checkpoint / "training_state.pt")
    (checkpoint / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    reference = model.cct_encoder.queries.detach().clone()
    with torch.no_grad():
        model.cct_encoder.queries.add_(1)
    model.load_controlnet_checkpoint(checkpoint)
    assert torch.equal(reference, model.cct_encoder.queries) and model.has_trained_context
    model.eval()
    batch = batches[0]
    model.set_context(batch["egoview"]["context"])
    with torch.inference_mode():
        result = model.predict_future(batch["instruction"], batch["egoview"]["image"].to(args.device),
                                      batch["condition_motion"][:, :100], diffusion_steps=1,
                                      history_mask=batch["gt_mask"][:, :100],
                                      history_feature_mask=batch["condition_motion_mask"][:, :100],
                                      hand_history=batch["gt_hand"][:, :100],
                                      text_feat=batch["text_embedding"], text_length=batch["text_length"])
    shapes = {k: list(v.shape) for k, v in result.items() if torch.is_tensor(v)}
    assert shapes and all(torch.isfinite(v).all() for v in result.values() if torch.is_tensor(v))
    evidence.update(passed=(final < initial * .5 if args.steps else True), initial_loss=initial, final_loss=final,
                    gradient_norms=gradients, prediction_shapes=shapes, checkpoint_roundtrip=True,
                    target_episode_ids=[data.episodes[data.index[i][0]]["episode_id"] for i in indices],
                    context_episode_ids=[s["context_episode_ids"] for s in samples],
                    elapsed_seconds=time.monotonic()-started, gpu_overfit_passed=bool(args.steps and final < initial*.5))
    (run / "result.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(json.dumps({k:v for k,v in evidence.items() if k not in ("gradient_norms","source_sha256")}), flush=True)
    if not evidence["passed"]:
        raise SystemExit("Overfit diagnostic did not pass")


if __name__ == "__main__":
    main()
