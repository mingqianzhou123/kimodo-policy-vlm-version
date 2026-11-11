"""Real-data preflight. Only a complete VLM + gradient + inference pass permits training.

Use --backbone-only to check the real pretrained Kimodo backbone while waiting
for gated Cosmos assets; that mode NEVER produces a passing training receipt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import default_collate

from groot_experiment import atomic_json, fingerprint
from run_config import REPO, load_yaml
from train import build_dataset, build_model
from kimodo_vla.model.modules.denoiser import TwostageDenoiser
from kimodo_vla.model.modules.groot_conditioning import GrootInjection, VLCondition
from kimodo_vla.model.planner import load_text_embeddings
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34


def grad_sum(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def check_backbone(model, batch, condition, report):
    representation, denoiser, injection = model.representation, model.denoiser, model.controlnet
    h = model.config.action_history
    x = representation.normalize(batch["gt_motion"])
    observed = representation.normalize(batch["condition_motion"])
    mask = batch["condition_motion_mask"].to(x.dtype)
    observed = observed.masked_fill(mask == 0, 0)
    text = batch["text_embedding"]
    t = torch.full((len(x),), 200, dtype=torch.long, device=x.device)
    heading_slice = representation.slice_dict["global_root_heading"]
    from kimodo_vla.model.kimodo_policy import KimodoPolicy
    heading = KimodoPolicy._first_history_heading(
        batch["condition_motion"][:, :h], batch["gt_mask"][:, :h], heading_slice)
    kwargs = dict(x=x, x_pad_mask=batch["gt_mask"], text_feat=text,
                  text_feat_pad_mask=torch.ones(text.shape[:2], dtype=torch.bool, device=x.device),
                  timesteps=t, first_heading_angle=heading, motion_mask=mask,
                  observed_motion=observed, detach_root_for_body=True)
    roots, bodies = injection(t, condition, x.shape[1], h)
    with torch.no_grad():
        base = denoiser(**kwargs)
        injected = denoiser(**kwargs, root_control_tokens=roots, body_control_tokens=bodies,
                            root_hint_fuser=injection.root_hint_fusion,
                            body_hint_fuser=injection.body_hint_fusion, control_future_start=h)
    error = float((base - injected).abs().max())
    if not torch.allclose(base, injected, atol=1e-5, rtol=1e-5):
        raise AssertionError(f"Zero-initialized injection changed pretrained Kimodo: {error}")
    report["zero_injection_max_error"] = error
    # Activate the zero-initialised injection heads so the detachment audit
    # cannot pass merely because their initial gate blocks upstream gradients.
    def activate_zero_stack(stack):
        for block in stack.blocks.values():
            if hasattr(block, "output_projection"):
                block.output_projection.weight.copy_(torch.eye(1024, device=x.device) * 0.01)
            elif hasattr(block, "modulation"):
                # Activate only the gate slice; this keeps the root/body
                # detach audit independent of the shift/scale initialisation.
                start = 2 * block.motion_dim
                gate = block.modulation.weight[start:start + block.motion_dim]
                gate.zero_()
                diagonal = min(gate.shape[0], gate.shape[1])
                gate[:diagonal, :diagonal].fill_diagonal_(0.01)
            else:
                raise AssertionError(f"Unsupported injection block {type(block).__name__}")

    def reset_zero_stack(stack):
        for block in stack.blocks.values():
            if hasattr(block, "output_projection"):
                block.output_projection.weight.zero_()
                block.output_projection.bias.zero_()
            else:
                block.modulation.weight.zero_()
                block.modulation.bias.zero_()

    with torch.no_grad():
        for stack in (injection.root_hint_fusion, injection.body_hint_fusion):
            activate_zero_stack(stack)
    prediction = denoiser(**kwargs, root_control_tokens=roots, body_control_tokens=bodies,
                          root_hint_fuser=injection.root_hint_fusion,
                          body_hint_fuser=injection.body_hint_fusion, control_future_start=h)
    prediction[:, h:, representation.body_slice].square().mean().backward()
    assert grad_sum(injection.root_hint_fusion) == 0, "Body loss crossed the Root detach boundary"
    assert grad_sum(injection.body_hint_fusion) > 0
    assert all(p.grad is None for p in denoiser.parameters())
    report["root_body_detach"] = True
    injection.zero_grad(set_to_none=True)
    with torch.no_grad():
        for stack in (injection.root_hint_fusion, injection.body_hint_fusion):
            reset_zero_stack(stack)


def run(args, config, report):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run in the GPU-enabled host environment")
    torch.manual_seed(config.main.seed)
    dataset = build_dataset(config, load_text_embeddings(config.main.text_cache))
    count = 1 if args.backbone_only else config.main.micro_batch_size
    indices = []
    expected_tasks = config.main.get("expected_tasks")
    if expected_tasks and not args.backbone_only:
        covered = set()
        for ordinal in range(10000):
            _, episode, _ = dataset._sample_record(dataset._rng_for_index(ordinal))
            if episode.task_name not in covered:
                indices.append(ordinal)
                covered.add(episode.task_name)
            if covered == set(expected_tasks):
                break
        assert covered == set(expected_tasks), "Preflight failed to cover all task instructions"
    indices += [i for i in range(count) if i not in indices]
    samples = [dataset[i] for i in indices[:count]]
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in default_collate(samples).items()}
    report["dataset"] = {"episodes": len(dataset._all_episode_records), "windows": len(dataset),
                         "instructions": dataset.task_instructions,
                         "motion_shape": list(batch["gt_motion"].shape),
                         "image_shape": list(batch["egoview"].shape),
                         "target_source": samples[0]["target_motion_source"], "sampling": dataset.sampling_mode}
    report["dataset"]["batch_tasks"] = sorted(set(s["task_id"] for s in samples))
    report["dataset"]["sample_ordinals"] = indices[:count]
    assert len(dataset._all_episode_records) == (sum(expected_tasks.values()) if expected_tasks else 100)
    assert samples[0]["target_motion_source"] == "action"
    assert dataset.sampling_mode == "episode_uniform"
    if args.backbone_only:
        stats = REPO / "checkpoints/Kimodo-G1-RP-v1"
        representation = KimodoMotionRep(G1Skeleton34(), fps=30, stats_path=str(stats / "stats/motion"))
        denoiser = TwostageDenoiser(representation, "concat", str(stats)).to(device).eval().requires_grad_(False)
        model = SimpleNamespace(
            representation=representation,
            denoiser=denoiser,
            controlnet=GrootInjection(
                denoiser,
                mode=config.model.get("groot_injection_mode", "cross_attention"),
                adaln_condition_dim=config.model.get("adaln_condition_dim", 512),
                adaln_pool=config.model.get("adaln_pool", "masked_mean"),
                max_diffusion_steps=config.model.get("adaln_max_diffusion_steps", 1000),
            ).to(device),
            config=config.main,
        )
        condition = VLCondition(torch.randn(1, 17, 2048, device=device),
                                torch.ones(1, 17, dtype=torch.bool, device=device))
        check_backbone(model, batch, condition, report)
        report["scope"] = "real Kimodo backbone + real Arena sample; synthetic VL tokens; NOT full VLM validation"
        return
    model = build_model(config).to(device).train()
    model.set_training_stage("injection")
    report["checkpoint_load"] = model.image_encoder.load_report
    report["official_source_hashes"] = model.image_encoder.source_hashes
    report["vlm_select_layer"] = model.image_encoder.checkpoint_config["select_layer"]
    assert not any(p.requires_grad for p in model.denoiser.parameters())
    assert not any(p.requires_grad for p in model.image_encoder.parameters())
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        condition = model._encode_condition(batch["egoview"], batch["instruction"])
    assert condition.tokens.shape[-1] == 2048 and torch.isfinite(condition.tokens).all()
    report["vl_condition_shape"] = list(condition.tokens.shape)
    report["vl_valid_token_lengths"] = condition.mask.sum(dim=1).tolist()
    if expected_tasks:
        assert len(report["dataset"]["batch_tasks"]) == len(expected_tasks)
        assert len(set(report["vl_valid_token_lengths"])) > 1, "Mixed instructions must exercise real VL padding"
    check_backbone(model, batch, condition.to(dtype=torch.float32), report)
    inputs = {k: batch[k] for k in ("instruction", "egoview", "gt_motion", "gt_mask",
               "condition_motion", "condition_motion_mask", "gt_hand", "gt_hand_mask")}
    inputs.update(text_feat=batch["text_embedding"], text_length=batch["text_length"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        losses = model(**inputs)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert grad_sum(model.controlnet) > 0 and grad_sum(model.hand_head) > 0
    assert all(p.grad is None for p in model.image_encoder.parameters())
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    report["stage1_loss"] = float(losses["loss"].detach())
    model.set_training_stage("adapter")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        losses = model(**inputs)
    losses["loss"].backward()
    assert grad_sum(model.image_encoder.adapter) > 0
    assert all(p.grad is None for p in model.image_encoder.backbone.parameters())
    assert all(p.grad is None for p in model.denoiser.parameters())
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    report["stage2_adapter_grad_sum"] = grad_sum(model.image_encoder.adapter)
    adapter_optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-5)
    adapter_optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    del adapter_optimizer
    model.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        losses = model(**inputs)
    losses["hand_loss"].backward()
    assert grad_sum(model.controlnet) == 0 and grad_sum(model.image_encoder) == 0
    assert grad_sum(model.hand_head) > 0
    report["hand_detach"] = True
    model.zero_grad(set_to_none=True)
    # Real checkpoint round-trip includes the adapter even when it is frozen.
    checkpoint_dir = Path(args.report).parent / "groot_preflight_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "training_state.pt"
    from omegaconf import OmegaConf
    atomic_json(checkpoint_dir / "config.json", OmegaConf.to_container(config, resolve=True))
    model.set_training_stage("injection")
    snapshot = {n: p.detach().cpu().clone() for n, p in model.checkpoint_parameters().items()}
    torch.save({"global_step": 1, "model": snapshot}, checkpoint)
    with torch.no_grad():
        next(model.image_encoder.adapter.parameters()).add_(1)
    model.load_controlnet_checkpoint(str(checkpoint))
    assert all(torch.equal(p.cpu(), snapshot[n]) for n, p in model.checkpoint_parameters().items())
    report["checkpoint_roundtrip"] = True
    report["smoke_checkpoint"] = str(checkpoint_dir.resolve())
    model.eval()
    calls = []
    hook = model.image_encoder.register_forward_hook(lambda *args: calls.append(1))
    prediction = model.predict_future(
        instruction=[batch["instruction"][0]], egoview=batch["egoview"][:1],
        history_motion=batch["condition_motion"][:1, :100],
        history_mask=batch["gt_mask"][:1, :100],
        history_feature_mask=batch["condition_motion_mask"][:1, :100],
        hand_history=batch["gt_hand"][:1, :100],
        text_feat=batch["text_embedding"][:1], diffusion_steps=10, squeeze_batch=False,
    )
    hook.remove()
    assert len(calls) == 1, "VLM must run once per plan, outside the diffusion loop"
    assert all(torch.isfinite(v).all() for v in prediction.values() if torch.is_tensor(v))
    report["inference_vlm_calls_per_plan"] = len(calls)
    report["inference_tensor_shapes"] = {k: list(v.shape) for k, v in prediction.items() if torch.is_tensor(v)}
    if device.type == "cuda":
        report["peak_cuda_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/arena_doubledesk_groot.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=REPO / "outputs/groot_preflight.json")
    parser.add_argument("--backbone-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    report = {"passed": False, "fingerprint": fingerprint(config), "backbone_only": args.backbone_only}
    try:
        report["repo_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        run(args, config, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        atomic_json(args.report, report)
        print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=2))


if __name__ == "__main__":
    main()
