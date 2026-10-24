"""Train Kimodo-Policy's ControlNet and hand head on one BiGym 2.0 task (one GPU).

    python scripts/train.py --task move_plate                       # three cameras
    python scripts/train.py --task move_plate --cameras head        # head only
    python scripts/train.py --task move_plate --cameras head --init-checkpoint <pt_419h checkpoint dir>
    python scripts/train.py --task move_plate main.max_steps=200    # any config field, OmegaConf dotlist
    python scripts/train.py --resume log/bigym2/<run>

Settings come from ``configs/bigym2.yaml``. Kimodo-G1, DINOv3 and LLM2Vec stay
frozen (LLM2Vec only through the cached embeddings of
``scripts/cache_text_embeddings.py``); the ControlNet and hand head start from
scratch unless ``--init-checkpoint`` is given. Samples are the task's
``KimodoWindowDataset`` windows, shuffled uniformly; the target is the operator's
command (``demos.commanded_motion``, written by ``prepare_bigym2.py``).

A run directory ``<save_root>/<task>_<n>cam_<time>/`` holds ``config.json``,
``train.log``, ``metrics.jsonl``, ``resume.pt`` (latest optimizer state) and
``checkpoint_<step>/{training_state.pt, config.json}``, which
``KimodoPlanner.from_checkpoint`` and ``scripts/eval_bigym2.py`` read;
``model.cameras`` in ``config.json`` tells the planner which cameras to feed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from kimodo_vla.data.episodes import KimodoWindowDataset
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.model.planner import load_text_embeddings

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "bigym2.yaml"
# Fields an --init-checkpoint must share with this run for its weights to mean the same thing.
ARCHITECTURE_FIELDS = (
    "fps", "motion_mask_mode", "dinov3_model_name", "controlnet_num_layers", "controlnet_scale",
    "detach_root_control_for_body", "control_fusion_mode",
    "hand_hidden_dim", "hand_num_layers", "hand_num_heads", "hand_ffn_dim",
)

log = logging.getLogger("train")


def load_config(args) -> tuple:
    """``(config, run_dir)``; a resumed run reads its own ``config.json`` and takes no overrides."""
    if args.resume:
        if args.task or args.cameras or args.init_checkpoint or args.overrides:
            raise SystemExit("--resume continues a run as configured; drop the other options")
        run_dir = Path(args.resume).expanduser().resolve()
        return OmegaConf.create(json.loads((run_dir / "config.json").read_text())), run_dir

    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if args.task:
        config.main.task = args.task
    if args.cameras:
        config.model.cameras = list(args.cameras)
    if not config.main.task:
        raise SystemExit("no task: pass --task")
    if config.main.batch_size % config.main.micro_batch_size:
        raise SystemExit("main.batch_size must be a multiple of main.micro_batch_size")
    for key in ("data_root", "text_cache", "save_root"):
        config.main[key] = str((REPO / os.path.expanduser(config.main[key])).resolve())
    if args.init_checkpoint:
        config.initialization = {"checkpoint": str(Path(args.init_checkpoint).expanduser().resolve())}
        check_initialization(config)

    name = args.name or f"{config.main.task}_{len(config.model.cameras)}cam{'_init' if args.init_checkpoint else ''}"
    run_dir = Path(config.main.save_root) / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    return config, run_dir


def build_model(config) -> KimodoPolicy:
    m, mse = config.model, config.training.loss.mse_weights
    return KimodoPolicy(KimodoPolicyConfig(
        fps=m.fps,
        motion_mask_mode=m.motion_mask_mode,
        dinov3_model_name=m.dinov3_model_name,
        dinov3_checkpoint=m.dinov3_checkpoint,
        action_chunk=config.main.action_chunk,
        action_history=config.main.action_history,
        load_text_encoder=False,
        controlnet_num_layers=m.controlnet_num_layers,
        controlnet_scale=m.controlnet_scale,
        detach_root_control_for_body=m.detach_root_control_for_body,
        control_fusion_mode=m.control_fusion_mode,
        motion_loss_type=config.training.loss.motion_loss_type,
        root_loss_weight=mse.root,
        body_loss_weight=mse.body,
        enable_hand_head=m.enable_hand_head,
        hand_hidden_dim=m.hand_hidden_dim,
        hand_num_layers=m.hand_num_layers,
        hand_num_heads=m.hand_num_heads,
        hand_ffn_dim=m.hand_ffn_dim,
        hand_loss_weight=mse.hand,
        hand_transition_loss_weight=mse.hand_transition,
        hand_init_seed=m.hand_init_seed,
        hand_control_mode=m.hand_control_mode,
        num_cameras=len(m.cameras),
    ))


def check_initialization(config) -> None:
    """Refuse an --init-checkpoint whose weights would mean something else in this run."""
    source = json.loads((Path(config.initialization.checkpoint) / "config.json").read_text())["model"]
    defaults = KimodoPolicyConfig()
    for field in ARCHITECTURE_FIELDS:
        theirs, ours = source.get(field, getattr(defaults, field)), config.model[field]
        if theirs != ours:
            raise SystemExit(f"--init-checkpoint has model.{field}={theirs!r}, this run {ours!r}")
    if len(source.get("cameras", ["head"])) != len(config.model.cameras):
        raise SystemExit(f"--init-checkpoint takes {len(source.get('cameras', ['head']))} camera(s), this run {len(config.model.cameras)}")


def initialize_from(model: KimodoPolicy, config) -> None:
    """Load ControlNet and hand-head weights of another run (e.g. pt_419h); nothing else carries over."""
    config.initialization.global_step = model.load_controlnet_checkpoint(config.initialization.checkpoint)


def param_groups(model: KimodoPolicy, config) -> list[dict]:
    """The original repository's groups: Kimodo-copied ControlNet layers, new adapters, hand head; no decay on biases."""
    opt = config.training.optimizer
    lr = {"backbone": opt.control_backbone_lr, "adapter": opt.control_adapter_lr, "hand": opt.hand_lr}
    groups = {
        (family, decay): {"params": [], "lr": lr[family], "weight_decay": opt.weight_decay if decay else 0.0, "name": f"{family}{'' if decay else '_no_decay'}"}
        for family in lr for decay in (True, False)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("hand_head."):
            family = "hand"
        elif name.startswith(("controlnet.root_layers.", "controlnet.body_layers.")):
            family = "backbone"
        else:
            family = "adapter"
        groups[family, parameter.ndim >= 2 and not name.endswith("bias")]["params"].append(parameter)
    return [g for g in groups.values() if g["params"]]


def lr_schedule(config):
    warmup, total, floor = config.training.scheduler.warmup_steps, config.main.max_steps, config.training.scheduler.min_lr_ratio

    def factor(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def checkpoint_steps(config) -> set[int]:
    main = config.main
    steps = set(range(main.checkpoint_every, main.max_steps + 1, main.checkpoint_every))
    steps.update(main.max_steps - i * main.final_checkpoint_spacing for i in range(main.final_checkpoints))
    return {s for s in steps if s > 0}


def batches(dataset, config, start_step: int):
    """Endless shuffled micro-batches; each pass over the windows gets its own seed."""
    for epoch in itertools.count():
        generator = torch.Generator().manual_seed(config.main.seed * 1_000_003 + start_step * 1_009 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.main.micro_batch_size,
            shuffle=True,
            drop_last=True,
            generator=generator,
            num_workers=config.main.num_workers,
            pin_memory=True,
        )
        yield from loader


def save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task")
    parser.add_argument("--cameras", nargs="+", help="default: model.cameras of the config (all three)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--init-checkpoint", help="start ControlNet and hand head from this checkpoint directory")
    parser.add_argument("--resume", help="run directory to continue")
    parser.add_argument("--name", help="run directory prefix; default <task>_<n>cam")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("overrides", nargs="*", help="config fields, e.g. main.max_steps=200")
    args = parser.parse_args()

    config, run_dir = load_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "train.log")],
    )
    main_cfg, device = config.main, torch.device(args.device)
    cameras = list(config.model.cameras)
    torch.manual_seed(main_cfg.seed)

    text = load_text_embeddings(main_cfg.text_cache)
    dataset = KimodoWindowDataset(
        [Path(main_cfg.data_root) / main_cfg.task],
        action_history=main_cfg.action_history,
        action_chunk=main_cfg.action_chunk,
        text_embeddings=text,
        cameras=cameras,
    )
    instruction = dataset.episodes[0]["instruction"]
    if instruction not in text:
        raise SystemExit(f"no cached text embedding for {instruction!r} in {main_cfg.text_cache}")
    log.info("%s: %d demos, %d windows, cameras %s, run %s", main_cfg.task, len(dataset.episodes), len(dataset), cameras, run_dir)

    model = build_model(config)
    if "initialization" in config and not args.resume:
        initialize_from(model, config)
        (run_dir / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
        log.info("initialized ControlNet and hand head from %s", config.initialization.checkpoint)
    model.to(device).train()
    groups = param_groups(model, config)
    opt = config.training.optimizer
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt.betas), eps=opt.eps, fused=device.type == "cuda")
    scheduler = LambdaLR(optimizer, lr_schedule(config))
    step = 0
    if args.resume:
        state = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["global_step"])
        log.info("resumed at step %d", step)

    hand_params = [p for g in groups if g["name"].startswith("hand") for p in g["params"]]
    control_params = [p for g in groups if not g["name"].startswith("hand") for p in g["params"]]
    log.info("trainable parameters: ControlNet %.1fM, hand head %.1fM",
             sum(p.numel() for p in control_params) / 1e6, sum(p.numel() for p in hand_params) / 1e6)
    accumulation = main_cfg.batch_size // main_cfg.micro_batch_size
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=main_cfg.dtype == "bf16")
    saves = checkpoint_steps(config)
    data = batches(dataset, config, step)
    metrics_file = open(run_dir / "metrics.jsonl", "a")
    window_start, window_data = time.time(), 0.0

    while step < main_cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, float] = {}
        for _ in range(accumulation):
            data_start = time.time()
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in next(data).items()}
            window_data += time.time() - data_start
            with autocast:
                output = model(
                    batch["instruction"],
                    batch["egoview"],
                    batch["gt_motion"],
                    batch["gt_mask"],
                    condition_motion=batch["condition_motion"],
                    condition_motion_mask=batch["condition_motion_mask"],
                    gt_hand=batch["gt_hand"],
                    gt_hand_mask=batch["gt_hand_mask"],
                    text_feat=batch["text_embedding"],
                    text_length=batch["text_length"],
                )
            (output["loss"] / accumulation).backward()
            for key, value in output.items():
                if key.endswith("loss") and torch.is_tensor(value) and value.ndim == 0:
                    losses[key] = losses.get(key, 0.0) + float(value.detach()) / accumulation
        if not math.isfinite(losses["loss"]):
            raise SystemExit(f"step {step + 1}: loss is {losses['loss']}")
        control_norm = float(torch.nn.utils.clip_grad_norm_(control_params, main_cfg.gradient.grad_clip_norm))
        hand_norm = float(torch.nn.utils.clip_grad_norm_(hand_params, main_cfg.gradient.hand_grad_clip_norm)) if hand_params else 0.0
        optimizer.step()
        scheduler.step()
        step += 1

        if step % main_cfg.log_every == 0 or step == main_cfg.max_steps:
            seconds = (time.time() - window_start) / main_cfg.log_every
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            record = {"step": step, **losses, "grad_control": control_norm, "grad_hand": hand_norm,
                      "lr_backbone": lrs["backbone"], "lr_adapter": lrs["adapter"], "s_per_step": seconds,
                      "data_s_per_step": window_data / main_cfg.log_every}
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            eta = seconds * (main_cfg.max_steps - step) / 3600
            log.info("step %d/%d | %s | grad %.2f / %.2f | lr %.1e | %.2fs/step (data %.2fs) | eta %.1fh",
                     step, main_cfg.max_steps, " ".join(f"{k.removesuffix('_loss') or k} {v:.4f}" for k, v in losses.items()),
                     control_norm, hand_norm, lrs["adapter"], seconds, record["data_s_per_step"], eta)
            window_start, window_data = time.time(), 0.0

        if step in saves:
            trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            checkpoint = run_dir / f"checkpoint_{step}"
            checkpoint.mkdir(exist_ok=True)
            save(checkpoint / "training_state.pt", {"global_step": step, "model": trainable})
            (checkpoint / "config.json").write_text((run_dir / "config.json").read_text())
            save(run_dir / "resume.pt", {"global_step": step, "model": trainable,
                                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()})
            log.info("saved %s", checkpoint)
    metrics_file.close()


if __name__ == "__main__":
    main()
