"""Train Kimodo-Policy's ControlNet and hand head on one BiGym 2.0 task (one or several GPUs).

    python scripts/train.py --task move_plate                       # three cameras
    python scripts/train.py --task move_plate --cameras head        # head only
    python scripts/train.py --task move_plate --cameras head --init-checkpoint <pt_419h checkpoint dir>
    python scripts/train.py --task move_plate main.max_steps=200    # any config field, OmegaConf dotlist
    python scripts/train.py --resume log/bigym2/<run>
    torchrun --standalone --nproc_per_node=8 scripts/train.py --task move_plate --config configs/bigym2_recipe.yaml

Settings come from ``configs/bigym2.yaml`` (a config may name another as ``_base_``
and list only what differs, like ``configs/bigym2_recipe.yaml``). Under ``torchrun``
every GPU takes ``main.micro_batch_size`` samples of each ``main.batch_size`` batch. Kimodo-G1, DINOv3 and LLM2Vec stay
frozen (LLM2Vec only through the cached embeddings of
``scripts/cache_text_embeddings.py``); the ControlNet and hand head start from
scratch unless ``--init-checkpoint`` is given. Samples are the task's
``KimodoWindowDataset`` windows, shuffled uniformly; the target is the operator's
command (``demos.commanded_motion``, written by ``prepare_bigym2.py``).

A run directory (``--run-dir``, or ``<save_root>/<task>_<n>cam_<time>/``) holds ``config.json``,
``train.log``, ``metrics.jsonl``, ``resume.pt`` (latest optimizer state) and
``checkpoint_<step>/{training_state.pt, config.json}``, which
``KimodoPlanner.from_checkpoint`` and ``scripts/eval_bigym2.py`` read;
``model.cameras`` in ``config.json`` tells the planner which cameras to feed.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

from kimodo_vla.data.episodes import KimodoWindowDataset
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.model.planner import load_text_embeddings
from run_config import DEFAULT_CONFIG, REPO, checkpoint_steps, load_yaml  # scripts/ is on sys.path when run as a script

# Fields an --init-checkpoint must share with this run for its weights to mean the same thing.
ARCHITECTURE_FIELDS = (
    "visual_conditioning",
    "groot_injection_mode", "adaln_condition_dim", "adaln_pool", "adaln_max_diffusion_steps",
    "fps", "motion_mask_mode", "dinov3_model_name", "controlnet_num_layers", "controlnet_scale",
    "detach_root_control_for_body", "control_fusion_mode",
    "hand_hidden_dim", "hand_num_layers", "hand_num_heads", "hand_ffn_dim",
)

log = logging.getLogger("train")


def load_config(args, rank: int = 0) -> tuple:
    """``(config, run_dir)``; a resumed run reads its own ``config.json`` and takes no overrides.

    Only rank 0 creates the run directory; with several processes, call ``share_run_dir`` next.
    """
    if args.resume:
        if args.task or args.cameras or args.init_checkpoint or args.overrides:
            raise SystemExit("--resume continues a run as configured; drop the other options")
        run_dir = Path(args.resume).expanduser().resolve()
        return OmegaConf.create(json.loads((run_dir / "config.json").read_text())), run_dir

    config = OmegaConf.merge(load_yaml(args.config), OmegaConf.from_dotlist(args.overrides))
    if args.task:
        config.main.task = args.task
    if args.cameras:
        config.model.cameras = list(args.cameras)
    if not config.main.task:
        raise SystemExit("no task: pass --task")
    for key in ("data_root", "text_cache", "save_root"):
        config.main[key] = str((REPO / os.path.expanduser(config.main[key])).resolve())
    if args.init_checkpoint:
        if config.model.get("visual_conditioning") == "groot_n1d7":
            raise SystemExit("This N1.7 scratch experiment must not load a policy/419h initialization")
        config.initialization = {"checkpoint": str(Path(args.init_checkpoint).expanduser().resolve())}
        check_initialization(config)

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    else:
        name = args.name or f"{config.main.task}_{len(config.model.cameras)}cam{'_init' if args.init_checkpoint else ''}"
        run_dir = Path(config.main.save_root) / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if rank == 0:
        run_dir.mkdir(parents=True)  # never mixes two runs; continue one with --resume
        (run_dir / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
    return config, run_dir


def share_run_dir(run_dir: Path) -> Path:
    """Rank 0's run directory (its name may carry rank 0's clock), once it exists."""
    shared = [str(run_dir)]
    dist.broadcast_object_list(shared, src=0)
    dist.barrier()
    return Path(shared[0])


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
        visual_conditioning=m.get("visual_conditioning", "dinov3_controlnet"),
        groot_checkpoint=m.get("groot_checkpoint"),
        groot_repo=m.get("groot_repo"),
        groot_vlm_model=m.get("groot_vlm_model"),
        groot_injection_mode=m.get("groot_injection_mode", "cross_attention"),
        adaln_condition_dim=m.get("adaln_condition_dim", 512),
        adaln_pool=m.get("adaln_pool", "masked_mean"),
        adaln_max_diffusion_steps=m.get("adaln_max_diffusion_steps", 1000),
    ))


def check_initialization(config) -> None:
    """Refuse an --init-checkpoint whose weights would mean something else in this run."""
    source = json.loads((Path(config.initialization.checkpoint) / "config.json").read_text())["model"]
    defaults = KimodoPolicyConfig()
    for field in ARCHITECTURE_FIELDS:
        theirs, ours = source.get(field, getattr(defaults, field)), config.model.get(field, getattr(defaults, field))
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
    lr = {"backbone": opt.control_backbone_lr, "adapter": opt.control_adapter_lr, "hand": opt.hand_lr,
          "vl_adapter": opt.get("vl_adapter_lr", opt.control_backbone_lr),
          "vision": opt.get("vision_lr", 1e-6),
          # AdaLN starts with zero modulation heads.  A mildly larger
          # dedicated rate lets the gates open in a few updates while the
          # frozen Kimodo backbone remains untouched.
          "adaln": opt.get("adaln_lr", opt.control_adapter_lr)}
    groups = {
        (family, decay): {"params": [], "lr": lr[family], "weight_decay": opt.weight_decay if decay else 0.0, "name": f"{family}{'' if decay else '_no_decay'}"}
        for family in lr for decay in (True, False)
    }
    model_config = getattr(model, "config", None)
    injection_mode = getattr(model_config, "groot_injection_mode", "cross_attention")
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("hand_head."):
            family = "hand"
        elif name.startswith("image_encoder.adapter."):
            family = "vl_adapter"
        elif name.startswith("image_encoder.backbone."):
            family = "vision"
        elif name.startswith(("controlnet.root_layers.", "controlnet.body_layers.")):
            family = "backbone"
        elif (injection_mode == "adaln_zero"
              and name.startswith(("controlnet.root_hint_fusion.", "controlnet.body_hint_fusion."))):
            family = "adaln"
        else:
            family = "adapter"
        groups[family, parameter.ndim >= 2 and not name.endswith("bias")]["params"].append(parameter)
    return [g for g in groups.values() if g["params"]]


def training_stage(config, step):
    stages = config.training.get("stages")
    if not stages:
        return None
    if config.model.get("visual_conditioning") != "groot_n1d7":
        raise ValueError("Staged training requires groot_n1d7")
    if not 0 < int(stages.adapter_start) < config.main.max_steps:
        raise ValueError("adapter_start must lie inside the training run")
    vision_start = stages.get("vision_start")
    if vision_start is not None and not int(stages.adapter_start) < int(vision_start) < config.main.max_steps:
        raise ValueError("vision_start must follow adapter_start and precede max_steps")
    if vision_start is not None and step >= int(vision_start):
        return "vision"
    return "adapter" if step >= int(stages.adapter_start) else "injection"


def make_optimizer(model, config, device, step=0, previous=None):
    groups = param_groups(model, config)
    for group in groups:
        group["initial_lr"] = group["lr"]
    opt = config.training.optimizer
    optimizer = torch.optim.AdamW(groups, betas=tuple(opt.betas), eps=opt.eps, fused=device.type == "cuda")
    if previous is not None:
        for parameter, state in previous.state.items():
            optimizer.state[parameter] = state
    base_schedule = lr_schedule(config)
    adapter_warmup = int(config.training.scheduler.get("adapter_warmup_steps", 0))
    adapter_start = int((config.training.get("stages") or {}).get("adapter_start", 0))
    def schedule_for(group):
        if adapter_warmup and group["name"].startswith("vl_adapter"):
            return lambda update: base_schedule(update) * min(1.0, max(0, update - adapter_start) / adapter_warmup)
        return base_schedule
    scheduler = LambdaLR(optimizer, [schedule_for(group) for group in groups], last_epoch=step - 1)
    return groups, optimizer, scheduler


def guarded_optimizer_step(optimizer, scheduler, loss, control_norm, hand_norm, main_cfg):
    max_loss = main_cfg.get("max_update_loss")
    max_grad = main_cfg.get("max_update_grad_norm")
    skip = ((max_loss is not None and loss > max_loss) or
            (max_grad is not None and max(control_norm, hand_norm) > max_grad))
    if skip:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
    scheduler.step()
    return skip


def build_dataset(config, text):
    if config.main.get("dataset_kind") == "original_arena":
        from kimodo_vla.benchmarks.arena.training import build_arena_dataset
        return build_arena_dataset(config, text)
    augmentation = config.training.get("augmentation") or {}
    return KimodoWindowDataset(
        [Path(config.main.data_root) / config.main.task],
        action_history=config.main.action_history, action_chunk=config.main.action_chunk,
        text_embeddings=text, cameras=list(config.model.cameras),
        image_shift=augmentation.get("image_shift", 0),
        proprio_mask_prob=augmentation.get("proprio_mask_prob", 0.0),
    )


def lr_schedule(config):
    warmup, total, floor = config.training.scheduler.warmup_steps, config.main.max_steps, config.training.scheduler.min_lr_ratio

    def factor(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor


def batches(dataset, config, start_step: int, rank: int = 0, world: int = 1):
    """Endless shuffled micro-batches of this rank; each pass over the windows gets its own seed."""
    if config.main.get("original_ordinal_stream", False):
        from kimodo_vla.benchmarks.arena.training import original_arena_batches
        yield from original_arena_batches(dataset, config, start_step, rank, world)
        return
    for epoch in itertools.count():
        seed = config.main.seed * 1_000_003 + start_step * 1_009 + epoch
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
        loader = DataLoader(
            dataset,
            batch_size=config.main.micro_batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=config.main.num_workers,
            pin_memory=True,
        )
        yield from loader


class WeightAverages:
    """Exponential moving averages of the trainable weights, one per decay in ``training.ema_decays``.

    Kept on rank 0 in fp32 and saved next to the raw weights as ``ema[<decay>]``; evaluate one
    with ``--weights ema<decay>``. Early on the rate is ``(1 + step) / (10 + step)``, so the first
    steps' weights do not linger in the average.
    """

    def __init__(self, model: torch.nn.Module, decays, state: dict | None = None):
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.decays = [float(d) for d in decays]
        self.averages = {}
        for decay in self.decays:
            saved = (state or {}).get(str(decay))
            self.averages[str(decay)] = [
                (saved[n] if saved else p.detach()).to(p.device, torch.float32).clone() for n, p in zip(self.names, self.params)
            ]

    @torch.no_grad()
    def update(self, step: int) -> None:
        current = [p.detach().float() for p in self.params]
        for decay in self.decays:
            rate = min(decay, (1 + step) / (10 + step))
            torch._foreach_mul_(self.averages[str(decay)], rate)
            torch._foreach_add_(self.averages[str(decay)], current, alpha=1 - rate)

    def state(self) -> dict:
        return {d: {n: t.detach().cpu() for n, t in zip(self.names, avg)} for d, avg in self.averages.items()}


def save(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def training_forward(model, config, device, world):
    forward = model
    if config.model.get("compile", False):
        if config.model.get("compile_scope") == "denoiser":
            # Compile tensor-heavy motion prediction, not tokenization/VL grouping/RNG.
            if not hasattr(model.denoiser, "_orig_mod"):
                model.denoiser = torch.compile(model.denoiser,
                    mode=config.model.get("compile_mode", "default"),
                    dynamic=config.model.get("compile_dynamic"))
        else:
            forward = torch.compile(model, mode=config.model.get("compile_mode", "reduce-overhead"), dynamic=False)
    return DistributedDataParallel(forward, device_ids=[device.index], gradient_as_bucket_view=True) if world > 1 else forward


def restore_training_rng(state, config, device, world, rank, step):
    seed = int(config.main.seed)
    torch.manual_seed(seed + (rank if config.main.get("original_ordinal_stream") else 7919 * rank))
    saved = (state or {}).get("rng_states", [])
    if state and saved and len(saved) != world:
        if not config.main.get("allow_world_size_change", False):
            raise RuntimeError(f"Checkpoint has {len(saved)} RNG ranks, requested {world}; explicit migration required")
        migration_seed = seed + 1_000_003 * step + rank
        torch.manual_seed(migration_seed)
        log.warning("RNG topology migration %d -> %d ranks at update %d; rank %d seed %d",
                    len(saved), world, step, rank, migration_seed)
    elif state and len(saved) == world:
        rng = saved[rank]
        torch.set_rng_state(rng["cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)
    elif state and world == 1:
        if "rng_cpu" in state:
            torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])


def main() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task")
    parser.add_argument("--cameras", nargs="+", help="default: model.cameras of the config (all three)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--init-checkpoint", help="start ControlNet and hand head from this checkpoint directory")
    parser.add_argument("--resume", help="run directory to continue")
    parser.add_argument("--name", help="run directory prefix; default <task>_<n>cam")
    parser.add_argument("--run-dir", help="exact run directory instead of <save_root>/<name>_<time>; must not exist")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int, help="save and stop at this update without changing the LR schedule")
    parser.add_argument("--audit-distributed", action="store_true", help="verify all saved model tensors are identical across ranks on exit")
    parser.add_argument("overrides", nargs="*", help="config fields, e.g. main.max_steps=200")
    args = parser.parse_args()

    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    device = torch.device(args.device)
    if world > 1:  # under torchrun
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", timeout=timedelta(seconds=3600))
    config, run_dir = load_config(args, rank)
    if world > 1:
        run_dir = share_run_dir(run_dir)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(run_dir / "train.log")] if rank == 0 else []),
    )
    main_cfg = config.main
    if main_cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    if config.model.get("visual_conditioning") == "groot_n1d7":
        from groot_experiment import check_preflight
        check_preflight(config)
    training_stage(config, 0)  # validate stage boundaries before loading weights
    if config.training.get("stages") and config.training.get("ema_decays"):
        raise SystemExit("Staged runs currently require ema_decays=[]")
    if config.model.get("visual_conditioning") == "groot_n1d7" and "initialization" in config:
        raise SystemExit("N1.7 scratch runs cannot initialize from policy/419h checkpoints")
    if main_cfg.batch_size % (main_cfg.micro_batch_size * world):
        raise SystemExit(f"main.batch_size {main_cfg.batch_size} is not a multiple of micro_batch_size x {world} GPUs")
    cameras = list(config.model.cameras)
    augmentation = config.training.get("augmentation") or {}
    torch.manual_seed(main_cfg.seed)  # the same initial weights on every rank

    text = load_text_embeddings(main_cfg.text_cache)
    dataset = build_dataset(config, text)
    is_arena = main_cfg.get("dataset_kind") == "original_arena"
    instruction = next(iter(dataset.task_instructions.values())) if is_arena else dataset.episodes[0]["instruction"]
    if instruction not in text:
        raise SystemExit(f"no cached text embedding for {instruction!r} in {main_cfg.text_cache}")
    log.info("%s: %d demos, %d windows, cameras %s, %d GPU(s), augmentation %s, run %s", main_cfg.task,
             len(dataset._all_episode_records) if is_arena else len(dataset.episodes), len(dataset), cameras, world, dict(augmentation), run_dir)

    model = build_model(config)
    if "initialization" in config and not args.resume:
        initialize_from(model, config)
        if rank == 0:
            (run_dir / "config.json").write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2))
        log.info("initialized ControlNet and hand head from %s", config.initialization.checkpoint)
    model.to(device).train()
    step, state = 0, None
    if args.resume:
        state = torch.load(run_dir / "resume.pt", map_location="cpu", weights_only=True)
        if config.model.get("visual_conditioning") == "groot_n1d7":
            model.load_controlnet_checkpoint(str(run_dir / "resume.pt"))
        else:
            model.load_state_dict(state["model"], strict=False)
        step = int(state["global_step"])
        log.info("resumed at step %d", step)
    stage = (state or {}).get("training_stage") or training_stage(config, step)
    if stage:
        model.set_training_stage(stage)
    groups, optimizer, scheduler = make_optimizer(model, config, device, step)
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
    decays = list(config.training.get("ema_decays") or [])
    averages = WeightAverages(model, decays, (state or {}).get("ema")) if decays and rank == 0 else None
    forward = training_forward(model, config, device, world)
    restore_training_rng(state, config, device, world, rank, step)

    hand_params = [p for g in groups if g["name"].startswith("hand") for p in g["params"]]
    control_params = [p for g in groups if not g["name"].startswith("hand") for p in g["params"]]
    log.info("trainable parameters: ControlNet %.1fM, hand head %.1fM",
             sum(p.numel() for p in control_params) / 1e6, sum(p.numel() for p in hand_params) / 1e6)
    accumulation = main_cfg.batch_size // (main_cfg.micro_batch_size * world)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=main_cfg.dtype == "bf16")
    saves = checkpoint_steps(config.main)
    saves.update(main_cfg.get("extra_checkpoints", []))
    stop_step = min(main_cfg.max_steps, args.stop_after or main_cfg.max_steps)
    if stop_step <= step:
        raise SystemExit(f"stop-after {stop_step} must exceed resumed step {step}")
    saves.add(stop_step)
    data = batches(dataset, config, step, rank, world)
    metrics_file = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None
    window_start, window_data = time.time(), 0.0
    consecutive_skips = 0
    skipped_updates = int((state or {}).get("skipped_updates", 0))

    while step < stop_step:
        # A file request is acted on at a full optimizer-update boundary on all
        # ranks. No asynchronous serialization of partially updated weights.
        requested = torch.tensor(int(rank == 0 and (run_dir / "CHECKPOINT_AND_STOP").exists()), device=device)
        if world > 1:
            dist.broadcast(requested, src=0)
        stop_requested = bool(requested.item())
        next_stage = training_stage(config, step)
        if next_stage != stage:
            # DDP only installs hooks for parameters trainable at construction.
            # Rebuild it after unfreezing; retain Adam moments of existing weights.
            if world > 1:
                dist.barrier()
            del forward
            model.set_training_stage(next_stage)
            groups, optimizer, scheduler = make_optimizer(model, config, device, step, optimizer)
            hand_params = [p for g in groups if g["name"].startswith("hand") for p in g["params"]]
            control_params = [p for g in groups if not g["name"].startswith("hand") for p in g["params"]]
            forward = training_forward(model, config, device, world)
            stage = next_stage
            log.info("step %d: training stage -> %s", step, stage)
        optimizer.zero_grad(set_to_none=True)
        losses: dict[str, float] = {}
        for micro in range(accumulation):
            data_start = time.time()
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in next(data).items()}
            window_data += time.time() - data_start
            sync = forward.no_sync() if world > 1 and micro < accumulation - 1 else contextlib.nullcontext()
            with sync:
                with autocast:
                    output = forward(
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
        if world > 1:  # log the mean over ranks; every rank then agrees on the NaN check
            values = torch.tensor(list(losses.values()), device=device)
            dist.all_reduce(values)
            losses = dict(zip(losses, (values / world).tolist()))
        if not math.isfinite(losses["loss"]):
            raise SystemExit(f"step {step + 1}: loss is {losses['loss']}")
        control_norm = float(torch.nn.utils.clip_grad_norm_(control_params, main_cfg.gradient.grad_clip_norm))
        hand_norm = float(torch.nn.utils.clip_grad_norm_(hand_params, main_cfg.gradient.hand_grad_clip_norm)) if hand_params else 0.0
        if not math.isfinite(control_norm) or not math.isfinite(hand_norm):
            raise SystemExit(f"step {step + 1}: nonfinite gradients {control_norm}, {hand_norm}")
        guard_cfg = main_cfg if step >= main_cfg.get("guard_start_step", 0) else {}
        skipped = guarded_optimizer_step(optimizer, scheduler, losses["loss"], control_norm,
                                         hand_norm, guard_cfg)
        consecutive_skips = consecutive_skips + 1 if skipped else 0
        skipped_updates += int(skipped)
        if skipped and rank == 0:
            log.warning("step %d: skipped abnormal update (loss %.4g, gradients %.4g / %.4g)",
                        step + 1, losses["loss"], control_norm, hand_norm)
        if consecutive_skips >= main_cfg.get("max_consecutive_skips", 10):
            raise SystemExit(f"step {step + 1}: {consecutive_skips} consecutive abnormal updates")
        step += 1
        if averages:
            averages.update(step)

        if rank == 0 and (step % main_cfg.log_every == 0 or step == main_cfg.max_steps):
            seconds = (time.time() - window_start) / main_cfg.log_every
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            record = {"step": step, **losses, "grad_control": control_norm, "grad_hand": hand_norm,
                      "lr_backbone": lrs.get("backbone", 0.0), "lr_adapter": lrs["adapter"], "stage": stage, "s_per_step": seconds,
                      "data_s_per_step": window_data / main_cfg.log_every,
                      "gpu_peak_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0}
            if main_cfg.get("max_update_loss") is not None or main_cfg.get("max_update_grad_norm") is not None:
                record.update(skipped_update=skipped, skipped_updates=skipped_updates)
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            eta = seconds * (main_cfg.max_steps - step) / 3600
            log.info("step %d/%d | %s | grad %.2f / %.2f | lr %.1e | %.2fs/step (data %.2fs) | eta %.1fh",
                     step, main_cfg.max_steps, " ".join(f"{k.removesuffix('_loss') or k} {v:.4f}" for k, v in losses.items()),
                     control_norm, hand_norm, lrs["adapter"], seconds, record["data_s_per_step"], eta)
            window_start, window_data = time.time(), 0.0

        if step in saves or stop_requested:
            local_rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            rng_states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(local_rng, rng_states, dst=0)
            else:
                rng_states = [local_rng]
        if rank == 0 and (step in saves or stop_requested):
            trainable = {n: p.detach().cpu() for n, p in model.checkpoint_parameters().items()}
            checkpoint = run_dir / f"checkpoint_{step}"
            checkpoint.mkdir(exist_ok=True)
            ema = {"ema": averages.state()} if averages else {}
            save(checkpoint / "training_state.pt", {"global_step": step, "model": trainable, **ema})
            (checkpoint / "config.json").write_text((run_dir / "config.json").read_text())
            if main_cfg.get("keep_previous_resume", False) and (run_dir / "resume.pt").exists():
                previous = run_dir / "resume_previous.pt"
                temporary = run_dir / "resume_previous.pt.tmp"
                temporary.unlink(missing_ok=True)
                os.link(run_dir / "resume.pt", temporary)
                os.replace(temporary, previous)
            save(run_dir / "resume.pt", {"global_step": step, "model": trainable, **ema,
                                         "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                         "training_stage": stage, "skipped_updates": skipped_updates,
                                         "rng_cpu": torch.get_rng_state(),
                                         "rng_states": rng_states,
                                         "rng_cuda": [torch.cuda.get_rng_state(device)] if device.type == "cuda" and world == 1 else []})
            log.info("saved %s", checkpoint)
        if stop_requested:
            log.info("checkpoint-and-stop request completed at step %d", step)
            break
    if metrics_file:
        metrics_file.close()
    if args.audit_distributed:
        import hashlib
        digest = hashlib.sha256()
        for name, parameter in model.checkpoint_parameters().items():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        hashes = [None] * world
        if world > 1:
            dist.all_gather_object(hashes, digest.hexdigest())
        else:
            hashes = [digest.hexdigest()]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"Model weights diverged across ranks: {hashes}")
        if rank == 0:
            (run_dir / f"distributed_audit_{step}.json").write_text(json.dumps({
                "passed": True, "world_size": world, "step": step, "stage": stage, "sha256_by_rank": hashes}, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
