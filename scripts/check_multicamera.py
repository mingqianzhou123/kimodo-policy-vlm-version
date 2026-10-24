"""Smoke check of the three-camera Kimodo-Policy on real data (run on a GPU server).

    MUJOCO_GL=egl python scripts/check_multicamera.py --prepared <prepared dir> --task move_plate

Builds a fresh KimodoPolicy with ``num_cameras=3`` (Kimodo-G1 and DINOv3 from
``checkpoints/``; ControlNet and hand head untrained), then:

1. a few training steps on prepared episodes (``scripts/prepare_bigym2.py``) with
   all three cameras;
2. a check that every camera reaches the output: blanking one camera's image
   changes the prediction (the ControlNet injection starts at zero, so this needs
   the training steps before it);
3. closed-loop control in the official BiGym env through the evaluation path
   (adapter -> Agent -> KimodoPlanner), long enough to replan.

The text feature is random: text embeddings are not cached yet, and this only
checks that three cameras flow through training and inference. Exits non-zero on
the first failed check.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from kimodo_vla.benchmarks.base import AgentPolicy
from kimodo_vla.benchmarks.bigym2.adapter import CAMERA_NAMES, BiGym2Adapter
from kimodo_vla.core.agent import Agent
from kimodo_vla.data.episodes import KimodoWindowDataset
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.model.planner import KimodoPlanner

SEED = 620000


def check(condition: bool, message: str) -> None:
    print(f"{'ok  ' if condition else 'FAIL'} {message}", flush=True)
    if not condition:
        raise SystemExit(1)


def train_steps(model, batch, text, steps: int) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        output = model.training_kimodo_policy_controlnet(
            instruction=batch["instruction"],
            egoview=batch["egoview"],
            gt_motion=batch["gt_motion"],
            gt_mask=batch["gt_mask"],
            condition_motion=batch["condition_motion"],
            condition_motion_mask=batch["condition_motion_mask"],
            gt_hand=batch["gt_hand"],
            gt_hand_mask=batch["gt_hand_mask"],
            text_feat=text,
        )
        loss = output["loss"]
        loss.backward()
        grad = model.controlnet.image_projection.weight.grad
        grad_norm = 0.0 if grad is None else float(grad.norm())
        check(bool(torch.isfinite(loss)), f"train step {step}: loss {float(loss):.4f}, image projection grad norm {grad_norm:.2e}")
        optimizer.step()
    check(grad_norm > 0, "gradient reaches the image tokens")


@torch.inference_mode()
def camera_influence(model, batch, text) -> None:
    model.eval()
    history = model.config.action_history
    egoview = batch["egoview"][:1].to(model.device)

    def predict(images):
        generator = torch.Generator(device=model.device).manual_seed(0)
        output = model.predict_future(
            instruction=batch["instruction"][0],
            egoview=images,
            history_motion=batch["condition_motion"][:1, :history].to(model.device),
            history_feature_mask=batch["condition_motion_mask"][:1, :history].to(model.device),
            diffusion_steps=10,
            squeeze_batch=True,
            text_feat=text[:1],
            hand_history=batch["gt_hand"][:1, :history].to(model.device),
            generator=generator,
        )
        prediction = torch.cat((output["motion_features"].flatten(), output["hand_probability"].flatten())).float()
        check(bool(torch.isfinite(prediction).all()), "prediction is finite")
        return prediction

    reference = predict(egoview)
    noise = float((predict(egoview) - reference).abs().max())  # GPU kernels need not be bit-exact
    print(f"     same input twice: max difference {noise:.2e}")
    for view, name in enumerate(CAMERA_NAMES):
        blanked = egoview.clone()
        blanked[:, view] = 0
        changed = float((predict(blanked) - reference).abs().max())
        check(changed > max(10 * noise, 1e-6), f"blanking {name} changes the prediction (max change {changed:.2e})")


def closed_loop(model, task: str, text_dim: int, steps: int) -> None:
    from bigym.loco import make

    env = make(task)  # official setup: all three cameras at 84x84
    try:
        adapter = BiGym2Adapter.from_env(env, task)
        check(adapter.cameras == CAMERA_NAMES, f"env provides {adapter.cameras}")
        timestep = env.reset(seed=SEED)
        instruction = adapter.to_observation(timestep).instruction
        embeddings = {instruction: torch.randn(1, text_dim, generator=torch.Generator().manual_seed(0))}
        planner = KimodoPlanner(model.eval(), embeddings, control_fps=adapter.control_fps, cameras=CAMERA_NAMES)
        agent = Agent(planner)
        policy = AgentPolicy(agent, adapter)
        policy.reset()
        start = time.time()
        for _ in range(steps):
            action = policy(timestep)
            if not np.isfinite(action).all():
                check(False, "env action is finite")
            timestep = env.step(action)
            if timestep.last():
                break
        seconds = time.time() - start
        check(agent.num_plans >= 2, f"{agent.num_plans} plans over {steps} control steps ({seconds:.1f}s)")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", required=True, help="output directory of prepare_bigym2.py")
    parser.add_argument("--task", default="move_plate")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=3)
    parser.add_argument("--control-steps", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(0)

    model = KimodoPolicy(KimodoPolicyConfig(load_text_encoder=False, num_cameras=len(CAMERA_NAMES)))
    model.to(args.device)
    tokens = model.image_encoder.token_count
    check(tokens == 196, f"DINOv3 gives {tokens} tokens per image")
    check(model.controlnet.image_token_count == 3 * tokens, f"ControlNet takes {model.controlnet.image_token_count} image tokens")
    check(model.hand_head.image_token_count == 3 * tokens, f"hand head takes {model.hand_head.image_token_count} image tokens")

    dataset = KimodoWindowDataset([args.prepared], tasks=[args.task], cameras=CAMERA_NAMES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in next(iter(loader)).items()}
    check(tuple(batch["egoview"].shape[1:3]) == (3, 3), f"egoview batch {tuple(batch['egoview'].shape)} from {len(dataset)} windows")

    text_dim = model.denoiser.root_model.embed_text.in_features
    text = torch.randn(args.batch_size, 1, text_dim, device=args.device)
    train_steps(model, batch, text, args.train_steps)
    camera_influence(model, batch, text)
    closed_loop(model, args.task, text_dim, args.control_steps)
    print("all checks passed")


if __name__ == "__main__":
    main()
