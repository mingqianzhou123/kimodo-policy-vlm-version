"""KimodoPlanner: Kimodo-Policy behind the ``Planner`` interface.

Ported from ``evaluation/humanoidarena_server.py`` of the original repository
(``KimodoHumanoidArenaRuntime.infer``), with the HTTP layer and trace flags
removed. With every joint observed it feeds the model the same motion history
as the original runtime. Two deliberate differences: predictions are sampled at
the true time of each control step (the original ran one model frame behind),
and the hand history is indexed by model frame. ``planner_original.OriginalTimingPlanner``
keeps the original runtime's decoding, for comparing with it and reproducing its
evaluations.

Per-benchmark options, all off by default (the original Arena + SONIC behavior):

* ``root_boundary``: where the first frame's root displacement starts.
  ``"history"`` is the end of the history the model generated in the same window
  (Arena with SONIC); ``"first_prediction"`` is the first predicted root (Arena
  with TWIST2, ``KIMODO_SERVER_ANCHOR_ROOT``); ``"extrapolate"`` is ``2 r0 - r1``
  from the first two predicted roots (SIMPLE, whose state has no root position).
* continuous hands (checkpoint ``hand_control_mode="continuous"``, SIMPLE and
  ToolBench): the hand condition is the measured closure in ``Observation.hand``
  and the output is a closure in [0, 1] instead of open/closed.
* ``hand_filter``: rewrites the executed binary hand prefix before it is sent and
  remembered (SIMPLE's debounce, ``benchmarks.simple.hands``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from kimodo_vla.core.types import ActionChunk, Observation, yaw_matrix, yaw_of
from kimodo_vla.model.features import KimodoFeatures, lerp_rows
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.motion.g1_reference import _interpolate_motion

ROOT_BOUNDARIES = ("history", "first_prediction", "extrapolate")
# (executed model-rate prefix [n, 2], whole prediction [T, 2]) -> the prefix to execute
HandFilter = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def load_text_embeddings(cache_dir: str | Path) -> dict[str, torch.Tensor]:
    """Read ``*.pt`` files of ``{task_id, instruction, embedding[, task_name]}``.

    The result is keyed by task id, task name and instruction text, so an
    adapter can identify the task by any of them.
    """
    files = sorted(Path(cache_dir).expanduser().glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no text embedding files in {cache_dir}")
    table: dict[str, torch.Tensor] = {}
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        embedding = torch.as_tensor(payload["embedding"]).float()
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError(f"{path}: embedding must be [1, D], got {tuple(embedding.shape)}")
        for key in (payload.get("task_id"), payload.get("task_name"), payload.get("instruction")):
            if key:
                table[str(key).strip()] = embedding
    return table


def _episode_noise_seed(episode_seed: int, plan_index: int) -> int:
    """SplitMix64 of (seed, index): the same per-plan noise as the original runtime."""
    mask64 = 0xFFFFFFFFFFFFFFFF
    value = ((episode_seed & mask64) + 0x9E3779B97F4A7C15 * (plan_index + 1)) & mask64
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & mask64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & mask64
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def _crossfade(old: ActionChunk, new: ActionChunk, steps: int) -> ActionChunk:
    """``new``, eased in from ``old`` (the same control steps) over its first ``steps`` frames."""
    k = min(steps, len(old), len(new))
    out = ActionChunk(new.root_delta_xy.copy(), new.root_height.copy(), new.root_rot.copy(),
                      new.joint_pos.copy(), new.hand.copy(), new.fps, dict(new.meta))
    if k == 0:
        return out
    w = 0.5 * (1.0 + np.cos(np.pi * np.arange(1, k + 1) / (k + 1)))  # weight of the old plan, 1 -> 0
    for name in ("root_delta_xy", "root_height", "joint_pos"):
        a, b = getattr(old, name)[:k], getattr(out, name)[:k]
        weight = w.reshape((k,) + (1,) * (a.ndim - 1))
        getattr(out, name)[:k] = weight * a + (1 - weight) * b
    mixed = w[:, None, None] * old.root_rot[:k] + (1 - w[:, None, None]) * out.root_rot[:k]
    u, _, vt = np.linalg.svd(mixed)  # back to the nearest rotation
    out.root_rot[:k] = (u @ vt).astype(np.float32)
    out.hand[:k] = np.where(w[:, None] >= 0.5, old.hand[:k], out.hand[:k])
    return out


class KimodoPlanner:
    """Kimodo-Policy as a ``Planner``.

    Options for replanning before a plan runs out (all off by default):
    ``noise="fixed"`` starts every plan of an episode from the same diffusion noise;
    ``rtc_overlap_frames`` > 0 guides each prediction towards the previous plan's
    unexecuted frames (real-time chunking, built into ``predict_future``; the original
    runtime used 12 overlap frames, 1 frozen; it starts over when the instruction
    changes); ``blend_steps`` > 0 cross-fades the first control targets of a new plan
    from the previous plan's targets for the same steps.
    """

    def __init__(
        self,
        model: KimodoPolicy,
        text_embeddings: dict[str, torch.Tensor],
        *,
        control_fps: float = 50.0,
        execution_frames: int = 0,
        diffusion_steps: int = 10,
        cameras: Sequence[str] = ("head",),
        noise: str = "per_plan",
        rtc_overlap_frames: int = 0,
        rtc_frozen_frames: int = 1,
        blend_steps: int = 0,
        root_boundary: str = "history",
        hand_filter: HandFilter | None = None,
    ):
        self.hand_mode = str(model.config.hand_control_mode).lower()
        if self.hand_mode not in ("binary", "continuous"):
            raise ValueError(f"hand_control_mode must be 'binary' or 'continuous', got {self.hand_mode!r}")
        if hand_filter is not None and self.hand_mode != "binary":
            raise ValueError("hand_filter works on binary hands only")
        if root_boundary not in ROOT_BOUNDARIES:
            raise ValueError(f"root_boundary must be one of {ROOT_BOUNDARIES}, got {root_boundary!r}")
        self.root_boundary, self.hand_filter = root_boundary, hand_filter
        num_cameras = getattr(model.config, "num_cameras", 1)
        if len(cameras) != num_cameras:
            raise ValueError(f"model takes {num_cameras} camera(s), got {tuple(cameras)}")
        self.model = model
        self.text_embeddings = text_embeddings
        self.control_fps = float(control_fps)
        self.model_fps = float(model.fps)
        self.diffusion_steps = int(diffusion_steps)
        self.cameras = tuple(cameras)
        if noise not in ("per_plan", "fixed"):
            raise ValueError(f"noise must be 'per_plan' or 'fixed', got {noise!r}")
        self.noise = noise
        self.rtc_overlap_frames = int(rtc_overlap_frames)
        self.rtc_frozen_frames = min(int(rtc_frozen_frames), self.rtc_overlap_frames)
        self.blend_steps = int(blend_steps)
        chunk = model.config.action_chunk
        self.execution_frames = chunk if execution_frames == 0 else int(execution_frames)
        if not 0 < self.execution_frames <= chunk:
            raise ValueError(f"execution_frames must be in (0, {chunk}], got {execution_frames}")
        self.history_frames = model.config.action_history
        # Margin: the newest observation is dropped at the prediction cut, and the grid may not align with the window.
        self.history_steps = math.ceil(self.history_frames * self.control_fps / self.model_fps) + 3
        self.kimodo = KimodoFeatures(self.control_fps, self.model_fps)
        self.reset()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        text_embedding_cache: str | Path,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        weights: str = "model",
        **kwargs,
    ) -> "KimodoPlanner":
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        saved = json.loads((checkpoint_dir / "config.json").read_text())
        model_cfg, main_cfg = saved["model"], saved["main"]
        cameras = tuple(model_cfg.get("cameras", ("head",)))  # released checkpoints: one egocentric camera
        config = KimodoPolicyConfig(
            fps=int(model_cfg["fps"]),
            motion_mask_mode=str(model_cfg["motion_mask_mode"]),
            dinov3_model_name=str(model_cfg["dinov3_model_name"]),
            dinov3_checkpoint=model_cfg.get("dinov3_checkpoint"),
            action_chunk=int(main_cfg["action_chunk"]),
            action_history=int(main_cfg["action_history"]),
            load_text_encoder=False,
            controlnet_num_layers=int(model_cfg.get("controlnet_num_layers", 8)),
            controlnet_scale=int(model_cfg.get("controlnet_scale", 1)),
            detach_root_control_for_body=bool(model_cfg.get("detach_root_control_for_body", False)),
            control_fusion_mode=str(model_cfg.get("control_fusion_mode", "both")),
            enable_hand_head=bool(model_cfg.get("enable_hand_head", True)),
            hand_hidden_dim=int(model_cfg.get("hand_hidden_dim", 256)),
            hand_num_layers=int(model_cfg.get("hand_num_layers", 4)),
            hand_num_heads=int(model_cfg.get("hand_num_heads", 4)),
            hand_ffn_dim=int(model_cfg.get("hand_ffn_dim", 1024)),
            hand_init_seed=int(model_cfg.get("hand_init_seed", 3407)),
            hand_control_mode=str(model_cfg.get("hand_control_mode", "binary")),
            num_cameras=len(cameras),
        )
        model = KimodoPolicy(config)
        model.load_controlnet_checkpoint(str(checkpoint_dir), weights=weights)
        model.to(device=torch.device(device), dtype=dtype).eval()
        kwargs.setdefault("cameras", cameras)
        return cls(model, load_text_embeddings(text_embedding_cache), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self.num_plans = 0
        self._first_heading: np.ndarray | None = None
        self._hand_track = torch.zeros(0, 2)  # commanded hand at every model frame since the episode start
        self._previous: tuple | None = None  # (cut, future features, clean hand, instruction) of the last prediction
        self._tail: tuple[int, ActionChunk] | None = None  # (first step, targets) the last plan did not execute

    def condition_history(self, history: Sequence[Observation]):
        """The model's history condition ``(motion [H', D], mask [H', D])`` plus the last observed pose.

        The newest observation sits at the prediction cut, and training conditions
        on frames ``[cut - H, cut)``, so it is left out (``KimodoWindowDataset`` slices
        the same way).
        """
        features, mask, local_rot, root = self.kimodo.history(history, self._first_heading)
        return features[:-1][-self.history_frames :], mask[:-1][-self.history_frames :], local_rot[-1], root[-1]

    def egoview(self, obs: Observation) -> torch.Tensor:
        """``[3, H, W]`` for one camera, ``[cameras, 3, H, W]`` for several (as ``KimodoWindowDataset``)."""
        images = [torch.from_numpy(np.ascontiguousarray(obs.images[c][..., :3])).permute(2, 0, 1) for c in self.cameras]
        return images[0] if len(images) == 1 else torch.stack(images)

    def _text_embedding(self, instruction: str) -> torch.Tensor | None:
        embedding = self.text_embeddings.get(instruction.strip())
        if embedding is None and self.model.text_encoder is None:
            raise KeyError(f"no cached text embedding for {instruction!r}")
        return embedding

    def control_offsets(self, step: int, frames: int, execution_frames: int | None = None) -> torch.Tensor:
        """Model-frame offsets from prediction frame 0 of the current control step and of the ones the plan covers.

        Frame 0 is at model time ``round(step * model_fps / control_fps)``, the history
        frame left out at the cut (``condition_history``). Row 0 is the current step;
        rows 1.. are the next steps, as many as ``execution_frames`` model frames last,
        but never past the last predicted frame.
        """
        ratio = self.model_fps / self.control_fps
        now = step * ratio - int(round(step * ratio))
        execution_frames = self.execution_frames if execution_frames is None else execution_frames
        steps = min(int(round(execution_frames / ratio)), int(math.floor((frames - 1 - now) / ratio + 1e-6)))
        return now + torch.arange(steps + 1, dtype=torch.float32) * ratio

    def hand_history(self, cut: int, frames: int) -> torch.Tensor:
        """``[1, frames, 2]``: the hand commands of model frames ``[cut - frames, cut)``, zeros before the first plan."""
        track = self._hand_track[:cut]
        if track.shape[0] < cut:
            track = torch.cat((track, torch.zeros(cut - track.shape[0], 2)))
        history = track[max(0, cut - frames) :]
        if history.shape[0] < frames:
            history = torch.cat((torch.zeros(frames - history.shape[0], 2), history))
        return history.unsqueeze(0)

    def _condition_hand(self, history: Sequence[Observation], cut: int, frames: int) -> torch.Tensor:
        """``[1, frames, 2]``: measured closure for continuous hands, else what earlier plans commanded."""
        if self.hand_mode == "binary":
            return self.hand_history(cut, frames)
        hand = self.kimodo.hand_history(history)[:-1][-self.history_frames :].unsqueeze(0)  # same grid as the motion
        hand = hand[:, -frames:] if frames else hand[:, :0]
        if hand.shape[1] < frames:
            hand = torch.cat((torch.zeros(1, frames - hand.shape[1], 2), hand), dim=1)
        return hand

    def _root_before(self, output: dict, roots: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        """The root one model frame before prediction frame 0, per ``root_boundary``."""
        if self.root_boundary == "first_prediction":
            return roots[0].clone()
        if self.root_boundary == "extrapolate":
            return roots[0].clone() if len(roots) == 1 else 2.0 * roots[0] - roots[1]
        before = output.get("history_last_root_position")
        # Root x/z is window-relative: the first delta must start from this window's history end.
        return fallback if before is None else before.float().cpu().reshape(3)

    @torch.inference_mode()
    def plan(self, history: Sequence[Observation]) -> ActionChunk:
        obs = history[-1]
        if self._first_heading is None:
            self._first_heading = yaw_matrix(float(yaw_of(history[0].root_rot)))
        device = self.model.device
        dtype = next(self.model.controlnet.parameters()).dtype
        cut = int(round(obs.step * self.model_fps / self.control_fps))

        motion, mask, last_rot, last_root = self.condition_history(history)
        history_motion = motion.unsqueeze(0).to(device)
        history_mask = mask.unsqueeze(0).to(device)
        text = self._text_embedding(obs.instruction)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(_episode_noise_seed(self.seed, 0 if self.noise == "fixed" else self.num_plans))
        # A filtered or measured hand must not be pulled back towards the raw previous prediction.
        rtc_hand = self.hand_mode == "binary" and self.hand_filter is None
        rtc = {}
        if self.rtc_overlap_frames and self._previous is not None and self._previous[3] == obs.instruction:
            previous_cut, previous_motion, previous_hand, _ = self._previous
            shift = cut - previous_cut  # the previous plan's frame for this plan's frame 0
            reference = previous_motion[shift : shift + self.rtc_overlap_frames]
            if 0 <= shift and reference.shape[0]:
                rtc = dict(rtc_motion_reference=reference.unsqueeze(0).to(device),
                           rtc_overlap_frames=self.rtc_overlap_frames, rtc_frozen_frames=self.rtc_frozen_frames)
                if previous_hand is not None:
                    rtc["rtc_hand_reference"] = previous_hand[shift : shift + reference.shape[0]].unsqueeze(0).to(device)

        output = self.model.predict_future(
            instruction=obs.instruction,
            egoview=self.egoview(obs).unsqueeze(0).to(device),
            history_motion=history_motion,
            history_feature_mask=history_mask,
            diffusion_steps=self.diffusion_steps,
            squeeze_batch=True,
            text_feat=None if text is None else text.to(device=device, dtype=dtype),
            hand_history=self._condition_hand(history, cut, history_motion.shape[1]).to(device),
            generator=generator,
            **rtc,
        )
        hand_clean = output.get("hand_clean") if rtc_hand else None
        self._previous = (cut, output["motion_features"].float().cpu(), None if hand_clean is None else hand_clean.float().cpu(),
                          obs.instruction)
        chunk = self._decode(output, obs, cut, last_rot, last_root)
        self.num_plans += 1
        return chunk

    def _decode(self, output: dict, obs: Observation, cut: int, last_rot: torch.Tensor, last_root: torch.Tensor) -> ActionChunk:
        """The plan, read off the prediction at the true time of each control step.

        ``last_rot`` / ``last_root`` (the newest observed pose) are unused here;
        ``OriginalTimingPlanner`` starts from them at reset.
        """
        local_rot = output["local_rot_mats"].float().cpu()
        root_pos = output["root_positions"].float().cpu()
        before = self._root_before(output, root_pos, root_pos[0]).reshape(1, 3)

        # Sample the prediction at the true time of each control step. Spreading the frames
        # over round(n * 5/3) steps from the previous plan's last frame instead lagged one
        # model frame behind the clock: with perfect predictions that cost drawer_top_open
        # and move_plate 30 points at 10-frame plans and pick_box 10 points at 50.
        frames = local_rot.shape[0]
        offsets = self.control_offsets(obs.step, frames, execution_frames=frames)  # all of it; the rest is the tail
        steps = len(self.control_offsets(obs.step, frames)) - 1
        control_rot, control_root = _interpolate_motion(
            torch.cat((local_rot[:1], local_rot)), torch.cat((before, root_pos)), offsets + 1.0
        )
        if self.hand_mode == "continuous":
            # The measured closure stands one frame before frame 0, like the root.
            closure = torch.cat((torch.as_tensor(obs.hand, dtype=torch.float32)[None],
                                 output["hand_closure"].clamp(0.0, 1.0).float().cpu()))
            control_hand = lerp_rows(closure, offsets[1:] + 1.0)
        else:
            hand = output["hand_binary"].float().cpu()
            if self.hand_filter is not None:
                executed = int(offsets[steps].round()) + 1
                hand = torch.cat((self.hand_filter(hand[:executed], hand), hand[executed:]))
            control_hand = hand[offsets[1:].round().clamp(0, hand.shape[0] - 1).long()]
            self._hand_track = torch.cat((self.hand_history(cut, cut)[0], hand))  # the next plan overwrites from its own cut
        actions = self.kimodo.codec.encode(
            control_rot[1:], control_root[1:], previous_root_position=control_root[0], hand_binary=control_hand
        )

        full = ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)
        full.root_rot = np.einsum("ij,tjk->tik", self._first_heading, full.root_rot).astype(np.float32)
        chunk = full[:steps]
        if self.blend_steps and self._tail is not None and self._tail[0] == obs.step + 1:
            chunk = _crossfade(self._tail[1], chunk, self.blend_steps)
        self._tail = (obs.step + 1 + steps, full[steps:])
        return chunk
