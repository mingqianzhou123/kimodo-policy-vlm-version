"""KimodoPlanner: Kimodo-Policy behind the ``Planner`` interface.

Ported from ``evaluation/humanoidarena_server.py`` of the original repository
(``KimodoHumanoidArenaRuntime.infer``), with the HTTP layer and trace flags
removed. With every joint observed it feeds the model the same motion history
as the original runtime. Two deliberate differences: predictions are sampled at
the true time of each control step (the original ran one model frame behind),
and the hand history is indexed by model frame. ``timing="original"`` keeps the
original runtime's decoding instead, for comparing with it (the Arena and SIMPLE
port tests) and for reproducing its evaluations.

RTC (real-time chunking) is kept as an option, off by default: every published
evaluation ran with it off. When on, the next plan's first ``rtc_overlap_frames``
model frames are pulled towards the part of the previous prediction that was
not executed, so consecutive chunks join smoothly.

Per-benchmark options, all off by default (the original Arena + SONIC behavior):

* ``root_boundary``: where the first frame's root displacement starts.
  ``"history"`` is the end of the history the model generated in the same window
  (Arena with SONIC); ``"first_prediction"`` is the first predicted root (Arena
  with TWIST2, ``KIMODO_SERVER_ANCHOR_ROOT``); ``"extrapolate"`` is ``2 r0 - r1``
  from the first two predicted roots (SIMPLE, whose state has no root position).
* continuous hands (checkpoint ``hand_control_mode="continuous"``, SIMPLE): the
  hand condition is the measured closure in ``Observation.hand`` and the output
  is a closure in [0, 1] instead of open/closed.
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
from kimodo_vla.motion.g1_reference import (
    _interpolate_motion,
    resample_hand_binary_chunk,
    resample_hand_continuous_chunk,
    resample_motion_chunk,
)

ROOT_BOUNDARIES = ("history", "first_prediction", "extrapolate")
TIMINGS = ("exact", "original")
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


class KimodoPlanner:
    def __init__(
        self,
        model: KimodoPolicy,
        text_embeddings: dict[str, torch.Tensor],
        *,
        control_fps: float = 50.0,
        execution_frames: int = 0,
        diffusion_steps: int = 10,
        cameras: Sequence[str] = ("head",),
        rtc: bool = False,
        rtc_overlap_frames: int = 12,
        rtc_frozen_frames: int = 1,
        rtc_ramp_power: float = 1.0,
        root_boundary: str = "history",
        hand_filter: HandFilter | None = None,
        timing: str = "exact",
    ):
        if timing not in TIMINGS:
            raise ValueError(f"timing must be one of {TIMINGS}, got {timing!r}")
        self.timing = timing
        self.hand_mode = str(model.config.hand_control_mode).lower()
        if self.hand_mode not in ("binary", "continuous"):
            raise ValueError(f"hand_control_mode must be 'binary' or 'continuous', got {self.hand_mode!r}")
        if hand_filter is not None and self.hand_mode != "binary":
            raise ValueError("hand_filter works on binary hands only")
        if root_boundary not in ROOT_BOUNDARIES:
            raise ValueError(f"root_boundary must be one of {ROOT_BOUNDARIES}, got {root_boundary!r}")
        self.root_boundary = root_boundary
        self.hand_filter = hand_filter
        num_cameras = getattr(model.config, "num_cameras", 1)
        if len(cameras) != num_cameras:
            raise ValueError(f"model takes {num_cameras} camera(s), got {tuple(cameras)}")
        self.model = model
        self.text_embeddings = text_embeddings
        self.control_fps = float(control_fps)
        self.model_fps = float(model.fps)
        self.diffusion_steps = int(diffusion_steps)
        self.cameras = tuple(cameras)
        chunk = model.config.action_chunk
        self.execution_frames = chunk if execution_frames == 0 else int(execution_frames)
        if not 0 < self.execution_frames <= chunk:
            raise ValueError(f"execution_frames must be in (0, {chunk}], got {execution_frames}")
        if not 0 <= rtc_frozen_frames <= rtc_overlap_frames:
            raise ValueError(f"need 0 <= rtc_frozen_frames <= rtc_overlap_frames, got {rtc_frozen_frames}, {rtc_overlap_frames}")
        if not (np.isfinite(rtc_ramp_power) and rtc_ramp_power > 0):
            raise ValueError(f"rtc_ramp_power must be finite and positive, got {rtc_ramp_power}")
        # The overlap can only cover the predicted frames that are not executed.
        self.rtc_overlap_frames = min(int(rtc_overlap_frames), chunk - self.execution_frames)
        self.rtc_frozen_frames = min(int(rtc_frozen_frames), self.rtc_overlap_frames)
        self.rtc_ramp_power = float(rtc_ramp_power)
        self.rtc = bool(rtc) and self.rtc_overlap_frames > 0
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
        model.load_controlnet_checkpoint(str(checkpoint_dir))
        model.to(device=torch.device(device), dtype=dtype).eval()
        kwargs.setdefault("cameras", cameras)
        return cls(model, load_text_embeddings(text_embedding_cache), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self.num_plans = 0
        self._first_heading: np.ndarray | None = None
        self._hand_track = torch.zeros(0, 2)  # commanded hand at every model frame since the episode start
        # timing="original": the last executed hand frames and pose, as the original runtime kept them
        self._hand_history = torch.zeros(1, 0, 2)
        self._previous_rot: torch.Tensor | None = None
        self._previous_root: torch.Tensor | None = None
        self._rtc_motion: torch.Tensor | None = None  # unexecuted prediction tail, and the instruction it was for
        self._rtc_hand: torch.Tensor | None = None
        self._rtc_instruction: str | None = None

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

    def control_offsets(self, step: int, frames: int) -> torch.Tensor:
        """Model-frame offsets from prediction frame 0 of the current control step and of the ones the plan covers.

        Frame 0 is at model time ``round(step * model_fps / control_fps)``, the history
        frame left out at the cut (``condition_history``). Row 0 is the current step;
        rows 1.. are the next steps, as many as ``execution_frames`` model frames last,
        but never past the last predicted frame.
        """
        ratio = self.model_fps / self.control_fps
        now = step * ratio - int(round(step * ratio))
        steps = min(int(round(self.execution_frames / ratio)), int(math.floor((frames - 1 - now) / ratio + 1e-6)))
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
        """``[1, frames, 2]``: measured closure for continuous hands, else the commanded hand."""
        if self.hand_mode == "continuous":  # on the same grid as the motion history
            hand = self.kimodo.hand_history(history)[:-1][-self.history_frames :].unsqueeze(0)
        elif self.timing == "exact":
            return self.hand_history(cut, frames)
        else:
            hand = self._hand_history
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
            generator.manual_seed(_episode_noise_seed(self.seed, self.num_plans))
        use_rtc = self.rtc and self._rtc_motion is not None and self._rtc_instruction == obs.instruction
        # A filtered or measured hand must not be pulled back towards the raw previous prediction.
        rtc_hand = self.hand_mode == "binary" and self.hand_filter is None

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
            rtc_motion_reference=self._rtc_motion if use_rtc else None,
            rtc_hand_reference=self._rtc_hand if use_rtc and rtc_hand else None,
            rtc_overlap_frames=self.rtc_overlap_frames,
            rtc_frozen_frames=self.rtc_frozen_frames,
            rtc_ramp_power=self.rtc_ramp_power,
        )
        if self.timing == "exact":
            actions = self._decode_exact(output, obs, cut)
        else:
            actions = self._decode_original(output, obs, last_rot, last_root)

        chunk = ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)
        chunk.root_rot = np.einsum("ij,tjk->tik", self._first_heading, chunk.root_rot).astype(np.float32)
        if self.rtc:
            tail = slice(self.execution_frames, self.execution_frames + self.rtc_overlap_frames)
            self._rtc_motion = output["motion_features"][tail].detach().float().clone()
            hand_clean = output.get("hand_clean") if rtc_hand else None
            self._rtc_hand = None if hand_clean is None else hand_clean[tail].detach().float().clone()
            self._rtc_instruction = obs.instruction
        self.num_plans += 1
        return chunk

    def _decode_exact(self, output: dict, obs: Observation, cut: int) -> torch.Tensor:
        """40-D actions read off the prediction at the true time of each control step."""
        local_rot = output["local_rot_mats"].float().cpu()
        root_pos = output["root_positions"].float().cpu()
        before = self._root_before(output, root_pos, root_pos[0]).reshape(1, 3)

        # Sample the prediction at the true time of each control step. Spreading the frames
        # over round(n * 5/3) steps from the previous plan's last frame instead lagged one
        # model frame behind the clock: with perfect predictions that cost drawer_top_open
        # and move_plate 30 points at 10-frame plans and pick_box 10 points at 50.
        offsets = self.control_offsets(obs.step, local_rot.shape[0])
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
                executed = int(offsets[-1].round()) + 1
                hand = torch.cat((self.hand_filter(hand[:executed], hand), hand[executed:]))
            control_hand = hand[offsets[1:].round().clamp(0, hand.shape[0] - 1).long()]
            self._hand_track = torch.cat((self.hand_history(cut, cut)[0], hand))  # the next plan overwrites from its own cut
        return self.kimodo.codec.encode(
            control_rot[1:], control_root[1:], previous_root_position=control_root[0], hand_binary=control_hand
        )

    def _decode_original(self, output: dict, obs: Observation, last_rot: torch.Tensor, last_root: torch.Tensor) -> torch.Tensor:
        """40-D actions as the original runtime made them: the first ``execution_frames`` predicted
        frames spread over the control steps from the previous plan's last frame."""
        n = self.execution_frames
        all_roots = output["root_positions"].float().cpu()
        local_rot = output["local_rot_mats"][:n].float().cpu()
        root_pos = all_roots[:n]
        # Continue from the last target actually handed out; fall back to the measurement at reset.
        previous_rot = self._previous_rot if self._previous_rot is not None else last_rot
        previous_root = self._previous_root if self._previous_root is not None else last_root
        previous_root = self._root_before(output, all_roots, previous_root)

        control_rot, control_root = resample_motion_chunk(
            local_rot, root_pos,
            source_fps=self.model_fps, target_fps=self.control_fps,
            previous_local_rot_mat=previous_rot, previous_root_position=previous_root,
        )
        if self.hand_mode == "continuous":
            hand = output["hand_closure"][:n].clamp(0.0, 1.0).float().cpu()
            control_hand = resample_hand_continuous_chunk(
                hand, source_fps=self.model_fps, target_fps=self.control_fps,
                previous_hand_closure=torch.as_tensor(obs.hand, dtype=torch.float32),
            )
        else:
            hand = output["hand_binary"][:n].float().cpu()
            if self.hand_filter is not None:
                hand = self.hand_filter(hand, output["hand_binary"].float().cpu())
            previous_hand = self._hand_history[0, -1] if self._hand_history.shape[1] else torch.zeros(2)
            control_hand = resample_hand_binary_chunk(
                hand, source_fps=self.model_fps, target_fps=self.control_fps, previous_hand_binary=previous_hand
            )
        self._hand_history = torch.cat((self._hand_history, hand.unsqueeze(0)), dim=1)[:, -self.history_frames :]
        self._previous_rot = local_rot[-1].clone()
        self._previous_root = root_pos[-1].clone()
        return self.kimodo.codec.encode(control_rot, control_root, previous_root_position=previous_root, hand_binary=control_hand)
