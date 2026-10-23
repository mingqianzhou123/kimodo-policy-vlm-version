"""KimodoPlanner: Kimodo-Policy behind the ``Planner`` interface.

Ported from ``evaluation/humanoidarena_server.py`` of the original repository
(``KimodoHumanoidArenaRuntime.infer``), with the HTTP layer, RTC and trace
flags removed. With every joint observed it feeds the model exactly what the
original runtime did, so released checkpoints behave the same.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from kimodo_vla.core.types import ActionChunk, G1_JOINT_INDEX, G1_JOINT_NAMES, Observation, yaw_matrix, yaw_of
from kimodo_vla.model.kimodo_policy import KimodoPolicy, KimodoPolicyConfig
from kimodo_vla.motion.g1_reference import (
    HumanoidArenaActionDecoder,
    _interpolate_motion,
    resample_hand_binary_chunk,
    resample_motion_chunk,
)
from kimodo_vla.motion.representation.kimodo_motionrep import KimodoMotionRep
from kimodo_vla.skeleton.definitions import G1Skeleton34

G1_XML_PATH = Path(__file__).resolve().parents[1] / "skeleton/assets/g1skel34/xml/g1.xml"


def history_feature_mask(representation: KimodoMotionRep, joint_mask: np.ndarray) -> torch.Tensor:
    """Which Kimodo features of a history frame are known from observations.

    Root position is never observed (the policy only sees orientation), so the
    mask covers the root heading plus the global rotation of every skeleton joint
    the benchmark reports. Joints it does not report must still be filled with a
    plausible value by the adapter, since they shift their children's rotations.
    """
    mask = torch.zeros(representation.motion_rep_dim, dtype=torch.bool)
    mask[representation.slice_dict["global_root_heading"]] = True
    rot = representation.slice_dict["global_rot_data"]
    skeleton = representation.skeleton
    observed: dict[str, bool] = {}
    for j, name in enumerate(skeleton.bone_order_names):
        parent = skeleton.bone_parents[name]
        joint = name.replace("_skel", "_joint")
        if parent is None:
            observed[name] = True
        elif joint in G1_JOINT_INDEX:
            observed[name] = bool(joint_mask[G1_JOINT_INDEX[joint]])
        else:  # toe and hand end points carry no joint of their own
            observed[name] = observed[parent]
        if observed[name]:
            mask[rot.start + 6 * j : rot.start + 6 * j + 6] = True
    return mask


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
        camera: str = "head",
    ):
        if model.config.hand_control_mode != "binary":
            raise NotImplementedError("KimodoPlanner currently supports binary hand control only")
        self.model = model
        self.text_embeddings = text_embeddings
        self.control_fps = float(control_fps)
        self.model_fps = float(model.fps)
        self.diffusion_steps = int(diffusion_steps)
        self.camera = camera
        chunk = model.config.action_chunk
        self.execution_frames = chunk if execution_frames == 0 else int(execution_frames)
        if not 0 < self.execution_frames <= chunk:
            raise ValueError(f"execution_frames must be in (0, {chunk}], got {execution_frames}")
        self.history_frames = model.config.action_history
        # Margin: the newest observation is dropped at the prediction cut, and the grid may not align with the window.
        self.history_steps = math.ceil(self.history_frames * self.control_fps / self.model_fps) + 3
        self.codec = HumanoidArenaActionDecoder(G1Skeleton34(), G1_XML_PATH, fps=self.control_fps)
        self.state_representation = KimodoMotionRep(G1Skeleton34(), fps=self.model_fps, stats_path=None)
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
        )
        model = KimodoPolicy(config)
        model.load_controlnet_checkpoint(str(checkpoint_dir))
        model.to(device=torch.device(device), dtype=dtype).eval()
        return cls(model, load_text_embeddings(text_embedding_cache), **kwargs)

    def reset(self, seed: int | None = None) -> None:
        self.seed = seed
        self.num_plans = 0
        self._first_heading: np.ndarray | None = None
        self._hand_history = torch.zeros(1, 0, 2)
        self._previous_rot: torch.Tensor | None = None
        self._previous_root: torch.Tensor | None = None

    def encode_history(self, history: Sequence[Observation]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Observations at control rate -> Kimodo features at model rate.

        Returns ``(features [T, D], feature_mask [T, D], local_rot_mats, root_positions)``;
        rotations are relative to the episode's first heading, root position is zero.
        """
        relative = self._first_heading.T
        rots = torch.from_numpy(np.stack([relative @ o.root_rot for o in history])).float()
        joints = torch.from_numpy(np.stack([o.joint_pos for o in history])).float()
        frames = joints.shape[0]
        if frames < 5:  # the smooth-root solver needs a few samples; pad, then drop the padding
            pad = 5 - frames
            joints = torch.cat((joints, joints[-1:].expand(pad, -1)))
            rots = torch.cat((rots, rots[-1:].expand(pad, -1, -1)))
        decoded = self.codec.decode_joint_configuration(
            joints,
            torch.zeros(joints.shape[0], 3),
            root_rotation_matrices=rots,
            joint_names=G1_JOINT_NAMES,
        )
        local_rot, root_pos = self._to_model_rate(
            decoded["local_rot_mats"][:frames], decoded["root_positions"][:frames], history[0].step
        )
        features = self._features(local_rot, root_pos)
        mask = history_feature_mask(self.state_representation, history[-1].joint_mask)
        return features, mask.expand_as(features), local_rot, root_pos

    def _to_model_rate(self, local_rot: torch.Tensor, root_pos: torch.Tensor, first_step: int):
        """Resample onto the model-rate grid anchored at the episode start.

        Training resamples whole episodes, so the grid must not shift with the
        window; with ``first_step == 0`` this equals ``resample_motion``.
        """
        last_step = first_step + local_rot.shape[0] - 1
        ratio = self.model_fps / self.control_fps
        first_k = math.ceil(first_step * ratio - 1e-6)
        last_k = int(round(last_step * ratio))
        times = torch.arange(first_k, last_k + 1, dtype=torch.float32) / self.model_fps
        positions = (times * self.control_fps - first_step).clamp(0, local_rot.shape[0] - 1)
        return _interpolate_motion(local_rot, root_pos, positions)

    def _features(self, local_rot: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        frames = local_rot.shape[0]
        if frames < 5:
            pad = 5 - frames
            local_rot = torch.cat((local_rot, local_rot[-1:].expand(pad, -1, -1, -1)))
            root_pos = torch.cat((root_pos, root_pos[-1:].expand(pad, -1)))
        return self.state_representation(local_rot, root_pos, to_normalize=False).cpu()[:frames]

    def _text_embedding(self, instruction: str) -> torch.Tensor | None:
        embedding = self.text_embeddings.get(instruction.strip())
        if embedding is None and self.model.text_encoder is None:
            raise KeyError(f"no cached text embedding for {instruction!r}")
        return embedding

    @torch.inference_mode()
    def plan(self, history: Sequence[Observation]) -> ActionChunk:
        obs = history[-1]
        if self._first_heading is None:
            self._first_heading = yaw_matrix(float(yaw_of(history[0].root_rot)))
        device = self.model.device
        dtype = next(self.model.controlnet.parameters()).dtype

        features, mask, state_rot, state_root = self.encode_history(history)
        # The newest observation sits at the prediction cut; training conditions on [cut - H, cut).
        history_motion = features[:-1][-self.history_frames :].unsqueeze(0).to(device)
        history_mask = mask[:-1][-self.history_frames :].unsqueeze(0).to(device)
        hand_history = self._hand_history[:, -history_motion.shape[1] :]
        if hand_history.shape[1] < history_motion.shape[1]:
            pad = torch.zeros(1, history_motion.shape[1] - hand_history.shape[1], 2)
            hand_history = torch.cat((pad, hand_history), dim=1)

        image = torch.from_numpy(np.ascontiguousarray(obs.images[self.camera][..., :3]))
        text = self._text_embedding(obs.instruction)
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(_episode_noise_seed(self.seed, self.num_plans))

        output = self.model.predict_future(
            instruction=obs.instruction,
            egoview=image.permute(2, 0, 1).unsqueeze(0).to(device),
            history_motion=history_motion,
            history_feature_mask=history_mask,
            diffusion_steps=self.diffusion_steps,
            squeeze_batch=True,
            text_feat=None if text is None else text.to(device=device, dtype=dtype),
            hand_history=hand_history.to(device),
            generator=generator,
        )

        n = self.execution_frames
        local_rot = output["local_rot_mats"][:n].float().cpu()
        root_pos = output["root_positions"][:n].float().cpu()
        hand = output["hand_binary"][:n].float().cpu()

        # Continue from the last target actually handed out; fall back to the measurement at reset.
        previous_rot = self._previous_rot if self._previous_rot is not None else state_rot[-1]
        previous_root = self._previous_root if self._previous_root is not None else state_root[-1]
        if output.get("history_last_root_position") is not None:
            # Root x/z is window-relative: the first delta must start from this window's history end.
            previous_root = output["history_last_root_position"].float().cpu().reshape(3)
        previous_hand = self._hand_history[0, -1] if self._hand_history.shape[1] else torch.zeros(2)

        control_rot, control_root = resample_motion_chunk(
            local_rot, root_pos,
            source_fps=self.model_fps, target_fps=self.control_fps,
            previous_local_rot_mat=previous_rot, previous_root_position=previous_root,
        )
        control_hand = resample_hand_binary_chunk(
            hand, source_fps=self.model_fps, target_fps=self.control_fps, previous_hand_binary=previous_hand
        )
        actions = self.codec.encode(control_rot, control_root, previous_root_position=previous_root, hand_binary=control_hand)

        chunk = ActionChunk.from_array40(actions.numpy(), fps=self.control_fps)
        chunk.root_rot = np.einsum("ij,tjk->tik", self._first_heading, chunk.root_rot).astype(np.float32)

        self._hand_history = torch.cat((self._hand_history, hand.unsqueeze(0)), dim=1)[:, -self.history_frames :]
        self._previous_rot = local_rot[-1].clone()
        self._previous_root = root_pos[-1].clone()
        self.num_plans += 1
        return chunk
