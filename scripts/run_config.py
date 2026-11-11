"""Training-config helpers that need no model code (train.py, submit_bigym2.py and summarize_bigym2.py share them)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "bigym2.yaml"


def current_observation_mode(config_path: Path, requested: str = "checkpoint") -> str:
    """Keep training and serving conditions matched unless explicitly overridden."""
    if requested in {"on", "off"}:
        return requested
    if requested != "checkpoint":
        raise ValueError(f"Unknown current-observation mode: {requested}")
    import json
    saved = json.loads(Path(config_path).read_text())
    return "on" if saved.get("main", {}).get("condition_current_observation", False) else "off"


def inference_fingerprint(config_path: Path, inference: dict) -> str:
    """Fingerprint the saved training config together with inference settings."""
    import hashlib
    import json
    value={"training":json.loads(config_path.read_text()),"inference":inference}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def load_yaml(path: Path):
    """A config file, on top of the one it names as ``_base_`` (relative to it), if any."""
    from omegaconf import OmegaConf  # here, so summarizing results needs nothing beyond the standard library

    config = OmegaConf.load(path)
    base = config.pop("_base_", None)
    return OmegaConf.merge(load_yaml(path.parent / base), config) if base else config


def score_steps(main) -> list[int]:
    """The checkpoints the headline score pools: ``score_checkpoints``, else the last ``final_checkpoints``.

    ``main`` is a config's ``main`` section, or the same as a plain dict from a run's ``config.json``.
    """
    if main.get("score_checkpoints"):
        return sorted(int(s) for s in main["score_checkpoints"])
    spacing = main.get("final_checkpoint_spacing", 1000)
    return sorted(main["max_steps"] - i * spacing for i in range(main["final_checkpoints"]))


def checkpoint_steps(main) -> set[int]:
    steps = set(range(main["checkpoint_every"], main["max_steps"] + 1, main["checkpoint_every"]))
    steps.update(score_steps(main))
    return {s for s in steps if 0 < s <= main["max_steps"]}
