"""Shared provenance checks for the doubledesk N1.7 experiment."""

import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def fingerprint(config):
    resolved = OmegaConf.to_container(config, resolve=True)
    spec = {"model": resolved["model"], "training": resolved["training"],
            "data": {k: resolved["main"].get(k) for k in (
                "task", "data_root", "dataset_kind", "original_repo", "text_cache",
                "action_history", "action_chunk", "micro_batch_size", "batch_size", "max_steps", "seed")}}
    spec["data"].update({k: resolved["main"][k] for k in (
        "arena_dataset_selection", "expected_tasks", "original_ordinal_stream", "sample_stride", "video_cache_size")
        if k in resolved["main"]})
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def check_preflight(config):
    if not config.main.get("require_preflight", False):
        return
    path = Path(config.main.preflight_report)
    if not path.is_file():
        raise RuntimeError(f"Run scripts/check_groot.py first; missing {path}")
    report = json.loads(path.read_text())
    if report.get("passed") is not True or report.get("fingerprint") != fingerprint(config):
        raise RuntimeError("Real N1.7 preflight has not passed for this exact experiment configuration")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)
