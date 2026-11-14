"""Explicit CCT evaluation loader; never silently use a plain Kimodo policy."""
from dataclasses import fields
import json
from pathlib import Path
import torch
from cct_data import combine_context
from cct_policy import CCTPolicy
from repaired_data import checksum
from kimodo_vla.model.kimodo_policy import KimodoPolicyConfig
from kimodo_vla.model.planner import KimodoPlanner, load_text_embeddings

ROOT = Path(__file__).resolve().parents[1]


def frozen_context(tracker):
    selection = json.loads((ROOT / "configs/cct-eval-context.json").read_text())
    cache = ROOT / "data/cache/cct-box-v1"
    report = json.loads((cache / "manifest.json").read_text())
    if not report["complete"] or selection["manifest_sha256"] != report["manifest_sha256"]:
        raise ValueError("Context split changed")
    ids = selection["episodes"][tracker]
    if len(set(ids)) != 3 or any(not i.startswith(f"HOI_pp_box/{tracker}/") for i in ids):
        raise ValueError("Incorrect context controller or trajectory count")
    records = []
    for episode_id in ids:
        name = report["episodes"][episode_id]["file"]
        if checksum(cache / name) != report["files"][name]:
            raise ValueError("Context episode changed")
        record = torch.load(cache / name, weights_only=True, map_location="cpu")
        if record["episode_id"] != episode_id or record["split"] != "train":
            raise ValueError("Non-training context")
        records.append(record)
    if checksum(cache / "metadata.pt") != report["files"]["metadata.pt"]:
        raise ValueError("Controller text embedding changed")
    metadata = torch.load(cache / "metadata.pt", weights_only=True, map_location="cpu")[tracker]
    result = combine_context(records, metadata, torch.tensor(report["mean"]), torch.tensor(report["std"]))
    return {key: value.unsqueeze(0) for key, value in result.items()}


def load_planner(checkpoint, tracker, text_cache, device="cuda", **kwargs):
    checkpoint = Path(checkpoint)
    saved = json.loads((checkpoint / "config.json").read_text())
    allowed = {f.name for f in fields(KimodoPolicyConfig)}
    config = {k: v for k, v in saved["model"].items() if k in allowed}
    config.update(action_chunk=saved["main"]["action_chunk"], action_history=saved["main"]["action_history"],
                  load_text_encoder=False, num_cameras=len(saved["model"].get("cameras", ["head"])))
    model = CCTPolicy(KimodoPolicyConfig(**config))
    model.load_controlnet_checkpoint(checkpoint)
    if not model.has_trained_context:
        raise ValueError("Evaluation requires a checkpoint containing CCT encoder weights")
    model.to(device=device, dtype=torch.float32).eval()
    model.set_context(frozen_context(tracker))
    kwargs.setdefault("cameras", tuple(saved["model"].get("cameras", ["head"])))
    return KimodoPlanner(model, load_text_embeddings(text_cache), **kwargs)
