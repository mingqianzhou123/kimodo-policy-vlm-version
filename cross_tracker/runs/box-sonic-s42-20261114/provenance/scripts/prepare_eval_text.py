"""Give verified task embeddings the simulator's task names without re-encoding."""
from pathlib import Path
import hashlib
import json
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = {
    "HOI_pp_box": "Isaac-Move-PickPlace-Box-G129-Dex3-Wholebody",
    "HSI_boxing": "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
    "HSI_vision_navi": "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
}


def main():
    source = ROOT / "data/cache/text"
    output = ROOT / "data/cache/text-eval"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for task, name in TASK_NAMES.items():
        original = source / f"{task}.pt"
        payload = torch.load(original, map_location="cpu", weights_only=False)
        assert payload["task_name"] == task
        derived = {**payload, "task_name": name}
        destination = output / original.name
        torch.save(derived, destination)
        loaded = torch.load(destination, map_location="cpu", weights_only=False)
        assert torch.equal(loaded["embedding"], payload["embedding"])
        assert loaded["instruction"] == payload["instruction"]
        records.append({"task": task, "simulator_task": name,
                        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "embedding_identical": True})
    (output / "manifest.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
