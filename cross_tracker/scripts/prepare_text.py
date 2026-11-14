"""Relocate released LLM2Vec paths without editing weights, then cache the three real instructions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
from fetch_assets import matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--arena-data", type=Path, default=ROOT / "data/cache/inputs-v1/audit/dataset")
    args = parser.parse_args()
    assets = json.loads((ROOT / "data/manifests/weights-v1.json").read_text())["files"]
    weights = ROOT / "data/cache/weights"
    for name, expected in assets.items():
        if name.startswith("checkpoints/"):
            if not (weights / name).is_file() or not matches(weights / name, expected):
                raise ValueError(f"Unverified text encoder asset: {name}")
    relocated = ROOT / "data/cache/text-encoder"
    changes = {}
    for name in assets:
        if not name.startswith("checkpoints/LLM2Vec"):
            continue
        source = weights / name
        dest = relocated / Path(name).relative_to("checkpoints")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "adapter_config.json":
            config = json.loads(source.read_text())
            base = str(weights / "checkpoints/Meta-Llama-3-8B-Instruct")
            changes[name] = {"old": config["base_model_name_or_path"], "new": base}
            config["base_model_name_or_path"] = base
            content = json.dumps(config, indent=2)+"\n"
            if dest.exists() and dest.read_text() != content:
                raise ValueError(f"Conflicting derived adapter config: {dest}")
            dest.write_text(content)
        elif not dest.exists():
            dest.symlink_to(source)
        elif dest.resolve() != source.resolve():
            raise ValueError(f"Unexpected derived asset: {dest}")
    (relocated / "relocations.json").write_text(json.dumps(changes, indent=2)+"\n")
    output = ROOT / "data/cache/text"
    if output.exists() and list(output.glob("*.pt")):
        raise FileExistsError("Text cache already exists; inspect before replacing it")
    command = [sys.executable, str(ROOT.parent / "scripts/cache_text_embeddings.py"),
               "--out", str(output), "--checkpoints", str(relocated),
               "--arena-data", str(args.arena_data.resolve()), "--device", args.device]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "OMP_NUM_THREADS": "4"}
    subprocess.run(command, cwd=ROOT.parent, env=env, check=True)


if __name__ == "__main__":
    main()
