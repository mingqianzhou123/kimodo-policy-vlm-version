"""Fetch the exact official checkpoints used by this experiment into a local directory."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from kimodo_vla.model.modules.groot_conditioning import GROOT_REVISION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/workspace/model-cache"))
    args = parser.parse_args()
    for repo, revision, name in [
        ("nvidia/GR00T-N1.7-3B", GROOT_REVISION, "GR00T-N1.7-3B"),
        ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561", "Cosmos-Reason2-2B"),
    ]:
        print(snapshot_download(repo, revision=revision, local_dir=str(args.root / name),
                                allow_patterns=["*.json", "*.txt", "*.safetensors"], max_workers=2))
