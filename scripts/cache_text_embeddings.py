"""Precompute the LLM2Vec embedding of every task instruction of a benchmark (run on a GPU server).

    python scripts/cache_text_embeddings.py --out data/cache/bigym2
    python scripts/cache_text_embeddings.py --out data/cache/arena --arena-data <HumanoidArena dataset>

BiGym 2.0: all tasks of BiGym's ``task_instructions.yaml`` (the sentences in the
demonstrations, and the ones the paper gave π0.5). HumanoidArena: each task's
instruction in the dataset's ``meta/tasks.parquet``, the string the simulator sends
as ``task`` (the SONIC, TWIST2 and mimic_lite datasets must agree). Writes ``<out>/<task>.pt`` =
``{version, source, task_id, task_name, instruction, embedding [1, 4096] bf16}``,
the per-task cache format of the original repository, which
``model.planner.load_text_embeddings`` reads. Training and evaluation both use
these files, so LLM2Vec (8B) never has to sit next to the policy.

Each instruction is encoded on its own, so an embedding does not depend on which
other sentences share its batch. The files are read back with
``load_text_embeddings`` at the end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder
from kimodo_vla.model.planner import load_text_embeddings

CACHE_VERSION = 3  # the original repository's _TEXT_EMBEDDING_CACHE_VERSION


def arena_instructions(root: Path) -> dict[str, str]:
    """Task folder -> instruction; refuses a task whose trackers' datasets disagree."""
    from kimodo_vla.benchmarks.arena import demos

    per_tracker = [demos.task_instructions(root, tracker) for tracker in demos.TRACKERS]
    instructions = {}
    for task in sorted(set().union(*per_tracker)):
        found = {table[task] for table in per_tracker if task in table}
        if len(found) != 1:
            raise SystemExit(f"{task}: the trackers' datasets give different instructions {sorted(found)}")
        instructions[task] = found.pop()
    if not instructions:
        raise SystemExit(f"no HumanoidArena task under {root}")
    return instructions
CHECKPOINTS = Path(__file__).resolve().parents[1] / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS, help="folder holding the two LLM2Vec checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arena-data", type=Path, help="HumanoidArena dataset root: cache its tasks instead of BiGym's")
    args = parser.parse_args()

    if args.arena_data:
        source, instructions = "HumanoidArena", arena_instructions(args.arena_data)
    else:
        from kimodo_vla.benchmarks.bigym2.adapter import task_instructions

        source, instructions = "bigym2", task_instructions()
    encoder = LLM2VecEncoder(checkpoint_path=str(args.checkpoints)).to(args.device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    for task, instruction in sorted(instructions.items()):
        with torch.inference_mode():
            embedding, _ = encoder(instruction)  # [1, 4096]
        embedding = embedding.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        torch.save(
            {
                "version": CACHE_VERSION,
                "source": source,
                "task_id": f"{source}::{task}",
                "task_name": task,
                "instruction": instruction,
                "embedding": embedding,
            },
            args.out / f"{task}.pt",
        )
        print(f"{task}: {tuple(embedding.shape)} norm {float(embedding.float().norm()):.2f}  {instruction}", flush=True)

    table = load_text_embeddings(args.out)
    for task, instruction in instructions.items():
        by_name, by_text = table[task], table[instruction]
        if not (torch.equal(by_name, by_text) and by_name.shape == (1, encoder.llm_dim) and torch.isfinite(by_name).all()):
            raise SystemExit(f"{task}: cache does not read back")
    stacked = torch.cat([table[task] for task in sorted(instructions)])
    if torch.unique(stacked, dim=0).shape[0] != len(instructions):
        raise SystemExit("two tasks share an embedding")
    print(f"{len(instructions)} task embeddings in {args.out.resolve()}")


if __name__ == "__main__":
    main()
