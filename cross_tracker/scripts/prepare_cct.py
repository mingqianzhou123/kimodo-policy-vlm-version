"""Cache frozen CCT image/text features and train-only transition statistics."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent / "scripts"), str(ROOT.parent)]
from cct_inputs import context_inputs
from repaired_data import checksum
from fetch_assets import matches


def save_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def text_encoder_path():
    weights = ROOT / "data/cache/weights"
    files = json.loads((ROOT / "data/manifests/weights-v1.json").read_text())["files"]
    target = ROOT / "data/cache/cct-text-encoder"
    for name, expected in files.items():
        if not name.startswith("checkpoints/"):
            continue
        source = (ROOT / "data/cache/cct-original-text-configs" / name
                  if name.endswith("adapter_config.json") else weights / name)
        if not matches(source, expected):
            raise ValueError(f"Unverified text encoder weight: {name}")
        if not name.startswith("checkpoints/LLM2Vec"):
            continue
        dest = target / Path(name).relative_to("checkpoints")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "adapter_config.json":
            config = json.loads(source.read_text())
            config["base_model_name_or_path"] = str(weights / "checkpoints/Meta-Llama-3-8B-Instruct")
            content = json.dumps(config, indent=2) + "\n"
            if dest.exists() and dest.read_text() != content:
                raise ValueError("Conflicting derived text encoder configuration")
            dest.write_text(content)
        elif not dest.exists():
            dest.symlink_to(source)
        elif dest.resolve() != source.resolve():
            raise ValueError("Conflicting text encoder symlink")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    out = ROOT / "data/cache/cct-box-v1"
    out.mkdir(parents=True, exist_ok=True)
    signature = {str(p.relative_to(ROOT)): checksum(p) for p in [
        ROOT / "data/manifests/balanced-v1.json", ROOT / "configs/cct-metadata.json",
        ROOT / "src/cct_inputs.py", Path(__file__)]}
    if (out / "manifest.json").exists():
        report = json.loads((out / "manifest.json").read_text())
        if report["inputs"] != signature:
            raise ValueError("Cache inputs changed; use a new version")
        for name, digest in report["files"].items():
            if checksum(out / name) != digest:
                raise ValueError(f"Cache changed: {name}")
        print("VERIFIED complete context cache", flush=True)
        return
    if (out / "inputs.json").exists() and json.loads((out / "inputs.json").read_text()) != signature:
        raise ValueError("Partial cache inputs changed")
    save_json(out / "inputs.json", signature)
    metadata = json.loads((ROOT / "configs/cct-metadata.json").read_text())
    if not (out / "metadata.json").exists():
        from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder
        encoder = LLM2VecEncoder(str(text_encoder_path())).to(args.device).eval()
        with torch.inference_mode():
            embeddings = {source: encoder(sentence)[0].cpu().to(torch.bfloat16) for source, sentence in metadata.items()}
        assert all(x.shape == (1, 4096) and torch.isfinite(x).all() for x in embeddings.values())
        torch.save(embeddings, out / "metadata.pt.partial")
        (out / "metadata.pt.partial").replace(out / "metadata.pt")
        save_json(out / "metadata.json", {"sha256": checksum(out / "metadata.pt")})
        del encoder
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print("Encoded controller metadata", flush=True)
    if checksum(out / "metadata.pt") != json.loads((out / "metadata.json").read_text())["sha256"]:
        raise ValueError("Metadata cache changed")
    from kimodo_vla.model.modules.dinov3_wrapper import DINOv3Encoder
    shared = json.loads((ROOT / "data/manifests/shared-weights-v1.json").read_text())["files"]
    for name, expected in shared.items():
        if "dinov3" in name.lower() and not matches(ROOT.parent / name, expected):
            raise ValueError(f"Unverified visual encoder: {name}")
    encoder = DINOv3Encoder(str(ROOT.parent / "checkpoints/dinov3-vitl16-pretrain-lvd1689m")).to(args.device).eval()
    encoder.requires_grad_(False)
    roots = json.loads(args.paths.read_text())
    episodes, transitions, hashes = {}, [], {"metadata.pt": checksum(out / "metadata.pt")}
    for record, images in context_inputs(roots):
        name = record["episode_id"].replace("/", "__") + ".pt"
        path, receipt = out / name, out / (name + ".json")
        if receipt.exists():
            if checksum(path) != json.loads(receipt.read_text())["sha256"]:
                raise ValueError(f"Context cache changed: {name}")
            cached = torch.load(path, map_location="cpu", weights_only=True)
            assert cached["episode_id"] == record["episode_id"]
            assert torch.equal(cached["transitions"], record["transitions"])
        else:
            with torch.inference_mode():
                visual = torch.cat([encoder(chunk.to(args.device)).cpu().to(torch.bfloat16)
                                    for chunk in images.split(4)])
            assert visual.shape == (43, 196, 1024) and torch.isfinite(visual).all()
            torch.save(dict(record, visual=visual), out / (name + ".partial"))
            (out / (name + ".partial")).replace(path)
            save_json(receipt, {"sha256": checksum(path)})
        transitions.append(record["transitions"].double())
        hashes[name] = checksum(path)
        episodes[record["episode_id"]] = {"file": name, "source_rows": record["source_rows"],
                                         "next_source_rows": record["next_source_rows"],
                                         "prepared_indices": record["prepared_indices"]}
        print(f"{len(episodes)}/86 {record['episode_id']}", flush=True)
    assert len(episodes) == 86
    values = torch.cat(transitions)
    report = dict(complete=True, inputs=signature, device=args.device,
                  manifest_sha256=checksum(ROOT / "data/manifests/balanced-v1.json"),
                  episodes=episodes, files=hashes, mean=values.mean(0).tolist(),
                  std=values.std(0, unbiased=False).clamp_min(1e-6).tolist(),
                  transition="q35[t], repaired_command40[t], q35[t+1]; adjacent 50 Hz rows")
    save_json(out / "manifest.json", report)
    evidence = ROOT / "runs/cct-box-preparation-20261114/context-cache.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    save_json(evidence, dict(complete=True, episodes=len(episodes),
                            manifest_sha256=checksum(out / "manifest.json"), files=hashes))
    print("COMPLETE 86 training episodes; validation excluded", flush=True)


if __name__ == "__main__":
    main()
