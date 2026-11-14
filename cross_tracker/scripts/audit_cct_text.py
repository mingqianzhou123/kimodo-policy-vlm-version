"""Verify both loaded LLM2Vec adapters and reproduce the existing box text cache."""
import json
import os
from pathlib import Path
import sys
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT / "scripts")]
from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder
from prepare_cct import text_encoder_path

torch.set_num_threads(4)
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
original = PeftModel.from_pretrained
checks = []


def audited(model, path, *args, **kwargs):
    loaded = original(model, path, *args, **kwargs)
    reference = load_file(str(Path(path) / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(loaded)
    assert set(reference) == set(actual), "Adapter parameter names differ"
    for name, tensor in reference.items():
        assert torch.equal(actual[name].cpu(), tensor.to(actual[name].dtype)), name
    checks.append({"adapter": Path(path).name, "matched_tensors": len(reference)})
    return loaded


PeftModel.from_pretrained = staticmethod(audited)
encoder = LLM2VecEncoder(str(text_encoder_path())).cpu().eval()
assert len(checks) == 2
cache = torch.load(ROOT / "data/cache/text/HOI_pp_box.pt", weights_only=True, map_location="cpu")
with torch.inference_mode():
    embedding = encoder(cache["instruction"])[0].cpu().to(cache["embedding"].dtype)
difference = float((embedding.float()-cache["embedding"].float()).abs().max())
cosine = float(torch.nn.functional.cosine_similarity(embedding.float(), cache["embedding"].float()).mean())
metadata = json.loads((ROOT / "configs/cct-metadata.json").read_text())
saved = torch.load(ROOT / "data/cache/cct-box-v1/metadata.pt", weights_only=True, map_location="cpu")
reproduced = {}
for source, sentence in metadata.items():
    with torch.inference_mode():
        value = encoder(sentence)[0].cpu().to(saved[source].dtype)
    delta = float((value.float()-saved[source].float()).abs().max())
    reproduced[source] = delta
    assert torch.equal(value, saved[source]), f"Metadata not reproducible: {source}"
report = {"passed": cosine > .999, "adapters": checks, "box_cache_max_absolute_difference": difference,
          "box_cache_cosine_similarity": cosine, "metadata_max_absolute_difference": reproduced,
          "note": "CPU BF16 task-cache comparison; no task embedding replaced"}
(ROOT / "runs/cct-box-preparation-20261114/text-adapter-audit.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report), flush=True)
assert report["passed"]
