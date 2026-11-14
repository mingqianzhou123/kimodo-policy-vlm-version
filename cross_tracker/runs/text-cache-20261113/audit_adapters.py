"""Verify both loaded adapters and reproduce the cache despite upstream loader warnings."""
from pathlib import Path
import json
import sys
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder

receipts = []
original = PeftModel.from_pretrained.__func__


def verified_load(cls, model, path, **kwargs):
    result = original(cls, model, path, **kwargs)
    expected = load_file(str(Path(path) / "adapter_model.safetensors"))
    actual = get_peft_model_state_dict(result)
    assert set(actual) == set(expected), (set(actual) - set(expected), set(expected) - set(actual))
    for key, value in actual.items():
        assert torch.equal(value.cpu(), expected[key].to(value.dtype)), key
    receipts.append({"adapter": Path(path).name, "matched_tensors": len(actual)})
    return result


PeftModel.from_pretrained = classmethod(verified_load)
encoder = LLM2VecEncoder(str(ROOT / "data/cache/text-encoder")).cuda().eval()
assert len(receipts) == 2
checks = []
for path in sorted((ROOT / "data/cache/text").glob("*.pt")):
    cached = torch.load(path, weights_only=True)
    with torch.inference_mode():
        value, _ = encoder(cached["instruction"])
    value = value.cpu().to(torch.bfloat16)
    difference = float((value.float() - cached["embedding"].float()).abs().max())
    assert difference < 0.01, (path.name, difference)
    checks.append({"task": path.stem, "max_absolute_difference": difference})
report = {"adapters": receipts, "cache_reproduction": checks, "passed": True}
(Path(__file__).parent / "adapter-audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
