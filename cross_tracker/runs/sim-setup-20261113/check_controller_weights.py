"""Load the available controllers and record their exact ONNX interfaces."""
from pathlib import Path
import hashlib
import json
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
twist = json.loads((RUN / "twist2-weight-pin.json").read_text())
paths = [("twist2", ROOT / twist["relative_path"])]
paths += [("sonic_" + name, ROOT / "data/cache/sim-assets/sonic" / ("model_" + name + ".onnx"))
          for name in ("encoder", "decoder")]
results = []
for name, path in paths:
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if name == "twist2":
        assert sha == twist["sha256"] and path.stat().st_size == twist["bytes"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    inputs = [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_inputs()]
    outputs = [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()]
    results.append({"controller": name, "sha256": sha, "inputs": inputs, "outputs": outputs,
                    "loaded": True})
    print(name, inputs, outputs, flush=True)
(RUN / "controller-interfaces.json").write_text(json.dumps({
    "controllers": results, "mimic_lite": "missing original ONNX",
    "meaning": "ONNX load and interface only; no control rollout",
}, indent=2))
