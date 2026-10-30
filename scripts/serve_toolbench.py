"""Serve a Kimodo checkpoint to HumanoidToolBench's evaluator (its /act, /info, /release protocol).

    python scripts/serve_toolbench.py --checkpoint log/.../checkpoint_100000 --text-cache <cache> --port 21000

Then point the evaluator at http://HOST:PORT. One server per evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from kimodo_vla.benchmarks.server import NUMPY_JSON, make_server
from kimodo_vla.benchmarks.toolbench.policy import make_kimodo_policy, routes
from run_config import inference_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 when the evaluator runs on another machine")
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--execution-frames", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--rtc-overlap-frames", type=int, default=0, help="real-time chunking; the original repository used 12")
    parser.add_argument("--turn", default="operator", help="'operator' (turn flag only while turning), or a constant flag")
    parser.add_argument("--seed", type=int, help="episode i uses seed + i for the diffusion noise")
    args = parser.parse_args()

    policy = make_kimodo_policy(
        args.checkpoint, args.text_cache, device=args.device, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
        execution_frames=args.execution_frames, diffusion_steps=args.diffusion_steps, rtc_overlap_frames=args.rtc_overlap_frames, turn=args.turn,
    )
    checkpoint = Path(args.checkpoint).resolve()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    repo = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(repo)): digest(p) for p in sorted((repo / "kimodo_vla").rglob("*.py"))}
    inference = {key: getattr(args, key) for key in ("dtype", "execution_frames", "diffusion_steps", "rtc_overlap_frames", "turn", "seed")}
    metadata = {"checkpoint_sha256": digest(checkpoint / "training_state.pt"),
                "config_sha256": inference_fingerprint(checkpoint/'config.json',inference),
                "adapter_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "inference": inference}
    server = make_server(routes(policy, checkpoint=str(checkpoint), base_seed=args.seed, metadata=metadata), args.host, args.port, NUMPY_JSON)
    print(f"Policy server: http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
