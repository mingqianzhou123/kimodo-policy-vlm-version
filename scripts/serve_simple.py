"""Serve a Kimodo checkpoint to SIMPLE over HTTP (the original server's /infer and /act).

    python scripts/serve_simple.py --checkpoint log/.../checkpoint_100000 --text-cache <cache> --port 18080

SIMPLE's official evaluation runs the policy inside its own process instead:
build it with ``kimodo_vla.benchmarks.simple.policy.make_kimodo_policy`` and call
it with the request described in ``kimodo_vla.benchmarks.simple.adapter``.
Formal runs used 50 executed frames and RTC off.
"""

from __future__ import annotations

import argparse

import torch

from kimodo_vla.benchmarks.server import NUMPY_JSON, make_server
from kimodo_vla.benchmarks.simple.policy import make_kimodo_policy, routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--execution-frames", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--rtc", action="store_true")
    parser.add_argument("--timing", choices=("exact", "original"), default="exact",
                        help="'original' decodes plans as the original repository did, one model frame behind")
    parser.add_argument("--max-navigation-speed", type=float, default=1.5)
    args = parser.parse_args()

    policy = make_kimodo_policy(
        args.checkpoint, args.text_cache, device=args.device, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
        execution_frames=args.execution_frames, diffusion_steps=args.diffusion_steps, rtc=args.rtc, timing=args.timing,
        max_navigation_speed=args.max_navigation_speed,
    )
    server = make_server(routes(policy), args.host, args.port, NUMPY_JSON)
    print(f"Serving SIMPLE Kimodo policy on http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
