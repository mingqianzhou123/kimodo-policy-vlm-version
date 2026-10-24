"""Serve a Kimodo checkpoint to HumanoidArena's evaluator (the original server's protocol).

    python scripts/serve_arena.py --checkpoint log/.../checkpoint_200000 --text-cache <cache> \
        --controller sonic --port 18080

Formal runs used 15 executed frames, RTC off, and a seed per episode via /reset.
"""

from __future__ import annotations

import argparse

import torch

from kimodo_vla.benchmarks.arena.policy import ROOT_BOUNDARY, make_kimodo_policy, routes
from kimodo_vla.benchmarks.server import PLAIN_JSON, make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--controller", choices=sorted(ROOT_BOUNDARY), default="sonic")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--execution-frames", type=int, default=15)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--rtc", action="store_true")
    args = parser.parse_args()

    policy = make_kimodo_policy(
        args.checkpoint, args.text_cache, controller=args.controller, device=args.device,
        dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
        execution_frames=args.execution_frames, diffusion_steps=args.diffusion_steps, rtc=args.rtc,
    )
    server = make_server(routes(policy), args.host, args.port, PLAIN_JSON)
    print(f"Serving on http://{args.host}:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
