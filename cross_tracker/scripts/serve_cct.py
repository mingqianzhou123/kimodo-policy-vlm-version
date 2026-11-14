"""Serve the CCT variant with fixed training context and the frozen box evaluation settings."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parent)]
from cct_eval import load_planner
from kimodo_vla.benchmarks.arena.policy import ROOT_BOUNDARY, make_policy, routes
from kimodo_vla.benchmarks.arena.adapter import CONTROL_FPS
from kimodo_vla.benchmarks.server import PLAIN_JSON, make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--controller", choices=["sonic", "twist2"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/box-screen.json").read_text())["evaluation_policy"]
    assert settings["timing"] == "exact" and settings["dtype"] == "fp32"
    planner = load_planner(args.checkpoint, args.controller, args.text_cache, device=args.device,
                           control_fps=CONTROL_FPS, execution_frames=settings["execution_frames_30hz"],
                           diffusion_steps=settings["diffusion_steps"], rtc_overlap_frames=settings["rtc_overlap_frames"],
                           root_boundary=ROOT_BOUNDARY[args.controller])
    server = make_server(routes(make_policy(planner)), "127.0.0.1", args.port, PLAIN_JSON)
    print(f"CCT {args.controller} serving on 127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
