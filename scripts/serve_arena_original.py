"""Evaluator entry point enforcing the original HumanoidArena SONIC execution settings."""

import sys

from serve_arena import main


if __name__ == "__main__":
    sys.argv.extend(["--timing", "original", "--controller", "sonic", "--dtype", "fp32",
                     "--diffusion-steps", "10", "--execution-frames", "15", "--rtc-overlap-frames", "0"])
    main()
