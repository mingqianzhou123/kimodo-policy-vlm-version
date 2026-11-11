"""Rebuild a legacy ToolBench cache from raw data using the current conversion.

Always use a new output directory. Existing RGB/target tensors are not reused.
The fullbody recipe normally calls prepare_toolbench.py directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='legacy cache, kept for provenance')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execution-frames', type=int, default=20, help='inference only; measured histories are independent of execution length')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--manifest', type=Path, default=Path('/root/workspace/ToolBook/manifest.json'))
    args = parser.parse_args()
    if args.out.resolve() == args.source.resolve():
        parser.error('--out must differ from --source; preserve the legacy experiment inputs')
    if not 1 <= args.execution_frames <= 50:
        parser.error('--execution-frames must be in [1, 50]')
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_toolbench.py')),
                    '--data-dir', str(args.manifest.parent / 'raw'), '--out', str(args.out),
                    '--manifest', str(args.manifest), '--workers', str(args.workers)], check=True)


if __name__ == '__main__':
    main()
