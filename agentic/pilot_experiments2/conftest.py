"""Make the shared package, the simulator hooks, the shared scripts and each experiment's own scripts importable."""
import sys
from pathlib import Path

PILOTS = Path(__file__).resolve().parent
AGENTIC = PILOTS.parent
for path in (AGENTIC.parent, AGENTIC, AGENTIC/'scripts', *sorted(p for p in PILOTS.iterdir() if p.is_dir())):
    sys.path.insert(0, str(path))
