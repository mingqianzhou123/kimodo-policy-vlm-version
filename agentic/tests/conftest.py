"""Make the agentic scripts, the simulator hooks and the shared package importable."""
import sys
from pathlib import Path

AGENTIC = Path(__file__).resolve().parents[1]
for path in (AGENTIC.parent, AGENTIC, AGENTIC/'scripts'):
    sys.path.insert(0, str(path))
