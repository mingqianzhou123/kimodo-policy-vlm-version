"""Keep the repository small and machine-independent.

Machine setup lives in scripts/sensecore/ and in untracked env.local.sh files;
experiment outputs, weights and caches live outside git.
"""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MACHINE_PATH = re.compile(r"(?<![\w.$}])/(root|mnt|home|Users)/\w")
ADDRESS = re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(\d{1,3}\.){3}\d{1,3}\b")
CODE = {".py", ".sh", ".yaml", ".yml", ".toml", ".json"}
OUTPUT_DIRS = {"runs", "outputs", "log", "checkpoints", "archives", "snapshots"}


def tracked():
    try:
        out = subprocess.check_output(["git", "-C", str(REPO), "ls-files", "-z"], text=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [Path(name) for name in out.split("\0") if name]


def test_code_names_no_machine():
    offenders = []
    for path in tracked():
        if path.suffix not in CODE or path.parts[:2] == ("scripts", "sensecore") or path.name == Path(__file__).name:
            continue
        for number, line in enumerate((REPO/path).read_text(errors="ignore").splitlines(), 1):
            if MACHINE_PATH.search(line) or ADDRESS.search(line):
                offenders.append(f"{path}:{number}: {line.strip()[:100]}")
    assert not offenders, "machine-specific paths or addresses in tracked code:\n" + "\n".join(offenders)


def test_no_outputs_or_large_files_are_tracked():
    files = tracked()
    outputs = [str(p) for p in files if OUTPUT_DIRS & set(p.parts[:-1])]
    large = [f"{p} ({(REPO/p).stat().st_size >> 10} KB)" for p in files if (REPO/p).stat().st_size > 1 << 20]
    assert not outputs, "experiment outputs are tracked:\n" + "\n".join(outputs)
    assert not large, "files over 1 MB are tracked:\n" + "\n".join(large)
