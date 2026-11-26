"""The adapter's look-ahead lag: unset is the protocol's 8 frames; KVLA_HEFT_LAG=0 only when asked for."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # agentic/, for the arena package


def load(monkeypatch, value=None):
    if value is None:
        monkeypatch.delenv('KVLA_HEFT_LAG', raising=False)
    else:
        monkeypatch.setenv('KVLA_HEFT_LAG', value)
    import arena.heft_bridge as bridge
    return importlib.reload(bridge)


def test_unset_is_the_protocol_lag(monkeypatch):
    assert load(monkeypatch).LAG == 8


def test_zero_when_set(monkeypatch):
    assert load(monkeypatch, '0').LAG == 0
    load(monkeypatch)   # leave the module as the protocol has it
