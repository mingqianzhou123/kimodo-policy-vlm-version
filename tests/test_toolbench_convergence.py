"""Guard the new experiment's seed identity, scratch recipe and stopping rule."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_toolbench_convergence import ConvergenceSearch, convergence


def records(means):
    return [dict(step=(i*2000+j+1)*10,loss=value)
            for i,value in enumerate(means) for j in range(2000)]


def test_plateau_requires_complete_windows_and_two_small_changes():
    assert convergence(records([.02,.0199,.01985]))['plateau']
    assert not convergence(records([.03,.025,.02]))['plateau']
    assert not convergence(records([.02,.025,.03]))['plateau']
    assert not convergence(records([.02,.0199,.01985])[-100:])['plateau']
    assert not convergence(records([.03,.02,.02]))['plateau']


def test_new_budget_evaluation_uses_canonical_100_episode_identity():
    search=object.__new__(ConvergenceSearch);seen={}
    def score(*args,**kwargs):seen.update(args=args,kwargs=kwargs);return 'ok'
    search.score=score
    assert search.official_score(300000,Path('/checkpoint_300000'),1)=='ok'
    assert seen['args'][2:]==(50,10,1)
    assert seen['kwargs']==dict(stage='official_budget_comparison',episodes=100,seed_start=10000)
