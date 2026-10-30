"""Revision10: scratch budget comparison with canonical ToolBench episode seeds."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import statistics
import time

from omegaconf import OmegaConf
from eval_toolbench import REPO, stop, stop_many
from tune_toolbench import Search, atomic, now, read
from toolbench_results import write_stage_reports


def convergence(rows, window=20000, tolerance=.01):
    """Three complete adjacent windows; plateau is a descriptive stopping rule."""
    end=rows[-1]['step']
    windows=[[r['loss'] for r in rows if end-(3-i)*window < r['step'] <= end-(2-i)*window]
             for i in range(3)]
    complete=all(len(w)>=.95*window/10 for w in windows)
    means=[statistics.mean(w) if w else None for w in windows]
    changes=[means[i+1]/means[i]-1 for i in range(2)] if complete else []
    return dict(last_step=end,window_steps=window,window_means=means,relative_changes=changes,
                tolerance=tolerance,complete_windows=complete,
                plateau=complete and all(abs(v)<tolerance for v in changes),
                interpretation='Descriptive training-loss plateau only; success rate determines policy ranking.')


class ConvergenceSearch(Search):
    def __init__(self,root):
        super().__init__(root)
        plan=read(self.root/'plan.json');assert plan['registered_revision']==10
        self.reference=Path(plan['convergence_experiment']['reference_checkpoint'])
        reference=read(self.reference/'config.json')
        # Copy the actual completed250k recipe, not a separately maintained YAML.
        self.base=OmegaConf.create(copy.deepcopy(reference))
        self.prepared_path=Path(reference['main']['data_root'])
        assert reference['main']['seed']==1000 and reference['main']['batch_size']==128
        assert reference['main']['micro_batch_size']==32
        assert reference['main']['require_scratch'] is True
        assert reference['training']['scheduler']['decay_steps']==250000
        assert read(self.reference.parent/'initialization.json')['policy_checkpoint'] is None
        assert read(self.reference.parent/'initialization.json')['policy_419h_loaded'] is False
        self.state['convergence_plan']=plan['convergence_experiment']
        self.state['convergence_results']=self.state.get('convergence_results',{})
        self.persist()

    def candidate(self,budget,slot):
        name=f'convergence_task_uniform_seed1000_e50_b{budget}'
        spec=dict(name=name,sampling='task_uniform',seed=1000,rates=[1e-5,3e-5,3e-5],
                  max_steps=budget,decay_steps=budget)
        checkpoint=self.train(spec,budget,50,slot)
        actual=read(checkpoint/'config.json');expected=read(self.reference/'config.json')
        expected['main']['max_steps']=budget;expected['training']['scheduler']['decay_steps']=budget
        assert actual==expected,'Only max_steps and explicit cosine horizon may differ from250k'
        rows=[json.loads(line) for line in (checkpoint.parent/'metrics.jsonl').read_text().splitlines()]
        evidence=convergence(rows)
        atomic(checkpoint.parent/'convergence.json',evidence)
        with self.mutex:
            self.state['convergence_results'][str(budget)]=dict(checkpoint=str(checkpoint),**evidence)
            self.persist()
            write_stage_reports(self.root,self.state['trials'])
        return self.official_score(budget,checkpoint,slot)

    def official_score(self,budget,checkpoint,slot):
        return self.score(f'convergence_b{budget}_seed1000_official_e100',checkpoint,50,10,slot,
                          stage='official_budget_comparison',episodes=100,seed_start=10000)

    def run(self):
        self.phase('convergence_scratch_300k_400k_500k_official100')
        write_stage_reports(self.root,self.state['trials'])
        # Launch250k evaluation and300k training first. Two extra workers let
        # CPU-only receipt validation overlap the next4-GPU training/evaluation.
        def guarded(fn,*args):
            try:return fn(*args)
            except BaseException:
                self.stopping=True
                raise
        with ThreadPoolExecutor(4) as pool:
            try:
                futures=[pool.submit(guarded,self.official_score,250000,self.reference,0),
                         pool.submit(guarded,self.candidate,300000,1)]
                deadline=time.monotonic()+120
                while len(self.state['active'])<2 and time.monotonic()<deadline:
                    for f in futures:
                        if f.done():f.result()
                    time.sleep(.5)
                futures += [pool.submit(guarded,self.candidate,400000,0),
                            pool.submit(guarded,self.candidate,500000,1)]
                for f in futures:f.result()
            except BaseException:
                self.stopping=True
                stop_many(self.children,grace=15)
                raise
        budget=500000
        while not self.state['convergence_results'][str(budget)]['plateau']:
            budget+=100000
            self.phase(f'convergence_extension_scratch_{budget}_official100')
            self.candidate(budget,0)
        self.phase('convergence_budget_comparison_complete')
        official=[r for r in self.state['trials'] if r['stage']=='official_budget_comparison']
        best=max(official,key=self.rank)
        atomic(self.root/'convergence_selection.json',dict(finished=now(),best=best,
            budgets=sorted(int(v) for v in self.state['convergence_results']),
            selection='primary L1/L2 mean, secondary all18 mean',
            canonical_seeds_used_for_budget_selection=True,independent_final_test=False))
        write_stage_reports(self.root,self.state['trials'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();os.chdir(REPO);search=ConvergenceSearch(a.root)
    def interrupted(signum,frame):
        search.stopping=True
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:search.run()
    except BaseException as exc:
        search.state.update(phase='failed',error=f'{type(exc).__name__}: {exc}');search.persist();raise
    finally:
        search.stopping=True
        for child in reversed(search.children):stop(child)


if __name__=='__main__':main()
