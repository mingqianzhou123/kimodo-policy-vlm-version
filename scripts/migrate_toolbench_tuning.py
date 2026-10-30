"""Drain revision4 sampling evaluation while revision5 confirms a guaranteed finalist.

Only the old scheduler is SIGSTOPped. Its last official evaluator continues on
group0. The guaranteed top-two window sampler is confirmed on otherwise idle
group1, then the other finalist uses group0. Finally exec the normal scheduler;
all completed evaluations are identity checked and reused.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import sys
import time

from eval_toolbench import stop_many
from tune_toolbench import Search, atomic, now, read


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0] != 'Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--old-supervisor',type=int,required=True)
    p.add_argument('--old-evaluator',type=int,required=True)
    a=p.parse_args();a.root=a.root.resolve()
    old_command=Path(f'/proc/{a.old_evaluator}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'scripts/eval_toolbench.py' in old_command and 'level_balanced_simple1_seed1000_e50' in old_command
    assert str(a.root) in old_command and '--gpus 0,1,2,3' in old_command
    search=Search(a.root,migration_from=a.old_supervisor)
    assert read(a.root/'plan.json')['registered_revision']==5
    window='window_uniform_simple1_seed1000_e50_s50000_screen'
    task='task_uniform_simple1_seed1000_e50_s50000_screen'
    level='level_balanced_simple1_seed1000_e50_s50000_screen'
    initial={r['name']:r for r in search.state['trials']}
    assert search.rank(initial[window])>=search.rank(initial[task]), 'window must be guaranteed to finish in top two'
    search.phase('sampling_confirmation_migration')
    retired=False

    def retire():
        nonlocal retired
        if retired:return
        if alive(a.old_supervisor):
            os.kill(a.old_supervisor,signal.SIGTERM)
            os.kill(a.old_supervisor,signal.SIGCONT)
            deadline=time.monotonic()+60
            while alive(a.old_supervisor) and time.monotonic()<deadline:time.sleep(.2)
            if alive(a.old_supervisor):raise RuntimeError('old scheduler did not retire cleanly; no concurrent restart')
        retired=True
        search.persist()

    def interrupted(signum,frame):
        search.stopping=True
        raise KeyboardInterrupt(f'migration signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)

    def confirm_window():
        r=initial[window]
        return search.score('sampling_confirm_'+window,Path(r['checkpoint']),50,10,1,stage='confirm')

    def drain_then_confirm():
        out=a.root/'evaluation'/level
        latest=-1;progress=time.monotonic()
        while alive(a.old_evaluator):
            if search.stopping:raise RuntimeError('migration stopping')
            count=len(list(out.glob('*/run-*/summaries/episode_*.json')))
            if count>latest:latest=count;progress=time.monotonic()
            if time.monotonic()-progress>2400:raise RuntimeError('old evaluation made no episode progress for40min')
            with search.mutex:
                search.state['active']['0']=dict(job='eval_'+level,pid=a.old_evaluator,episodes=count,
                    validated=len(list(out.glob('*/run-*/benchmark_result.json'))),migration='draining revision4 evaluator')
                search.persist()
            time.sleep(5)
        assert (out/'selection_score.json').exists(),'old evaluator failed before official score was validated'
        retire()
        with search.mutex:search.state['active'].pop('0',None)
        cp=a.root/'training'/'level_balanced_simple1_seed1000_e50'/'checkpoint_50000'
        level_row=search.score(level,cp,50,10,0)
        ordered=sorted([initial[window],initial[task],level_row],key=search.rank,reverse=True)[:2]
        assert any(r['name']==window for r in ordered)
        second=next(r for r in ordered if r['name']!=window)
        return search.score('sampling_confirm_'+second['name'],Path(second['checkpoint']),50,10,0,stage='confirm')

    def guarded(fn):
        try:return fn()
        except BaseException:
            search.stopping=True;raise
    try:
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(guarded,fn) for fn in [confirm_window,drain_then_confirm]]
            confirmations=[f.result() for f in futures]
        atomic(a.root/'migration_v5_complete.json',dict(completed=now(),old_supervisor=a.old_supervisor,
            old_evaluator=a.old_evaluator,confirmations=confirmations,
            note='No in-flight episodes interrupted; guaranteed finalist confirmed on free GPU group.'))
        search.phase('migration_complete')
        search.lockfile.close()
        os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).with_name('tune_toolbench.py')),'--root',str(a.root)])
    except BaseException as exc:
        search.stopping=True
        stop_many(search.children)
        retire()
        search.state.update(phase='failed',error=f'migration: {type(exc).__name__}: {exc}')
        search.persist()
        raise


if __name__=='__main__':main()
