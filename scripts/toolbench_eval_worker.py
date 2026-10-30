"""Run official simulation/scoring with a CPU validation pipeline.

The benchmark files and success predicate remain unchanged. Every video is
decoded by the original validator, in a bounded thread pool, before the original
validate_run performs all coverage, camera, configuration and provenance checks.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def atomic(path, value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


@contextmanager
def validation_slot():
    """At most24 validators (48 video threads) across this search's evaluators."""
    directory=Path(os.environ['TOOLBENCH_VALIDATION_ROOT']);directory.mkdir(exist_ok=True,parents=True)
    lock=None
    try:
        while lock is None:
            for i in range(24):
                candidate=(directory/f'{i}.lock').open('a')
                try:fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:candidate.close();continue
                lock=candidate;break
            if lock is None:time.sleep(.25)
        yield
    finally:
        if lock is not None:lock.close()


def parallel_validation(root, config, original_run, original_video, module, workers):
    """Only scheduling changes: use each original video result exactly once."""
    root=Path(root);jobs=[]
    for path in sorted((root/'summaries').glob('episode_*.json')):
        summary=json.loads(path.read_text())
        steps=summary.get('step')
        if type(steps) is not int:continue  # original validator emits its error
        folder=root/'videos'/path.stem
        for video in sorted(folder.glob('*.mp4')):
            jobs.append((video,steps))
    def check(job):
        path,steps=job;before=path.stat()
        value=original_video(path,steps)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise RuntimeError(f'Video changed during validation: {path}')
        return ((str(path),steps),(value,after.st_size,after.st_mtime_ns))
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:cache=dict(pool.map(check,jobs))
    def checked_video(path,steps):
        value,size,mtime=cache[(str(path),steps)];stat=path.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(size,mtime):raise RuntimeError(f'Video changed: {path}')
        return value
    module.validate_video=checked_video
    try:receipt=original_run(root,config)
    finally:module.validate_video=original_video
    atomic(root/'validation_timing.json',dict(video_workers=workers,video_count=len(jobs),
        seconds=time.monotonic()-started,original_video_validator=True,all_original_run_checks=True))
    return receipt


def main():
    # Invoke through the official runtime wrapper first, so its re-exec never
    # drops these hooks. We do not bypass GPU/EGL selection or asset validation.
    assert os.environ.get('HUMANOIDTOOLBENCH_RUNTIME_WRAPPER')=='1'
    import humanoidtoolbench.cli.eval_decoupled_wbc as simulation
    import humanoidtoolbench.evals.public_validation as validation
    from humanoidtoolbench.policies.http_client import release_sessions
    original_simulation=simulation.run_eval
    original_run,original_video=validation.validate_run,validation.validate_video
    marker=Path(os.environ['TOOLBENCH_SIMULATION_MARKER'])
    workers=int(os.environ.get('TOOLBENCH_VIDEO_WORKERS','2'))
    assert 1<=workers<=4
    def run_simulation(config):
        result=original_simulation(config)
        # run_eval has closed environments/video writers and atomically written
        # evaluation_result.json. Release this exact session before another
        # condition can acquire the policy server; no episode is interleaved.
        release_sessions()
        atomic(marker,dict(simulation_complete=True,run_dir=str(result.eval_dir),
            pid=os.getpid(),time=time.time()))
        return result
    simulation.run_eval=run_simulation
    def validate(root,config):
        with validation_slot():
            return parallel_validation(root,config,original_run,original_video,validation,workers)
    validation.validate_run=validate
    import humanoidtoolbench.cli.public_eval as official
    official.main(sys.argv[1:])


if __name__=='__main__':main()
