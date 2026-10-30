"""Search changes that can otherwise waste multi-day GPU experiments."""
import json
from pathlib import Path
import sys
import threading

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from eval_toolbench import matching_receipt, lock_evaluation_output
from toolbench_results import table_values
from tune_toolbench import Search, AdoptedTraining, reusable_training_prefix
from toolbench_eval_worker import parallel_validation


def row(step, primary, auxiliary=0.1):
    return dict(step=step, primary_l12=primary, auxiliary_all18=auxiliary,
                primary_episodes=1200, execution=50, diffusion=10)


def test_budget_shortlist_uses_confirmation_and_adds_explored_neighbor():
    confirmed = [row(100000, .11), row(200000, .10), row(150000, .09)]
    # A noisy screening estimate cannot outrank confirmation.
    explored = [row(50000, .99), row(100000, .05), row(150000, .07), row(200000, .08)]
    assert Search.budget_candidates(confirmed, explored) == [100000, 200000, 50000]
    assert Search.budget_candidates([row(150000, .11), row(100000, .10)], explored) == [150000, 100000]


def test_close_budget_results_request_second_training_seed():
    assert Search.near_tie(row(100000, .001), row(150000, .003))
    assert not Search.near_tie(row(100000, .1), row(150000, .2))


def test_table3_mapping_counts_and_duplicate_rejection():
    rows = []
    for scenario in ['G1BallMove', 'G1BallRetrieve', 'G1IceBreak']:
        for level in range(3):
            for mode in ['S', 'R']:
                n = level + (3 if mode == 'R' else 0)
                rows.append(dict(env_id=f'humanoidtoolbench/{scenario}-L{level}-{mode}',
                                 episodes=10, successes=n, success_rate=n / 10, status='completed'))
    summary = dict(failed=0, conditions=rows)
    assert table_values(summary) == pytest.approx([0, .1, .2, .3, .4, .5, .3, .25])
    rows[-1] = dict(rows[0])
    with pytest.raises(AssertionError):table_values(summary)


def test_training_prefix_reuse_only_ignores_stop_limit_with_explicit_lr_horizon():
    import copy
    original=dict(main=dict(max_steps=300000,seed=1000,sampling='task_uniform',data_root='e50'),
                  training=dict(scheduler=dict(decay_steps=200000),optimizer=dict(lr=1e-5)))
    requested=copy.deepcopy(original);requested['main']['max_steps']=200000
    assert reusable_training_prefix(original,requested,200000)
    assert original['main']['max_steps']==300000
    assert not reusable_training_prefix(original,requested,250000)
    for section,key,value in [('main','seed',1001),('main','sampling','window_uniform'),('main','data_root','e20')]:
        changed=copy.deepcopy(requested);changed[section][key]=value
        assert not reusable_training_prefix(original,changed,200000)
    changed=copy.deepcopy(requested);changed['training']['scheduler']['decay_steps']=250000
    assert not reusable_training_prefix(original,changed,200000)
    changed=copy.deepcopy(requested);changed['training']['optimizer']['lr']=3e-5
    assert not reusable_training_prefix(original,changed,200000)
    for config in [original,requested]:config['training']['scheduler'].pop('decay_steps')
    assert not reusable_training_prefix(original,requested,200000)


def test_reused_budget_inference_prefetch_does_not_change_budget_candidates(tmp_path):
    search=fake_search(tmp_path)
    name='budget_fit_task_uniform_seed1000_e50_b200000'
    (tmp_path/'training_reuse.json').write_text(json.dumps({name:{}}))
    (tmp_path/'plan.json').write_text(json.dumps(dict(prefetch_reused_budget_inference=True)))
    (tmp_path/'logs').mkdir()
    prefetched=threading.Event();started=threading.Event()
    def train(spec,step,execution,slot):
        if step==250000:
            started.set()
            assert prefetched.wait(8),'free group must prefetch while250k trains'
        else:assert started.wait(8)
        return tmp_path/f'checkpoint_{step}'
    def score(candidate,checkpoint,execution,diffusion,slot):
        if candidate.startswith('trained_budget'):
            assert execution in [30,20,10]
            prefetched.set()
        return dict(row(int(checkpoint.name.split('_')[-1]),.01),name=candidate)
    search.train=train;search.score=score
    specs=[dict(name='budget_fit_250k'),dict(name=name)]
    results=search.train_scores([(spec,step,50,10) for spec,step in zip(specs,[250000,200000])])
    assert len(results)==2
    assert [r['step'] for r in results]==[250000,200000]
    assert all(not r['name'].startswith('trained_budget') for r in results)


def test_training_reuse_normalizes_paths_like_training_loader(tmp_path,monkeypatch):
    import copy
    import hashlib
    import torch
    from omegaconf import OmegaConf
    import tune_toolbench
    search=fake_search(tmp_path)
    search.prepared_path=tmp_path/'data'
    spec=dict(name='budget_fit_200k',sampling='task_uniform',seed=1000,rates=[1e-5,3e-5,3e-5],max_steps=200000,decay_steps=200000)
    search.base=OmegaConf.create(dict(main=dict(data_root='data',text_cache='cache',save_root='log',max_steps=300000,seed=1000,sampling='task_uniform'),training=dict(scheduler=dict(decay_steps=200000),optimizer=dict(control_backbone_lr=1e-5,control_adapter_lr=3e-5,hand_lr=3e-5))))
    source=tmp_path/'source';cp=source/'checkpoint_200000';cp.mkdir(parents=True);(tmp_path/'logs').mkdir()
    config=OmegaConf.to_container(search.base,resolve=True)
    config['main']['data_root']=str(search.prepared_path)
    for k in ('text_cache','save_root'):config['main'][k]=str((tune_toolbench.REPO/config['main'][k]).resolve())
    (cp/'config.json').write_text(json.dumps(config))
    torch.save(dict(global_step=200000,model={'weight':torch.tensor([1.])}),cp/'training_state.pt')
    (source/'initialization.json').write_text(json.dumps(dict(policy_checkpoint=None,policy_419h_loaded=False)))
    reuse=dict(checkpoint=str(cp),checkpoint_config_sha256=hashlib.sha256((cp/'config.json').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256((cp/'training_state.pt').read_bytes()).hexdigest())
    (tmp_path/'training_reuse.json').write_text(json.dumps({spec['name']:reuse}))
    assert search.train(spec,200000,50,1)==cp
    assert json.loads((cp/'config.json').read_text())['main']['max_steps']==300000


def test_condition_resume_requires_exact_official_identity(tmp_path):
    settings = dict(episodes=2, seed_start=22000, max_steps=3000,
                    checkpoint_sha256='weights', policy_config_sha256='config',
                    adapter_sha256='adapter', execution_frames=50, diffusion_steps=10)
    info = dict(checkpoint_sha256='weights', config_sha256='config', adapter_sha256='adapter',
                inference=dict(execution_frames=50, diffusion_steps=10))
    result = dict(status='validated', episodes=2, seeds=[22000, 22001],
                  configuration=dict(max_episode_steps=3000), policy_info=info)
    path = tmp_path / 'run-first' / 'benchmark_result.json'
    path.parent.mkdir();path.write_text(json.dumps(result))
    assert matching_receipt(tmp_path, settings) == path
    for key, value in [('episodes', 3), ('seed_start', 22001), ('max_steps', 2000),
                       ('checkpoint_sha256', 'other'), ('adapter_sha256', 'other'),
                       ('execution_frames', 10), ('diffusion_steps', 20)]:
        assert matching_receipt(tmp_path, dict(settings, **{key: value})) is None


def fake_search(tmp_path):
    search = object.__new__(Search)
    search.root = tmp_path
    search.mutex = threading.RLock()
    search.stopping = False
    search.budget_report = lambda rows: None
    return search


def test_budget_pipeline_evaluates_before_training_finishes(tmp_path):
    search = fake_search(tmp_path)
    observed = threading.Event();finished = threading.Event();evaluated = []
    run = tmp_path / 'training' / 'candidate'
    def publish(step):
        cp = run / f'checkpoint_{step}';cp.mkdir(parents=True)
        (cp / 'training_state.pt').write_text('complete')
        (cp / 'config.json').write_text('{}')
    def train(spec, step, execution, slot):
        assert slot == 0 and step == 200000
        publish(100000)
        assert observed.wait(8), 'evaluation did not overlap training'
        publish(150000);publish(200000);finished.set()
    def score(name, checkpoint, execution, diffusion, slot):
        step = int(checkpoint.name.removeprefix('checkpoint_'))
        if step == 100000:
            assert slot == 1 and not finished.is_set()
            observed.set()
        evaluated.append(step)
        return row(step, .1)
    search.train = train;search.score = score
    result = search.budget_pipeline(dict(name='candidate'), [100000, 150000, 200000], 50, 10)
    assert [r['step'] for r in result] == [100000, 150000, 200000]
    assert sorted(evaluated) == [100000, 150000, 200000]


def test_budget_pipeline_failure_cancels_waiters(tmp_path):
    search = fake_search(tmp_path)
    def fail(*args):raise ValueError('training failed')
    search.train = fail
    with pytest.raises(ValueError, match='training failed'):
        search.budget_pipeline(dict(name='candidate'), [100000, 150000], 50, 10)
    assert search.stopping


def test_provisional_sampling_bypasses_all_sampling_evaluations(tmp_path, monkeypatch):
    import tune_toolbench
    search = fake_search(tmp_path)
    search.provisional_sampling = dict(sampling='task_uniform', reason='User prioritizes budget exploration')
    search.state = dict(trials=[])
    search.prepared = lambda execution: tmp_path
    phases = []
    search.phase = phases.append
    monkeypatch.setattr(tune_toolbench, 'read', lambda path: dict(passed=True, failed=0))
    def train_scores(jobs):
        assert len(jobs) == 1
        spec, step, execution, diffusion = jobs[0]
        assert spec['sampling'] == 'task_uniform' and step == 50000
        return [dict(spec=spec, step=step)]
    search.train_scores = train_scores
    def no_confirmation(*args):raise AssertionError('sampling confirmation must be skipped')
    search.confirm = no_confirmation
    def budget(spec, steps, execution, diffusion):
        assert spec['sampling'] == 'task_uniform'
        assert spec['rates'] == [1e-5, 3e-5, 3e-5]
        assert steps == [100000, 150000, 200000]
        raise RuntimeError('reached budget exploration')
    search.budget_pipeline = budget
    with pytest.raises(RuntimeError, match='reached budget exploration'):
        search.run()
    choices = json.loads((tmp_path / 'fixed_training_choices.json').read_text())
    assert choices['provisional'] is True and choices['sampling'] == 'task_uniform'
    assert phases == ['sampling_provisionally_fixed', 'budget_exploration_100k_150k_200k']


def test_gpu_lease_can_pass_to_next_job_during_cpu_validation(tmp_path):
    search=fake_search(tmp_path)
    search.gpu_condition=threading.Condition();search.gpu_busy=set()
    assert search.acquire_gpus(0)==[0]
    assert search.acquire_gpus(0)==[1]
    acquired=threading.Event();lease=[]
    def queued():lease.extend(search.acquire_gpus(1));acquired.set()
    thread=threading.Thread(target=queued);thread.start()
    assert not acquired.wait(.05)
    search.release_gpus([1]);assert acquired.wait(2);thread.join()
    assert lease==[1] and search.gpu_busy=={0,1}
    search.release_gpus([0,1]);assert search.acquire_gpus(0,all_gpus=True)==[0,1]
    search.release_gpus([0,1])


def test_adopted_training_requires_complete_optimizer_checkpoint(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint_200000';checkpoint.mkdir()
    child=AdoptedTraining(999999999,checkpoint)
    assert child.poll()==1
    (checkpoint/'training_state.pt').write_text('weights')
    assert AdoptedTraining(999999999,checkpoint).poll()==1
    (checkpoint/'optimizer_state.pt').write_text('optimizer and rng')
    assert AdoptedTraining(999999999,checkpoint).poll()==0


def test_parallel_validation_uses_original_outputs_and_restores_validator(tmp_path):
    from types import SimpleNamespace
    summaries=tmp_path/'summaries';summaries.mkdir()
    (summaries/'episode_20000.json').write_text(json.dumps(dict(step=100)))
    videos=tmp_path/'videos/episode_20000';videos.mkdir(parents=True)
    paths=[videos/f'{i}.mp4' for i in range(4)]
    for i,p in enumerate(paths):p.write_text(str(i))
    calls=[]
    def video(path,steps):calls.append((path,steps));return dict(path=str(path),hash=path.read_text())
    module=SimpleNamespace(validate_video=video)
    def run(root,config):return dict(videos=[module.validate_video(p,100) for p in paths],all_checks=True)
    expected=run(tmp_path,None);calls.clear()
    actual=parallel_validation(tmp_path,None,run,video,module,workers=2)
    assert actual==expected and len(calls)==4 and module.validate_video is video


def test_parallel_validation_propagates_bad_video(tmp_path):
    from types import SimpleNamespace
    (tmp_path/'summaries').mkdir();(tmp_path/'summaries/episode_1.json').write_text('{"step": 3}')
    videos=tmp_path/'videos/episode_1';videos.mkdir(parents=True);(videos/'bad.mp4').write_text('bad')
    def bad(*args):raise ValueError('corrupt video')
    module=SimpleNamespace(validate_video=bad)
    with pytest.raises(ValueError,match='corrupt video'):
        parallel_validation(tmp_path,None,lambda *a:None,bad,module,2)
    assert module.validate_video is bad


def test_duplicate_evaluator_is_rejected_before_starting_servers(tmp_path):
    owner=lock_evaluation_output(tmp_path)
    try:
        with pytest.raises(RuntimeError,match='Evaluation already running'):
            lock_evaluation_output(tmp_path)
    finally:owner.close()
    following=lock_evaluation_output(tmp_path);following.close()


def test_retiring_frozen_old_scheduler_never_resumes_its_queue(tmp_path,monkeypatch):
    import os,signal
    search=fake_search(tmp_path)
    search.adoption=dict(old_supervisor=999999998,training_pid=123,target_step=200000)
    search.lockfile=(tmp_path/'migration.lock').open('a')
    search.persist=lambda:None
    # A simulated stopped scheduler vanishes after SIGKILL, so none of its
    # waiting worker threads can run a second copy of the evaluation queue.
    original_exists=Path.exists;original_read=Path.read_text;sent=[]
    def exists(path):
        if str(path)=='/proc/999999998':return not sent
        return original_exists(path)
    def read_text(path,*args,**kwargs):
        if str(path)=='/proc/999999998/stat':return '999999998 (python) T 1 0'
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'exists',exists);monkeypatch.setattr(Path,'read_text',read_text)
    monkeypatch.setattr(os,'kill',lambda pid,sig:sent.append((pid,sig)))
    search.retire_old_supervisor()
    assert sent==[(999999998,signal.SIGKILL)] and search.adoption['retired']
    search.lockfile.close()
