"""Build one page: how far each planned experiment is, and every run's numbers, curves and video.

The plan (experiments, their steps and which run names count toward each step) is
agentic/configs/experiments.json. Runs are read from RUNS; the page is written to RUNS/overview/.
"""
import argparse
import fnmatch
import functools
import hashlib
import http.server
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from paths import AGENTIC, RUNS

LIMIT_CM = 3
STATUS = {'completed': '完成', 'failed': '失败'}
SWITCHES = {'reference_source': '参考来源', 'variant': '衔接', 'support_constraints': '支撑约束',
            'hand_constraints': '手部约束', 'trajectory_projection': '轨迹投影', 'dynamics_profile': '动力学配置',
            'root_support': '根部支撑', 'target_source': '目标来源', 'vlm_localizer': '定位器'}
# What an experiment's record holds: the first five are written before it runs, the last four after.
RECORD = ('question', 'success', 'prediction', 'judgment', 'stop', 'conclusion', 'confidence', 'changed', 'next')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def rows(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:  # a run killed mid-write leaves a cut last line
            pass
    return out


def optional(make):
    """Old runs miss fields or log other shapes; whatever cannot be computed is left out."""
    try:
        return make()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def numbers(values):
    values = np.asarray(values, float)
    if values.ndim > 1:
        return [numbers(v) for v in values]
    return [round(float(v), 3) if math.isfinite(v) else None for v in values]


def rate(wins, total):
    """A success rate with its 95% Wilson interval; with few runs the interval is wide, and that is worth seeing."""
    if wins is None or not total:
        return None
    z, p = 1.96, wins/total
    scale = 1+z*z/total
    middle, half = (p+z*z/(2*total))/scale, z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/scale
    return {'wins': wins, 'total': total, 'low': round(max(middle-half, 0), 4), 'high': round(min(middle+half, 1), 4)}


def measure(run):
    """Wrist and root errors of one run in cm: the post-arrival numbers and the curves behind them.

    Same definitions as report_settled.measure: frames after target.json's arrival_time, actual
    minus consumed reference; a run passes when both wrists stay within 3 cm on every such frame,
    of the reference and, where the run has marker truth, of the markers too.
    """
    steps = rows(Path(run)/'execution.jsonl')
    target = read(Path(run)/'target.json') or {}
    truth = read(Path(run)/'visual-evaluation-truth.json') or {}

    def logged():
        after = [s['after'] for s in steps]
        return (after, np.array([a['time'] for a in after], float), np.array([a['wrist_pos'] for a in after], float),
                np.array([s['reference_wrist_world'] for s in steps], float))
    found = optional(logged) if steps else None
    if found is None:
        return None
    after, t, wrist, reference = found
    tracking = optional(lambda: 100*np.linalg.norm(wrist-reference, axis=-1).reshape(len(t), 2))
    if tracking is None:
        return None
    root = optional(lambda: 100*(np.array([a['root_pos'] for a in after], float)
                                 - np.array([s['reference_root_world'] for s in steps], float)).reshape(len(t), 3))
    marker = optional(lambda: np.array(truth['targets_world'], float).reshape(2, 3))
    goal = optional(lambda: np.array(target['targets_world'], float)[:, 0].reshape(2, 3))
    aim = marker if marker is not None else goal
    to_aim = None if aim is None else optional(lambda: 100*np.linalg.norm(wrist-aim, axis=-1).reshape(len(t), 2))
    start = optional(lambda: float(steps[0]['before']['time']))
    start = t[0] if start is None else start
    arrival = target.get('arrival_time')
    arrival = arrival if isinstance(arrival, (int, float)) and math.isfinite(arrival) else None
    late = t > arrival+1e-6 if arrival is not None else np.zeros(len(t), bool)

    metrics = {'frames': len(steps), 'late_frames': int(late.sum())}
    if late.any():
        metrics['wrist_max_cm'] = numbers(tracking[late].max(axis=0))
        metrics['wrist_mean_cm'] = numbers(tracking[late].mean(axis=0))
        passed = bool(np.all(tracking[late] <= LIMIT_CM))
        if marker is not None and to_aim is not None:
            metrics['marker_max_cm'] = numbers(to_aim[late].max(axis=0))
            passed = passed and bool(np.all(to_aim[late] <= LIMIT_CM))
        if root is not None:
            metrics['root_mean_cm'] = numbers([np.linalg.norm(root[late], axis=-1).mean()])[0]
        metrics['pass'] = passed
    series = {'t': numbers(t-start), 'arrival': None if arrival is None else round(arrival-start, 3),
              'wrist': numbers(tracking.T)}
    if to_aim is not None:
        series['aim'] = numbers(to_aim.T)
        series['aim_is_marker'] = marker is not None
    if root is not None:
        series['root'] = numbers(np.linalg.norm(root, axis=-1))
        series['root_z'] = numbers(root[:, 2])
    return {'metrics': metrics, 'series': series}


def playable(source, out):
    """Where the page finds a run's video, and whether a browser can play it.

    Browsers play H.264. The simulator writes MPEG-4 Part 2, so those get one H.264 copy under
    out/videos/, made once and reused; without ffmpeg the page links the original instead.
    """
    original = {'url': Path(os.path.relpath(source, out)).as_posix(), 'plays': b'avc1' in source.read_bytes()}
    if original['plays'] or not shutil.which('ffmpeg'):
        return original
    copy = out/'videos'/(source.parents[1].name+'.mp4')
    if not copy.exists() or copy.stat().st_mtime < source.stat().st_mtime:
        copy.parent.mkdir(parents=True, exist_ok=True)
        part = copy.with_name(copy.stem+'.part.mp4')
        done = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(source), '-an', '-c:v', 'libx264',
                               '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', str(part)])
        if done.returncode:
            part.unlink(missing_ok=True)
            return original
        part.replace(copy)
    return {'url': copy.relative_to(out).as_posix(), 'plays': True}


def describe(run, out):
    """Everything the page shows about one run directory. Missing pieces stay None."""
    meta = read(run/'run.json') or {}
    stamp = re.search(r'(\d{4})(\d{2})(\d{2})', run.name)
    started, ended = meta.get('started_unix'), meta.get('ended_unix')
    date = '-'.join(stamp.groups()) if stamp else optional(lambda: time.strftime('%Y-%m-%d', time.gmtime(started)))
    version = None
    if meta.get('git_commit'):
        version = {'commit': str(meta['git_commit'])[:8], 'dirty': bool(meta.get('git_dirty'))}
    elif meta.get('source_sha256'):
        blob = json.dumps(meta['source_sha256'], sort_keys=True).encode()
        version = {'fingerprint': hashlib.sha256(blob).hexdigest()[:8]}
    switches = {label: meta[key] for key, label in SWITCHES.items() if meta.get(key) is not None}
    if meta.get('archived_goal_sha256'):
        switches['起步'] = '站稳后'
    episodes = (read(run/'episodes.json') or {}).get('episodes')
    episodes = episodes if isinstance(episodes, list) and episodes else None
    episode = read(run/'episode.json') or {}
    success = episode.get('success')
    grasp = read(run/'grasp-result.json')
    if isinstance(grasp, dict) and isinstance(grasp.get('success'), bool):
        # Grasp runs are judged by the physical criterion; Arena's own verdict is about the shelf.
        success = grasp['success']
    if meta.get('success_rate_evaluation') is False:  # the run never tried the task, so its "failure" is not a result
        wins = None
    elif episodes and all(isinstance(e, dict) and isinstance(e.get('success'), bool) for e in episodes):
        wins = sum(e['success'] for e in episodes)
    else:
        wins = int(success) if isinstance(success, bool) and not (episodes and len(episodes) > 1) else None
    videos = sorted((run/'videos').glob('*.mp4')) if (run/'videos').is_dir() else []
    measured = measure(run) or {}
    verdict = (read(run/'motion-result.json') or {}).get('pass')  # written by analyze_motion.py for moving goals
    if isinstance(verdict, bool) and measured.get('metrics') is not None:
        measured['metrics']['pass'] = verdict
    return {'name': run.name, 'date': date, 'version': version, 'switches': switches,
            'status': STATUS.get(meta.get('status'), '未收尾' if meta.get('status') else '未知'), 'error': meta.get('error'),
            'minutes': optional(lambda: round((ended-started)/60, 1)),
            'episodes': len(episodes) if episodes else 1, 'wins': wins, 'seed': episode.get('seed'),
            'video': playable(videos[0], out) if videos else None,
            'm': measured.get('metrics'), 's': measured.get('series')}


def summarize(runs):
    """What a set of runs says together, and what in it needs a second look."""
    measured = [r['m'] for r in runs if r['m']]
    judged = [m['pass'] for m in measured if m.get('pass') is not None]
    scored = [r for r in runs if r['wins'] is not None and r['status'] == STATUS['completed']]
    def worst(key):
        values = [max(m[key]) for m in measured if m.get(key)]
        return max(values) if values else None
    roots = [m['root_mean_cm'] for m in measured if m.get('root_mean_cm') is not None]
    notes = [f"{r['name']} {r['status']}" + (f"：{r['error']}" if r['error'] else '')
             for r in runs if r['status'] != STATUS['completed']]
    versions = {json.dumps(r['version'], sort_keys=True) for r in runs if r['version']}
    if len(versions) > 1:
        notes.append(f'这些实验来自 {len(versions)} 个不同的代码版本，比较时先确认差别只在想比的那一项')
    silent = sum(r['status'] == STATUS['completed'] and not r['m'] for r in runs)
    if silent:
        notes.append(f'{silent} 个实验跑完了但没有逐步日志，没有数字')
    return {'runs': len(runs), 'passed': rate(sum(judged), len(judged)),
            'success': rate(sum(r['wins'] for r in scored), sum(r['episodes'] for r in scored)),
            'worst_wrist_cm': worst('wrist_max_cm'), 'worst_marker_cm': worst('marker_max_cm'),
            'root_mean_cm': round(sum(roots)/len(roots), 3) if roots else None, 'notes': notes}


def progress(plan, runs):
    """Each planned experiment with what has been run so far.

    A step is done / planned, where done is either typed into the plan or the episodes of finished
    runs whose name matches the step's pattern. An experiment's progress is the mean over its steps;
    an optional step only counts once it has been started. Progress says how much has been run,
    not whether it passed: that is each step's success rate and the conclusion typed into the plan.
    """
    experiments = []
    for experiment in plan['experiments']:
        steps = []
        for step in experiment['steps']:
            matched = [r for r in runs if 'runs' in step and fnmatch.fnmatch(r['name'], step['runs'])]
            finished = [r for r in matched if r['status'] == STATUS['completed']]
            planned = step.get('planned', 1)
            done = sum(r['episodes'] for r in finished) if 'runs' in step else step.get('done', 0)
            steps.append({'name': step['name'], 'planned': planned, 'done': done, 'share': min(done/planned, 1),
                          'optional': bool(step.get('optional')), 'pattern': step.get('runs'),
                          'predict': step.get('predict'), 'summary': summarize(matched),
                          'runs': [r['name'] for r in matched]})
        counted = [s for s in steps if not s['optional'] or s['done']]
        basis = [r for r in runs if any(fnmatch.fnmatch(r['name'], p) for p in experiment.get('basis', []))]
        experiments.append({**{k: experiment.get(k) for k in ('id', 'title', 'when', 'owner', *RECORD)},
                            'steps': steps, 'basis': [r['name'] for r in basis], 'basis_summary': summarize(basis),
                            'steps_done': sum(s['share'] == 1 for s in counted), 'steps_counted': len(counted),
                            'progress': sum(s['share'] for s in counted)/len(counted) if counted else 0})
    return experiments


def stage(name):
    """Runs made before the plan are grouped by topic and day: settled-20261123-r1-far-left -> settled-20261123."""
    found = re.match(r'([a-z0-9]+).*?-(\d{8})', name)
    return '-'.join(found.groups()) if found else name


def collect(runs_dir, out, plan):
    out.mkdir(parents=True, exist_ok=True)
    folders = [d for d in sorted(runs_dir.iterdir()) if d.is_dir() and d != out]
    runs = [describe(d, out) for d in folders if (d/'run.json').is_file()]
    experiments = progress(plan, runs)
    planned = {name for e in experiments for s in e['steps'] for name in s['runs']}
    groups = {}
    for run in runs:
        if run['name'] not in planned:
            groups.setdefault(stage(run['name']), []).append(run['name'])
    by_name = {r['name']: r for r in runs}
    groups = [{'id': key, 'date': by_name[names[0]]['date'], 'runs': names,
               'summary': summarize([by_name[n] for n in names])} for key, names in groups.items()]
    groups.sort(key=lambda g: (g['date'] or '', g['id']), reverse=True)
    return {'generated': time.strftime('%Y-%m-%d %H:%M'), 'plan': plan.get('plan'), 'limit_cm': LIMIT_CM,
            'progress': sum(e['progress'] for e in experiments)/len(experiments) if experiments else 0,
            'experiments': experiments, 'groups': groups, 'runs': by_name,
            'other_folders': len(folders)-len(runs)}


def write(data, out):
    template = (Path(__file__).with_suffix('.html')).read_text()
    blob = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')
    page = out/'index.html'
    page.write_text(template.replace('__DATA__', blob))
    return page


class Ranged(http.server.SimpleHTTPRequestHandler):
    """Static files with byte ranges, which a browser needs to seek inside a video."""

    def send_head(self):
        path = Path(self.translate_path(self.path))
        asked = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        if not asked or not any(asked.groups()) or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        first, last = asked.groups()
        start = int(first) if first else max(size-int(last), 0)
        end = min(int(last), size-1) if first and last else size-1
        if start > end:
            self.send_error(416)
            return None
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end-start+1))
        self.end_headers()
        self.left = end-start+1
        source = path.open('rb')
        source.seek(start)
        return source

    def copyfile(self, source, outputfile):
        if getattr(self, 'left', None) is None:
            return super().copyfile(source, outputfile)
        outputfile.write(source.read(self.left))
        self.left = None

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, default=AGENTIC/'configs/experiments.json')
    p.add_argument('--serve', type=int, metavar='PORT', help='after building, serve the page on this local port')
    args = p.parse_args()
    out = RUNS/'overview'
    page = write(collect(RUNS, out, json.loads(args.plan.read_text())), out)
    print(page)
    if args.serve:
        print(f'http://localhost:{args.serve}/{page.relative_to(RUNS).as_posix()}', flush=True)
        http.server.ThreadingHTTPServer(('127.0.0.1', args.serve), functools.partial(Ranged, directory=str(RUNS))).serve_forever()
