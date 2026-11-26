"""Judge command-loop episodes: success, the first failed link (1-5) or unclassified, reach margins.

    python agentic/pilot_experiments2/experiment3/exp3_judge.py 'exp3dev-pp_box-scripted-20261126-d1-*'
    python agentic/pilot_experiments2/experiment3/exp3_judge.py <run folder> --attribution last_attempt --out verdicts.json

Arguments are run folders or glob patterns under the runs folder (KVLA_AGENTIC_RUNS). Reads only
each run's loop/ logs and episode.json, and the thresholds in exp3_thresholds.json. Prints one line
per episode and counts per link; with --out writes every verdict with all its checks.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))
from paths import AGENTIC, REPO, RUNS

from kimodo_vla.agent.episode_log import load_episode
from kimodo_vla.agent.judge import judge
from kimodo_vla.core.types import G1_JOINT_NAMES
from kimodo_vla.oracle.geometry import XmlFK


def runs(patterns):
    out = []
    for pattern in patterns:
        path = Path(pattern)
        out += [path] if path.is_dir() else sorted(p for p in RUNS.glob(pattern) if (p/'loop').is_dir())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--thresholds', type=Path, default=HERE/'exp3_thresholds.json')
    p.add_argument('--attribution', choices=['earliest', 'last_attempt'], default='earliest')
    p.add_argument('--out', type=Path, help='Write all verdicts here (outside the repository)')
    a = p.parse_args()
    thresholds = json.loads(a.thresholds.read_text())
    grasp = json.loads((AGENTIC/'configs'/'grasp-loop.json').read_text())
    fk = XmlFK(REPO/'kimodo_vla/skeleton/assets/g1skel34/xml/g1.xml', G1_JOINT_NAMES)
    verdicts = []
    for run in runs(a.runs):
        try:
            v = judge(load_episode(run), thresholds, fk=fk, grasp_config=grasp, attribution=a.attribution)
        except Exception as exc:   # one unreadable run must not hide the others
            print(f'{run.name}: NOT JUDGED ({exc!r})')
            continue
        verdicts.append(v)
        margins = ' '.join(f'{k}={x:+.3f}' for k, x in v['margins'].items())
        what = 'success' if v['success'] else f"link {v['link']} ({v.get('phase')}: {v['reason']})"
        print(f"{run.name}: {what}; attempts {v['attempts']}; end {v['end_reason']} / arena {v['arena_failure_reason']}; {margins}")
    counts = Counter('success' if v['success'] else str(v['link']) for v in verdicts)
    print('counts:', dict(sorted(counts.items())))
    if a.out:
        a.out.write_text(json.dumps(verdicts, indent=1, default=str))


if __name__ == '__main__':
    main()
