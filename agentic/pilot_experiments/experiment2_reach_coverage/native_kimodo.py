"""Our port of official Kimodo generation (``kimodo_vla/model/segment_sampler.py``) checked against NVIDIA's pipeline.

Subcommands (policy environment; nothing from the official repository is imported):

  check  replay the official fixtures (made by ``official_kimodo.py --make-fixtures``) with our port
         from the same noise and print the largest differences:
             native_kimodo.py check --checkpoint <Kimodo-G1-RP-v1 dir> [--fixtures <dir>]
  walks  generate the official comparison's walks with our port (the official default: text only,
         one 8 s segment, 100 DDIM steps, guidance 2.0 / 2.0, same seeds as ``official_kimodo.py``,
         whose noise comes from the global CUDA generator seeded per walk), convert them to 50 Hz
         actions the way our planners send them to SONIC, and score the gait:
             native_kimodo.py walks --checkpoint <dir> --text-cache <phrase embeddings> --out <segments dir>

Fixture noise is drawn on the CPU from ``torch.Generator().manual_seed(seed)`` in segment order, so
any device can repeat it. Fixture files hold unnormalized features.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'scripts'))  # agentic/scripts: paths.py
import paths  # noqa: F401,E402  (puts this checkout's package first)
from kimodo_vla.model import segment_sampler as ss  # noqa: E402
from kimodo_vla.oracle import demo_walks as dw  # noqa: E402

FIXTURES = HERE/'fixtures'/'official_kimodo'
PROMPTS = {'forward': 'A person walks forward.', 'backward': 'A person walks backward.',
           'left': 'A person side steps to the left.', 'right': 'A person side steps to the right.',
           'turn': 'A person turns to the left in place.'}


def fixture_noise(seed, shapes):
    """Noise of each segment, drawn on the CPU in order (as the fixtures were made)."""
    g = torch.Generator().manual_seed(int(seed))
    return [torch.randn(tuple(s), generator=g) for s in shapes]


def load_fixture(path):
    with np.load(path, allow_pickle=False) as z:
        fx = {k: z[k] for k in z.files}
    fx['meta'] = json.loads(str(fx['meta']))
    return fx


def official_condition(fx, i, dim):
    """The i-th official call's constraints: (values [T, D] unnormalized, mask [T, D] bool), or None."""
    if f'cond{i}_shape' not in fx:
        return None
    shape = tuple(int(x) for x in fx[f'cond{i}_shape'])
    mask = np.unpackbits(fx[f'cond{i}_mask'])[:shape[0]*shape[1]].reshape(shape).astype(bool)
    values = np.zeros(shape, dtype=np.float32)
    values[mask] = fx[f'cond{i}_values']
    return torch.from_numpy(values), torch.from_numpy(mask)


def our_conditions(rep, fx):
    """Our builders' constraints for each official call of a fixture (unnormalized values, bool masks)."""
    meta, out = fx['meta'], []
    for i, frames in enumerate(meta['frames']):
        if i == 0:
            v, m = ss.empty_condition(rep, frames)
            if meta.get('root_path'):
                p = meta['root_path']
                ss.root_path_condition(rep, v, m, torch.arange(frames), torch.tensor(p['xz']), torch.tensor(p['heading']))
            out.append((v, m, 0.))
        else:
            v, m, heading, _ = ss.transition_condition(rep, torch.from_numpy(fx[f'raw{i-1}']), frames,
                                                       transition=meta['transition'])
            out.append((v, m, heading))
    return out


def stitched(rep, fx):
    """Join the fixture's official raw segments with our ``join_segment``."""
    meta, seq = fx['meta'], torch.from_numpy(fx['raw0'])
    for i in range(1, len(meta['frames'])):
        _, _, _, origin = ss.transition_condition(rep, seq, meta['frames'][i], transition=meta['transition'])
        seq = ss.join_segment(rep, seq, torch.from_numpy(fx[f'raw{i}']), origin, transition=meta['transition'])
    return seq


def decoded_joints(rep, feats):
    return ss.decode(rep, feats)['posed_joints']


def replay(model, fx):
    """Our port from the fixture's noise: raw segment outputs (unnormalized) and the stitched sequence."""
    meta, rep = fx['meta'], model.representation
    shapes = [(1, f + (meta['transition'] if i else 0), rep.motion_rep_dim) for i, f in enumerate(meta['frames'])]
    noises = fixture_noise(meta['seed'], shapes)
    raws, seq = [], None
    for i, frames in enumerate(meta['frames']):
        text = torch.from_numpy(fx[f'text{i}']) if f'text{i}' in fx else None
        kw = dict(text_feat=text, text_weight=meta['cfg'][0], constraint_weight=meta['cfg'][1], steps=meta['steps'],
                  noise=noises[i])
        if i == 0:
            v, m, _ = our_conditions(rep, fx)[0]
            raw, _ = ss.sample_segment(model, v, m, first_heading=0., **kw)
            seq = raw.cpu()
        else:
            v, m, heading, origin = ss.transition_condition(rep, seq, frames, transition=meta['transition'])
            raw, _ = ss.sample_segment(model, v, m, first_heading=heading, **kw)
            seq = ss.join_segment(rep, seq, raw, origin, transition=meta['transition'])
        raws.append(raw.cpu())
    return raws, seq


def compare(model, fx):
    """Largest differences, ours vs official: raw outputs in normalized units, joints in metres."""
    rep = model.representation
    raws, seq = replay(model, fx)
    out = {'case': fx['meta']['name']}
    out['raw_max_abs_normalized'] = max(float((rep.normalize(r.to(model.device)) - rep.normalize(torch.from_numpy(fx[f'raw{i}']).to(model.device))).abs().max())
                                        for i, r in enumerate(raws))
    ours, theirs = decoded_joints(rep, seq), torch.from_numpy(fx['posed_joints'])
    out['joints_max_abs_m'] = float((ours - theirs).abs().max())
    out['joints_mean_abs_m'] = float((ours - theirs).abs().mean())
    return out


def load_model(checkpoint, device):
    from kimodo_vla.model.native_policy import NativeKimodoPolicy
    return NativeKimodoPolicy(checkpoint).to(device).eval()


def check(args):
    model = load_model(args.checkpoint, args.device)
    rows = [compare(model, load_fixture(p)) for p in sorted(Path(args.fixtures).glob('*.npz'))]
    for r in rows:
        print(json.dumps(r), flush=True)
    return rows


def to_actions(features):
    """Unnormalized 30 fps features -> 50 Hz actions, the way our planners send motion to SONIC."""
    from kimodo_vla.model.features import KimodoFeatures
    from kimodo_vla.motion.g1_reference import _interpolate_motion
    from kimodo_vla.oracle.continuity import bounded_hinge_trajectory
    F = KimodoFeatures(dw.RATE)
    d = F.representation.inverse(torch.as_tensor(features).float()[None], is_normalized=False)
    local, root = d['local_rot_mats'][0], d['root_positions'][0]
    times = torch.arange(0., (len(local)-1)*dw.RATE/30+1e-6)*30/dw.RATE
    local50, root50 = _interpolate_motion(local, root, times)
    a = F.codec.encode(local50, root50, previous_root_position=root50[0], hand_binary=torch.zeros(len(times), 2)).numpy()
    limits = dw.g1_fk().joint_limits
    a[:, 9:38] = bounded_hinge_trajectory(local50, F.codec, limits, np.clip(a[0, 9:38], limits[:, 0], limits[:, 1]), 4.)
    return a


def walks(args):
    """The official comparison's walks (official_kimodo.py, variant 'official') through our port."""
    model = load_model(args.checkpoint, 'cuda')
    table = {}
    for path in Path(args.text_cache).glob('*.pt'):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        table[payload['instruction'].strip()] = torch.as_tensor(payload['embedding']).float()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    frames, fk, rows = int(args.seconds*30), dw.g1_fk(), []
    for direction in args.directions:
        prompt = PROMPTS[direction]
        for seed in args.seeds:
            began = time.time()
            torch.manual_seed(seed)       # official_kimodo.py: seed_everything(seed), then torch.randn on cuda
            v, m = ss.empty_condition(model.representation, frames)
            feats, _ = ss.sample_segment(model, v, m, text_feat=table[prompt], text_weight=2., constraint_weight=2.,
                                         steps=args.steps)
            actions = to_actions(feats.cpu())
            window = [int(dw.RATE), len(actions)-1]
            seg = {'name': f'native-{direction}-s{seed}', 'direction': direction, 'window': window,
                   'source': {'pipeline': 'kimodo_vla.model.segment_sampler', 'prompt': prompt, 'seed': seed,
                              'frames': frames, 'diffusion_steps': args.steps, 'cfg_weight': [2., 2.]},
                   'actions': actions.tolist()}
            (out/f"{seg['name']}.json").write_text(json.dumps(seg))
            summary = dw.window_summary(actions, *window)
            g = dw.stepping(dw.reference_feet(actions[window[0]:], fk))
            rows.append({'name': seg['name'], 'direction': direction, 'seed': seed, 'window': summary, 'gait': g,
                         'seconds_to_generate': time.time()-began})
            print(seg['name'], f"{summary['speed_m_s']:.2f} m/s, yaw {summary['yaw_deg']:.0f}, slide {g['slide_m_s'] or 0:.3f}, "
                  f"both {g['both_moving']:.2f}, step {g['step_length_m'] or 0:.2f}", flush=True)
    (out/'segments.json').write_text(json.dumps({'segments': rows}, indent=1))
    for d in args.directions:
        group = [r for r in rows if r['direction'] == d]
        print(d, 'median slide / both / step / speed / yaw:',
              f"{np.median([r['gait']['slide_m_s'] or 0 for r in group]):.3f}",
              f"{np.median([r['gait']['both_moving'] for r in group]):.2f}",
              f"{np.median([r['gait']['step_length_m'] or 0 for r in group]):.2f}",
              f"{np.median([r['window']['speed_m_s'] for r in group]):.2f}",
              f"{np.median([r['window']['yaw_deg'] for r in group]):.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--fixtures', default=str(FIXTURES))
    c.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    w = sub.add_parser('walks')
    w.add_argument('--checkpoint', required=True)
    w.add_argument('--text-cache', required=True)
    w.add_argument('--out', required=True)
    w.add_argument('--seconds', type=float, default=8.)
    w.add_argument('--steps', type=int, default=100)
    w.add_argument('--seeds', type=int, nargs='+', default=[1000, 1001, 1002, 1003, 1004])
    w.add_argument('--directions', nargs='+', default=list(PROMPTS))
    args = p.parse_args()
    {'check': check, 'walks': walks}[args.cmd](args)


if __name__ == '__main__':
    main()
