"""Fixed-noise input interventions on a real demonstration; numerical pathway check, not task scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kimodo_vla.benchmarks.toolbench import demos
from kimodo_vla.model.planner import KimodoPlanner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path('/root/workspace/ToolBook/manifest.json'))
    p.add_argument('--text-cache', type=Path, default=Path('data/cache/toolbench'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    entry = json.loads(a.manifest.read_text())['episodes'][0]
    data = a.manifest.parent/entry['data']
    episode = int(data.stem.split('_')[-1])
    demo = demos.ToolBenchDemo(demos.read_recording(data.parents[2], episode),entry['instruction'],a.manifest.parent/entry['video'])
    cut = min(250,len(demo)-2)
    history = demo.observations()[:cut+1]
    image = demo.images(np.array([cut]))[0,0]
    history[-1].images = {'head':image}
    planner = KimodoPlanner.from_checkpoint(a.checkpoint,a.text_cache,device='cuda',dtype=torch.bfloat16,
        control_fps=50,execution_frames=50,diffusion_steps=10,root_boundary='extrapolate')

    def predict(instruction, picture):
        for obs in history:
            obs.instruction = instruction
        history[-1].images['head'] = picture
        planner.reset(1234)
        chunk = planner.plan(history)
        return np.concatenate([chunk.joint_pos,chunk.root_delta_xy,chunk.root_height[:,None],chunk.hand],axis=1)

    baseline = predict(entry['instruction'],image)
    repeated = predict(entry['instruction'],image)
    blank = predict(entry['instruction'],np.zeros_like(image))
    alternative = demos.INSTRUCTIONS['G1IceBreak'][0]
    changed_text = predict(alternative,image)
    result = {'checkpoint':str(a.checkpoint),'source_episode':entry['episode_id'],'control_step':cut,
        'fixed_diffusion_seed':1234,'original_instruction':entry['instruction'],'alternative_instruction':alternative,
        'repeat_max_error':float(np.abs(baseline-repeated).max()),
        'blank_image_rms_change':float(np.sqrt(np.mean((baseline-blank)**2))),
        'changed_instruction_rms_change':float(np.sqrt(np.mean((baseline-changed_text)**2))),
        'output_shape':list(baseline.shape),
        'scope':'Numerical visual/language conditioning check; does not establish task success.'}
    assert np.isfinite(baseline).all() and np.isfinite(blank).all() and np.isfinite(changed_text).all()
    assert result['repeat_max_error'] < 1e-6
    assert result['blank_image_rms_change'] > 1e-7 and result['changed_instruction_rms_change'] > 1e-7
    result['passed'] = True
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
