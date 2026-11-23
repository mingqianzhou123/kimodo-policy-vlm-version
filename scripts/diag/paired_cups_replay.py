"""Re-record selected validation episodes without changing the evaluation protocol.

Uses original env and policy seeds, fp32/10 diffusion steps/50 execution frames.
Records collisions and grasp predicates read-only; never changes task success.
"""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch
from bigym.loco import make
from bigym.loco.eval import is_success
from kimodo_vla.benchmarks.bigym2.policy import make_kimodo_policy

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)

def compose(env, ts):
    outside = np.asarray(env.render())
    if outside.ndim != 3:
        raise RuntimeError('Expected RGB render; no software-render fallback')
    outside = np.asarray(Image.fromarray(outside).resize((640, 480)))
    cams = [np.asarray(Image.fromarray(np.asarray(c[-3:]).transpose(1,2,0)).resize((208,208))) for c in ts.rgb_obs]
    strip = np.pad(np.concatenate(cams,axis=1), ((0,0),(8,8),(0,0)))
    return np.concatenate([outside, strip], axis=0)

def state(env, ts, step, action=None):
    inner = env.inner_env
    cups=[]
    for cup in inner.cups:
        cups.append(dict(held={str(side): bool(inner.robot.is_gripper_holding_object(cup, side)) for side in inner.robot.grippers},
            tray=bool(cup.is_colliding(inner.dishwasher.tray_middle.colliders)),
            cabinet=any(bool(cup.is_colliding(c)) for c in inner.cabinets),
            floor=bool(cup.is_colliding(inner.floor))))
    return dict(step=step, seconds=step/50, cups=cups,
                hand_action=None if action is None else np.asarray(action[-2:]).tolist(),
                event_progress=float(ts.event_progress) if np.isfinite(ts.event_progress) else None, success=bool(is_success(env)))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs='*')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.run/'config.json').read_text())
    original={k:{r['seed']:r for r in json.loads((a.run/'val700000'/f'checkpoint_{k}.json').read_text())['episode_records']} for k in (30000,100000)}
    regress=[s for s,r in original[30000].items() if r['success'] and not original[100000][s]['success']]
    seeds=a.seeds or regress
    assert all(700000 <= s < 700050 for s in seeds), 'Validation seeds only'
    torch.set_num_threads(2)
    env=make('dishwasher_load_cups')
    policy=make_kimodo_policy(env,'dishwasher_load_cups',str(a.run/f'checkpoint_{a.step}'),cfg['main']['text_cache'],device='cuda',dtype=torch.float32,execution_frames=50,diffusion_steps=10,noise='per_plan',rtc_overlap_frames=0,rtc_frozen_frames=1,blend_steps=0)
    renderer=None
    results=[]
    try:
        for seed in seeds:
            stem=f'cp{a.step}_seed{seed}'
            if (a.out/f'{stem}.json').exists():
                results.append(json.loads((a.out/f'{stem}.json').read_text())); continue
            # AgentPolicy.reset seeds diffusion as base_seed + _episode.
            policy.base_seed=700000; policy._episode=seed-700000
            policy.reset(); ts=env.reset(seed=seed)
            initial=np.asarray(env.inner_env.data.qpos).copy().tolist()
            track=[state(env,ts,0)]; thumbnails=[]; start=time.time(); length=0
            writer=imageio.get_writer(a.out/f'{stem}.mp4',fps=10,quality=6,macro_block_size=16,ffmpeg_params=['-threads','1'])
            try:
                while True:
                    if length % 5 == 0 or ts.last():
                        frame=compose(env,ts)
                        if renderer is None:
                            from OpenGL.GL import glGetString, GL_RENDERER
                            renderer=glGetString(GL_RENDERER).decode()
                            assert 'NVIDIA' in renderer, renderer
                            print('RENDERER',renderer,flush=True)
                        im=Image.fromarray(frame); ImageDraw.Draw(im).text((10,10),f'{a.step//1000}k | seed {seed} | t={length/50:.1f}s',fill='black',stroke_width=1,stroke_fill='white')
                        writer.append_data(np.asarray(im))
                        if length % 100 == 0 or ts.last():
                            thumbnails.append((length,im.resize((320,344))))
                    if ts.last(): break
                    action=policy(ts); ts=env.step(action); length+=1
                    track.append(state(env,ts,length,action))
            finally: writer.close()
            record=dict(seed=seed,step=a.step,success=int(is_success(env)),length=length,termination=env.episode_termination(),fell=int(env.episode_fell()),wall_seconds=round(time.time()-start,2),renderer=renderer,initial_qpos=initial,original=original[a.step][seed],video=f'{stem}.mp4')
            record['matches_original']=all(record[k]==record['original'][k] for k in ['success','length','termination','fell'])
            save(a.out/f'{stem}.trace.json',track)
            cols=5; rows=(len(thumbnails)+cols-1)//cols
            sheet=Image.new('RGB',(320*cols,368*rows),'#f5f4ed'); draw=ImageDraw.Draw(sheet)
            for i,(t,img) in enumerate(thumbnails):
                x=(i%cols)*320; y=(i//cols)*368
                sheet.paste(img,(x,y+24)); draw.text((x+8,y+5),f'{a.step//1000}k / {seed} / {t/50:.1f}s',fill='black')
            sheet.save(a.out/f'{stem}.jpg',quality=87)
            save(a.out/f'{stem}.json',record); results.append(record)
            save(a.out/f'records_{a.step}_{seeds[0]}.json',results)
            print('EPISODE',json.dumps({k:v for k,v in record.items() if k!='initial_qpos'}),flush=True)
        print('DONE',a.step,len(results),flush=True)
    finally: env.close()

if __name__=='__main__': main()
