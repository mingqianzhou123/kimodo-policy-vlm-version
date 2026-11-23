"""Qwen coarse-to-fine localization with two crop scales and RGB rejection."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from kimodo_vla.vlm.geometry import MOTION_TEXT,parse_points,load_capture
from kimodo_vla.vlm.refinement import CROP_SIZES,MAX_DISAGREEMENT_PX,crop_box,crop_to_full,parse_crop_point,combine_and_verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch,transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    data=load_capture(a.capture);rgb=data['rgb'][...,:3];image=Image.fromarray(rgb)
    image.save(a.out.with_suffix('.input.png'))
    prompt=('Locate the center of the small solid RED square target marker for the left hand and the small solid BLUE square target marker for the right hand. '
        'Use the marker colors, not image left/right. Return ONLY JSON with keys left, right, motion_text. '
        'Each point is [x,y], normalized from 0 to 1000 over the full image. '
        'If a marker is not visible use null. Do not guess an occluded marker. '
        f'motion_text must be exactly "{MOTION_TEXT}".')
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    loaded=time.perf_counter();calls=[]
    def infer(img,prompt,max_tokens):
        inputs=processor.apply_chat_template([{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':prompt}]}],
            tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to(model.device)
        tic=time.perf_counter()
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False)
        text=processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        calls.append({'prompt':prompt,'raw_output':text,'seconds':time.perf_counter()-tic,
                      'generated_tokens':int(output.shape[1]-inputs['input_ids'].shape[1])})
        return text
    result={'model_path':str(a.model.resolve()),'prompt':prompt,'raw_output':'',
        'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),'localizer':'crop-consensus-v1',
        'load_s':loaded-start,'transformers_version':transformers.__version__,'torch_version':torch.__version__,
        'model_manifest':json.loads((a.model/'download-manifest.json').read_text()),
        'crop_sizes':list(CROP_SIZES),'max_disagreement_px':MAX_DISAGREEMENT_PX,
        'color_check':'3x3 RGB majority; verification only, no point snapping','calls':calls,'refinement':{}}
    candidate={'motion_text':MOTION_TEXT};errors=[]
    try:
        result['raw_output']=infer(image,prompt,128)
        coarse=parse_points(result['raw_output']);result['coarse_intent']=coarse
        for side,color in [('left','RED'),('right','BLUE')]:
            estimates=[];records=[]
            for size in CROP_SIZES:
                box=crop_box(coarse[side],image.size,size)
                crop=image.crop(box).resize((640,640),Image.Resampling.BICUBIC)
                crop.save(a.out.with_suffix(f'.{side}-{size}.png'))
                instruction=(f'Find the exact center of the small solid {color} square target marker in this image crop. '
                    'Do not point to the robot hand, background, or a nearby object. '
                    'Return ONLY JSON {"point":[x,y]}, coordinates normalized 0 to 1000 over this crop. '
                    'If the colored square is absent or occluded, return {"point":null}.')
                raw=infer(crop,instruction,64);record={'box':list(box),'raw_output':raw}
                records.append(record)
                try:
                    full=crop_to_full(parse_crop_point(raw),box);record['full_pixel']=full.tolist();estimates.append(full)
                except (ValueError,TypeError) as exc:record['error']=str(exc);errors.append(f'{side}: {exc}')
            result['refinement'][side]={'crops':records}
            if len(estimates)!=2:continue
            # Candidate is stored even if quality checks reject; evaluator must count it.
            point=np.mean(estimates,axis=0)
            candidate[side]=(point/np.array([image.width-1,image.height-1])*1000).tolist()
            try:
                point,checks=combine_and_verify(rgb,estimates,side);result['refinement'][side].update(checks)
            except ValueError as exc:errors.append(str(exc))
        if set(candidate)=={'left','right','motion_text'}:
            result['candidate_intent']=parse_points(json.dumps(candidate))
            result['raw_output']=json.dumps(candidate)
        if errors or 'candidate_intent' not in result:raise ValueError('; '.join(errors) or 'Incomplete refined target')
        result['intent']=result['candidate_intent']
    except (ValueError,TypeError) as exc:
        result['error']=str(exc)
    result.update(inference_s=time.perf_counter()-loaded,generated_tokens=sum(c['generated_tokens'] for c in calls))
    a.out.write_text(json.dumps(result,indent=2))
    if 'candidate_intent' in result:
        draw=ImageDraw.Draw(image)
        for side,color in [('left','red'),('right','blue')]:
            x,y=np.array(candidate[side])/1000*np.array([image.width-1,image.height-1])
            draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2);draw.text((x+7,y),side,fill=color)
        image.save(a.out.with_suffix('.points.png'))
    print(json.dumps(result),flush=True)
    if 'error' in result:raise SystemExit(2)

if __name__=='__main__':main()
