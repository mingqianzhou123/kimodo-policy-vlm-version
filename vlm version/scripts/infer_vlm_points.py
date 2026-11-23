"""Run real Qwen weights on one Arena RGB frame; no simulator targets enter the prompt."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from kimodo_vla.vlm.geometry import MOTION_TEXT,parse_points,load_capture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    data=load_capture(a.capture)
    prompt=('Locate the center of the small solid RED square target marker for the left hand and the small solid BLUE square target marker for the right hand. '
        'Use the marker colors, not image left/right. Return ONLY JSON with keys left, right, motion_text. '
        'Each point is [x,y], normalized from 0 to 1000 over the full image. '
        'If a marker is not visible use null. Do not guess an occluded marker. '
        f'motion_text must be exactly "{MOTION_TEXT}".')
    image=Image.fromarray(data['rgb'][...,:3]);image.save(a.out.with_suffix('.input.png'))
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt').to(model.device)
    loaded=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=128,do_sample=False)
    text=processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    result={'model_path':str(a.model.resolve()),'prompt':prompt,'raw_output':text,
        'capture_sha256':hashlib.sha256(a.capture.read_bytes()).hexdigest(),
        'load_s':loaded-start,'inference_s':time.perf_counter()-loaded,
        'generated_tokens':int(output.shape[1]-inputs['input_ids'].shape[1]),
        'transformers_version':transformers.__version__,'torch_version':torch.__version__,
        'model_manifest':json.loads((a.model/'download-manifest.json').read_text())}
    try:
        result['intent']=parse_points(text)
    except (ValueError,TypeError) as exc:
        result['error']=str(exc);a.out.write_text(json.dumps(result,indent=2));raise
    a.out.write_text(json.dumps(result,indent=2))
    draw=ImageDraw.Draw(image)
    for side,color in [('left','red'),('right','blue')]:
        x,y=np.array(result['intent'][side])/1000*np.array([image.width-1,image.height-1])
        draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2);draw.text((x+7,y),side,fill=color)
    image.save(a.out.with_suffix('.points.png'))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
