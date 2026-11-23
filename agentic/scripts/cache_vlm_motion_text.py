"""Encode the exact first-stage VLM motion phrase with original LLM2Vec weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
import paths  # noqa: F401  (puts this checkout on sys.path)
from kimodo_vla.vlm.geometry import MOTION_TEXT
from kimodo_vla.model.modules.llm2vec.llm2vec_wreapper import LLM2VecEncoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('data/vlm-text'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    encoder=LLM2VecEncoder(str(a.checkpoints)).to('cuda').eval()
    with torch.inference_mode():embedding,_=encoder(MOTION_TEXT)
    embedding=embedding.cpu().to(torch.bfloat16)
    assert embedding.shape==(1,4096) and torch.isfinite(embedding).all()
    dest=a.out/'reach_hold.pt'
    torch.save({'version':3,'source':'VLM motion phrase','instruction':MOTION_TEXT,'embedding':embedding},dest)
    (a.out/'manifest.json').write_text(json.dumps({'text':MOTION_TEXT,'encoder':str(a.checkpoints.resolve()),
        'embedding_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))

if __name__=='__main__':main()
