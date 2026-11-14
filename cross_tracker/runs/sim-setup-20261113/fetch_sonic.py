from pathlib import Path
import json,subprocess,hashlib,zipfile
r=Path(__file__).resolve().parent;root=r.parents[1]
meta=json.loads((r/'sonic-repository.json').read_text());out=root/'data/cache/sim-assets/sonic';out.mkdir(parents=True,exist_ok=True)
receipt={'repo':'nvidia/GEAR-SONIC','revision':meta['sha'],'files':{}}
for f in meta['siblings']:
 name=f['rfilename']
 if name not in ['model_encoder.onnx','model_decoder.onnx','observation_config.yaml']:continue
 p=out/name
 subprocess.run(['curl','-fsSL','--retry','3','--connect-timeout','20','--max-time','300','-o',str(p),'https://huggingface.co/nvidia/GEAR-SONIC/resolve/'+meta['sha']+'/'+name],check=True)
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert len(b)==f['size']
 if 'lfs' in f:assert sha==f['lfs']['sha256']
 else:assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['blobId']
 receipt['files'][name]={'bytes':len(b),'sha256':sha}
(r/'sonic-weights.json').write_text(json.dumps(receipt,indent=2))
z=root/'data/cache/relay/sonic-weights.zip'
with zipfile.ZipFile(z,'x') as archive:
 for p in out.iterdir():archive.write(p,'sonic/'+p.name)
 archive.write(r/'sonic-weights.json','sonic/manifest.json')
print('SONIC_WEIGHTS_VERIFIED',z.stat().st_size,hashlib.sha256(z.read_bytes()).hexdigest())
