from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parents[2]/'data/cache/relay'
spec=[('text-weights-01.zip',1804206080,'bef63c94c64b8e8da66461cec61e9b1475990f34cc4f1493df5f3a2e983ae321'),('text-weights-02.zip',1805647872,'66368e29d9c2744a04784b84401aa71b89764dcb2b09e68662c7c6fd19b49e77'),('text-weights-03.zip',1828913152,'4710f3642f2f92d6a0c5df42ff5d98bb593f3b26b1f96fc27360781c83feb4ff')]
out=r/'remaining-text';out.mkdir(exist_ok=False);manifest={}
for name,size,sha in spec:
 with (r/name).open('rb') as f:
  h=hashlib.sha256();remaining=size
  while remaining:
   b=f.read(min(16*1024**2,remaining));h.update(b);remaining-=len(b)
  assert h.hexdigest()==sha,(name,h.hexdigest())
  chunks=[]
  while b:=f.read(512*1024**2):
   p=out/f'{name}.remaining-{len(chunks):02d}';p.write_bytes(b)
   chunks.append({'name':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
  manifest[name]={'prefix_bytes':size,'prefix_sha256':sha,'chunks':chunks}
(out/'remaining.json').write_text(json.dumps(manifest,indent=2))
print('VERIFIED_PREFIXES_AND_CHUNKS',sum(len(x['chunks']) for x in manifest.values()))
