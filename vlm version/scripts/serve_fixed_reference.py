import argparse
from kimodo_vla.oracle.fixed_reference import FixedReference
from kimodo_vla.benchmarks.server import make_server

p=argparse.ArgumentParser(description='CPU fixed-reference source; no Kimodo model')
p.add_argument('--run',required=True);p.add_argument('--mode',choices=['hold','reach'],required=True)
p.add_argument('--port',type=int,required=True)
a=p.parse_args(); policy=FixedReference(a.run,a.mode)
make_server({('GET','/health'):lambda _:(200,{'status':'ok','native':False,'source':'fixed'}),
             ('POST','/reset'):lambda r:(200,policy.reset(r.get('seed'))),
             ('POST','/infer'):lambda r:(200,policy(r))},'127.0.0.1',a.port).serve_forever()
