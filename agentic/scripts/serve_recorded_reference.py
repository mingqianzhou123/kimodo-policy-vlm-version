import argparse
from kimodo_vla.oracle.recorded_reference import RecordedReference
from kimodo_vla.benchmarks.server import make_server

p=argparse.ArgumentParser(description='CPU replay of archived commands; no VLM, Kimodo or IK')
p.add_argument('--run',required=True)
p.add_argument('--fixture',required=True)
p.add_argument('--port',type=int,required=True)
a=p.parse_args()
policy=RecordedReference(a.run,a.fixture)
make_server({('GET','/health'):lambda _:(200,{'status':'ok','native':False,'source':'replay'}),
             ('POST','/reset'):lambda r:(200,policy.reset(r.get('seed'))),
             ('POST','/infer'):lambda r:(200,policy(r))},'127.0.0.1',a.port).serve_forever()
