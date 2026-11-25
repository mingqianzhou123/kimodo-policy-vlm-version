import argparse
from kimodo_vla.oracle.fixed_reference import FixedReference
from kimodo_vla.oracle.demo_walks import DemoReference
from kimodo_vla.benchmarks.server import make_server

p=argparse.ArgumentParser(description='CPU fixed-reference source; no Kimodo model')
p.add_argument('--run',required=True);p.add_argument('--mode',choices=['hold','reach'],required=True)
p.add_argument('--port',type=int,required=True)
p.add_argument('--demo-segment',help='Replay this walking segment instead (kimodo_vla/oracle/demo_walks.py)')
a=p.parse_args()
policy=DemoReference(a.run,a.demo_segment) if a.demo_segment else FixedReference(a.run,a.mode)
source='demo' if a.demo_segment else 'fixed'
make_server({('GET','/health'):lambda _:(200,{'status':'ok','native':False,'source':source}),
             ('POST','/reset'):lambda r:(200,policy.reset(r.get('seed'))),
             ('POST','/infer'):lambda r:(200,policy(r))},'127.0.0.1',a.port).serve_forever()
