import argparse
import json
from pathlib import Path
from kimodo_vla.vlm.stability import review_suite

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('suite',type=Path)
    result=review_suite(p.parse_args().suite)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
