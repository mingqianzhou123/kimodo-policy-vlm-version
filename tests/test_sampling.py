from collections import Counter
import importlib.util
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from kimodo_vla.data.sampling import WindowSampler


class Toy:
    def __init__(self, sizes=(100,900,9000)):
        self.episodes=[{'task':f'G1Toy-L{i}-S'} for i in range(3)]
        self.index=[(i,j) for i,n in enumerate(sizes) for j in range(n)]
    def __len__(self):return len(self.index)
    def __getitem__(self,i):return i


def test_distributed_and_resume():
    d=Toy()
    a=list(WindowSampler(d,world=2,rank=0,seed=17,epoch=3))
    b=list(WindowSampler(d,world=2,rank=1,seed=17,epoch=3))
    assert not set(a)&set(b) and len(set(a+b))==len(d)
    assert list(WindowSampler(d,world=2,rank=0,seed=17,epoch=3,offset=47))==a[47:]


@pytest.mark.parametrize('mode,target',[('task_uniform',[1/3]*3),('level_balanced',[.25,.375,.375])])
def test_sampling_probability_independent_of_recording_length(mode,target):
    d=Toy();indices=list(WindowSampler(d,mode=mode,seed=19));counts=Counter(d.index[i][0] for i in indices)
    for i,t in enumerate(target):assert abs(counts[i]/len(indices)-t)<.025


def test_batch_resume_crosses_epoch_at_same_sample():
    scripts=Path(__file__).resolve().parents[1]/'scripts';sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('train_sampling_test',scripts/'train.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=SimpleNamespace(main=Attr(seed=7,batch_size=8,micro_batch_size=2,num_workers=0,sampling='task_uniform'))
        d=Toy((7,8,9));full=list(islice(module.batches(d,config,0,rank=1,world=2),14))
        resumed=list(islice(module.batches(d,config,3,rank=1,world=2),8))
        assert [v.tolist() for v in full[6:]]==[v.tolist() for v in resumed]
    finally:sys.path.remove(str(scripts))


class Attr(SimpleNamespace):
    def get(self,key,default=None):return getattr(self,key,default)
