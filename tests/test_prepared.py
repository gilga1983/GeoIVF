"""Prepared query lifetime, selection, coalescing and full-search invariants."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from test_speed import ready, COUNTS
from geoivf.cells import CellIndex
from geoivf.index import Index
from geoivf.io import MemoryReplay
from geoivf.prepared import PreparedPlanner
from geoivf.search import search, coalesce


@pytest.fixture(scope='module', autouse=True)
def compile_prepared():
    subprocess.run(['bash',str(Path(__file__).resolve().parents[1]/'scripts/build_prepared.sh')],check=True)


@pytest.mark.parametrize('bits',[4,6,8])
@pytest.mark.parametrize('mode',['none','balls','radial','combined'])
@pytest.mark.parametrize('window,gap,extent',[(0,0,256),(1,0,1),(3,1,4),(64,2,8)])
def test_exact_plans(ready,bits,mode,window,gap,extent):
    x,layout,paths=ready
    idx=CellIndex(layout,paths[bits],shape='ball')
    class Recorder:
        def __init__(self,r):self.r=r;self.plans=[]
        def read(self,requests):self.plans.append(requests);return self.r.read(requests)
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for q in [x[0],x[88],np.full(128,40,dtype=np.float32)]:
            before=Recorder(reader);after=Recorder(reader)
            kw=dict(nprobe=2,preassigned_lists=[0,1],filtering=mode,window_pages=window,
                    gap_pages=gap,max_extent_pages=extent,scan='native')
            expected,base=search(idx,q,before,selection='adaptive',**kw)
            audit=[]
            got,stat=search(idx,q,after,selection='prepared',audit_sink=audit,**kw)
            np.testing.assert_array_equal(got,expected)
            assert before.plans==after.plans
            assert tuple(base[k] for k in COUNTS)==tuple(stat[k] for k in COUNTS)
            for p,tau,lb in audit:
                actual=np.linalg.norm(x[idx.page_ids[p,:idx.valid[p]]].astype(float)-q,axis=1).min()
                assert actual>tau and lb<=actual+1e-8
    finally:reader.close()


def test_context_independence_reuse_and_errors(ready):
    x,layout,paths=ready;idx=CellIndex(layout,paths[8],shape='ball')
    q=x[0].copy();a=PreparedPlanner(idx,q,window_pages=1);b=PreparedPlanner(idx,x[88])
    q[:]=0  # Context owns its query, not an alias into caller memory.
    first,last=map(int,idx.ranges[0]);pages=np.arange(first,last)
    for tau in [0,100,200,400,float('inf')]:
        mask=idx.select(x[0],pages,0,'combined',tau,strategy=2)
        ext,n=a.plan(first,last,0,tau)
        assert ext==coalesce(pages[mask]) and n==int(mask.sum())
        b.plan(first,last,0,tau)
        assert a.plan(first,last,0,tau)==(ext,n)
    assert a.capacity>=len(pages) and len(a.radial_cache)==1
    assert a.scratch_bytes < 10000
    for args in [(-1,2,0,1),(0,last+1,0,1),(0,last,2,1),(0,last,0,-1),(0,last,0,float('nan'))]:
        with pytest.raises(ValueError):a.plan(*args)
    assert a.plan(first,first,0,1)==([],0)
    with pytest.raises(ValueError):PreparedPlanner(idx,np.zeros(4))
    with pytest.raises(ValueError):PreparedPlanner(idx,x[0],gap_pages=-1)
    with pytest.raises(ValueError):PreparedPlanner(idx,x[0],max_extent_pages=0)
    with pytest.raises(ValueError):PreparedPlanner(CellIndex(layout,paths[8],shape='box'),x[0])
    plain=Index(layout)
    p=PreparedPlanner(plain,x[0],mode='none')
    assert p.plan(first,last,0,0)[0]==coalesce(pages)
    with pytest.raises(ValueError):PreparedPlanner(plain,x[0])


@pytest.mark.parametrize('bits',[4,6,8])
def test_noninteger_and_boundary_predicates(ready,bits):
    x,layout,paths=ready;idx=CellIndex(layout,paths[bits],shape='ball')
    q=np.random.default_rng(718).normal(size=128).astype(np.float32)
    plan=PreparedPlanner(idx,q,mode='balls')
    first,last=map(int,idx.ranges[0]);pages=np.arange(first,last)
    lb=idx.bounds(q,pages,0,'balls')
    # Compare against the SAME squared-threshold predicate even immediately
    # around a full-bound boundary. Full sqrt evaluation can round differently.
    taus=np.r_[lb,np.nextafter(lb,-np.inf),np.nextafter(lb,np.inf)]
    for tau in taus:
        expected=idx.select(q,pages,0,'balls',float(tau),strategy=2)
        ext,n=plan.plan(first,last,0,float(tau))
        np.testing.assert_array_equal(plan.keep[:len(pages)].astype(bool),expected)
        assert ext==coalesce(pages[expected]) and n==expected.sum()
