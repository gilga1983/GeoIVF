"""RAM-first shortlist, two-wave certificate, ties and bounded I/O regressions."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from test_speed import ready
from geoivf.cells import CellIndex
from geoivf.dependent import unpack_codes
from geoivf.io import MemoryReplay
from geoivf.ramfirst import rank,search_ramfirst,page_extents,batches

@pytest.fixture(scope='module',autouse=True)
def build_ramfirst():
    subprocess.run(['bash',str(Path(__file__).resolve().parents[1]/'scripts/build_ramfirst.sh')],check=True)

class Recorder:
    def __init__(self,reader):self.reader=reader;self.pages=[]
    def read(self,req):
        for o,n in req:self.pages.extend(range(o//4096,(o+n)//4096))
        return self.reader.read(req)

@pytest.mark.parametrize('bits',[4,6,8])
def test_rank_and_lower_bounds(ready,bits):
    x,layout,paths=ready;idx=CellIndex(layout,paths[bits],shape='ball')
    q=np.random.default_rng(344).normal(size=128).astype(np.float32)
    slots,dd,pages,lb,n=rank(idx,q,[1,0],31,certified=True)
    code=unpack_codes(idx.arrays['packed'],bits,(8,64))
    centers=idx.arrays['origin']+idx.arrays['scale']*code
    proj=(q.astype(float)-idx.arrays['mean'])@idx.arrays['matrix']
    ds=((centers-proj[:64])**2).sum(2)
    valid=idx.page_ids>=0
    order=np.lexsort((idx.page_ids[valid],ds[valid]))[:31]
    expected=np.flatnonzero(valid)[order]
    np.testing.assert_array_equal(slots,expected)
    np.testing.assert_allclose(dd,ds.ravel()[slots],rtol=1e-14)
    assert n==len(x)
    for li in [1,0]:
        pp=np.arange(*idx.ranges[li]);at=np.isin(pages,pp)
        np.testing.assert_allclose(lb[at],idx.bounds(q,pp,li,'combined'),atol=1e-10)
    s2,d2,p2,l2,n2=rank(idx,q,[1,0],31,certified=False)
    np.testing.assert_array_equal(s2,slots);np.testing.assert_array_equal(d2,dd)
    assert len(l2)==0 and n2==n

@pytest.mark.parametrize('bits',[4,6,8])
@pytest.mark.parametrize('shortlist',[10,16,64])
@pytest.mark.parametrize('certified',[False,True])
def test_exact_verified_pool_and_certificate(ready,bits,shortlist,certified):
    x,layout,paths=ready;idx=CellIndex(layout,paths[bits],shape='ball')
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for q in [x[0],x[88],np.full(128,40,dtype=np.float32)]:
            recorder=Recorder(reader);audit=[]
            ids,stat=search_ramfirst(idx,q,recorder,shortlist=shortlist,certified=certified,
                k=10,nprobe=2,preassigned_lists=[1,0],scan='native',gap_pages=2,
                max_extent_pages=2,io_batch_bytes=8192,audit_sink=audit)
            assert len(set(recorder.pages))==len(recorder.pages)==stat['read_pages']
            assert stat['read_pages']==stat['first_wave_pages']+stat['second_wave_pages']
            assert stat['read_pages']==stat['selected_pages']+stat['gap_pages_read']
            assert stat['verification_waves']<= (2 if certified else 1)
            fetched=np.concatenate([idx.page_ids[p,:idx.valid[p]] for p in recorder.pages])
            dd=((x.astype(float)-q)**2).sum(1)
            pool=fetched[np.lexsort((fetched,dd[fetched]))[:10]]
            np.testing.assert_array_equal(ids,pool)
            if certified:np.testing.assert_array_equal(ids,np.lexsort((np.arange(len(x)),dd))[:10])
            for p,tau,lb in audit:
                actual=np.sqrt(dd[idx.page_ids[p,:idx.valid[p]]].min())
                assert actual>tau and lb<=actual+1e-8
    finally:reader.close()

def test_wave_does_not_mean_one_io_batch(ready):
    x,layout,paths=ready;idx=CellIndex(layout,paths[8],shape='ball')
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        ids,s=search_ramfirst(idx,x[0],reader,shortlist=149,certified=True,nprobe=2,
            preassigned_lists=[0,1],scan='native',max_extent_pages=1,io_batch_bytes=4096)
        assert s['verification_waves']==1 and s['read_stages']>2 and s['first_already_certified']
    finally:reader.close()

def test_bad_inputs_and_no_cross_list_bridges(ready):
    x,layout,paths=ready;idx=CellIndex(layout,paths[8],shape='ball')
    boundary=int(idx.ranges[0,1])
    assert page_extents(idx,[boundary-1,boundary],set(),gap=2)==[(boundary-1,1),(boundary,1)]
    assert page_extents(idx,[0,2],{1},gap=2)==[(0,1),(2,1)]
    with pytest.raises(ValueError):page_extents(idx,[1],{1})
    assert list(batches([(0,3),(3,1)],4096,16384))==[[(0,3)],[(3,1)]]
    for lists in [[0,0],[-1],[2],[]]:
        with pytest.raises(ValueError):rank(idx,x[0],lists,16)
    with pytest.raises(ValueError):rank(idx,np.zeros(3),[0],16)
    with pytest.raises(ValueError):rank(idx,x[0],[0],0)
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        with pytest.raises(ValueError):search_ramfirst(idx,x[0],reader,shortlist=9)
        with pytest.raises(ValueError):search_ramfirst(idx,x[0],reader,scan='unknown')
    finally:reader.close()
