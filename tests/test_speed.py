"""Regressions for optional native scan/selection and borrowed aligned buffers."""
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from geoivf.index import Index
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex, certify_cells
from geoivf.io import MemoryReplay, PooledNative
from geoivf.search import search
from geoivf.native_scan import NativeTopK

COUNTS=('candidate_pages','selected_pages','read_pages','gap_pages_read','read_requests',
        'read_bytes','distance_evals','read_stages')

@pytest.fixture(scope='module')
def ready(tmp_path_factory):
    root=Path(__file__).resolve().parents[1]
    subprocess.run(['bash',str(root/'scripts/build_speed.sh')],check=True)
    subprocess.run(['make','-C',str(root)],check=True)
    tmp=tmp_path_factory.mktemp('speed')
    rng=np.random.default_rng(46319)
    x=rng.integers(-15,16,size=(149,128)).astype(np.float32)
    x[0]=x[1]  # true tie across native heap insertion orders
    c=np.stack([x[:75].mean(0),x[75:].mean(0)])
    labels=np.r_[np.zeros(75,dtype=np.int64),np.ones(74,dtype=np.int64)]
    layout=tmp/'pages';freeze_layout(x,c,labels,layout,pack_dims=16)
    basis=Basis.fit(x);z=basis.project(x);paths={}
    for bits in (4,6,8):
        out=tmp/f'b{bits}'
        summarize_dependencies(layout,out,basis,dims=64,bits=bits,scheme='independent',projected_by_id=z)
        certify_cells(layout,out,z,out/'cells.json');paths[bits]=out
    return x,layout,paths

@pytest.mark.parametrize('bits',[4,6,8])
@pytest.mark.parametrize('shape',['ball','box','hybrid'])
@pytest.mark.parametrize('mode',['balls','radial','combined'])
def test_online_selection_and_scan(ready,bits,shape,mode):
    x,layout,paths=ready;idx=CellIndex(layout,paths[bits],shape=shape)
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for q in [x[0],x[88],np.full(128,40,dtype=np.float32)]:
            expected,base=search(idx,q,reader,nprobe=2,preassigned_lists=[0,1],
                                 filtering=mode,window_pages=3,gap_pages=1)
            oracle=np.lexsort((np.arange(len(x)),((x.astype(float)-q)**2).sum(1)))[:10]
            np.testing.assert_array_equal(expected,oracle)
            for selection in ['bounds','fixed','adaptive']:
                audit=[]
                got,stat=search(idx,q,reader,nprobe=2,preassigned_lists=[0,1],
                    filtering=mode,window_pages=3,gap_pages=1,selection=selection,scan='native',audit_sink=audit)
                np.testing.assert_array_equal(got,oracle)
                assert {k:stat[k] for k in COUNTS}=={k:base[k] for k in COUNTS}
                for p,tau,lb in audit:
                    v=x[idx.page_ids[p,:idx.valid[p]]].astype(float)
                    actual=np.linalg.norm(v-q,axis=1).min()
                    assert actual>tau and lb<=actual+1e-9
    finally: reader.close()

@pytest.mark.parametrize('k',[1,10,149])
def test_heap_scan_and_ties(ready,k):
    x,layout,_=ready;idx=Index(layout);reader=MemoryReplay(layout/'vectors.pages')
    acc=NativeTopK(idx,x[0],k)
    try:
        # Reverse physical-page order exercises insertion-order independence.
        for p in range(idx.meta['n_pages']-1,-1,-1):
            data=reader.read([(p*4096,4096)])
            assert acc.consume(data,[(p,1)])==idx.valid[p]
        ids,dd=acc.finish()
        true=np.sum((x.astype(float)-x[0])**2,axis=1)
        oracle=np.lexsort((np.arange(len(x)),true))[:k]
        np.testing.assert_array_equal(ids,oracle);np.testing.assert_array_equal(dd,true[oracle])
    finally: acc.close();reader.close()
    acc.close()
    with pytest.raises(ValueError): acc.finish()


def test_bad_scan_buffers_and_modes(ready):
    x,layout,paths=ready;idx=Index(layout);acc=NativeTopK(idx,x[0],10)
    try:
        with pytest.raises(EOFError):acc.consume([b'abc'],[(0,1)])
        with pytest.raises(ValueError):acc.consume([bytes(4096)],[(-1,1)])
        with pytest.raises(ValueError):acc.consume([],[(0,1)])
    finally: acc.close()
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        with pytest.raises(ValueError):search(idx,x[0],reader,selection='adaptive')
        with pytest.raises(ValueError):search(idx,x[0],reader,scan='nonsense')
    finally: reader.close()
    cell=CellIndex(layout,paths[8],shape='ball')
    for tau in [-1,float('nan')]:
        with pytest.raises(ValueError):cell.select(x[0],[0],0,'combined',tau)


@pytest.mark.parametrize('direct',[False,True])
@pytest.mark.parametrize('uring',[False,True])
def test_pooled_reader_lifetime(ready,direct,uring):
    if direct and os.environ.get('GEOIVF_TEST_DIRECT')!='1': pytest.skip('optional direct I/O')
    if uring and os.environ.get('GEOIVF_TEST_URING')!='1': pytest.skip('optional io_uring')
    x,layout,paths=ready
    idx=CellIndex(layout,paths[8],shape='ball')
    reader=PooledNative(layout/'vectors.pages',direct=direct,uring=uring,depth=2)
    try:
        b=reader.read([(0,4096),(8192,4096)])
        expected=(layout/'vectors.pages').read_bytes()
        assert bytes(b[0])==expected[:4096] and bytes(b[1])==expected[8192:12288]
        allocations=reader.allocations
        with pytest.raises(RuntimeError):reader.read([(0,4096)])
        reader.release(b)
        for _ in range(3):
            b=reader.read([(8192,4096),(0,4096)]);reader.release(b)
        assert reader.allocations==allocations
        for scan in ['python','native']:
            got,_=search(idx,x[0],reader,nprobe=2,preassigned_lists=[0,1],
                selection='adaptive',scan=scan,window_pages=3)
            true=np.sum((x.astype(float)-x[0])**2,axis=1)
            np.testing.assert_array_equal(got,np.lexsort((np.arange(len(x)),true))[:10])
        assert reader.reserved_bytes<=reader.max_pool_bytes
        with pytest.raises(MemoryError):reader.read([(0,reader.max_pool_bytes+4096)])
        with pytest.raises(OSError):reader.read([(expected.__len__()+4096,4096)])
    finally: reader.close()
    reader.close()


def test_noninteger_scan_agreement(ready):
    # Non-integer distances are checked with tolerance, not a false bitwise claim.
    x,layout,_=ready;idx=Index(layout);q=np.random.default_rng(92).normal(size=128).astype(np.float32)
    reader=MemoryReplay(layout/'vectors.pages');acc=NativeTopK(idx,q,10)
    try:
        count=idx.meta['n_pages'];acc.consume(reader.read([(0,4096*count)]),[(0,count)])
        got,dd=acc.finish();dist=np.sum((x.astype(float)-q)**2,axis=1)
        oracle=np.lexsort((np.arange(len(x)),dist))[:10]
        np.testing.assert_array_equal(got,oracle)
        np.testing.assert_allclose(dd,dist[oracle],rtol=1e-13,atol=1e-10)
    finally:acc.close();reader.close()
