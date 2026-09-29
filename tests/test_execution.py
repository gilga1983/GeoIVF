"""Same-order SIMD distances, staged plans, and rolling read ownership/errors."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from test_speed import ready, COUNTS
from geoivf.native_scan import NativeTopK
from geoivf.execution import SIMDTopK, RollingPooled
from geoivf.cells import CellIndex
from geoivf.io import MemoryReplay, PooledNative
from geoivf.search import search

@pytest.fixture(scope='module',autouse=True)
def build():
    subprocess.run(['bash',str(Path(__file__).resolve().parents[1]/'scripts/build_execution.sh')],check=True)

@pytest.mark.parametrize('dims',[1,3,4,7,18,33,64,127,128])
@pytest.mark.parametrize('k',[1,10,21])
def test_same_double_sums(dims,k):
    rng=np.random.default_rng(780+dims)
    x=(rng.normal(size=(21,dims))*1000).astype(np.float32); x[5]=x[0]
    q=rng.normal(size=dims).astype(np.float32)
    idx=SimpleNamespace(page_ids=np.arange(24,dtype=np.int64).reshape(3,8),
        valid=np.array([8,8,5],dtype=np.uint16),
        meta=dict(n=21,d=dims,n_pages=3,capacity=8,page_size=4096))
    scalar=NativeTopK(idx,q,k);simd=SIMDTopK(idx,q,k)
    try:
        for p in [2,0,1]:
            payload=x[8*p:min(21,8*p+8)].tobytes()
            raw=b'!'+payload+bytes(4096-len(payload))
            # Deliberately unaligned, read-only byte view.
            view=memoryview(raw)[1:]
            assert scalar.consume([view],[(p,1)])==simd.consume([view],[(p,1)])
            assert scalar.tau2==simd.tau2 and scalar.count==simd.count
            a,da=scalar.finish();b,db=simd.finish()
            np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(da,db)
    finally:scalar.close();simd.close()

@pytest.mark.parametrize('bits',[4,6,8])
@pytest.mark.parametrize('window,gap',[(16,0),(64,0),(256,2),(0,0)])
def test_simd_online_exact_plans(ready,bits,window,gap):
    x,layout,paths=ready;idx=CellIndex(layout,paths[bits],shape='ball')
    class Recorder:
        def __init__(self,r):self.r=r;self.plans=[]
        def read(self,req):self.plans.append(req);return self.r.read(req)
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for mode in ['none','combined']:
            aa,bb=Recorder(reader),Recorder(reader)
            opts=dict(nprobe=2,preassigned_lists=[0,1],filtering=mode,window_pages=window,
                      gap_pages=gap,selection='prepared')
            ids,s=search(idx,x[0],aa,scan='native',**opts)
            other,t=search(idx,x[0],bb,scan='simd',**opts)
            np.testing.assert_array_equal(ids,other);assert aa.plans==bb.plans
            assert [s[k] for k in COUNTS]==[t[k] for k in COUNTS]
    finally:reader.close()

def test_simd_invalid_payload(ready):
    x,layout,_=ready;idx=CellIndex(layout,ready[2][8],shape='ball')
    a=SIMDTopK(idx,x[0]); raw=bytearray(4096)
    raw[:4]=np.array([np.nan],dtype=np.float32).tobytes()
    try:
        with pytest.raises(ValueError,match='nonfinite'):a.consume([raw],[(0,1)])
        with pytest.raises(ValueError):a.finish()
    finally:a.close()

@pytest.mark.parametrize('depth',[1,2,8,32])
@pytest.mark.parametrize('direct',[False,True])
def test_rolling_completions(ready,depth,direct):
    if os.environ.get('GEOIVF_TEST_URING')!='1':pytest.skip('optional io_uring backend')
    if direct and os.environ.get('GEOIVF_TEST_DIRECT')!='1':pytest.skip('optional direct I/O')
    x,layout,paths=ready;path=layout/'vectors.pages';raw=path.read_bytes()
    reader=RollingPooled(path,uring=True,direct=direct,depth=depth)
    reference=PooledNative(path,uring=True,direct=direct,depth=depth)
    try:
        # More than one queue-depth group; duplicates, lengths and reversed IDs.
        rng=np.random.default_rng(79);requests=[]
        for i in range(3*depth+17):
            count=1+i%3; p=int(rng.integers(0,len(raw)//4096-count+1))
            requests.append((p*4096,count*4096))
        for repeat in range(2):
            b=reader.read(requests)
            assert all(bytes(v)==raw[o:o+n] for v,(o,n) in zip(b,requests))
            with pytest.raises(RuntimeError):reader.read(requests)
            reader.release(b)
        idx=CellIndex(layout,paths[8],shape='ball')
        opts=dict(nprobe=2,preassigned_lists=[0,1],selection='prepared',scan='simd',window_pages=16)
        ids,s=search(idx,x[0],reference,**opts);other,t=search(idx,x[0],reader,**opts)
        np.testing.assert_array_equal(ids,other)
        assert [s[k] for k in COUNTS]==[t[k] for k in COUNTS]
        # Error mixed with valid requests; reader cannot reuse failed state.
        with pytest.raises(OSError):reader.read([(len(raw)+4096,4096)]+requests)
        with pytest.raises(OSError):reader.read(requests)
    finally:reader.close();reference.close()
    reader.close()

def test_rolling_requires_async(tmp_path):
    with pytest.raises(ValueError):RollingPooled(tmp_path/'not-opened',uring=False)
