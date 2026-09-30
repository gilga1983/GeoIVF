"""Transform safety, packed code budgets and exact verification contracts."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pytest
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.rotations import (transform_limits,random_rotation,build_rotation,
    RotationIndex,search_rotation,KINDS)
from geoivf.io import MemoryReplay
from geoivf.dependent import unpack_codes

@pytest.fixture(scope='module')
def data(tmp_path_factory):
    subprocess.run(['bash',str(Path(__file__).resolve().parents[1]/'scripts/build_rotations.sh')],check=True)
    rng=np.random.default_rng(128333)
    x=rng.normal(size=(149,128)).astype(np.float32);x[1]=x[0]
    labels=np.r_[np.zeros(75,dtype=np.int64),np.ones(74,dtype=np.int64)]
    centers=np.stack([x[:75].mean(0),x[75:].mean(0)]).astype(np.float32)
    root=tmp_path_factory.mktemp('rotations');layout=root/'pages'
    freeze_layout(x,centers,labels,layout);basis=Basis.fit(x)
    paths={}
    for kind in KINDS[:5]:
        p=root/kind;build_rotation(layout,p,x,x,basis,kind);paths[kind]=p
    return root,layout,x,basis,paths

@pytest.mark.parametrize('d',[3,8,64,128])
def test_limits(d):
    r=random_rotation(d,43)
    for a in [r,0.5*r,r[:,:max(1,d//2)],r.astype(np.float32).astype(float)]:
        hi,lo=transform_limits(a);sv=np.linalg.svd(a,compute_uv=False)
        assert hi>=sv.max()-1e-14
        if a.shape[0]==a.shape[1]:assert 0<lo<=sv.min()+1e-14
        else:assert lo==0
    with pytest.raises(ValueError):transform_limits(np.zeros((d,d)))
    with pytest.raises(ValueError):transform_limits(np.full((d,d),np.nan))

@pytest.mark.parametrize('kind',KINDS[:5])
@pytest.mark.parametrize('r',[10,32,149])
def test_scalar_rank_and_certificate(data,kind,r):
    root,layout,x,basis,paths=data;idx=RotationIndex(layout,paths[kind]);reader=MemoryReplay(layout/'vectors.pages')
    try:
        assert idx.summary['geometry_bytes_per_page']==544
        assert idx.summary['all_points_cover_audited']==len(x)
        for q in [x[0],np.zeros(128,dtype=np.float32),np.full(128,4,dtype=np.float32)]:
            q64=q.astype(float);dist=np.linalg.norm(x.astype(float)-q64,axis=1)
            expected=np.lexsort((np.arange(len(x)),dist))[:10]
            slots,scores,pages,lb,upper,n=idx.rank(q,[0,1],r,k=10)
            a=idx.arrays;s=idx.summary
            codes=unpack_codes(a['packed'],s['bits'],(8,s['groups']))
            recon=a['origin']+codes.astype(float)*a['scale']
            dd=np.sum((recon-((q64-a['mean'])@a['matrix']))**2,axis=2)
            dd[np.arange(8)[None]>=idx.valid[:,None]]=np.inf
            eligible=np.flatnonzero(np.isfinite(dd.ravel()))
            order=np.lexsort((idx.page_ids.ravel()[eligible],dd.ravel()[eligible]))[:r]
            np.testing.assert_array_equal(slots,eligible[order]);np.testing.assert_allclose(scores,dd.ravel()[slots],rtol=1e-14)
            for p,v in zip(pages,lb):assert v<=dist[idx.page_ids[p,:idx.valid[p]]].min()+1e-10
            if s['norm_lower']>0:assert upper>=np.sort(dist)[9]-1e-10
            else:assert np.isinf(upper)
            for mode in ['approx','certified']+(['upper'] if s['norm_lower'] else []):
                audit=[];ids,stat=search_rotation(idx,q,reader,shortlist=r,mode=mode,nprobe=2,
                    preassigned_lists=[0,1],io_batch_bytes=4096,scan='native',audit_sink=audit)
                if mode!='approx':np.testing.assert_array_equal(ids,expected)
                assert stat['read_bytes']==stat['read_pages']*4096
                assert stat['verification_waves']<= (2 if mode=='certified' else 1)
                assert stat['read_pages']<=idx.meta['n_pages']
                for p,tau,bound in audit:assert bound<=dist[idx.page_ids[p,:idx.valid[p]]].min()+1e-10 and dist[idx.page_ids[p,:idx.valid[p]]].min()>tau
    finally:reader.close()

def test_invalid(data):
    _,layout,x,_,paths=data;idx=RotationIndex(layout,paths['pca64-sq8'])
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        with pytest.raises(ValueError):search_rotation(idx,x[0],reader,mode='upper',nprobe=2)
        with pytest.raises(ValueError):idx.rank(x[0],[0,0],32)
        with pytest.raises(ValueError):idx.rank(np.zeros(3),[0],32)
        with pytest.raises(ValueError):idx.rank(x[0],[2],32)
    finally:reader.close()

@pytest.mark.parametrize('kind',['pq64x8','opq64x8'])
def test_upstream_pq_decode_and_safety(data,kind):
    faiss=pytest.importorskip('faiss');faiss.omp_set_num_threads(1)
    root,layout,x,basis,_=data
    learn=np.random.default_rng(891).normal(size=(1024,128)).astype(np.float32)
    path=root/kind;meta=build_rotation(layout,path,x,learn,basis,kind,train_count=1024,opq_iters=1)
    idx=RotationIndex(layout,path);reader=MemoryReplay(layout/'vectors.pages')
    try:
        assert meta['geometry_bytes_per_page']==544 and meta['norm_lower']>0
        for mode in ['certified','upper']:
            ids,_=search_rotation(idx,x[0],reader,mode=mode,shortlist=16,nprobe=2,preassigned_lists=[0,1],scan='native')
            dd=np.sum((x.astype(float)-x[0])**2,axis=1);expected=np.lexsort((np.arange(len(x)),dd))[:10]
            np.testing.assert_array_equal(ids,expected)
    finally:reader.close()
