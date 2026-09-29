from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
from geoivf.index import build_assigned, Index
from geoivf.lines import (fit_lines, coordinates, summarize_lines, LineIndex,
                         refine_layout, digest, decode_axes, scalar_intervals)
from geoivf.io import MemoryReplay
from geoivf.search import search


def fixture(tmp_path, n=65):
    rng=np.random.default_rng(90210)
    x=rng.normal(size=(n,128)).astype(np.float32)
    labels=np.arange(n)%3
    centers=np.stack([x[labels==i].mean(axis=0) for i in range(3)])
    physical=tmp_path/'physical'
    meta=build_assigned(x,centers,labels,physical,dims=16,balls=1)
    meta.update(physical_layout_frozen=True,layout_payload_sha256=digest(physical/'vectors.pages'))
    (physical/'manifest.json').write_text(json.dumps(meta))
    basis=SimpleNamespace(mean=np.zeros(128),matrix=np.eye(128),fingerprint=lambda:'identity')
    return x,labels,physical,basis


def test_exact_line_geometry():
    z=np.array([[[0.,0.],[2.,0.],[4.,0.]],[[1.,1.],[1.,1.],[1.,1.]]])
    a,u=fit_lines(z,np.ones((2,3),bool))
    _,rho=coordinates(z,a,u)
    np.testing.assert_allclose(rho,0,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(u,axis=1),1)
    t,r=coordinates(np.array([[[0.,3.],[10.,0.]]]),np.zeros((1,2)),np.array([[1.,0.]]))
    assert np.hypot(t[0,0]-t[0,1],r[0,0]-r[0,1]) == np.sqrt(109.)


@pytest.mark.parametrize('dims',[1,16,64,128])
@pytest.mark.parametrize('precision',['uint8','float32'])
def test_bounds_coverage_and_bytes(tmp_path,dims,precision):
    x,labels,physical,basis=fixture(tmp_path)
    out=tmp_path/'summary'
    info=summarize_lines(physical,out,basis,x.astype(float),dims=dims,precision=precision)
    shell=LineIndex(physical,out,bound='shell'); ball=LineIndex(physical,out,bound='ball')
    expected=2*dims*(1 if precision=='uint8' else 4)+4*8+12
    assert info['bytes_per_page']==expected
    assert sum(info['component_bytes'].values())==info['summary_bytes']
    assert info['all_point_intervals_audited']==len(x)
    for q in list(x[:3])+[np.full(128,1e7,dtype=np.float32)]:
        for li,(first,last) in enumerate(shell.ranges):
            pp=np.arange(first,last)
            ids=shell.page_ids[pp]
            actual=np.linalg.norm(x[np.maximum(ids,0)].astype(float)-q,axis=2)
            actual[ids<0]=np.inf; truth=actual.min(axis=1)
            for mode in ['balls','combined','radial']:
                sb=shell.bounds(q,pp,li,mode); bb=ball.bounds(q,pp,li,mode)
                assert np.all(sb<=truth+1e-7)
                assert np.all(bb<=truth+1e-7)
                assert np.all(sb>=bb-1e-7)
    assert not hasattr(shell,'codes')
    assert digest(physical/'vectors.pages')==info['layout_payload_sha256']


@pytest.mark.parametrize('bound',['shell','ball'])
def test_search_equivalence(tmp_path,bound):
    x,labels,physical,basis=fixture(tmp_path)
    out=tmp_path/'summary'
    summarize_lines(physical,out,basis,x.astype(float),dims=64)
    idx=LineIndex(physical,out,bound=bound)
    reader=MemoryReplay(physical/'vectors.pages')
    try:
        for q in x[:5]:
            ids,stats=search(idx,q,reader,k=10,nprobe=3,numpy_test=True,window_pages=2)
            dd=np.sum((x.astype(float)-q)**2,axis=1)
            np.testing.assert_array_equal(ids,np.lexsort((np.arange(len(x)),dd))[:10])
            assert stats['read_bytes']==4096*stats['read_pages']
    finally: reader.close()


def test_repacking_conserves_lists_and_payload(tmp_path):
    x,labels,physical,basis=fixture(tmp_path,n=101)
    before=Index(physical); new=tmp_path/'refined'
    info=refine_layout(physical,new,x,x.astype(float),dims=64,passes=3)
    after=Index(new)
    assert all(v>=0 for v in info['refinement']['objective_decreases'])
    raw=np.fromfile(new/'vectors.pages',dtype='<f4').reshape(-1,8,128)
    for li,(first,last) in enumerate(before.ranges):
        np.testing.assert_array_equal(np.sort(before.page_ids[first:last].ravel()),np.sort(after.page_ids[first:last].ravel()))
    for p,n in enumerate(after.valid):
        np.testing.assert_array_equal(raw[p,:n],x[after.page_ids[p,:n]])
    out=tmp_path/'new-summary'
    summarize_lines(new,out,basis,x.astype(float),dims=64)
    idx=LineIndex(new,out)
    assert idx.meta['n']==len(x)


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):fit_lines(np.zeros((1,3,4)),np.zeros((1,3),bool))
    with pytest.raises(ValueError):fit_lines(np.full((1,3,4),np.nan),np.ones((1,3),bool))
    x,labels,physical,basis=fixture(tmp_path)
    with pytest.raises(ValueError):summarize_lines(physical,tmp_path/'bad',basis,x,dims=0)
    with pytest.raises(ValueError):summarize_lines(physical,tmp_path/'bad',basis,x,precision='int4')
    out=tmp_path/'summary';summarize_lines(physical,out,basis,x.astype(float))
    with pytest.raises(FileExistsError):summarize_lines(physical,out,basis,x.astype(float))
    with pytest.raises(ValueError):LineIndex(physical,out,bound='unknown')
