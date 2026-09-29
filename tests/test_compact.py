"""Tests for compact segment packing, not evidence of ANN performance."""
import json
import numpy as np
import pytest
from geoivf.compact import geometry, diameter2, constrained_pair, rounds, compact_layout
from geoivf.index import build_assigned, Index


def test_rounds_cover_every_pair_once():
    pairs=[]
    for r in rounds():
        assert sorted(j for pair in r for j in pair)==list(range(8))
        pairs.extend(tuple(sorted(p)) for p in r)
    assert len(set(pairs))==28


def test_short_but_thick_is_penalized():
    z=np.zeros((1,8,3));z[0,:,0]=np.arange(8)
    g=geometry(z)
    assert g['worst'][0]<1e-10
    np.testing.assert_allclose(g['span'],7)
    np.testing.assert_allclose(g['score'],49)


def test_tube_encloses_diameter():
    z=np.random.default_rng(3).normal(size=(32,8,12))
    g=geometry(z)
    assert np.all(diameter2(z)<=g['score']+1e-9)


@pytest.mark.parametrize('relaxation',[0.,.05])
def test_pairs_preserve_ids_diameter_and_objective(relaxation):
    rng=np.random.default_rng(11)
    x=rng.normal(size=(32,16,32));z=x[:,:,:16]
    budgets=diameter2(x.reshape(-1,8,32)).reshape(-1,2)*(1+relaxation)**2
    order,use,stats=constrained_pair(z,x,budgets)
    np.testing.assert_array_equal(np.sort(order,axis=1),np.tile(np.arange(16),(len(x),1)))
    xp=np.take_along_axis(x,order[...,None],axis=1)
    zp=np.take_along_axis(z,order[...,None],axis=1)
    assert np.all(diameter2(xp.reshape(-1,8,32)).reshape(-1,2)<=budgets+1e-8)
    old=geometry(z.reshape(-1,8,16))['score'].reshape(-1,2).sum(1)
    new=geometry(zp.reshape(-1,8,16))['score'].reshape(-1,2).sum(1)
    assert np.all(new<=old+1e-8)
    assert stats['feasible']>=len(x)


def test_hidden_full_space_dimension_protected():
    x=np.zeros((1,16,4));x[0,:,0]=np.tile(np.arange(8),2)
    x[0,8:,3]=1000.;z=x[:,:,:2]
    b=diameter2(x.reshape(-1,8,4)).reshape(-1,2)
    order,_,_=constrained_pair(z,x,b)
    xx=np.take_along_axis(x,order[...,None],axis=1)
    assert np.all(diameter2(xx.reshape(-1,8,4)).reshape(-1,2)<=b+1e-8)


def test_identical_points_fallback():
    x=np.ones((2,16,8));order,use,_=constrained_pair(x,x,np.zeros((2,2)))
    assert not use.any()
    np.testing.assert_array_equal(order,np.tile(np.arange(16),(2,1)))


def test_layout_keeps_partial_pages_and_lists(tmp_path):
    rng=np.random.default_rng(20)
    x=rng.normal(size=(139,128)).astype('float32')
    labels=np.r_[np.zeros(67,dtype=int),np.ones(72,dtype=int)]
    centers=np.stack([x[labels==k].mean(0) for k in range(2)])
    source=tmp_path/'source';out=tmp_path/'compact'
    meta=build_assigned(x,centers,labels,source,dims=16,balls=1)
    from geoivf.compact import digest
    meta.update(physical_layout_frozen=True,layout_payload_sha256=digest(source/'vectors.pages'))
    (source/'manifest.json').write_text(json.dumps(meta))
    m=compact_layout(source,out,x,x.astype(float),dims=16)
    a,b=Index(source),Index(out)
    np.testing.assert_array_equal(a.valid,b.valid)
    for first,last in a.ranges:
        np.testing.assert_array_equal(np.sort(a.page_ids[first:last].ravel()),np.sort(b.page_ids[first:last].ravel()))
    partial=a.valid<8
    np.testing.assert_array_equal(a.page_ids[partial],b.page_ids[partial])
    assert m['refinement']['full_space_diameter_violations']==0
    raw=np.memmap(out/'vectors.pages',dtype='<f4',mode='r',shape=(b.meta['n_pages'],1024))
    for p in range(b.meta['n_pages']):
        n=int(b.valid[p]);np.testing.assert_array_equal(raw[p,:n*128].reshape(n,128),x[b.page_ids[p,:n]])


def test_invalid_budgets():
    with pytest.raises(ValueError):
        constrained_pair(np.zeros((1,16,4)),np.zeros((1,16,4)),-np.ones((1,2)))
