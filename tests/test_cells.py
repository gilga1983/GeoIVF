from pathlib import Path
import json
import subprocess
import numpy as np
import pytest
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.dependent import summarize_dependencies, DependentIndex, unpack_codes
from geoivf.cells import CellIndex, certify_cells
from geoivf.io import MemoryReplay
from geoivf.search import search


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    repo = Path(__file__).resolve().parents[1]
    if not (repo/'build/libgeoivf_cells.so').exists():
        subprocess.run(['bash',str(repo/'scripts/build_cells.sh')],check=True)
    root = tmp_path_factory.mktemp('cells')
    rng = np.random.default_rng(6782)
    x = (rng.normal(size=(139,128))*7).astype(np.float32)
    # Include exact zero/constant coordinates and a duplicate point.
    x[:,0] = 3; x[1] = x[0]
    centers = np.stack((x[:70].mean(0), x[70:].mean(0)))
    labels = np.r_[np.zeros(70,dtype=np.int64), np.ones(69,dtype=np.int64)]
    layout = root/'pages'; freeze_layout(x, centers, labels, layout, pack_dims=16)
    basis = Basis.fit(x)
    projected = basis.project(x)
    paths = {}
    for bits in [4,6,8]:
        path = root/f'bits{bits}'
        summarize_dependencies(layout,path,basis,dims=64,bits=bits,scheme='independent',projected_by_id=projected)
        certify_cells(layout,path,projected,path/'cells.json'); paths[bits] = path
    return root,x,labels,layout,projected,paths


@pytest.mark.parametrize('bits',[4,6,8])
def test_native_bounds_and_selection(prepared,bits):
    _,x,_,layout,z,paths = prepared
    indices = {s:CellIndex(layout,paths[bits],shape=s) for s in ['ball','box','hybrid']}
    base = indices['ball']; pp=np.arange(base.meta['n_pages'])
    old = DependentIndex(layout,paths[bits])
    for q in [x[0],x[57],np.zeros(128,dtype=np.float32),np.ones(128,dtype=np.float32)*100]:
        vals={s:idx.geometry(q,pp) for s,idx in indices.items()}
        np.testing.assert_allclose(vals['ball'],old.bounds(q,pp,0,'balls'),atol=1e-11,rtol=1e-13)
        assert np.all(vals['hybrid'] >= vals['ball']-1e-12)
        assert np.all(vals['hybrid'] >= vals['box']-1e-12)
        true=np.linalg.norm(x[np.maximum(base.page_ids,0)].astype(float)-q,axis=2)
        true[base.page_ids<0]=np.inf
        for shape,idx in indices.items():
            assert np.all(vals[shape] <= true.min(axis=1)+1e-9)
            for tau in [0.,1.,40.,80.,float('inf')]:
                full=idx.geometry(q,pp,strategy=1,tau=tau)
                early=idx.geometry(q,pp,strategy=2,tau=tau)
                np.testing.assert_array_equal(full,early)
                np.testing.assert_array_equal(full,vals[shape] <= tau)
    assert 'radii' not in indices['box'].arrays
    assert indices['ball'].geometry_bytes-indices['box'].geometry_bytes == len(pp)*8*4


@pytest.mark.parametrize('bits',[4,6,8])
@pytest.mark.parametrize('shape',['ball','box','hybrid'])
def test_search_equivalence(prepared,bits,shape):
    _,x,labels,layout,_,paths=prepared
    idx=CellIndex(layout,paths[bits],shape=shape)
    with_reader=MemoryReplay(layout/'vectors.pages')
    for q in x[[4,30,78]]:
        ref=np.lexsort((np.arange(len(x)),np.sum((x.astype(float)-q)**2,axis=1)))[:10]
        got,_=search(idx,q,with_reader,nprobe=2,preassigned_lists=[0,1],window_pages=3,filtering='combined')
        np.testing.assert_array_equal(got,ref)
    with_reader.close()


@pytest.mark.parametrize('bits',[4,6,8])
def test_numpy_cell_distance(prepared,bits):
    _,x,_,layout,_,paths=prepared
    idx=CellIndex(layout,paths[bits],shape='box');pp=np.arange(idx.meta['n_pages'])
    a=idx.arrays;code=unpack_codes(a['packed'],bits,(8,64))
    center=a['origin']+a['scale']*code
    q=x[20]; guard=idx.prepare(q)
    gap=np.maximum(0,np.abs(center-idx._zq[:64])-a['scale']/2-idx.padding)
    d=np.linalg.norm(gap,axis=2);d[idx.page_ids<0]=np.inf
    ref=np.maximum(0,d.min(axis=1)-guard)
    np.testing.assert_allclose(idx.geometry(q,pp),ref,atol=1e-11,rtol=1e-13)


def test_overflow_fails_certification(prepared,tmp_path):
    _,_,_,layout,z,paths=prepared
    wrong=z.copy();wrong[0,:64]+=1e9
    with pytest.raises(AssertionError,match='containment'):
        certify_cells(layout,paths[8],wrong,tmp_path/'invalid.json')


def test_invalid_shape_pages(prepared):
    _,x,_,layout,_,paths=prepared
    with pytest.raises(ValueError): CellIndex(layout,paths[8],shape='cone')
    idx=CellIndex(layout,paths[8],shape='box')
    for pages in [[-1],[idx.meta['n_pages']],[[0]]]:
        with pytest.raises(ValueError): idx.geometry(x[0],pages)
    assert len(idx.geometry(x[0],[])) == 0
    with pytest.raises(ValueError):idx.geometry(x[0],[0],strategy=3)


def test_code_boundaries_and_ties():
    # Actual nearest-rounding cells with closed boundaries and no clipping of outliers.
    for bits in [4,6,8]:
        step=1.25;origin=-7.;n=2**bits
        z=origin+np.arange(n)*step
        z=np.r_[z,origin+(np.arange(n-1)+.5)*step]
        code=np.clip(np.rint((z-origin)/step),0,n-1)
        assert np.all(np.abs(z-(origin+code*step))<=step/2+1e-13)
