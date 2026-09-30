"""Discrete optima, packed records, scalar-control equivalence and safe verification."""
from pathlib import Path
import itertools
import subprocess
import numpy as np
import pytest
from test_rotations import data
from geoivf.mixedbits import (WIDTHS,MIXED_KINDS,allocate_bits,allocate_grouped,segments,
    pack_mixed,unpack_mixed,calibration_tables,choose_allocation,build_mixed,MixedIndex)
from geoivf.rotations import RotationIndex,search_rotation
from geoivf.io import MemoryReplay
import hashlib,struct
COUNTS=('candidate_pages','selected_pages','read_pages','gap_pages_read','read_requests','read_bytes','distance_evals','read_stages')
class RecordedReader:
    def __init__(self,reader): self.reader=reader;self.hash=hashlib.sha256()
    def read(self,requests):
        self.hash.update(struct.pack('<Q',len(requests)))
        for o,n in requests:self.hash.update(struct.pack('<QQ',o,n))
        return self.reader.read(requests)

@pytest.fixture(scope='module',autouse=True)
def compiled():
    subprocess.run(['bash',str(Path(__file__).resolve().parents[1]/'scripts/build_mixedbits.sh')],check=True)

@pytest.mark.parametrize('seed',range(8))
def test_optimal_dp(seed):
    rng=np.random.default_rng(seed);e=rng.uniform(0,10,(6,4));budget=24
    bit=allocate_bits(e,budget)
    loss=lambda bs:sum(e[j,WIDTHS.index(int(b))] for j,b in enumerate(bs))
    possibilities=[b for b in itertools.product(WIDTHS,repeat=6) if sum(b)==budget]
    assert loss(bit)==pytest.approx(min(map(loss,possibilities)))
    grouped=allocate_grouped(e,budget,tile=2,max_segments=2)
    legal=[b for b in possibilities if all(b[i]==b[i+1] for i in range(0,6,2)) and len(segments(b))<=2]
    assert loss(grouped)==pytest.approx(min(map(loss,legal)))
    np.testing.assert_array_equal(bit,allocate_bits(e,budget))

@pytest.mark.parametrize('seed',range(6))
def test_pack_roundtrip(seed):
    rng=np.random.default_rng(seed);bits=rng.choice(WIDTHS,size=128).astype(np.uint8)
    codes=np.column_stack([rng.integers(0,1<<int(b),19) for b in bits]).astype(np.uint8)
    raw=pack_mixed(codes,bits)
    assert raw.shape==(19,(int(bits.sum())+7)//8)
    np.testing.assert_array_equal(unpack_mixed(raw,bits),codes)
    # Includes 6-bit coordinates crossing bytes and unaligned 8-bit segments.
    with pytest.raises(ValueError):pack_mixed(codes,np.full(128,1))
    with pytest.raises(ValueError):unpack_mixed(raw[:,:-1],bits)

@pytest.fixture(scope='module')
def mixed(data):
    root,layout,x,basis,uniform=data
    projection=basis.project(x);tables=calibration_tables(projection,projection)
    paths={}
    for kind in MIXED_KINDS:
        out=root/kind;build_mixed(layout,out,projection,basis,tables,kind);paths[kind]=out
    return layout,x,paths,uniform,tables

@pytest.mark.parametrize('kind',MIXED_KINDS)
def test_allocation_and_certificates(mixed,kind):
    layout,x,paths,_,tables=mixed;idx=MixedIndex(layout,paths[kind]);s=idx.summary;a=idx.arrays
    b=a['bit_widths'];assert int(b.sum())==512 and np.all(b>=2)
    assert s['geometry_bytes_per_page']==544 and s['all_points_cover_audited']==len(x)
    if kind in ('mixed-mse','mixed-grouped'):
        assert s['allocation_mse']<=s['uniform4_calibration_mse']+1e-10
    if kind=='mixed-grouped':assert len(a['segments'])<=4 and np.all(a['segments'][:,0]%8==0)
    raw=a['packed'].reshape(-1,64);dec=(a['origin']+unpack_mixed(raw,b)*a['scale']).reshape(-1,8,128)
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for q in (x[0],np.zeros(128,dtype=np.float32),np.full(128,2.1,dtype=np.float32)):
            true=np.linalg.norm(x.astype(float)-q,axis=1);expected=np.lexsort((np.arange(len(x)),true))[:10]
            slots,score,pages,lb,upper,compared=idx.rank(q,[0,1],32)
            z=(q.astype(float)-a['mean'])@a['matrix'];dd=np.sum((dec-z)**2,axis=2)
            dd[np.arange(8)[None]>=idx.valid[:,None]]=np.inf
            good=np.flatnonzero(np.isfinite(dd.ravel()));order=np.lexsort((idx.page_ids.ravel()[good],dd.ravel()[good]))[:32]
            np.testing.assert_array_equal(slots,good[order]);np.testing.assert_allclose(score,dd.ravel()[slots],rtol=1e-14)
            assert compared==len(x) and upper>=np.sort(true)[9]-1e-10
            for p,v in zip(pages,lb):assert v<=true[idx.page_ids[p,:idx.valid[p]]].min()+1e-10
            for mode in ('approx','certified','upper'):
                audit=[];record=RecordedReader(reader)
                ids,st=search_rotation(idx,q,record,shortlist=32,mode=mode,nprobe=2,
                    preassigned_lists=[0,1],io_batch_bytes=4096,scan='native',audit_sink=audit)
                if mode!='approx':np.testing.assert_array_equal(ids,expected)
                assert st['read_bytes']==4096*st['read_pages'] and st['read_pages']<=idx.meta['n_pages']
                for p,tau,v in audit:assert true[idx.page_ids[p,:idx.valid[p]]].min()>tau
    finally:reader.close()

@pytest.mark.parametrize('kind',['pca64-sq8','identity128-sq4','pca128-sq4'])
def test_uniform_scanner_equivalence(mixed,kind):
    layout,x,_,paths,_=mixed;old=RotationIndex(layout,paths[kind]);new=MixedIndex(layout,paths[kind])
    reader=MemoryReplay(layout/'vectors.pages')
    try:
        for q in (x[0],np.zeros(128,dtype=np.float32)):
            for cert in (False,True):
                aa=old.rank(q,[0,1],32,certified=cert);bb=new.rank(q,[0,1],32,certified=cert)
                for i in (0,1,2):np.testing.assert_array_equal(aa[i],bb[i])
                if cert:np.testing.assert_array_equal(aa[3],bb[3]);assert aa[4]==bb[4]
            for mode in ('approx','certified'):
                a=RecordedReader(reader);b=RecordedReader(reader)
                ids,sa=search_rotation(old,q,a,shortlist=32,mode=mode,nprobe=2,preassigned_lists=[0,1],scan='native')
                other,sb=search_rotation(new,q,b,shortlist=32,mode=mode,nprobe=2,preassigned_lists=[0,1],scan='native')
                np.testing.assert_array_equal(ids,other);assert a.hash.hexdigest()==b.hash.hexdigest()
                assert [sa[k] for k in COUNTS]==[sb[k] for k in COUNTS]
    finally:reader.close()

def test_invalid_and_schema(mixed):
    layout,x,paths,_,tables=mixed;idx=MixedIndex(layout,paths['mixed-mse'])
    for err,budget in [(np.ones((3,4)),1),(np.full((3,4),np.nan),12)]:
        with pytest.raises(ValueError):allocate_bits(err,budget)
    with pytest.raises(ValueError):allocate_grouped(np.ones((3,4)),12,tile=2)
    with pytest.raises(ValueError):choose_allocation(tables,'unknown')
    for q,lists in [(np.zeros(4),[0]),(np.full(128,np.nan),[0]),(x[0],[0,0]),(x[0],[2])]:
        with pytest.raises(ValueError):idx.rank(q,lists,32)
    idx.arrays['segments'][0,3]=4
    with pytest.raises(ValueError):idx.rank(x[0],[0],32)
