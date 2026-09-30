"""Fixed-page, equal-code-budget scalar/PQ transformations and safe intervals.

No inverse transform or original-vector access occurs during ranking. Training
uses the learning set; scalar min/max calibration uses the indexed base vectors.
"""
from __future__ import annotations
import ctypes as ct
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .index import Index
from .projections import Basis
from .dependent import pack_codes, unpack_codes, file_hash
from .ramfirst import page_extents, batches
from .execution import SIMDTopK
from .native_scan import NativeTopK
from .search import Stats
from dataclasses import asdict

EPS=np.finfo(np.float64).eps
KINDS=('pca64-sq8','pcar64-sq8','identity128-sq4','pca128-sq4',
       'random128-sq4','pq64x8','opq64x8')


def transform_limits(matrix):
    """Gershgorin bounds plus FP64 dot/row-sum error allowance.

    A rectangular projection has no full-input-space positive minimum gain.
    The upper bound applies to all transforms; a positive lower bound is used
    only for square, sufficiently nonsingular transforms.
    """
    a=np.asarray(matrix,dtype=np.float64)
    if a.ndim!=2 or not min(a.shape) or not np.isfinite(a).all():raise ValueError('invalid transform')
    gram=a.T@a
    err=128*EPS*max(a.shape)*float(np.max((np.abs(a).T@np.abs(a)).sum(axis=1)))
    absolute=np.abs(gram)
    hi=float(absolute.sum(axis=1).max())+err
    lo=float((np.diag(gram)-(absolute.sum(axis=1)-np.abs(np.diag(gram)))).min())-err
    if hi<=0 or not np.isfinite(hi):raise ValueError('degenerate transform')
    upper=float(np.nextafter(np.sqrt(hi),np.inf))
    lower=float(np.nextafter(np.sqrt(lo),0.)) if a.shape[0]==a.shape[1] and lo>0 else 0.
    return upper,lower


def random_rotation(d,seed):
    q,r=np.linalg.qr(np.random.default_rng(seed).normal(size=(d,d)))
    return q*np.where(np.diag(r)<0,-1.,1.)[None]


def build_rotation(layout_dir,out,x,learn,basis,kind,*,train_count=50000,opq_iters=20):
    if kind not in KINDS:raise ValueError('unknown encoding')
    base=Index(layout_dir);m=base.meta;out=Path(out)
    if x.shape!=(m['n'],128) or learn.shape[1]!=128 or basis.matrix.shape!=(128,128):
        raise ValueError('this qualification requires 128-dimensional data')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    if file_hash(Path(layout_dir)/'vectors.pages')!=m['layout_payload_sha256']:raise ValueError('payload changed')
    out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    mean=basis.mean.copy();matrix=np.eye(128);pq=None;book=np.empty((0,),dtype=np.float32)
    groups=64 if kind.endswith('sq8') or 'pq' in kind else 128
    bits=8 if groups==64 else 4
    train_ids=np.random.default_rng(94021).permutation(len(learn))[:min(train_count,len(learn))]
    train=(learn[train_ids].astype(float)-mean).astype(np.float32)
    if kind=='pca64-sq8':matrix=basis.matrix[:,:64].copy()
    elif kind=='pcar64-sq8':matrix=basis.matrix[:,:64]@random_rotation(64,74193)
    elif kind=='pca128-sq4':matrix=basis.matrix.copy()
    elif kind=='random128-sq4':matrix=random_rotation(128,74201)
    if kind in ('pq64x8','opq64x8'):
        import faiss
        if kind=='opq64x8':
            opq=faiss.OPQMatrix(128,64)
            opq.niter=opq_iters;opq.niter_pq=4;opq.niter_pq_0=20
            opq.max_train_points=len(train);opq.verbose=False
            opq.train(train)
            if opq.have_bias:raise ValueError('unexpected OPQ bias')
            matrix=faiss.vector_to_array(opq.A).reshape(128,128).astype(float).T.copy()
        pq=faiss.ProductQuantizer(128,64,8)
        pq.cp.niter=20;pq.cp.seed=94021;pq.cp.max_points_per_centroid=max(256,len(train))
        pq.train(np.ascontiguousarray(train.astype(float)@matrix,dtype=np.float32))
        book=faiss.vector_to_array(pq.centroids).reshape(64,256,2).copy()
    norm_upper,norm_lower=transform_limits(matrix)
    dims=matrix.shape[1]
    projection=np.empty((len(x),dims),dtype=np.float64)
    for s in range(0,len(x),32768):projection[s:s+32768]=(x[s:s+32768].astype(float)-mean)@matrix
    origin=projection.min(axis=0);scale=(projection.max(axis=0)-origin)/(2**bits-1)
    scale[scale==0]=1.
    guard=8192*EPS*128**2*(1+float(np.max(np.abs(projection)))+float(np.max(np.abs(mean))))
    raw=np.empty((len(x),64),dtype=np.uint8);decoded_error=np.empty(len(x),dtype=np.float64)
    omitted=np.zeros(len(x))
    for s in range(0,len(x),32768):
        z=projection[s:s+32768]
        if pq is None:
            codes=np.clip(np.rint((z-origin)/scale),0,2**bits-1).astype(np.uint8)
            raw[s:s+len(z)]=pack_codes(codes,bits)
            decoded=origin+codes.astype(float)*scale
        else:
            codes=pq.compute_codes(np.ascontiguousarray(z,dtype=np.float32))
            raw[s:s+len(z)]=codes
            decoded=pq.decode(codes).astype(float)
            # Verify codebook/decoded convention, not just library self-consistency.
            check=book[np.arange(groups)[None],codes].reshape(-1,128).astype(float)
            np.testing.assert_array_equal(check,decoded)
        decoded_error[s:s+len(z)]=np.linalg.norm(z-decoded,axis=1)
        if dims<128:
            # Diagnostic only: full ideal centered norm minus retained norm.
            full=np.sum((x[s:s+len(z)].astype(float)-mean)**2,axis=1)
            omitted[s:s+len(z)]=np.sqrt(np.maximum(0.,full-np.sum(z*z,axis=1)))
    padded=np.maximum(base.page_ids,0);active=np.arange(m['capacity'])[None]<base.valid[:,None]
    packed=raw[padded].reshape(m['n_pages'],-1).copy()
    radii=np.nextafter((decoded_error[padded]+guard).astype(np.float32),np.float32(np.inf))
    radii[~active]=-1
    if np.any(radii[active].astype(float)<decoded_error[padded][active]):raise AssertionError('radius undercoverage')
    arrays=dict(packed=packed,radii=radii,mean=mean,matrix=np.ascontiguousarray(matrix),
                origin=origin,scale=scale,book=book)
    # Re-read actual packed page records and certify every point after decoding.
    worst=-float('inf');audited=0
    for s in range(0,m['n_pages'],1024):
        codes=unpack_codes(packed[s:s+1024],bits,(m['capacity'],groups))
        if pq is None:dec=origin+codes.astype(float)*scale
        else:dec=book[np.arange(groups)[None,None],codes].reshape(len(codes),m['capacity'],128).astype(float)
        ids=padded[s:s+len(codes)];mask=active[s:s+len(codes)]
        err=np.linalg.norm(projection[ids]-dec,axis=2)-radii[s:s+len(codes)]
        worst=max(worst,float(err[mask].max()));audited+=int(mask.sum())
        if np.any(err[mask]>0):raise AssertionError('packed cover failure')
    np.savez(out/'rotation.npz',**arrays)
    structural=sum(getattr(base,key).nbytes for key in ('centers','page_ids','valid','radial','ranges'))
    geometry=packed.nbytes+radii.nbytes;shared=sum(v.nbytes for key,v in arrays.items() if key not in ('packed','radii'))
    meta=dict(kind=kind,dims=dims,groups=groups,bits=bits,pq=pq is not None,
        norm_upper=norm_upper,norm_lower=norm_lower,build_guard=guard,
        code_bytes_per_vector=64,geometry_bytes_per_page=geometry/m['n_pages'],
        directory_array_bytes=structural+geometry+shared,geometry_bytes=geometry,shared_bytes=shared,
        radius_mean=float(decoded_error.mean()),radius_p95=float(np.quantile(decoded_error,.95)),
        radius_max=float(decoded_error.max()),omitted_norm_mean=float(omitted.mean()),
        build_seconds=time.monotonic()-began,learning_rows=len(learn),pq_training_rows=len(train) if pq else 0,
        opq_iterations=opq_iters if kind=='opq64x8' else 0,pq_iterations=20 if pq else 0,
        all_points_cover_audited=audited,maximum_cover_excess=worst,
        layout_payload_sha256=m['layout_payload_sha256'],
        transform_sha256=hashlib.sha256(mean.tobytes()+matrix.tobytes()).hexdigest(),
        code_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
        data_queries_used_for_training=False,scalar_ranges_calibrated_on_base=True)
    (out/'rotation.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


@lru_cache(maxsize=4)
def load_library(path):
    lib=ct.CDLL(str(path));f=lib.grot_rank
    f.argtypes=[ct.c_void_p]*5+[ct.c_uint64]*2+[ct.c_uint]*6+[ct.c_void_p]+[ct.c_double]*3+[ct.c_int]+[ct.c_void_p]*5
    f.restype=ct.c_int
    return lib


class RotationIndex(Index):
    def __init__(self,layout_dir,summary_dir):
        super().__init__(layout_dir)
        self.summary=json.loads((Path(summary_dir)/'rotation.json').read_text())
        if self.summary['layout_payload_sha256']!=self.meta['layout_payload_sha256']:raise ValueError('wrong layout')
        for key in ('codes','radii','coordinates','origin','scale'):delattr(self,key)
        with np.load(Path(summary_dir)/'rotation.npz',allow_pickle=False) as f:self.arrays={k:np.ascontiguousarray(f[k]) for k in f.files}
        self.meta=dict(self.meta,summary_bytes=self.summary['geometry_bytes'],directory_array_bytes=self.summary['directory_array_bytes'])
        self.fn=load_library(Path(__file__).resolve().parents[1]/'build/libgeoivf_rotation.so').grot_rank

    def rank(self,q,lists,wanted,*,certified=True,k=10):
        q=np.ascontiguousarray(q,dtype=np.float64);li=np.asarray(lists,dtype=np.int64)
        m=self.meta;s=self.summary;a=self.arrays
        if q.shape!=(m['d'],) or not np.isfinite(q).all() or not 1<=k<=wanted<=1000000:raise ValueError('invalid query/shortlist')
        if li.ndim!=1 or not len(li) or np.any(li<0) or np.any(li>=m['nlist']) or len(np.unique(li))!=len(li):raise ValueError('invalid lists')
        pages=np.concatenate([np.arange(*self.ranges[l],dtype=np.int64) for l in li])
        z=(q-a['mean'])@a['matrix']
        if s['pq']:
            delta=a['book'].astype(float)-z.reshape(s['groups'],1,-1)
            lut=np.ascontiguousarray(np.sum(delta*delta,axis=2))
        else:
            centers=a['origin'][:,None]+a['scale'][:,None]*np.arange(2**s['bits'])[None]
            lut=np.ascontiguousarray((z[:,None]-centers)**2)
        guard=float(s['build_guard']+8192*EPS*m['d']**2*(1+np.max(np.abs(q))))
        slots=np.empty(wanted,dtype=np.int64);scores=np.empty(wanted,dtype=float)
        lbs=np.empty(len(pages),dtype=float);ub=ct.c_double();compared=ct.c_uint64()
        n=self.fn(a['packed'].ctypes.data,a['radii'].ctypes.data if certified else None,
            self.valid.ctypes.data,self.page_ids.ctypes.data,pages.ctypes.data,len(pages),m['n_pages'],
            m['capacity'],s['groups'],s['bits'],a['packed'].shape[1],wanted,k,lut.ctypes.data,
            guard,s['norm_upper'],s['norm_lower'],int(certified),slots.ctypes.data,scores.ctypes.data,
            lbs.ctypes.data,ct.byref(ub),ct.byref(compared))
        if n<0:raise ValueError(f'native rank failed ({n})')
        if certified:
            offset=0
            for l in li:
                lo,hi=map(int,self.ranges[l]);nn=hi-lo;r=np.linalg.norm(q-self.centers[l])
                rb=np.maximum(self.radial[lo:hi,0]-r,r-self.radial[lo:hi,1])
                lbs[offset:offset+nn]=np.maximum(lbs[offset:offset+nn],np.maximum(0.,rb-guard));offset+=nn
        return slots[:n],scores[:n],pages,lbs,float(ub.value),int(compared.value)


def search_rotation(index,q,reader,*,shortlist=64,mode='certified',k=10,nprobe=64,
                    preassigned_lists=None,io_batch_bytes=16<<20,scan='simd',audit_sink=None):
    if mode not in ('approx','certified','upper') or scan not in ('simd','native') or not 1<=k<=shortlist:
        raise ValueError('invalid mode or shortlist')
    if mode=='upper' and index.summary['norm_lower']<=0:raise ValueError('one-wave upper bound requires full rank')
    budget=min(int(io_batch_bytes),int(getattr(reader,'max_pool_bytes',io_batch_bytes)))
    if budget<index.meta['page_size']:raise ValueError('I/O budget smaller than a page')
    max_extent=min(256,(1<<(budget.bit_length()-1))//index.meta['page_size'])
    started=time.perf_counter_ns();stat=Stats();q=np.asarray(q,dtype=np.float32)
    t=time.perf_counter_ns();lists=index.route(q,nprobe) if preassigned_lists is None else np.asarray(preassigned_lists,dtype=np.int64)
    if lists.shape!=(nprobe,):raise ValueError('invalid routing size')
    stat.route_us=(time.perf_counter_ns()-t)/1000
    t=time.perf_counter_ns();slots,scores,pages,lbs,upper,compared=index.rank(q,lists,shortlist,certified=mode!='approx',k=k)
    rank_us=(time.perf_counter_ns()-t)/1000;stat.filter_us+=rank_us;stat.candidate_pages=len(pages)
    acc=(SIMDTopK if scan=='simd' else NativeTopK)(index,q,k)
    visited=set();waves=0;first_pages=0;second_pages=0;certificate_us=0.;threshold=None
    def verify(selected):
        nonlocal waves
        t=time.perf_counter_ns();selected=np.unique(selected)
        ext=page_extents(index,selected,visited,gap=0,max_extent=max_extent)
        stat.filter_us+=(time.perf_counter_ns()-t)/1000;stat.selected_pages+=len(selected)
        if not ext:return 0
        waves+=1;fetched=0
        for batch in batches(ext,index.meta['page_size'],budget):
            requests=[(p*index.meta['page_size'],n*index.meta['page_size']) for p,n in batch]
            t=time.perf_counter_ns();buffers=reader.read(requests);stat.io_us+=(time.perf_counter_ns()-t)/1000
            t=time.perf_counter_ns()
            try:stat.distance_evals+=acc.consume(buffers,batch)
            finally:
                release=getattr(reader,'release',None)
                if release:release(buffers)
            stat.distance_us+=(time.perf_counter_ns()-t)/1000
            for p,n in batch:visited.update(range(p,p+n));fetched+=n
            stat.read_requests+=len(batch);stat.read_stages+=1
        stat.read_pages+=fetched;stat.read_bytes+=fetched*index.meta['page_size']
        return fetched
    try:
        if mode=='upper':
            threshold=upper
            if audit_sink is not None:audit_sink.extend((int(p),upper,float(v)) for p,v in zip(pages[lbs>upper],lbs[lbs>upper]))
            first_pages=verify(pages[lbs<=upper])
        else:
            first_pages=verify(slots//index.meta['capacity'])
            if mode=='certified':
                t=time.perf_counter_ns();threshold=np.sqrt(acc.tau2)
                unread=np.fromiter((int(p) not in visited for p in pages),dtype=bool,count=len(pages))
                eligible=unread&(lbs<=threshold)
                if audit_sink is not None:audit_sink.extend((int(p),float(threshold),float(v)) for p,v in zip(pages[unread&~eligible],lbs[unread&~eligible]))
                certificate_us=(time.perf_counter_ns()-t)/1000;stat.filter_us+=certificate_us
                second_pages=verify(pages[eligible])
        t=time.perf_counter_ns();ids,_=acc.finish();stat.distance_us+=(time.perf_counter_ns()-t)/1000
    finally:acc.close()
    stat.total_us=(time.perf_counter_ns()-started)/1000
    return ids,dict(asdict(stat),verification_waves=waves,first_wave_pages=first_pages,
        second_wave_pages=second_pages,rank_us=rank_us,certificate_us=certificate_us,
        shortlist=shortlist,mode=mode,approximate_comparisons=compared)
