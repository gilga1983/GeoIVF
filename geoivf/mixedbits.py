"""Fixed-length mixed-precision scalar codes over frozen IVF pages.

Allocation uses learning vectors, never evaluation queries. The min/max bins
retain the preceding experiment's base-corpus calibration policy. Radii cover
actual decoded points in ALL represented dimensions, including low-bit tails.
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
from .rotations import EPS, transform_limits
from .dependent import file_hash

WIDTHS = (2, 4, 6, 8)
MIXED_KINDS = ('mixed-manual', 'mixed-variance', 'mixed-mse', 'mixed-grouped')


def validate_bits(bits):
    raw = np.asarray(bits)
    if raw.ndim != 1 or not len(raw) or not np.isin(raw, WIDTHS).all():
        raise ValueError('bit widths must be a nonempty vector of 2/4/6/8')
    return raw.astype(np.uint8)


def segments(bits):
    """first coordinate, coordinate count, width, bit offset, LUT offset."""
    b = validate_bits(bits); rows = []; bit = 0; table = 0; first = 0
    for end in range(1, len(b)+1):
        if end == len(b) or b[end] != b[first]:
            width = int(b[first]); n = end-first
            rows.append((first, n, width, bit, table))
            bit += n*width; table += n*(1 << width); first = end
    return np.asarray(rows, dtype=np.uint32)


def pack_mixed(codes, bits):
    b = validate_bits(bits); a = np.asarray(codes)
    if a.ndim != 2 or a.shape[1] != len(b) or not np.issubdtype(a.dtype,np.integer):
        raise ValueError('invalid scalar codes')
    if np.any(a < 0) or np.any(a >= (1 << b.astype(int))):
        raise ValueError('code outside its quantization bin range')
    out = np.zeros((len(a), (int(b.sum())+7)//8), dtype=np.uint8); pos = 0
    for j, width in enumerate(map(int,b)):
        byte, shift = divmod(pos,8); v = a[:,j].astype(np.uint16) << shift
        out[:,byte] |= (v & 255).astype(np.uint8)
        if shift+width > 8: out[:,byte+1] |= (v >> 8).astype(np.uint8)
        pos += width
    return out


def unpack_mixed(packed, bits):
    b = validate_bits(bits); a = np.asarray(packed)
    if a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != (int(b.sum())+7)//8:
        raise ValueError('invalid packed array')
    out = np.empty((len(a),len(b)),dtype=np.uint8); pos = 0
    for j, width in enumerate(map(int,b)):
        byte, shift = divmod(pos,8); v = a[:,byte].astype(np.uint16)
        if shift+width > 8: v |= a[:,byte+1].astype(np.uint16) << 8
        out[:,j] = (v >> shift) & ((1 << width)-1); pos += width
    return out


def allocate_bits(errors, budget, widths=WIDTHS):
    """Exact discrete separable-MSE minimizer; not a recall optimizer."""
    e = np.asarray(errors,dtype=float); w = np.asarray(widths,dtype=int)
    if (e.ndim != 2 or e.shape[1] != len(w) or not len(e) or budget < 0 or
        not np.isfinite(e).all() or np.any(e < 0) or np.any(w <= 0) or len(np.unique(w)) != len(w)):
        raise ValueError('invalid allocation objective')
    dp = np.full(budget+1,np.inf); dp[0] = 0.
    back = np.full((len(e),budget+1),-1,dtype=np.int16)
    for j in range(len(e)):
        nxt = np.full_like(dp,np.inf)
        for wi, cost in enumerate(w):
            if cost > budget: continue
            cand = dp[:budget+1-cost] + e[j,wi]
            better = cand < nxt[cost:]
            nxt[cost:][better] = cand[better]
            back[j,cost:][better] = wi
        dp = nxt
    if not np.isfinite(dp[budget]): raise ValueError('infeasible bit budget')
    b = np.empty(len(e),dtype=np.uint8); used = budget
    for j in range(len(e)-1,-1,-1):
        wi = int(back[j,used]); b[j] = w[wi]; used -= int(w[wi])
    assert used == 0
    return b


def allocate_grouped(errors, budget, *, tile=8, max_segments=4, widths=WIDTHS):
    """Exact optimum with <=max_segments and boundaries at multiples of tile."""
    e = np.asarray(errors,dtype=float); w = tuple(map(int,widths))
    if (e.ndim != 2 or e.shape[1] != len(w) or not len(e) or tile < 1 or len(e)%tile or
        max_segments < 1 or budget < 0 or not np.isfinite(e).all() or np.any(e < 0) or min(w)<=0):
        raise ValueError('invalid grouped objective')
    n = len(e)//tile; prefix = np.vstack((np.zeros(len(w)),np.cumsum(e,axis=0)))
    dp = {(0,0,0): 0.}; parent = {}
    for group in range(max_segments):
        states = [(key,val) for key,val in dp.items() if key[0] == group]
        for (_,start,used), val in states:
            for end in range(start+1,n+1):
                for wi,b in enumerate(w):
                    newused = used+(end-start)*tile*b
                    if newused>budget: continue
                    key=(group+1,end,newused)
                    loss=val+prefix[end*tile,wi]-prefix[start*tile,wi]
                    if loss < dp.get(key,np.inf):
                        dp[key]=loss; parent[key]=(start,used,b)
    finals=[key for key in dp if key[1:]==(n,budget)]
    if not finals: raise ValueError('infeasible grouped bit budget')
    key=min(finals,key=lambda k:(dp[k],k[0])); b=np.empty(len(e),dtype=np.uint8)
    while key[0]:
        start,oldused,width=parent[key]; b[start*tile:key[1]*tile]=width
        key=(key[0]-1,start,oldused)
    return b


def calibration_tables(projection, learning_projection):
    z=np.asarray(learning_projection,dtype=float)
    if z.ndim!=2 or not len(z) or projection.shape[1]!=z.shape[1] or not np.isfinite(z).all():
        raise ValueError('invalid calibration data')
    low=np.asarray(projection.min(axis=0),dtype=float); high=np.asarray(projection.max(axis=0),dtype=float)
    errors=[]
    for bits in WIDTHS:
        step=(high-low)/(2**bits-1); step[step==0]=1.
        code=np.clip(np.rint((z-low)/step),0,2**bits-1)
        errors.append(np.mean((z-(low+step*code))**2,axis=0))
    return dict(low=low,high=high,errors=np.column_stack(errors),variance=np.var(z,axis=0),learning_rows=len(z))


def choose_allocation(tables, kind, budget=512):
    errors=tables['errors']; d=len(errors)
    if kind=='mixed-manual':
        if d!=128 or budget!=512: raise ValueError('manual schedule requires 128D/512 bits')
        b=np.r_[np.full(32,8),np.full(32,4),np.full(64,2)].astype(np.uint8)
    elif kind=='mixed-variance':
        surrogate=tables['variance'][:,None]*2.**(-2*np.asarray(WIDTHS)[None])
        b=allocate_bits(surrogate,budget)
    elif kind=='mixed-mse': b=allocate_bits(errors,budget)
    elif kind=='mixed-grouped': b=allocate_grouped(errors,budget,tile=8,max_segments=4)
    else: raise ValueError('unknown mixed allocation')
    return b


def build_mixed(layout_dir,out,projection,basis,tables,kind):
    started=time.monotonic(); base=Index(layout_dir); m=base.meta; out=Path(out)
    if projection.shape!=(m['n'],128) or basis.matrix.shape!=(128,128): raise ValueError('expected full PCA128')
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    if file_hash(Path(layout_dir)/'vectors.pages')!=m['layout_payload_sha256']: raise ValueError('payload changed')
    out.mkdir(parents=True,exist_ok=True)
    bits=choose_allocation(tables,kind); assert int(bits.sum())==512
    dims=len(bits); levels=(1 << bits.astype(int))-1
    origin=tables['low'].copy(); scale=(tables['high']-origin)/levels; scale[scale==0]=1.
    guard=8192*EPS*m['d']**2*(1+float(np.max(np.abs(projection)))+float(np.max(np.abs(basis.mean))))
    packed=np.zeros((m['n_pages'],m['capacity']*64),dtype=np.uint8)
    radii=np.full((m['n_pages'],m['capacity']),-1,dtype=np.float32)
    errors=np.empty(m['n'],dtype=float); worst=-float('inf'); audited=0
    for first in range(0,m['n_pages'],512):
        ids=base.page_ids[first:first+512]; active=np.arange(m['capacity'])[None]<base.valid[first:first+len(ids),None]
        z=np.asarray(projection[np.maximum(ids,0)])
        code=np.clip(np.rint((z-origin)/scale),0,levels).astype(np.uint8)
        raw=pack_mixed(code.reshape(-1,dims),bits)
        decoded=origin+scale*unpack_mixed(raw,bits).reshape(z.shape)
        err=np.linalg.norm(z-decoded,axis=2)
        rr=np.nextafter((err+guard).astype(np.float32),np.float32(np.inf)); rr[~active]=-1
        packed[first:first+len(ids)]=raw.reshape(len(ids),-1); radii[first:first+len(ids)]=rr
        excess=err[active]-rr[active]; worst=max(worst,float(excess.max()))
        if np.any(excess>0): raise AssertionError('decoded-point undercoverage')
        errors[ids[active]]=err[active]; audited+=int(active.sum())
    norm_upper,norm_lower=transform_limits(basis.matrix)
    arrays=dict(packed=packed,radii=radii,mean=basis.mean.copy(),matrix=basis.matrix.copy(),
                origin=origin,scale=scale,bit_widths=bits,segments=segments(bits))
    geometry=packed.nbytes+radii.nbytes
    shared=sum(v.nbytes for key,v in arrays.items() if key not in ('packed','radii'))
    structural=sum(getattr(base,key).nbytes for key in ('centers','page_ids','valid','radial','ranges'))
    allocation_error=float(sum(tables['errors'][j,WIDTHS.index(int(b))] for j,b in enumerate(bits)))
    meta=dict(kind=kind,dims=dims,groups=dims,pq=False,norm_upper=norm_upper,norm_lower=norm_lower,
        build_guard=guard,code_bytes_per_vector=64,geometry_bytes_per_page=geometry/m['n_pages'],
        geometry_bytes=geometry,shared_bytes=shared,directory_array_bytes=structural+geometry+shared,
        radius_mean=float(errors.mean()),radius_p95=float(np.quantile(errors,.95)),radius_max=float(errors.max()),
        mean_squared_error=float(np.mean(errors**2)),omitted_norm_mean=0.,build_seconds=time.monotonic()-started,
        learning_rows=int(tables['learning_rows']),allocation_bits=bits.tolist(),allocation_segments=segments(bits).tolist(),
        allocation_mse=allocation_error,uniform4_calibration_mse=float(tables['errors'][:,1].sum()),
        all_points_cover_audited=audited,maximum_cover_excess=worst,scalar_ranges_calibrated_on_base=True,
        data_queries_used_for_training=False,allocation_source='learning vectors; base-derived scalar ranges',
        layout_payload_sha256=m['layout_payload_sha256'],
        transform_sha256=hashlib.sha256(basis.mean.tobytes()+basis.matrix.tobytes()).hexdigest(),
        code_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
    np.savez(out/'mixed.npz',**arrays);(out/'mixed.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


@lru_cache(maxsize=4)
def load_library(path):
    lib=ct.CDLL(str(path));fn=lib.gmix_rank
    fn.argtypes=([ct.c_void_p]*5+[ct.c_uint64]*2+[ct.c_uint]*6+
        [ct.c_void_p]+[ct.c_uint]+[ct.c_void_p]*3+[ct.c_double]*3+[ct.c_int]+[ct.c_void_p]*5)
    fn.restype=ct.c_int
    return lib


class MixedIndex(Index):
    """Same grouped native scanner for uniform and nonuniform scalar controls."""
    def __init__(self,layout_dir,summary_dir):
        super().__init__(layout_dir); side=Path(summary_dir)
        stem='mixed' if (side/'mixed.json').exists() else 'rotation'
        self.summary=json.loads((side/f'{stem}.json').read_text());s=self.summary
        if s['pq'] or s['layout_payload_sha256']!=self.meta['layout_payload_sha256']:
            raise ValueError('expected scalar sidecar for this layout')
        for key in ('codes','radii','coordinates','origin','scale'): delattr(self,key)
        with np.load(side/f'{stem}.npz',allow_pickle=False) as f:
            self.arrays={key:np.ascontiguousarray(f[key]) for key in f.files}
        a=self.arrays; dims=s['dims'];m=self.meta
        if 'bit_widths' not in a:
            a['bit_widths']=np.full(dims,s['bits'],dtype=np.uint8)
            a['segments']=segments(a['bit_widths'])
        bits=validate_bits(a['bit_widths']); seg=segments(bits)
        if int(bits.sum())!=512 or not np.array_equal(seg,a['segments']): raise ValueError('invalid layout descriptors')
        expected={'packed':(np.uint8,(m['n_pages'],m['capacity']*64)),
                  'radii':(np.float32,(m['n_pages'],m['capacity'])),
                  'mean':(np.float64,(m['d'],)),'matrix':(np.float64,(m['d'],dims)),
                  'origin':(np.float64,(dims,)),'scale':(np.float64,(dims,))}
        for key,(dtype,shape) in expected.items():
            if a[key].dtype!=dtype or a[key].shape!=shape: raise ValueError('invalid sidecar array '+key)
        if not np.isfinite(a['scale']).all() or np.any(a['scale']<=0): raise ValueError('invalid scales')
        self.meta=dict(m,summary_bytes=s['geometry_bytes'],directory_array_bytes=s['directory_array_bytes'])
        self.extra_descriptor_bytes=0 if stem=='mixed' else bits.nbytes+seg.nbytes
        self.fn=load_library(Path(__file__).resolve().parents[1]/'build/libgeoivf_mixed.so').gmix_rank

    def rank(self,q,lists,wanted,*,certified=True,k=10):
        q=np.ascontiguousarray(q,dtype=np.float64);li=np.asarray(lists,dtype=np.int64)
        m=self.meta;s=self.summary;a=self.arrays
        if (q.shape!=(m['d'],) or not np.isfinite(q).all() or np.max(np.abs(q))>1e10 or
                not 1<=k<=wanted<=1000000): raise ValueError('invalid query/shortlist')
        if (li.ndim!=1 or not len(li) or np.any(li<0) or np.any(li>=m['nlist']) or
                len(np.unique(li))!=len(li)): raise ValueError('invalid IVF lists')
        pages=np.concatenate([np.arange(*self.ranges[l],dtype=np.int64) for l in li])
        z=np.ascontiguousarray((q-a['mean'])@a['matrix'])
        guard=float(s['build_guard']+8192*EPS*m['d']**2*(1+np.max(np.abs(q))))
        slots=np.empty(wanted,dtype=np.int64);scores=np.empty(wanted,dtype=float)
        lbs=np.empty(len(pages),dtype=float);ub=ct.c_double();compared=ct.c_uint64()
        seg=a['segments']
        n=self.fn(a['packed'].ctypes.data,a['radii'].ctypes.data if certified else None,
            self.valid.ctypes.data,self.page_ids.ctypes.data,pages.ctypes.data,len(pages),m['n_pages'],
            m['capacity'],s['dims'],64,a['packed'].shape[1],wanted,k,
            seg.ctypes.data,len(seg),a['origin'].ctypes.data,a['scale'].ctypes.data,z.ctypes.data,
            guard,s['norm_upper'],s['norm_lower'],int(certified),slots.ctypes.data,scores.ctypes.data,
            lbs.ctypes.data,ct.byref(ub),ct.byref(compared))
        if n<0: raise ValueError(f'mixed native rank failed ({n})')
        if certified:
            offset=0
            for l in li:
                first,last=map(int,self.ranges[l]);npage=last-first;r=np.linalg.norm(q-self.centers[l])
                radial=np.maximum(self.radial[first:last,0]-r,r-self.radial[first:last,1])
                lbs[offset:offset+npage]=np.maximum(lbs[offset:offset+npage],np.maximum(0.,radial-guard));offset+=npage
        return slots[:n],scores[:n],pages,lbs,float(ub.value),int(compared.value)
