"""RAM ranking followed by exact verification; optional certified second wave.

No full vectors or oracle distances are consulted before reading. The second
wave is fixed using the first wave's exact kth distance, even if its I/O needs
multiple bounded batches. Points/pages are never scanned twice in one query.
"""
from __future__ import annotations
import ctypes as ct
from functools import lru_cache
import math
from pathlib import Path
import time
import numpy as np
from .index import checked
from .native_scan import NativeTopK
from .execution import SIMDTopK
from .search import Stats
from dataclasses import asdict


@lru_cache(maxsize=4)
def library(path):
    lib=ct.CDLL(str(path));f=lib.gr_rank
    f.argtypes=([ct.c_void_p]*5+[ct.c_uint64]*2+[ct.c_uint]*5+
                [ct.c_void_p]*3+[ct.c_double,ct.c_int]+[ct.c_void_p]*4)
    f.restype=ct.c_int
    return lib


def rank(index,q,lists,wanted,*,certified=True):
    """One compressed scan, O(R) native heap and O(candidate pages) scratch."""
    m=index.meta;a=index.arrays;info=index.summary
    if (getattr(index,'shape',None)!='ball' or info['scheme']!='independent' or
        info['bits'] not in (4,6,8) or not 1<=wanted<=1000000):
        raise ValueError('requires independently encoded 4/6/8-bit balls and valid shortlist')
    q=np.ascontiguousarray(q,dtype=np.float64)
    if q.shape!=(m['d'],) or not np.isfinite(q).all():raise ValueError('invalid query')
    lists=np.asarray(lists,dtype=np.int64)
    if (lists.ndim!=1 or not len(lists) or len(np.unique(lists))!=len(lists) or
        np.any(lists<0) or np.any(lists>=m['nlist'])):raise ValueError('invalid IVF lists')
    pages=np.concatenate([np.arange(*index.ranges[li],dtype=np.int64) for li in lists])
    qproj=np.ascontiguousarray((q-a['mean'])@a['matrix'])
    guard=float(info['build_guard']+2048*np.finfo(float).eps*m['d']**2*(1+np.max(np.abs(q))))
    slots=np.empty(wanted,dtype=np.int64);dd=np.empty(wanted,dtype=np.float64)
    lb=np.empty(len(pages),dtype=np.float64) if certified else np.empty(0,dtype=np.float64)
    count=ct.c_uint64();f=library(Path(__file__).resolve().parents[1]/'build/libgeoivf_ramrank.so').gr_rank
    rc=f(a['packed'].ctypes.data,a['radii'].ctypes.data if certified else None,
        index.valid.ctypes.data,index.page_ids.ctypes.data,pages.ctypes.data,len(pages),m['n_pages'],
        a['packed'].shape[1],m['capacity'],m['dims'],info['bits'],wanted,
        a['origin'].ctypes.data,a['scale'].ctypes.data,qproj.ctypes.data,guard,int(certified),
        slots.ctypes.data,dd.ctypes.data,lb.ctypes.data if certified else None,ct.byref(count))
    if rc<0:raise ValueError(f'native ranker rejected inputs ({rc})')
    if certified:
        at=0
        for li in lists:
            first,last=map(int,index.ranges[li]);n=last-first
            r=float(np.linalg.norm(q-index.centers[li]))
            radial=np.maximum(index.radial[first:last,0]-r,r-index.radial[first:last,1])
            lb[at:at+n]=np.maximum(lb[at:at+n],np.maximum(0.,radial-guard));at+=n
    return slots[:rc],dd[:rc],pages,lb,int(count.value)


def page_extents(index,pages,visited,*,gap=0,max_extent=256):
    """Coalesce only within one IVF list, without rereading earlier-wave pages."""
    if gap<0 or max_extent<1:raise ValueError('invalid coalescing parameters')
    pages=np.unique(np.asarray(pages,dtype=np.int64))
    if pages.ndim!=1 or np.any(pages<0) or np.any(pages>=index.meta['n_pages']):raise ValueError('invalid page')
    if any(int(p) in visited for p in pages):raise ValueError('selected already visited page')
    ext=[];last_li=-1
    for p in map(int,pages):
        li=int(np.searchsorted(index.ranges[:,1],p,side='right'))
        if ext and li==last_li:
            start,n=ext[-1];end=start+n
            if p-start<max_extent and p-end<=gap and not any(t in visited for t in range(end,p)):
                ext[-1]=(start,p-start+1);continue
        ext.append((p,1));last_li=li
    return ext


def batches(extents,page_size,budget):
    """Limit actual power-of-two buffer reservation, not only requested bytes."""
    out=[];used=0
    for p,n in extents:
        reserve=1<<max(12,(n*page_size-1).bit_length())
        if reserve>budget:raise ValueError('one extent exceeds I/O batch budget')
        if out and used+reserve>budget:
            yield out;out=[];used=0
        out.append((p,n));used+=reserve
    if out:yield out


def search_ramfirst(index,q,reader,*,shortlist=64,certified=True,k=10,nprobe=64,
                    preassigned_lists=None,gap_pages=0,max_extent_pages=256,
                    io_batch_bytes=16<<20,scan='simd',audit_sink=None):
    if not 1<=k<=shortlist<=1000000 or k>index.meta['n'] or scan not in ('native','simd'):
        raise ValueError('invalid k, shortlist or scan')
    if gap_pages<0 or max_extent_pages<1 or io_batch_bytes<4096:raise ValueError('invalid I/O options')
    started=time.perf_counter_ns();q=checked(np.asarray(q)[None])[0]
    t=time.perf_counter_ns()
    lists=index.route(q,nprobe) if preassigned_lists is None else np.asarray(preassigned_lists,dtype=np.int64)
    if lists.shape!=(nprobe,):raise ValueError('incorrect nprobe/list shape')
    stat=Stats(route_us=(time.perf_counter_ns()-t)/1000)
    t=time.perf_counter_ns()
    slots,scores,pages,lb,compared=rank(index,q,lists,shortlist,certified=certified)
    rank_us=(time.perf_counter_ns()-t)/1000;stat.filter_us+=rank_us
    stat.candidate_pages=len(pages)
    acc=(SIMDTopK if scan=='simd' else NativeTopK)(index,q,k)
    visited=set();waves=0;first_pages=0;second_pages=0;certificate_us=0.;tau_first=None
    budget=min(io_batch_bytes,getattr(reader,'max_pool_bytes',io_batch_bytes))
    eligible_after_first=0;first_already_certified=False

    def verify(selected):
        nonlocal waves
        t=time.perf_counter_ns();selected=np.unique(np.asarray(selected,dtype=np.int64))
        ext=page_extents(index,selected,visited,gap=gap_pages,max_extent=max_extent_pages)
        stat.selected_pages+=len(selected);stat.filter_us+=(time.perf_counter_ns()-t)/1000
        if not ext:return 0
        waves+=1;fetched=0
        for batch in batches(ext,index.meta['page_size'],budget):
            req=[(p*index.meta['page_size'],n*index.meta['page_size']) for p,n in batch]
            t=time.perf_counter_ns();buffers=reader.read(req);stat.io_us+=(time.perf_counter_ns()-t)/1000
            t=time.perf_counter_ns()
            try:stat.distance_evals+=acc.consume(buffers,batch)
            finally:
                release=getattr(reader,'release',None)
                if release:release(buffers)
            stat.distance_us+=(time.perf_counter_ns()-t)/1000
            for p,n in batch:visited.update(range(p,p+n));fetched+=n
            stat.read_stages+=1;stat.read_requests+=len(batch)
        stat.read_pages+=fetched;stat.read_bytes+=fetched*index.meta['page_size']
        stat.gap_pages_read+=fetched-len(selected)
        return fetched

    try:
        first_pages=verify(slots//index.meta['capacity'])
        if certified:
            t=time.perf_counter_ns();tau_first=math.sqrt(acc.tau2)
            unread=np.fromiter((int(p) not in visited for p in pages),dtype=bool,count=len(pages))
            eligible=unread&(lb<=tau_first);eligible_after_first=int(eligible.sum())
            first_already_certified=not np.any(eligible)
            if audit_sink is not None:
                audit_sink.extend((int(p),float(tau_first),float(v)) for p,v in zip(pages[unread&~eligible],lb[unread&~eligible]))
            certificate_us=(time.perf_counter_ns()-t)/1000;stat.filter_us+=certificate_us
            # Freeze the whole unresolved set. No repeated threshold-dependent planning.
            second_pages=verify(pages[eligible])
        t=time.perf_counter_ns();ids,_=acc.finish();stat.distance_us+=(time.perf_counter_ns()-t)/1000
    finally:acc.close()
    stat.total_us=(time.perf_counter_ns()-started)/1000
    extra=dict(shortlist=shortlist,certified=certified,verification_waves=waves,
               first_wave_pages=first_pages,second_wave_pages=second_pages,
               eligible_pages_after_first=eligible_after_first,first_already_certified=first_already_certified,
               approximate_comparisons=compared,rank_us=rank_us,certificate_us=certificate_us)
    return ids,dict(asdict(stat),**extra)
