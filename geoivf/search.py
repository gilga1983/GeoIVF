"""Completion-driven, staged IVF execution shared by all byte backends."""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, asdict
import numpy as np
from .index import Index, checked


def coalesce(pages, *, gap_pages: int = 0, max_pages: int = 256):
    """Return (first_page, page_count); optional gaps count as actual reads."""
    if gap_pages < 0 or max_pages < 1:
        raise ValueError('invalid coalescing parameters')
    result = []
    for p in sorted(set(map(int, pages))):
        if p < 0:
            raise ValueError('negative page')
        if result:
            first, count = result[-1]
            if p-first < max_pages and p-(first+count) <= gap_pages:
                result[-1] = (first, p-first+1)
                continue
        result.append((p, 1))
    return result


@dataclass
class Stats:
    candidate_pages: int = 0
    selected_pages: int = 0
    read_pages: int = 0
    gap_pages_read: int = 0
    read_requests: int = 0
    read_bytes: int = 0
    distance_evals: int = 0
    read_stages: int = 0
    route_us: float = 0
    filter_us: float = 0
    io_us: float = 0
    distance_us: float = 0
    total_us: float = 0


class Trace:
    def __init__(self, path):
        self.f = open(path, 'w')
        self.epoch = time.perf_counter_ns()
        self.request_id = 0

    def stage(self, *, qid, stage, requests, submitted, completed, tau, mode):
        for offset, size in requests:
            record = dict(query_id=qid, stage_id=stage,
                          depends_on_stage=stage-1 if stage else None,
                          request_id=self.request_id, offset_bytes=offset,
                          length_bytes=size, operation='read', backend=mode,
                          submit_ns=submitted-self.epoch,
                          stage_completed_ns=completed-self.epoch,
                          threshold_before=None if math.isinf(tau) else tau)
            self.f.write(json.dumps(record, allow_nan=False)+'\n')
            self.request_id += 1

    def close(self):
        self.f.close()


def search(index: Index, q: np.ndarray, reader, *, k: int = 10, nprobe: int = 8,
           filtering: str = 'combined', window_pages: int = 64,
           gap_pages: int = 0, max_extent_pages: int = 256,
           trace: Trace | None = None, qid: int = 0,
           numpy_test: bool = False, preassigned_lists=None, audit_sink=None,
           selection: str = 'bounds', scan: str = 'python'):
    if not 1 <= k <= index.meta['n'] or window_pages < 0:
        raise ValueError('invalid k or window size')
    q = checked(np.asarray(q)[None, :])[0]
    if len(q) != index.meta['d']:
        raise ValueError('query dimension mismatch')
    if filtering not in ('none', 'radial', 'balls', 'combined'):
        raise ValueError('invalid filtering mode')
    if scan not in ('python','native') or selection not in ('bounds','fixed','adaptive','prepared'):
        raise ValueError('invalid scan/selection engine')
    if selection not in ('bounds','prepared') and filtering != 'none' and not callable(getattr(index,'select',None)):
        raise ValueError('threshold selection requires a compatible index')
    t0 = time.perf_counter_ns()
    if preassigned_lists is None:
        lists = index.route(q, nprobe, numpy_test=numpy_test)
    else:
        lists = np.asarray(preassigned_lists, dtype=np.int64)
        if (lists.shape != (nprobe,) or len(np.unique(lists)) != nprobe
                or np.any(lists < 0) or np.any(lists >= index.meta['nlist'])):
            raise ValueError('invalid preassigned IVF lists')
    stat = Stats(route_us=(time.perf_counter_ns()-t0)/1000)
    page_size = index.meta['page_size']
    capacity, d = index.meta['capacity'], index.meta['d']
    native = None
    if scan == 'native':
        from .native_scan import NativeTopK
        native = NativeTopK(index, q, k)
    best = []  # Exact distances of fetched FP32 vectors, with ID tie-breaking.
    q64 = q.astype(np.float64)
    try:
        planner = None
        if selection == 'prepared':
            from .prepared import PreparedPlanner
            setup = time.perf_counter_ns()
            planner = PreparedPlanner(index,q,mode=filtering,window_pages=window_pages,
                gap_pages=gap_pages,max_extent_pages=max_extent_pages)
            stat.filter_us += (time.perf_counter_ns()-setup)/1000
        stage = 0
        for li in lists:
            start, end = map(int, index.ranges[li])
            stat.candidate_pages += end-start
            cursor = start
            while cursor < end:
                # Never obtain tau from ground truth. Seed from genuine reads.
                take = window_pages or (end-cursor)
                found = native.count if native is not None else len(best)
                if filtering != 'none' and found < k:
                    take = max(1, math.ceil((k-found)/capacity))
                stop = min(end, cursor+take)
                tau = (math.sqrt(native.tau2) if native is not None else
                       math.sqrt(best[-1][0]) if len(best) == k else math.inf)
                t = time.perf_counter_ns()
                if planner is not None:
                    extents, selected_count = planner.plan(cursor,stop,int(li),tau)
                    if audit_sink is not None:
                        pages = np.arange(cursor,stop,dtype=np.int64)
                        rejected = pages[planner.keep[:len(pages)] == 0]
                        rejected_lb = index.bounds(q,rejected,int(li),filtering)
                        audit_sink.extend((int(p),float(tau),float(v))
                                          for p,v in zip(rejected,rejected_lb))
                else:
                    pages = np.arange(cursor, stop, dtype=np.int64)
                    if selection == 'bounds' or filtering == 'none':
                        lb = index.bounds(q, pages, int(li), filtering)
                        keep = lb <= tau
                    else:
                        keep = index.select(q, pages, int(li), filtering, tau,
                                            strategy=2 if selection == 'adaptive' else 1)
                        lb = None
                    selected = pages[keep]
                    selected_count = len(selected)
                    if audit_sink is not None:
                        rejected = pages[~keep]
                        rejected_lb = (lb[~keep] if lb is not None else
                                       index.bounds(q,rejected,int(li),filtering))
                        audit_sink.extend((int(p),float(tau),float(v))
                                          for p,v in zip(rejected,rejected_lb))
                    extents = coalesce(selected, gap_pages=gap_pages, max_pages=max_extent_pages)
                stat.filter_us += (time.perf_counter_ns()-t)/1000
                stat.selected_pages += selected_count
                cursor = stop
                if not extents:
                    continue
                requests = [(p*page_size, count*page_size) for p, count in extents]
                submitted = time.perf_counter_ns()
                buffers = reader.read(requests)
                completed = time.perf_counter_ns()
                stat.io_us += (completed-submitted)/1000
                t = time.perf_counter_ns()
                try:
                    if len(buffers) != len(requests):
                        raise ValueError('backend returned the wrong number of buffers')
                    # Complete a stage in canonical request order on both scan paths.
                    if native is not None:
                        stat.distance_evals += native.consume(buffers,extents)
                    else:
                        for data,(first,count) in zip(buffers,extents):
                            if len(data) != count*page_size:
                                raise EOFError('backend returned a short extent')
                            for j in range(count):
                                p=first+j
                                n=int(index.valid[p])
                                a=np.frombuffer(data,dtype='<f4',count=n*d,
                                      offset=j*page_size).reshape(n,d).astype(np.float64)
                                diff=a-q64
                                distances=np.einsum('ij,ij->i',diff,diff)
                                best.extend(zip(map(float,distances),map(int,index.page_ids[p,:n])))
                                best=sorted(best)[:k]
                                stat.distance_evals += n
                finally:
                    release=getattr(reader,'release',None)
                    if release is not None: release(buffers)
                stat.distance_us += (time.perf_counter_ns()-t)/1000
                fetched = sum(count for _, count in extents)
                stat.read_pages += fetched
                stat.gap_pages_read += fetched-selected_count
                stat.read_requests += len(extents)
                stat.read_bytes += fetched*page_size
                stat.read_stages += 1
                if trace:
                    trace.stage(qid=qid, stage=stage, requests=requests,
                                submitted=submitted, completed=completed, tau=tau, mode=reader.mode)
                stage += 1
        t=time.perf_counter_ns()
        result = native.finish()[0] if native is not None else np.array([i for _,i in best],dtype=np.int64)
        stat.distance_us += (time.perf_counter_ns()-t)/1000
    finally:
        if native is not None: native.close()
    stat.total_us = (time.perf_counter_ns()-t0)/1000
    return result, asdict(stat)
