#!/usr/bin/env python3
"""Online speed qualification. Real thresholds; common native scan; direct I/O optional.

This is a small, reused development workload on a shared host, not final
out-of-RAM-scale evaluation or a CLIP/Faiss production speed comparison.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import faiss
from geoivf.index import Index,vectors,train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis,digest_file
from geoivf.dependent import DependentIndex,summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.search import search
from geoivf.io import MemoryReplay,Native,PooledNative
from scripts.qualify_sift1m import reference,audit_rejections

COUNTS=('candidate_pages','selected_pages','read_pages','gap_pages_read','read_requests',
        'read_bytes','distance_evals','read_stages')


def save(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


class RecordedReader:
    """Count actual staged request plans without retaining payload copies."""
    def __init__(self,reader): self.reader=reader;self.hash=hashlib.sha256()
    @property
    def mode(self): return self.reader.mode
    def read(self,requests):
        self.hash.update(struct.pack('<Q',len(requests)))
        for o,n in requests: self.hash.update(struct.pack('<QQ',o,n))
        return self.reader.read(requests)
    def release(self,buffers):
        release=getattr(self.reader,'release',None)
        if release is not None: release(buffers)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=64)
    ap.add_argument('--timed-queries',type=int,default=32)
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--with-direct',action='store_true')
    ap.add_argument('--depth',type=int,default=16)
    ap.add_argument('--prepared',action='store_true',help='compare prepared/fused planning with the native-speed baseline')
    args=ap.parse_args()
    if not 1<=args.timed_queries<=args.queries<=1000 or args.repeats<3 or not 1<=args.depth<=1024:
        ap.error('invalid campaign dimensions')
    for path in (args.work,args.out):
        if path.exists() and any(path.iterdir()): raise FileExistsError(path)
        path.mkdir(parents=True,exist_ok=True)
    lockpath=Path.home()/'.cache/geoivf/speed-device.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        # Coordinates GeoIVF speed campaigns on this host only, not other projects.
        fcntl.flock(lock,fcntl.LOCK_EX)
        run(args)


def run(args):
    started=time.monotonic();faiss.omp_set_num_threads(1)
    dataset=json.loads((args.data/'dataset.json').read_text())
    expected='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
    if digest_file(args.data/'sift_base.fvecs')!=expected:raise ValueError('unexpected SIFT corpus')
    x=vectors(args.data/'sift_base.fvecs');allq=vectors(args.data/'sift_query.fvecs')
    learn=vectors(args.data/'sift_learn.fvecs')
    if x.shape!=(1000000,128) or allq.shape!=(10000,128) or learn.shape!=(100000,128):
        raise ValueError('wrong canonical corpus dimensions')
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    qids=np.random.default_rng(20260929).permutation(10000)[:1000][:args.queries]
    queries=allq[qids]
    print('Build unchanged IVF, frozen pages, and PCA64 6/8-bit summaries',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    quantizer=faiss.IndexFlatL2(128);quantizer.add(centers)
    routes=quantizer.search(queries,64)[1]
    refs=[reference(x,labels,q,routes[qi]) for qi,q in enumerate(queries)]
    np.savez(args.out/'shared-candidates.npz',query_ids=qids,lists=routes,centers=centers)
    basis=Basis.fit(learn,'pca');del learn;basis.save(args.out/'pca-basis.npz')
    projected=np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush()
    layout=args.work/'geopack';frozen=freeze_layout(x,centers,labels,layout,layout='geopack',pack_dims=16)
    cells={};legacy={};builds=[]
    for bits in (6,8):
        out=args.work/f'bits{bits}'
        meta=summarize_dependencies(layout,out,basis,dims=64,scheme='independent',bits=bits,projected_by_id=projected)
        cert=certify_cells(layout,out,projected,out/'cells.json')
        save(args.out/f'certificate-b{bits}.json',cert)
        cells[bits]=CellIndex(layout,out,shape='ball');legacy[bits]=DependentIndex(layout,out)
        builds.append(meta)
    del projected
    plain=Index(layout)
    # Python remains an implementation diagnostic, not a competitive IVF baseline.
    methods={
      'unfiltered-python':(plain,'none','bounds','python',0),
      'unfiltered-native':(plain,'none','bounds','native',0),
      'legacy-b8-python':(legacy[8],'combined','bounds','python',64),
      'full-b8-python':(cells[8],'combined','bounds','python',64),
      'full-b8-native':(cells[8],'combined','bounds','native',64),
      'adaptive-b8-native':(cells[8],'combined','adaptive','native',64),
      'full-b6-native':(cells[6],'combined','bounds','native',64),
      'adaptive-b6-native':(cells[6],'combined','adaptive','native',64)}
    if args.prepared:
        methods = {name: methods[name] for name in (
            'unfiltered-native','full-b8-native','adaptive-b8-native','adaptive-b6-native')}
        methods.update({
            'prepared-unfiltered':(plain,'none','prepared','native',0),
            'prepared-b8-native':(cells[8],'combined','prepared','native',64),
            'prepared-b6-native':(cells[6],'combined','prepared','native',64)})
    memory=MemoryReplay(layout/'vectors.pages')
    correctness=[];audits=0;plans={}
    for name,(idx,mode,selection,scan,window) in methods.items():
        group='none' if mode=='none' else ('b6' if 'b6' in name else 'b8')
        for qi,q in enumerate(queries):
            reader=RecordedReader(memory);audit=[] if qi<2 else None
            got,stat=search(idx,q,reader,nprobe=64,preassigned_lists=routes[qi],
                filtering=mode,selection=selection,scan=scan,window_pages=window,audit_sink=audit)
            np.testing.assert_array_equal(got,refs[qi])
            if audit: audits+=audit_rejections(idx,x,q,audit)
            ph=reader.hash.hexdigest();key=(group,qi)
            values=tuple(stat[k] for k in COUNTS)
            if key in plans and plans[key]!=(ph,values):
                raise AssertionError('optimization changed staged reads or exact-distance counts')
            plans[key]=(ph,values)
            rec=len(set(map(int,got))&set(map(int,gt[qids[qi],:10])))/10
            correctness.append(dict(method=name,query_id=int(qids[qi]),recall_at_10=rec,
                                    plan_sha256=ph,**{k:stat[k] for k in COUNTS}))
        print('Correctness and request-plan equivalence: '+name,flush=True)
    csv_write(args.out/'correctness.csv',correctness)
    save(args.out/'correctness.json',dict(comparisons=len(correctness),independent_queries=len(queries),
        audited_rejection_decisions=audits,exact_ids_and_plans_agree=True))

    # Pin this process for repeatability, not exclusivity. Other host jobs continue.
    original_affinity=sorted(os.sched_getaffinity(0))
    cpu=original_affinity[0];os.sched_setaffinity(0,{cpu})
    load_start=os.getloadavg();records=[];pools=[];readers=[]

    def measure(phase,arms):
        nonlocal records
        # Warm each execution path on real online queries, not final-radius masks.
        for name,reader in arms:
            idx,mode,sel,scan,window=methods[name]
            if hasattr(idx,'_query'):idx._query=None
            search(idx,queries[0],reader,nprobe=64,preassigned_lists=routes[0],
                   filtering=mode,selection=sel,scan=scan,window_pages=window)
        rng=np.random.default_rng(987314)
        for repeat in range(args.repeats):
            for qi in rng.permutation(args.timed_queries):
                for arm in rng.permutation(len(arms)):
                    name,reader=arms[arm];idx,mode,sel,scan,window=methods[name]
                    # Do not let another arm donate a cached query projection.
                    if hasattr(idx,'_query'):idx._query=None
                    before=time.perf_counter_ns();cpu_before=time.process_time_ns()
                    route_start=time.perf_counter_ns()
                    online_lists=quantizer.search(queries[qi:qi+1],64)[1][0]
                    route_us=(time.perf_counter_ns()-route_start)/1000
                    got,stat=search(idx,queries[qi],reader,nprobe=64,preassigned_lists=online_lists,
                        filtering=mode,selection=sel,scan=scan,window_pages=window)
                    elapsed=(time.perf_counter_ns()-before)/1e6
                    cpu_ms=(time.process_time_ns()-cpu_before)/1e6
                    # Checksums and assertions are outside the timed interval.
                    np.testing.assert_array_equal(online_lists,routes[qi])
                    np.testing.assert_array_equal(got,refs[qi])
                    group='none' if mode=='none' else ('b6' if 'b6' in name else 'b8')
                    assert tuple(stat[k] for k in COUNTS)==plans[group,qi][1]
                    records.append(dict(phase=phase,method=name,backend=reader.mode,round=repeat,
                        query_id=int(qids[qi]),wall_ms=elapsed,process_cpu_ms=cpu_ms,
                        coarse_route_us=route_us,**stat))
            csv_write(args.out/'timings.csv',records)
            print(f'{phase}: repeat {repeat+1}/{args.repeats} completed',flush=True)
    try:
        measure('memory',[(name,memory) for name in methods])
        if args.with_direct:
            pooled=PooledNative(layout/'vectors.pages',direct=True,uring=True,depth=args.depth)
            readers.append(pooled);pools.append(pooled)
            copying=None
            if not args.prepared:
                copying=Native(layout/'vectors.pages',direct=True,uring=True,depth=args.depth)
                readers.append(copying)
            if args.prepared:
                measure('direct',[(name,pooled) for name in methods])
            else:
                measure('direct',[(name,pooled) for name in [
                'unfiltered-native','full-b8-native','adaptive-b8-native','adaptive-b6-native']]+
                [('unfiltered-native',copying),('adaptive-b8-native',copying)])
        pool_info=[dict(mode=p.mode,reserved_bytes=p.reserved_bytes,allocation_count=p.allocations,
                       budget_bytes=p.max_pool_bytes) for p in pools]
    finally:
        for reader in readers:reader.close()
        memory.close()
        os.sched_setaffinity(0,original_affinity)
    summaries=[]
    groups=sorted(set((r['phase'],r['method'],r['backend']) for r in records))
    for phase,name,backend in groups:
        rr=[r for r in records if (r['phase'],r['method'],r['backend'])==(phase,name,backend)]
        wall=np.array([r['wall_ms'] for r in rr])
        summaries.append(dict(phase=phase,method=name,backend=backend,
            samples=len(rr),distinct_queries=args.timed_queries,repeats=args.repeats,
            median_wall_ms=float(np.median(wall)),mean_wall_ms=float(wall.mean()),
            p95_wall_ms=float(np.quantile(wall,.95)),p99_wall_ms=float(np.quantile(wall,.99)),
            round_mean_wall_ms=[float(np.mean([r['wall_ms'] for r in rr if r['round']==i])) for i in range(args.repeats)],
            mean_process_cpu_ms=float(np.mean([r['process_cpu_ms'] for r in rr])),
            means={k:float(np.mean([r[k] for r in rr])) for k in COUNTS+('filter_us','io_us','distance_us','coarse_route_us')}))
    report=dict(dataset=dataset,source_layout_sha256=frozen['layout_payload_sha256'],
        pca_basis_sha256=basis.fingerprint(),candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
        query_ids=qids.tolist(),timed_query_ids=qids[:args.timed_queries].tolist(),
        query_split='reused development-only',seed=12345,nlist=1024,nprobe=64,k=10,
        correctness_comparisons=len(correctness),timing_correctness_comparisons=len(records),
        audited_rejection_decisions=audits,all_tested_answers_and_plans_preserved=True,
        builds=builds,timings=summaries,pools=pool_info,
        cpu_affinity_during_timings=[cpu],original_cpu_affinity=original_affinity,
        load_average_start=load_start,load_average_end=os.getloadavg(),
        platform=platform.platform(),numpy=np.__version__,faiss=faiss.__version__,
        elapsed_seconds=time.monotonic()-started,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_direct_io_executed=args.with_direct,exclusive_host=False,
        prepared_planner_executed=args.prepared,
        production_speedup_valid=False,
        limitations=['one small SIFT1M index, one seed, reused development queries',
          'shared host, no frequency control, pinned process does not reserve core',
          'CPU and device caches warm; O_DIRECT bypasses host page cache only',
          'payload file is not larger than host RAM; this is an execution microbenchmark',
          'application requested bytes/extents are not instrumented physical NVMe commands',
          'Python coordinates stages; native heap/kernel not final Faiss/SIMD optimization',
          'no CLIP comparison, concurrent QPS, MQSim latency, or held-out generalization',
          'FP64 accumulation order may differ at numerical ties; SIFT IDs/plans checked exactly',
          'benchmark RSS includes oracle/database/scratch and is not deployment memory'])
    save(args.out/'speed-results.json',report)
    for row in summaries:print(json.dumps({k:row[k] for k in ('phase','method','backend','median_wall_ms','mean_wall_ms')}),flush=True)
    print(f'PASS: {len(correctness)} initial and {len(records)} timed answer/count checks; {audits} audited decisions',flush=True)

if __name__=='__main__':main()
