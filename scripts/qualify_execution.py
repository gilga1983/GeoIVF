#!/usr/bin/env python3
"""Same-plan scanner/reader controls, then separately labeled scheduling changes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import Index,vectors,train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis,digest_file
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.io import MemoryReplay,PooledNative
from geoivf.execution import RollingPooled
from geoivf.search import search
from scripts.qualify_sift1m import reference,audit_rejections
from scripts.qualify_speed import RecordedReader,COUNTS,csv_write,save

# name, precision (0 means no filtering), scanner, I/O schedule, window, gap.
ARMS=[('unfiltered-scalar-batch',0,'native','batch',0,0),
      ('unfiltered-simd-batch',0,'simd','batch',0,0),
      ('unfiltered-simd-rolling',0,'simd','rolling',0,0),
      ('b8-scalar-batch-w64',8,'native','batch',64,0),
      ('b8-simd-batch-w64',8,'simd','batch',64,0),
      ('b8-simd-rolling-w64',8,'simd','rolling',64,0),
      ('b6-scalar-batch-w64',6,'native','batch',64,0),
      ('b6-simd-rolling-w64',6,'simd','rolling',64,0),
      ('b8-simd-rolling-w16',8,'simd','rolling',16,0),
      ('b8-simd-rolling-w256',8,'simd','rolling',256,0),
      ('b8-simd-rolling-whole',8,'simd','rolling',0,0),
      ('b8-simd-rolling-w256-g2',8,'simd','rolling',256,2)]


def run(args):
    start=time.monotonic();faiss.omp_set_num_threads(1)
    dataset=json.loads((args.data/'dataset.json').read_text())
    if digest_file(args.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':
        raise ValueError('unexpected canonical SIFT1M corpus')
    x=vectors(args.data/'sift_base.fvecs');allq=vectors(args.data/'sift_query.fvecs')
    learn=vectors(args.data/'sift_learn.fvecs')
    if (x.shape,allq.shape,learn.shape)!=((1000000,128),(10000,128),(100000,128)):
        raise ValueError('unexpected corpus shapes')
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    qids=np.random.default_rng(20260929).permutation(10000)[:args.queries];queries=allq[qids]
    print('Build shared IVF and unchanged independent PCA64 geometry',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    quantizer=faiss.IndexFlatL2(128);quantizer.add(centers)
    routes=quantizer.search(queries,64)[1]
    refs=[reference(x,labels,q,routes[qi]) for qi,q in enumerate(queries)]
    basis=Basis.fit(learn,'pca');del learn;basis.save(args.out/'pca-basis.npz')
    np.savez(args.out/'shared-candidates.npz',query_ids=qids,lists=routes,centers=centers)
    projected=np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush()
    layout=args.work/'geopack';frozen=freeze_layout(x,centers,labels,layout,layout='geopack',pack_dims=16)
    indices={0:Index(layout)};builds=[]
    for bits in (6,8):
        out=args.work/f'b{bits}'
        meta=summarize_dependencies(layout,out,basis,dims=64,bits=bits,scheme='independent',projected_by_id=projected)
        cert=certify_cells(layout,out,projected,out/'cells.json')
        save(args.out/f'certificate-b{bits}.json',cert)
        indices[bits]=CellIndex(layout,out,shape='ball');builds.append(meta)
    del projected
    memory=MemoryReplay(layout/'vectors.pages');plans={};counts={};checks=[];audited=0

    def execute(arm,qi,reader,*,preassigned=True,audit=None):
        name,bits,scan,schedule,window,gap=arm
        return search(indices[bits],queries[qi],reader,nprobe=64,
            preassigned_lists=routes[qi] if preassigned else None,
            filtering='combined' if bits else 'none',selection='prepared',scan=scan,
            window_pages=window,gap_pages=gap,audit_sink=audit)

    for arm in ARMS:
        name,bits,scan,schedule,window,gap=arm
        for qi in range(len(queries)):
            reader=RecordedReader(memory);audit=[] if bits and qi<2 else None
            ids,stat=execute(arm,qi,reader,audit=audit)
            np.testing.assert_array_equal(ids,refs[qi])
            if audit:audited+=audit_rejections(indices[bits],x,queries[qi],audit)
            ph=reader.hash.hexdigest();values=tuple(stat[k] for k in COUNTS)
            key=(bits,window,gap,qi)
            if key in plans and plans[key]!=(ph,values):raise AssertionError('same-plan optimization changed requests')
            plans[key]=(ph,values);counts[name,qi]=values
            rec=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10
            checks.append(dict(method=name,query_id=int(qids[qi]),recall_at_10=rec,plan_sha256=ph,
                **{k:stat[k] for k in COUNTS}))
        print('PASS canonical answers/plans: '+name,flush=True)
    csv_write(args.out/'correctness.csv',checks)
    save(args.out/'correctness.json',dict(comparisons=len(checks),distinct_queries=len(queries),
        audited_decisions=audited,all_same_group_plan_hashes_agree=True))

    affinity=sorted(os.sched_getaffinity(0));cpu=affinity[0];os.sched_setaffinity(0,{cpu})
    load_start=os.getloadavg();records=[];readers={};pool_info=[]
    try:
        for phase in ['memory','direct']:
            if phase=='direct':
                readers['batch']=PooledNative(layout/'vectors.pages',direct=True,uring=True,depth=args.depth)
                readers['rolling']=RollingPooled(layout/'vectors.pages',direct=True,uring=True,depth=args.depth)
            # Initialize routing once outside timings; every timed search still routes afresh.
            for idx in indices.values():idx.route(queries[0],64)
            for arm in ARMS:
                execute(arm,0,memory if phase=='memory' else readers[arm[3]],preassigned=False)
            rng=np.random.default_rng(829473)
            for rep in range(args.repeats):
                for qi in rng.permutation(len(queries)):
                    for ai in rng.permutation(len(ARMS)):
                        arm=ARMS[ai];reader=memory if phase=='memory' else readers[arm[3]]
                        t=time.perf_counter_ns();c=time.process_time_ns()
                        ids,stat=execute(arm,qi,reader,preassigned=False)
                        wall=(time.perf_counter_ns()-t)/1e6;cp=(time.process_time_ns()-c)/1e6
                        np.testing.assert_array_equal(ids,refs[qi])
                        if tuple(stat[k] for k in COUNTS)!=counts[arm[0],qi]:raise AssertionError('timed operation count changed')
                        records.append(dict(phase=phase,method=arm[0],backend=reader.mode,round=rep,
                            query_id=int(qids[qi]),wall_ms=wall,process_cpu_ms=cp,**stat))
                csv_write(args.out/'timings.csv',records)
                print(f'{phase}: completed round {rep+1}/{args.repeats}',flush=True)
        for name,r in readers.items():pool_info.append(dict(schedule=name,reserved_bytes=r.reserved_bytes,
            allocations=r.allocations,budget_bytes=r.max_pool_bytes))
    finally:
        for r in readers.values():r.close()
        memory.close();os.sched_setaffinity(0,affinity)
    summaries=[]
    for phase in ['memory','direct']:
        for arm in ARMS:
            rr=[r for r in records if r['phase']==phase and r['method']==arm[0]]
            wall=np.array([r['wall_ms'] for r in rr])
            summaries.append(dict(phase=phase,method=arm[0],samples=len(rr),distinct_queries=len(queries),
                mean_ms=float(wall.mean()),median_ms=float(np.median(wall)),p95_ms=float(np.quantile(wall,.95)),
                round_means=[float(np.mean([r['wall_ms'] for r in rr if r['round']==i])) for i in range(args.repeats)],
                mean_cpu_ms=float(np.mean([r['process_cpu_ms'] for r in rr])),
                means={k:float(np.mean([r[k] for r in rr])) for k in COUNTS+('route_us','filter_us','io_us','distance_us')}))
    report=dict(dataset=dataset,source_layout_sha256=frozen['layout_payload_sha256'],
        pca_basis_sha256=basis.fingerprint(),candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
        query_ids=qids.tolist(),query_split='reused development queries',nlist=1024,nprobe=64,k=10,seed=12345,
        arms=[dict(name=n,bits=b,scan=s,io_schedule=io,window=w,gap=g) for n,b,s,io,w,g in ARMS],
        builds=builds,results=summaries,pools=pool_info,depth=args.depth,repeats=args.repeats,
        correctness_comparisons=len(checks),timed_answer_count_checks=len(records),audited_decisions=audited,
        same_plan_groups='equal bits, window and gap; scan and I/O schedule may differ',
        cpu_affinity=[cpu],load_average_start=load_start,load_average_end=os.getloadavg(),
        elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform=platform.platform(),numpy=np.__version__,faiss=faiss.__version__,
        real_direct_io=True,production_speedup_valid=False,
        limitations=['same-engine comparison, not Faiss/CLIP/DiskANN','one dataset/seed, 64 reused development queries',
            'SIMD preserves per-vector FP64 sum order; no SIMD across-coordinate reassociation',
            'window/gap sweep changes request plans; separate from same-plan optimizations',
            'host shared and frequency/device caches uncontrolled; CPU affinity is not exclusive',
            'small payload fits RAM; O_DIRECT not cold-device or out-of-RAM-scale evidence',
            'query-to-query concurrency not measured; block-level commands not instrumented',
            'benchmark RSS includes data/oracle/variants; not deployed memory'])
    save(args.out/'execution-results.json',report)
    for r in summaries:print(json.dumps({k:r[k] for k in ('phase','method','mean_ms','median_ms')}),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=64);ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--depth',type=int,default=16);args=ap.parse_args()
    if not 1<=args.queries<=1000 or args.repeats<3 or not 1<=args.depth<=1024:ap.error('invalid campaign size')
    for p in (args.work,args.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    lock=Path.home()/'.cache/geoivf/speed-device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);run(args)

if __name__=='__main__':main()
