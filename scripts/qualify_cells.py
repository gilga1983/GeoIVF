#!/usr/bin/env python3
"""Canonical same-code ball/cell/hybrid comparison and native filter microbenchmarks."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import faiss
from geoivf.index import Index, vectors, train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.dependent import summarize_dependencies, DependentIndex, file_hash
from geoivf.cells import CellIndex, certify_cells
from geoivf.search import search
from geoivf.io import MemoryReplay
from scripts.qualify_sift1m import reference, audit_rejections


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def cpu_benchmark(indices, queries, pages, taus, out, bits, repeats=7):
    """Equal candidate/threshold work, no I/O, projection and wrapper included.

    Uses final-radius thresholds only for this explicitly OFFLINE microbenchmark.
    Both shapes use squared rejection tests; no per-vector square roots. Full
    kernels evaluate all 64 coordinates of ALL active centers; early16 kernels
    allow coordinate rejection and stop after a surviving center. Each
    trial scans all query candidate pages, in query/list order. Libraries and
    data are warm, not guaranteed to fit in cache. No CPU exclusivity is claimed.
    """
    count=min(32,len(queries)); qs=queries[:count]; ps=pages[:count]; ts=taus[:count]
    variants=[(shape,strategy) for shape in indices for strategy in [1,2]]
    checks={}
    for shape,strategy in variants:
        got=[]
        for q,pp,tau in zip(qs,ps,ts):
            mask=indices[shape].geometry(q,pp,strategy=strategy,tau=tau)
            ref=indices[shape].geometry(q,pp)<=tau
            np.testing.assert_array_equal(mask,ref)
            got.append(mask)
        checks[f'{shape}-{strategy}']=int(sum(v.sum() for v in got))
    observations=[]; rng=np.random.default_rng(5109)
    for rep in range(repeats):
        for ix in rng.permutation(len(variants)):
            shape,strategy=variants[ix]; idx=indices[shape];checksum=0
            cpu0=time.process_time_ns();wall0=time.perf_counter_ns()
            for q,pp,tau in zip(qs,ps,ts):
                mask=idx.geometry(q,pp,strategy=strategy,tau=tau)
                checksum+=int(mask.sum())
            wall=time.perf_counter_ns()-wall0;cpu=time.process_time_ns()-cpu0
            assert checksum==checks[f'{shape}-{strategy}']
            observations.append(dict(bits=bits,shape=shape,strategy='full' if strategy==1 else 'early16',
                                     repeat=rep,queries=count,candidate_pages=sum(map(len,ps)),
                                     wall_ns=wall,cpu_ns=cpu,retained_pages=checksum))
    with Path(out).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(observations[0]));w.writeheader();w.writerows(observations)
    summary=[]
    for shape,strategy in variants:
        name='full' if strategy==1 else 'early16'
        rows=[r for r in observations if r['shape']==shape and r['strategy']==name]
        summary.append(dict(bits=bits,shape=shape,strategy=name,queries=count,repeats=repeats,
                            median_wall_us_per_query=float(np.median([r['wall_ns']/count/1000 for r in rows])),
                            min_wall_us_per_query=min(r['wall_ns']/count/1000 for r in rows),
                            max_wall_us_per_query=max(r['wall_ns']/count/1000 for r in rows),
                            median_cpu_us_per_query=float(np.median([r['cpu_ns']/count/1000 for r in rows])),
                            retained_pages=checks[f'{shape}-{strategy}']))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=64);ap.add_argument('--cpu-repeats',type=int,default=7)
    args=ap.parse_args()
    if not 1<=args.queries<=1000 or args.cpu_repeats<1:ap.error('invalid development scope')
    for p in [args.work,args.out]:
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    faiss.omp_set_num_threads(1); started=time.monotonic()
    dataset=json.loads((args.data/'dataset.json').read_text())
    if file_hash(args.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':
        raise ValueError('wrong corpus')
    x=vectors(args.data/'sift_base.fvecs');allq=vectors(args.data/'sift_query.fvecs')
    learn=vectors(args.data/'sift_learn.fvecs')
    if x.shape!=(1000000,128) or allq.shape!=(10000,128) or learn.shape!=(100000,128):
        raise ValueError('wrong canonical shape')
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    qids=np.random.default_rng(20260929).permutation(10000)[:1000][:args.queries];queries=allq[qids]
    print('Fit shared IVF and PCA; preserve original GeoPack pages',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    quantizer=faiss.IndexFlatL2(128);quantizer.add(centers);routes=quantizer.search(queries,64)[1]
    refs=[reference(x,labels,q,routes[i]) for i,q in enumerate(queries)]
    np.savez(args.out/'shared-candidates.npz',query_ids=qids,lists=routes,centers=centers)
    basis=Basis.fit(learn,'pca');del learn;basis.save(args.out/'pca-basis.npz')
    projected=np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush()
    physical=args.work/'geopack';frozen=freeze_layout(x,centers,labels,physical,pack_dims=16)
    if frozen['layout_payload_sha256']!='8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7':
        raise AssertionError('unexpected layout change')
    base=Index(physical);reader=MemoryReplay(physical/'vectors.pages')
    pages=[np.concatenate([np.arange(*base.ranges[li],dtype=np.int64) for li in route]) for route in routes]
    # Only the microbenchmark gets the final radius. Online search receives no tau.
    taus=[float(np.linalg.norm(x[ref[-1]].astype(float)-q)) for ref,q in zip(refs,queries)]
    results=[];builds=[];cpu=[];all_records={};matches=0;audits=0

    def evaluate(name,idx,mode='combined',window=64):
        nonlocal matches,audits
        records=[];checked=0
        for i,q in enumerate(queries):
            decisions=[] if i<2 else None
            ids,stat=search(idx,q,reader,nprobe=64,filtering=mode,window_pages=window,
                            preassigned_lists=routes[i],audit_sink=decisions)
            np.testing.assert_array_equal(ids,refs[i])
            if decisions: checked+=audit_rejections(idx,x,q,decisions)
            recall=len(set(map(int,ids))&set(map(int,gt[qids[i],:10])))/10
            records.append(dict(query_id=int(qids[i]),recall_at_10=recall,
                                saved_fraction=1-stat['read_pages']/stat['candidate_pages'],**stat))
        with (args.out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        row=dict(name=name,queries=len(queries),exact_matches=len(queries),audited_rejections=checked,
                 geometry_bytes_per_page=idx.meta['summary_bytes']/idx.meta['n_pages'],
                 directory_array_bytes=idx.meta['directory_array_bytes'],
                 means={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='query_id'},
                 median_saved_fraction=float(np.median([r['saved_fraction'] for r in records])),csv=name+'.csv')
        all_records[name]=records;results.append(row);matches+=len(queries);audits+=checked
        save(args.out/'partial-results.json',results)
        print(json.dumps(row),flush=True)
    evaluate('none-whole-list',base,mode='none',window=0)
    for bits in [4,6,8]:
        path=args.work/f'codes{bits}'
        meta=summarize_dependencies(physical,path,basis,dims=64,scheme='independent',bits=bits,projected_by_id=projected)
        certificate=certify_cells(physical,path,projected,path/'cells.json')
        save(args.out/f'cells{bits}-certificate.json',certificate)
        builds.append(dict(meta))
        # Legacy and native ball controls verify the common decoder/kernel change.
        legacy=DependentIndex(physical,path);evaluate(f'legacy-ball-b{bits}',legacy);del legacy
        indices={shape:CellIndex(physical,path,shape=shape) for shape in ['ball','box','hybrid']}
        for shape,idx in indices.items(): evaluate(f'{shape}-b{bits}',idx)
        for i in range(len(queries)):
            a=all_records[f'legacy-ball-b{bits}'][i];b=all_records[f'ball-b{bits}'][i]
            for k in ['candidate_pages','read_pages','read_requests','read_stages','distance_evals']:
                assert a[k]==b[k],(bits,i,k)
            assert all_records[f'hybrid-b{bits}'][i]['read_pages']<=b['read_pages']
            assert all_records[f'hybrid-b{bits}'][i]['read_pages']<=all_records[f'box-b{bits}'][i]['read_pages']
        print(f'Native CPU microbenchmark, {bits} bits, fixed identical workload',flush=True)
        cpu.extend(cpu_benchmark(indices,queries,pages,taus,args.out/f'cpu-b{bits}.csv',bits,args.cpu_repeats))
        save(args.out/'partial-cpu.json',cpu)
        del indices
    reader.close()
    report=dict(dataset=dataset,query_ids=qids.tolist(),independent_queries=len(queries),
                query_split='development-only',nlist=1024,nprobe=64,k=10,seed=12345,
                layout_payload_sha256=frozen['layout_payload_sha256'],
                pca_basis_sha256=basis.fingerprint(),candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                configuration_query_matches=matches,audited_rejection_decisions=audits,
                builds=builds,results=results,native_cpu=cpu,
                host=platform.uname()._asdict(),load_average=list(os.getloadavg()),
                allowed_cpus=sorted(os.sched_getaffinity(0)),numpy=np.__version__,faiss=faiss.__version__,
                elapsed_seconds=time.monotonic()-started,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                storage_latency_valid=False,production_speedup_valid=False,
                limitations=['one dataset, construction seed, 64 reused development queries',
                             'online search is RAM replay with Python exact scan, not SSD timing',
                             'native CPU uses fixed final-radius diagnostic thresholds, not query latency',
                             'native CPU includes Python call/projection/allocation overhead; no I/O or radial filter',
                             'warm randomized repeated CPU trials on one shared host, not exclusive hardware',
                             'engineering FP64 guards, not formal interval arithmetic',
                             'hybrid is max of per-point ball/cell lower bounds, not exact intersection distance'])
    save(args.out/'cell-results.json',report)
    print(f'PASS: {matches} configuration-query comparisons, {audits} rejection audits',flush=True)

if __name__=='__main__':main()
