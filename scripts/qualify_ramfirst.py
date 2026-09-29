#!/usr/bin/env python3
"""RAM shortlist vs staged search, with fresh frozen-choice held-out evaluation."""
from __future__ import annotations
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
from geoivf.index import vectors,train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis,digest_file
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.io import MemoryReplay
from geoivf.execution import RollingPooled
from geoivf.search import search
from geoivf.ramfirst import search_ramfirst
from scripts.qualify_sift1m import reference,audit_rejections
from scripts.qualify_speed import COUNTS,csv_write,save

SIZES=(16,32,64,128,256)
ARMS=['staged']+[f'{mode}-r{r}' for mode in ('approx','certified') for r in SIZES]


def summarize(rows):
    result=[]
    for name in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==name];v=np.array([r['wall_ms'] for r in rr])
        result.append(dict(method=name,samples=len(rr),distinct_queries=len(set(r['query_id'] for r in rr)),
            mean_ms=float(v.mean()),median_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95)),
            exact_match_fraction=float(np.mean([r['exact_match'] for r in rr])),
            round_mean_ms=[float(np.mean([r['wall_ms'] for r in rr if r['round']==i])) for i in sorted(set(r['round'] for r in rr))],
            means={k:float(np.mean([r[k] for r in rr])) for k in (
                'recall','ivf_recall','read_pages','read_bytes','selected_pages','gap_pages_read','read_requests',
                'read_stages','verification_waves','first_wave_pages','second_wave_pages','first_already_certified',
                'candidate_pages','distance_evals','rank_us','certificate_us','filter_us','io_us','distance_us','cpu_ms')}))
    return result


def choose(development,target=.99):
    """No test results enter this function. Report infeasible approximate targets."""
    choices={}
    for mode in ('approx','certified'):
        family=[r for r in development if r['method'].startswith(mode+'-')]
        eligible=[r for r in family if r['means']['recall']>=target]
        if eligible:
            best=min(eligible,key=lambda r:(r['mean_ms'],r['method']))
        else:
            best=max(family,key=lambda r:(r['means']['recall'],-r['mean_ms']))
        choices[mode]=dict(method=best['method'],development_target_met=bool(eligible),
                           development_recall=best['means']['recall'],development_mean_ms=best['mean_ms'])
    return choices


def run(args):
    began=time.monotonic();faiss.omp_set_num_threads(1)
    dataset=json.loads((args.data/'dataset.json').read_text())
    if digest_file(args.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':
        raise ValueError('unexpected SIFT1M corpus')
    x=vectors(args.data/'sift_base.fvecs');allq=vectors(args.data/'sift_query.fvecs');learn=vectors(args.data/'sift_learn.fvecs')
    if (x.shape,allq.shape,learn.shape)!=((1000000,128),(10000,128),(100000,128)):raise ValueError('unexpected shapes')
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    permutation=np.random.default_rng(20260929).permutation(10000)
    # Exclude entire old 1000-query development pool AND the 256 previously examined test queries.
    dev_ids=permutation[:args.development];held_ids=permutation[1256:1256+args.heldout]
    assert not set(held_ids)&set(permutation[:1256])
    qids=np.r_[dev_ids,held_ids];queries=allq[qids]
    print('Build unchanged SIFT1M IVF, page layout and PCA64 eight-bit directory',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    quantizer=faiss.IndexFlatL2(128);quantizer.add(centers);routes=quantizer.search(queries,64)[1]
    refs=[reference(x,labels,q,routes[i]) for i,q in enumerate(queries)]
    np.savez(args.out/'queries-and-reference.npz',query_ids=qids,dev_ids=dev_ids,heldout_ids=held_ids,
             routes=routes,reference_ids=np.asarray(refs),supplied_gt=gt[qids,:10])
    basis=Basis.fit(learn,'pca');del learn;basis.save(args.out/'pca-basis.npz')
    projected=np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush();layout=args.work/'geopack'
    frozen=freeze_layout(x,centers,labels,layout,layout='geopack',pack_dims=16)
    if frozen['layout_payload_sha256']!='8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7':
        raise AssertionError('physical layout changed')
    side=args.work/'b8'
    meta=summarize_dependencies(layout,side,basis,dims=64,bits=8,scheme='independent',projected_by_id=projected)
    cert=certify_cells(layout,side,projected,side/'cells.json');save(args.out/'cells-certificate.json',cert)
    del projected
    idx=CellIndex(layout,side,shape='ball');memory=MemoryReplay(layout/'vectors.pages')
    checks=[];expected={};answers={};audits=0

    def execute(name,qi,reader,*,online_route=True,audit=None):
        common=dict(nprobe=64,scan='simd',preassigned_lists=None if online_route else routes[qi])
        if name=='staged':
            ids,stat=search(idx,queries[qi],reader,selection='prepared',filtering='combined',window_pages=256,
                            gap_pages=2,audit_sink=audit,**common)
            stat.update(verification_waves=stat['read_stages'],first_wave_pages=0,second_wave_pages=0,
                        first_already_certified=False,rank_us=0.,certificate_us=0.)
        else:
            mode,r=name.split('-r')
            ids,stat=search_ramfirst(idx,queries[qi],reader,shortlist=int(r),certified=mode=='certified',
                                    gap_pages=0,io_batch_bytes=16<<20,audit_sink=audit,**common)
        return ids,stat

    # Development mechanics and certificates are verified before timing.
    for name in ARMS:
        for qi in range(args.development):
            audit=[] if qi<2 and not name.startswith('approx-') else None
            ids,stat=execute(name,qi,memory,online_route=False,audit=audit)
            eq=bool(np.array_equal(ids,refs[qi]))
            if not name.startswith('approx-') and not eq:raise AssertionError('certified/reference mismatch')
            if audit:audits+=audit_rejections(idx,x,queries[qi],audit)
            expected[name,qi]=tuple(stat[k] for k in COUNTS);answers[name,qi]=ids.copy()
            checks.append(dict(method=name,query_id=int(qids[qi]),exact_match=eq,**{k:stat[k] for k in COUNTS}))
        print('Development correctness/recall recorded: '+name,flush=True)
    csv_write(args.out/'development-checks.csv',checks)
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]});load0=os.getloadavg()
    direct=RollingPooled(layout/'vectors.pages',direct=True,uring=True,depth=16)
    records=[];neighbor_rows=[]

    def measure(phase,names,cohort,reader):
        rows=[];rng=np.random.default_rng(772169)
        idx.route(queries[cohort[0]],64)
        for name in names:execute(name,cohort[0],reader)
        for rep in range(args.repeats):
            for qi in rng.permutation(cohort):
                for name in rng.permutation(names):
                    before=time.perf_counter_ns();c=time.process_time_ns()
                    ids,stat=execute(str(name),int(qi),reader)
                    wall=(time.perf_counter_ns()-before)/1e6;cpu=(time.process_time_ns()-c)/1e6
                    eq=bool(np.array_equal(ids,refs[qi]))
                    if not str(name).startswith('approx-') and not eq:raise AssertionError('timed exact answer mismatch')
                    key=(str(name),int(qi));cc=tuple(stat[k] for k in COUNTS)
                    if key in expected:
                        if expected[key]!=cc or not np.array_equal(ids,answers[key]):raise AssertionError('nonrepeatable result or read count')
                    else:expected[key]=cc;answers[key]=ids.copy()
                    recall=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10
                    ivfrec=len(set(map(int,ids))&set(map(int,refs[qi])))/10
                    row=dict(phase=phase,method=str(name),round=rep,query_id=int(qids[qi]),wall_ms=wall,cpu_ms=cpu,
                             exact_match=eq,recall=recall,ivf_recall=ivfrec,**stat)
                    # Uniform CSV schema for both strategies.
                    row={k:row[k] for k in ('phase','method','round','query_id','wall_ms','cpu_ms','exact_match','recall','ivf_recall')+
                         COUNTS+('route_us','filter_us','io_us','distance_us','total_us','verification_waves',
                         'first_wave_pages','second_wave_pages','first_already_certified','rank_us','certificate_us')}
                    rows.append(row);records.append(row)
                    if rep==0:neighbor_rows.append(dict(phase=phase,method=str(name),query_id=int(qids[qi]),ids=ids.tolist()))
            csv_write(args.out/'timings.csv',records)
            print(f'{phase}: round {rep+1}/{args.repeats} complete',flush=True)
        return summarize(rows)

    try:
        dev=measure('development-direct',ARMS,list(range(args.development)),direct)
        choices=choose(dev,args.target)
        frozen_choice=dict(target=args.target,nprobe=64,choices=choices,heldout_used_for_selection=False)
        save(args.out/'frozen-selection.json',frozen_choice)
        print('Frozen choices: '+json.dumps(frozen_choice),flush=True)
        selected=['staged',choices['approx']['method'],choices['certified']['method']]
        # Only frozen configurations see fresh test queries. No retuning.
        held=measure('heldout-direct',selected,list(range(args.development,len(qids))),direct)
        mem=measure('heldout-memory',selected,list(range(args.development,len(qids))),memory)
        pool=dict(reserved_bytes=direct.reserved_bytes,allocations=direct.allocations,cap=direct.max_pool_bytes)
    finally:
        direct.close();memory.close();os.sched_setaffinity(0,affinity)
    save(args.out/'returned-neighbors.json',neighbor_rows)
    report=dict(dataset=dataset,development_query_ids=dev_ids.tolist(),heldout_query_ids=held_ids.tolist(),
        excluded_previous_query_ids=permutation[:1256].tolist(),heldout_slice=[1256,1256+args.heldout],
        seed=12345,nlist=1024,nprobe=64,k=10,shortlists=list(SIZES),repeats=args.repeats,
        source_layout_sha256=frozen['layout_payload_sha256'],pca_basis_sha256=basis.fingerprint(),
        candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),directory=meta,
        development=dev,heldout_direct=held,heldout_memory=mem,frozen_selection=frozen_choice,
        initial_comparisons=len(checks),timed_searches=len(records),audited_rejections=audits,
        io_pool=pool,rank_scratch='O(R) candidate heap plus O(selected-IVF-pages) temporary IDs/bounds',
        load_average_start=load0,load_average_end=os.getloadavg(),cpu_affinity=[affinity[0]],
        numpy=np.__version__,faiss=faiss.__version__,platform=platform.platform(),
        elapsed_seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_direct_io=True,production_sota_claim_valid=False,
        limitations=['same-engine comparison, no fresh external baseline run',
            'one dataset/seed, small corpus fits host RAM, shared host/device cache uncontrolled',
            'all directory/oracle/data coexist in campaign RSS; not deployment RAM',
            'approximate mode may lose IVF recall; certified mode exhausts unresolved page set',
            'two decision waves may require multiple bounded reader batches',
            'conservative numeric guards are engineering safeguards, not formally verified arithmetic',
            'no query concurrency or independent physical device I/O accounting'])
    save(args.out/'ramfirst-results.json',report)
    print(json.dumps(dict(development=dev,heldout_direct=held,heldout_memory=mem)),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--development',type=int,default=128);ap.add_argument('--heldout',type=int,default=256)
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--target',type=float,default=.99)
    args=ap.parse_args()
    if not 1<=args.development<=1000 or not 1<=args.heldout<=512 or args.repeats<3 or not 0<args.target<=1:
        ap.error('invalid scope')
    for p in (args.work,args.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    lock=Path.home()/'.cache/geoivf/speed-device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);run(args)

if __name__=='__main__':main()
