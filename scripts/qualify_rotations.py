#!/usr/bin/env python3
"""Equal 64-byte codes on fixed pages: rotation, truncation, PQ and certification."""
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
from geoivf.rotations import KINDS,build_rotation,RotationIndex,search_rotation
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.ramfirst import search_ramfirst
from geoivf.search import search
from geoivf.io import MemoryReplay
from geoivf.execution import RollingPooled
from scripts.qualify_sift1m import reference,audit_rejections
from scripts.qualify_speed import COUNTS,csv_write,save

SIZES=(16,32,64,128,256)
EXTRA=('verification_waves','first_wave_pages','second_wave_pages','rank_us','certificate_us')
BASELINES=('legacy-staged','legacy-approx-r64','legacy-certified-r32')


def summarize(rows):
    out=[]
    for name in dict.fromkeys(r['method'] for r in rows):
        rr=[r for r in rows if r['method']==name];wall=np.array([r['wall_ms'] for r in rr])
        out.append(dict(method=name,samples=len(rr),distinct_queries=len(set(r['query_id'] for r in rr)),
            mean_ms=float(wall.mean()),median_ms=float(np.median(wall)),p95_ms=float(np.quantile(wall,.95)),
            exact_match_fraction=float(np.mean([r['exact_match'] for r in rr])),
            means={k:float(np.mean([r[k] for r in rr])) for k in COUNTS+EXTRA+('recall','ivf_recall','cpu_ms','filter_us','io_us','distance_us')},
            round_mean_ms=[float(np.mean([r['wall_ms'] for r in rr if r['round']==i])) for i in sorted(set(r['round'] for r in rr))]))
    return out


def choose(rows,target):
    result={}
    for kind in KINDS:
        result[kind]={}
        for mode in ('approx','certified'):
            family=[r for r in rows if r['method'].startswith(kind+'/'+mode+'/')]
            eligible=[r for r in family if r['means']['recall']>=target]
            pick=min(eligible,key=lambda r:(r['mean_ms'],r['method'])) if eligible else max(family,key=lambda r:(r['means']['recall'],-r['mean_ms']))
            result[kind][mode]=dict(method=pick['method'],target_met=bool(eligible),
                development_recall=pick['means']['recall'],development_mean_ms=pick['mean_ms'])
    return result


def run(args):
    started=time.monotonic();faiss.omp_set_num_threads(1)
    if digest_file(args.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':raise ValueError('wrong SIFT1M hash')
    x=vectors(args.data/'sift_base.fvecs');learn=vectors(args.data/'sift_learn.fvecs');allq=vectors(args.data/'sift_query.fvecs')
    if (x.shape,learn.shape,allq.shape)!=((1000000,128),(100000,128),(10000,128)):raise ValueError('wrong shapes')
    dataset=json.loads((args.data/'dataset.json').read_text())
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    perm=np.random.default_rng(20260929).permutation(10000)
    dev_ids=perm[:args.development];held_ids=perm[1512:1512+args.heldout]
    if set(held_ids)&set(perm[:1512]):raise AssertionError('test leakage')
    qids=np.r_[dev_ids,held_ids];queries=allq[qids]
    centers,labels=train_faiss(x,1024,12345,100000)
    router=faiss.IndexFlatL2(128);router.add(centers);routes=router.search(queries,64)[1]
    refs=np.asarray([reference(x,labels,q,routes[i]) for i,q in enumerate(queries)])
    np.savez(args.out/'queries-and-reference.npz',query_ids=qids,development_ids=dev_ids,
        heldout_ids=held_ids,routes=routes,reference_ids=refs,groundtruth=gt[qids,:10])
    basis=Basis.fit(learn,'pca');basis.save(args.out/'pca-basis.npz')
    layout=args.work/'geopack';frozen=freeze_layout(x,centers,labels,layout,pack_dims=16)
    if frozen['layout_payload_sha256']!='8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7':raise AssertionError('layout changed')
    models={};builds=[]
    for kind in KINDS:
        print('Build and all-point certify: '+kind,flush=True)
        side=args.work/kind;meta=build_rotation(layout,side,x,learn,basis,kind,train_count=50000,opq_iters=20)
        models[kind]=RotationIndex(layout,side);builds.append(meta)
        save(args.out/'builds.json',builds)
        print(json.dumps(meta),flush=True)
    # Old execution controls use their actual original code/sidecar and ranker.
    projected=np.lib.format.open_memmap(args.work/'legacy_projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush();side=args.work/'legacy'
    legacy_meta=summarize_dependencies(layout,side,basis,dims=64,bits=8,scheme='independent',projected_by_id=projected)
    certify_cells(layout,side,projected,side/'cells.json');del projected
    legacy=CellIndex(layout,side,shape='ball')
    np.testing.assert_array_equal(models['pca64-sq8'].arrays['packed'],legacy.arrays['packed'])
    save(args.out/'baseline-code-equality.json',dict(equal=True,kind='pca64-sq8',
        note='same packed codes; common LUT scoring and guard bookkeeping are separate execution changes'))
    arms=list(BASELINES)+[f'{kind}/{mode}/{r}' for kind in KINDS for mode in ('approx','certified') for r in SIZES]
    arms += [f'{kind}/upper/64' for kind in KINDS if models[kind].summary['norm_lower']>0]
    memory=MemoryReplay(layout/'vectors.pages');answers={};operations={};checks=[];audits=0
    def execute(name,qi,reader,*,audit=None,online=True):
        common=dict(nprobe=64,scan='simd',preassigned_lists=None if online else routes[qi])
        if name=='legacy-staged':
            ids,stat=search(legacy,queries[qi],reader,selection='prepared',filtering='combined',
                window_pages=256,gap_pages=2,audit_sink=audit,**common)
            stat.update(verification_waves=stat['read_stages'],first_wave_pages=0,second_wave_pages=0,rank_us=0.,certificate_us=0.)
        elif name.startswith('legacy-'):
            ids,stat=search_ramfirst(legacy,queries[qi],reader,shortlist=64 if 'approx' in name else 32,
                certified='certified' in name,gap_pages=0,audit_sink=audit,**common)
        else:
            kind,mode,r=name.split('/')
            ids,stat=search_rotation(models[kind],queries[qi],reader,shortlist=int(r),mode=mode,audit_sink=audit,**common)
        return ids,stat
    # Audit the first two development queries for every exact arm before timing.
    for name in arms:
        exact='approx' not in name
        for qi in range(2):
            audit=[] if exact else None
            ids,stat=execute(name,qi,memory,audit=audit,online=False)
            if exact:np.testing.assert_array_equal(ids,refs[qi])
            if audit:
                idx=legacy if name.startswith('legacy') else models[name.split('/')[0]]
                audits+=audit_rejections(idx,x,queries[qi],audit)
            checks.append(dict(method=name,query_id=int(qids[qi]),exact_match=bool(np.array_equal(ids,refs[qi])),**{k:stat[k] for k in COUNTS}))
    csv_write(args.out/'audit-checks.csv',checks)
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]});load0=os.getloadavg()
    for idx in list(models.values())+[legacy]:idx.route(queries[0],64)
    direct=RollingPooled(layout/'vectors.pages',direct=True,uring=True,depth=16)
    records=[];neighbor_rows=[]
    def measure(phase,names,cohort,reader):
        rows=[];rng=np.random.default_rng(382017)
        for name in names:execute(name,cohort[0],reader)
        for rep in range(args.repeats):
            for qi in rng.permutation(cohort):
                for name in rng.permutation(names):
                    qi=int(qi);name=str(name)
                    if name.startswith('legacy'):legacy._query=None
                    t=time.perf_counter_ns();c=time.process_time_ns()
                    ids,stat=execute(name,qi,reader)
                    wall=(time.perf_counter_ns()-t)/1e6;cpu=(time.process_time_ns()-c)/1e6
                    equal=bool(np.array_equal(ids,refs[qi]))
                    if 'approx' not in name and not equal:raise AssertionError('exact candidate-set answer mismatch')
                    key=name,qi;counts=tuple(stat[k] for k in COUNTS)
                    if key in answers:
                        np.testing.assert_array_equal(ids,answers[key])
                        if operations[key]!=counts:raise AssertionError('nonrepeatable operation counts')
                    else:answers[key]=ids.copy();operations[key]=counts
                    recall=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10
                    ivfrec=len(set(map(int,ids))&set(map(int,refs[qi])))/10
                    row=dict(phase=phase,method=name,round=rep,query_id=int(qids[qi]),wall_ms=wall,cpu_ms=cpu,
                        exact_match=equal,recall=recall,ivf_recall=ivfrec,
                        **{k:stat[k] for k in COUNTS+EXTRA+('route_us','filter_us','io_us','distance_us','total_us')})
                    rows.append(row);records.append(row)
                    if rep==0:neighbor_rows.append(dict(phase=phase,method=name,query_id=int(qids[qi]),ids=ids.tolist()))
            csv_write(args.out/'timings.csv',records)
            print(f'{phase}: round {rep+1}/{args.repeats} complete',flush=True)
        return summarize(rows)
    try:
        dev=measure('development-direct',arms,list(range(args.development)),direct)
        frozen_choices=choose(dev,args.target);save(args.out/'frozen-selection.json',frozen_choices)
        print('FROZEN CHOICES '+json.dumps(frozen_choices),flush=True)
        selected=list(BASELINES)
        for kind in KINDS:
            selected += [frozen_choices[kind][mode]['method'] for mode in ('approx','certified')]
            selected.append(f'{kind}/approx/64') # predeclared fixed-R shape comparison, never retuned.
            if models[kind].summary['norm_lower']>0:selected.append(f'{kind}/upper/64')
        selected=list(dict.fromkeys(selected))
        held=measure('heldout-direct',selected,list(range(args.development,len(qids))),direct)
        replay=measure('heldout-memory',selected,list(range(args.development,len(qids))),memory)
        pool=dict(reserved_bytes=direct.reserved_bytes,allocations=direct.allocations,cap=direct.max_pool_bytes)
    finally:
        direct.close();memory.close();os.sched_setaffinity(0,affinity)
    save(args.out/'returned-neighbors.json',neighbor_rows)
    report=dict(dataset=dataset,development_query_ids=dev_ids.tolist(),heldout_query_ids=held_ids.tolist(),
        heldout_slice=[1512,1512+args.heldout],excluded_prior_ids=perm[:1512].tolist(),
        seed=12345,nlist=1024,nprobe=64,k=10,code_bytes_per_vector=64,shortlists=list(SIZES),
        repeats=args.repeats,target=args.target,builds=builds,legacy_directory=legacy_meta,
        development=dev,heldout_direct=held,heldout_memory=replay,frozen_choices=frozen_choices,
        audit_comparisons=len(checks),audited_rejections=audits,timed_searches=len(records),
        all_exact_answers_passed=True,baseline_codes_equal=True,
        source_payload_sha256=frozen['layout_payload_sha256'],pca_sha256=basis.fingerprint(),
        route_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
        cpu_affinity=[affinity[0]],load_start=load0,load_end=os.getloadavg(),io_pool=pool,
        elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        faiss=faiss.__version__,numpy=np.__version__,platform=platform.platform(),
        real_direct_io=True,production_sota_claim=False,
        limitations=['one SIFT1M index/seed, no external SOTA rerun',
            'equal 64-byte codes plus radius; shared matrices/codebooks separately charged',
            'new common FP64 LUT ranker must not be conflated with geometric improvement',
            'scalar ranges use base data, PQ/OPQ use 50000 learning vectors and fixed training budgets',
            'OPQ default iterations not used; this is a pilot, not exhaustive tuning',
            'shared CPU/device, frequency and device cache uncontrolled, payload fits RAM',
            'base/oracle/all variants retained for experiment, RSS not deployment RAM',
            'one verification wave can require multiple bounded reader calls',
            'engineering numerical guards and transform bounds, not formally verified arithmetic'])
    save(args.out/'rotation-results.json',report)
    print(json.dumps(dict(heldout_direct=held,heldout_memory=replay)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--development',type=int,default=128);ap.add_argument('--heldout',type=int,default=256)
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--target',type=float,default=.99)
    args=ap.parse_args()
    if not 2<=args.development<=1000 or not 1<=args.heldout<=512 or args.repeats<3 or not 0<args.target<=1:ap.error('invalid scope')
    for p in (args.work,args.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    lock=Path.home()/'.cache/geoivf/speed-device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:fcntl.flock(f,fcntl.LOCK_EX);run(args)
if __name__=='__main__':main()
