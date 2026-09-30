#!/usr/bin/env python3
"""Unequal scalar bit allocation, fixed pages/code budget and held-out verification."""
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
from geoivf.rotations import build_rotation,RotationIndex,search_rotation
from geoivf.mixedbits import MixedIndex,MIXED_KINDS,calibration_tables,build_mixed
from geoivf.io import MemoryReplay
from geoivf.execution import RollingPooled
from scripts.qualify_sift1m import reference,audit_rejections
from scripts.qualify_speed import COUNTS,csv_write,save,RecordedReader
from scripts.qualify_rotations import summarize

UNIFORM=('pca64-sq8','pca128-sq4','identity128-sq4','pq64x8')
KINDS=UNIFORM+MIXED_KINDS
SIZES=(16,32,64,128,256)
EXTRA=('verification_waves','first_wave_pages','second_wave_pages','rank_us','certificate_us')


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
    dev_ids=perm[:args.development];held_ids=perm[1768:1768+args.heldout]
    if set(held_ids)&set(perm[:1768]):raise AssertionError('test leakage')
    qids=np.r_[dev_ids,held_ids];queries=allq[qids]
    print('Build identical coarse IVF and frozen GeoPack pages',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    router=faiss.IndexFlatL2(128);router.add(centers);routes=router.search(queries,64)[1]
    refs=np.asarray([reference(x,labels,q,routes[i]) for i,q in enumerate(queries)])
    np.savez(args.out/'queries-and-reference.npz',query_ids=qids,development_ids=dev_ids,
        heldout_ids=held_ids,routes=routes,reference_ids=refs,groundtruth=gt[qids,:10])
    basis=Basis.fit(learn,'pca');basis.save(args.out/'pca-basis.npz')
    layout=args.work/'geopack';frozen=freeze_layout(x,centers,labels,layout,pack_dims=16)
    if frozen['layout_payload_sha256']!='8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7':raise AssertionError('layout changed')
    models={};builds=[];old_controls={}
    for kind in UNIFORM:
        print('Build unchanged reference: '+kind,flush=True)
        side=args.work/kind;meta=build_rotation(layout,side,x,learn,basis,kind,train_count=50000)
        idx=RotationIndex(layout,side) if kind=='pq64x8' else MixedIndex(layout,side)
        if kind!='pq64x8':
            meta['extra_uniform_descriptor_bytes']=idx.extra_descriptor_bytes
            meta['directory_array_bytes']+=idx.extra_descriptor_bytes
            if kind=='pca64-sq8':old_controls['old-pca64-sq8']=RotationIndex(layout,side)
        models[kind]=idx;builds.append(meta);save(args.out/'builds.json',builds)
    cal_start=time.monotonic()
    projection=np.lib.format.open_memmap(args.work/'full-projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projection[s:s+32768]=basis.project(x[s:s+32768])
    projection.flush();tables=calibration_tables(projection,basis.project(learn))
    calibration_seconds=time.monotonic()-cal_start
    np.savez(args.out/'allocation-calibration.npz',**tables,widths=np.array([2,4,6,8]))
    for kind in MIXED_KINDS:
        side=args.work/kind;meta=build_mixed(layout,side,projection,basis,tables,kind)
        models[kind]=MixedIndex(layout,side);builds.append(meta);save(args.out/'builds.json',builds)
        print(json.dumps(meta),flush=True)
    del projection
    models.update(old_controls)
    arms=[f'{kind}/{mode}/{r}' for kind in KINDS for mode in ('approx','certified') for r in SIZES]
    controls=['old-pca64-sq8/approx/64','old-pca64-sq8/certified/128']
    arms+=controls
    memory=MemoryReplay(layout/'vectors.pages');checks=[];audits=0;answers={};operations={}
    def execute(name,qi,reader,*,audit=None,online=True):
        kind,mode,r=name.split('/')
        return search_rotation(models[kind],queries[qi],reader,shortlist=int(r),mode=mode,
            nprobe=64,scan='simd',preassigned_lists=None if online else routes[qi],audit_sink=audit)
    # Same packed codes and complete request plans across old/new scalar scanners.
    equivalent=[]
    for mode,r in [('approx',64),('certified',128)]:
        for qi in range(min(8,args.development)):
            aa=RecordedReader(memory);bb=RecordedReader(memory)
            ids,sa=execute(f'pca64-sq8/{mode}/{r}',qi,aa,online=False)
            other,sb=execute(f'old-pca64-sq8/{mode}/{r}',qi,bb,online=False)
            np.testing.assert_array_equal(ids,other)
            if aa.hash.hexdigest()!=bb.hash.hexdigest() or tuple(sa[k] for k in COUNTS)!=tuple(sb[k] for k in COUNTS):
                raise AssertionError('uniform scanner changed complete request plan')
            equivalent.append(dict(mode=mode,query_id=int(qids[qi]),plan_sha256=aa.hash.hexdigest()))
    save(args.out/'uniform-plan-equivalence.json',equivalent)
    for name in arms:
        for qi in range(2):
            exact='/certified/' in name;audit=[] if exact else None
            ids,stat=execute(name,qi,memory,audit=audit,online=False)
            if exact:np.testing.assert_array_equal(ids,refs[qi])
            if audit:audits+=audit_rejections(models[name.split('/')[0]],x,queries[qi],audit)
            checks.append(dict(method=name,query_id=int(qids[qi]),exact_match=bool(np.array_equal(ids,refs[qi])),**{k:stat[k] for k in COUNTS}))
    csv_write(args.out/'audit-checks.csv',checks)
    affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[0]});load0=os.getloadavg()
    for idx in models.values():idx.route(queries[0],64)
    direct=RollingPooled(layout/'vectors.pages',direct=True,uring=True,depth=16)
    records=[];neighbor_rows=[]
    def measure(phase,names,cohort,reader):
        rows=[];rng=np.random.default_rng(448073)
        for name in names:execute(name,cohort[0],reader)
        for rep in range(args.repeats):
            for qi in rng.permutation(cohort):
                for name in rng.permutation(names):
                    qi=int(qi);name=str(name);t=time.perf_counter_ns();c=time.process_time_ns()
                    ids,stat=execute(name,qi,reader)
                    wall=(time.perf_counter_ns()-t)/1e6;cpu=(time.process_time_ns()-c)/1e6
                    equal=bool(np.array_equal(ids,refs[qi]))
                    if '/certified/' in name and not equal:raise AssertionError('certified answer mismatch')
                    key=name,qi;cc=tuple(stat[k] for k in COUNTS)
                    if key in answers:
                        np.testing.assert_array_equal(ids,answers[key])
                        if operations[key]!=cc:raise AssertionError('nonrepeatable read counts')
                    else:answers[key]=ids.copy();operations[key]=cc
                    recall=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10
                    ivfrec=len(set(map(int,ids))&set(map(int,refs[qi])))/10
                    row=dict(phase=phase,method=name,round=rep,query_id=int(qids[qi]),wall_ms=wall,cpu_ms=cpu,
                        exact_match=equal,recall=recall,ivf_recall=ivfrec,
                        **{k:stat[k] for k in COUNTS+EXTRA+('route_us','filter_us','io_us','distance_us','total_us')})
                    rows.append(row);records.append(row)
                    if rep==0:neighbor_rows.append(dict(phase=phase,method=name,query_id=int(qids[qi]),ids=ids.tolist()))
            csv_write(args.out/'timings.csv',records)
            print(f'{phase}: round {rep+1}/{args.repeats} completed',flush=True)
        return summarize(rows)
    try:
        dev=measure('development-direct',arms,list(range(args.development)),direct)
        choices=choose(dev,args.target)
        save(args.out/'frozen-selection.json',dict(target=args.target,choices=choices,heldout_used=False))
        print('FROZEN '+json.dumps(choices),flush=True)
        selected=controls+[v['method'] for family in choices.values() for v in family.values()]
        # Fixed-R64 approximate arms were specified before looking at test data.
        selected=list(dict.fromkeys(selected+[f'{kind}/approx/64' for kind in KINDS]))
        cohort=list(range(args.development,len(qids)))
        held=measure('heldout-direct',selected,cohort,direct)
        mem=measure('heldout-memory',selected,cohort,memory)
        pool=dict(reserved_bytes=direct.reserved_bytes,allocations=direct.allocations,cap=direct.max_pool_bytes)
    finally:
        direct.close();memory.close();os.sched_setaffinity(0,affinity)
    save(args.out/'returned-neighbors.json',neighbor_rows)
    report=dict(dataset=dataset,development_query_ids=dev_ids.tolist(),heldout_query_ids=held_ids.tolist(),
        heldout_slice=[1768,1768+args.heldout],excluded_previous_query_ids=perm[:1768].tolist(),
        nlist=1024,nprobe=64,k=10,seed=12345,shortlists=list(SIZES),repeats=args.repeats,
        source_layout_sha256=frozen['layout_payload_sha256'],pca_basis_sha256=basis.fingerprint(),
        candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),builds=builds,
        calibration_seconds=calibration_seconds,development=dev,heldout_direct=held,heldout_memory=mem,
        choices=choices,initial_comparisons=len(checks),timed_searches=len(records),audited_rejections=audits,
        uniform_plan_equivalence=equivalent,io_pool=pool,cpu_affinity=[affinity[0]],
        load_average_start=load0,load_average_end=os.getloadavg(),platform=platform.platform(),
        faiss=faiss.__version__,numpy=np.__version__,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        elapsed_seconds=time.monotonic()-started,real_direct_io=True,production_sota_claim_valid=False,
        limitations=['one SIFT1M index/seed; held-out queries are fresh for this campaign only',
        'all codes use 64 bytes plus FP32 radius; shared allocation descriptors counted separately',
        'allocation objective is learning-set reconstruction MSE, not optimal recall or page I/O',
        'all mixed variants retain all 128 PCA dimensions with 2/4/6/8 bits',
        'shared host and device caches uncontrolled; CPU affinity does not reserve a core',
        'dataset fits host RAM; benchmark RSS includes all variants, oracle and base data',
        'uniform/mixed scalar controls share grouped ranker; PQ uses existing uniform-code LUT ranker',
        'no SAQ implementation, external SOTA rerun, concurrency test or matched RSS cap',
        'FP64 numerical safeguards are engineering measures, not formally verified interval arithmetic'])
    save(args.out/'mixedbits-results.json',report)
    print(json.dumps(dict(heldout_direct=held,heldout_memory=mem)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--development',type=int,default=128);ap.add_argument('--heldout',type=int,default=256)
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--target',type=float,default=.99);args=ap.parse_args()
    if not 2<=args.development<=1000 or not 1<=args.heldout<=512 or args.repeats<3 or not 0<args.target<=1:ap.error('invalid scope')
    for p in (args.work,args.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    lock=Path.home()/'.cache/geoivf/speed-device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX);run(args)
if __name__=='__main__':main()
