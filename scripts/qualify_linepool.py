#!/usr/bin/env python3
"""Broader line-first grouping qualification; counts, not SSD timings."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import vectors, train_faiss, Index
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.dependent import summarize_dependencies, DependentIndex
from geoivf.lines import summarize_lines, LineIndex, digest
from geoivf.linepool import pool_layout
from geoivf.search import search
from geoivf.io import MemoryReplay
from scripts.qualify_sift1m import reference, audit_rejections


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=64)
    ap.add_argument('--nprobe',type=int,default=64)
    args=ap.parse_args()
    if not 1<=args.queries<=1000 or not 1<=args.nprobe<=1024:ap.error('invalid scope')
    for p in (args.work,args.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    faiss.omp_set_num_threads(1); started=time.monotonic()
    dataset=json.loads((args.data/'dataset.json').read_text())
    expected='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
    if digest(args.data/'sift_base.fvecs')!=expected:raise ValueError('unexpected corpus hash')
    x=vectors(args.data/'sift_base.fvecs'); allq=vectors(args.data/'sift_query.fvecs')
    learn=vectors(args.data/'sift_learn.fvecs')
    if x.shape!=(1000000,128) or learn.shape!=(100000,128) or allq.shape!=(10000,128):
        raise ValueError('unexpected canonical dataset shapes')
    gt=np.memmap(args.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    qids=np.random.default_rng(20260929).permutation(10000)[:1000][:args.queries]; queries=allq[qids]
    centers,labels=train_faiss(x,1024,12345,100000)
    quantizer=faiss.IndexFlatL2(128);quantizer.add(centers)
    routes=quantizer.search(queries,args.nprobe)[1]
    np.savez(args.out/'shared-candidates.npz',query_ids=qids,lists=routes,centers=centers)
    refs=[reference(x,labels,q,routes[i]) for i,q in enumerate(queries)]
    basis=Basis.fit(learn,'pca');del learn;basis.save(args.out/'pca-basis.npz')
    projected=np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush()
    physical=args.work/'geopack'
    original=freeze_layout(x,centers,labels,physical,layout='geopack',pack_dims=16)
    print('Build line-first pages from 64-vector pools',flush=True)
    refined=args.work/'line-pool64'
    t=time.monotonic(); new=pool_layout(physical,refined,x,projected,dims=64)
    new['refinement_seconds']=time.monotonic()-t
    save(args.out/'layout-refinement.json',new)
    builds=[];results=[]; comparisons=0;audits=0

    def evaluate(layout_name,name,idx,reader,mode='combined',window=64):
        nonlocal comparisons,audits
        records=[];checked_pages=0
        for qi,q in enumerate(queries):
            audit=[] if qi<2 else None
            ids,stat=search(idx,q,reader,nprobe=args.nprobe,filtering=mode,window_pages=window,
                            preassigned_lists=routes[qi],audit_sink=audit)
            np.testing.assert_array_equal(ids,refs[qi])
            if audit:checked_pages+=audit_rejections(idx,x,q,audit)
            recall=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10.
            records.append(dict(query_id=int(qids[qi]),recall_at_10=recall,
                                saved_fraction=(stat['candidate_pages']-stat['read_pages'])/max(1,stat['candidate_pages']),**stat))
        tag=layout_name+'-'+name
        with (args.out/(tag+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        savings=np.array([r['candidate_pages']-r['read_pages'] for r in records])
        row=dict(layout=layout_name,name=name,queries=len(queries),exact_matches=len(queries),
                 audited_rejections=checked_pages,geometry_bytes_per_page=idx.meta['summary_bytes']/idx.meta['n_pages'],
                 directory_array_bytes=idx.meta['directory_array_bytes'],
                 means={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='query_id'},
                 median_saved_fraction=float(np.median([r['saved_fraction'] for r in records])),
                 largest_query_share_of_savings=float(savings.max()/max(1,savings.sum())),
                 zero_saving_queries=int(np.sum(savings==0)),csv=tag+'.csv')
        results.append(row);comparisons+=len(queries);audits+=checked_pages
        save(args.out/'partial-results.json',results)
        print(json.dumps(dict(name=tag,pages=row['means']['read_pages'],extents=row['means']['read_requests'],
                              geometry=row['geometry_bytes_per_page'])),flush=True)

    for lname,layout in [('geopack',physical),('line-pool64',refined)]:
        reader=MemoryReplay(layout/'vectors.pages')
        idx=Index(layout)
        evaluate(lname,'none-whole-list',idx,reader,mode='none',window=0)
        del idx
        for dims,bits in [(18,8),(64,4)]:
            name=f'independent-d{dims}-b{bits}';out=args.work/(lname+'-'+name)
            print('Build '+lname+' '+name,flush=True);t=time.monotonic()
            meta=summarize_dependencies(layout,out,basis,dims=dims,scheme='independent',bits=bits,projected_by_id=projected)
            builds.append(dict(layout=lname,name=name,build_seconds=time.monotonic()-t,**meta))
            idx=DependentIndex(layout,out);evaluate(lname,name,idx,reader);del idx
        for dims,precision in [(64,'uint8')]:
            name=f'line-d{dims}-{precision}';out=args.work/(lname+'-'+name)
            print('Build '+lname+' '+name,flush=True);t=time.monotonic()
            meta=summarize_lines(layout,out,basis,projected,dims=dims,precision=precision)
            builds.append(dict(layout=lname,name=name,build_seconds=time.monotonic()-t,**meta))
            for bound in ['ball','shell']:
                idx=LineIndex(layout,out,bound=bound);evaluate(lname,name+'-'+bound,idx,reader);del idx
        reader.close()
    for layout,meta in [(physical,original),(refined,new)]:
        if digest(layout/'vectors.pages')!=meta['layout_payload_sha256']:raise AssertionError('payload changed')
    report=dict(dataset=dataset,query_split='development-only',query_ids=qids.tolist(),independent_queries=len(queries),
                nlist=1024,nprobe=args.nprobe,k=10,seed=12345,
                source_layout_sha256=original['layout_payload_sha256'],refined_layout_sha256=new['layout_payload_sha256'],
                pca_basis_sha256=basis.fingerprint(),candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                configuration_query_matches=comparisons,audited_rejection_decisions=audits,
                refinement=new['refinement'],builds=builds,results=results,
                elapsed_seconds=time.monotonic()-started,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                numpy=np.__version__,faiss=faiss.__version__,python=platform.python_version(),
                storage_latency_valid=False,production_speedup_valid=False,
                limitations=['RAM replay, not SSD timing','64 development queries, one construction seed',
                             'line-first 64-vector pools, not globally optimal grouping',
                             'engineering numerical guards','full-dimensional line scalar map loses residual angle',
                             'no CLIP result','no timing isolation or native common scan kernel'])
    save(args.out/'line-results.json',report)
    print(f'PASS: {comparisons} comparisons, {audits} audited rejections',flush=True)

if __name__=='__main__':main()
