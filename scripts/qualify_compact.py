#!/usr/bin/env python3
"""Canonical short-segment qualification with separate layout/filter diagnostics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
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
from geoivf.compact import compact_layout, layout_metrics
from geoivf.search import search
from geoivf.io import MemoryReplay
from scripts.qualify_sift1m import reference, audit_rejections


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def offline_layout(index, x, queries, routes, refs):
    """Fixed final-radius diagnostics, never used to plan the online reads."""
    records = []
    for qi, q in enumerate(queries):
        delta = x[refs[qi][-1]].astype(float)-q
        tau2 = float(np.einsum('i,i->', delta, delta))
        pages, minima = [], []
        for li in routes[qi]:
            first, last = map(int, index.ranges[li])
            pp = np.arange(first, last)
            ids = index.page_ids[pp]
            d = x[np.maximum(ids, 0)].astype(float)-q
            dd = np.einsum('bnd,bnd->bn', d, d)
            dd[ids < 0] = np.inf
            pages.append(pp); minima.append(dd.min(axis=1))
        pp = np.concatenate(pages); minimum = np.concatenate(minima)
        records.append(dict(pages=pp, minimum_squared=minimum, tau_squared=tau2,
                            near_pages={str(s):int(np.sum(minimum<=tau2*s*s)) for s in (1., 1.1, 1.25)}))
    return records


def final_admissions(index, q, lists, diag):
    # Entire operation is an AFTER-search diagnostic at a hindsight threshold.
    lb = np.concatenate([index.bounds(q, np.arange(*map(int, index.ranges[li])), int(li), 'combined') for li in lists])
    tau = np.sqrt(diag['tau_squared'])
    true = diag['minimum_squared']<=diag['tau_squared']
    keep = lb<=tau
    if np.any(true & ~keep):
        raise AssertionError('final-radius summary rejected a true in-radius page')
    return int(keep.sum()), int((keep & ~true).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--queries', type=int, default=64)
    args = ap.parse_args()
    if not 1<=args.queries<=1000: ap.error('invalid development query count')
    for p in (args.work, args.out):
        if p.exists() and any(p.iterdir()): raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1); started = time.monotonic()
    dataset = json.loads((args.data/'dataset.json').read_text())
    if digest(args.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':
        raise ValueError('unexpected canonical corpus hash')
    x = vectors(args.data/'sift_base.fvecs'); allq = vectors(args.data/'sift_query.fvecs')
    learn = vectors(args.data/'sift_learn.fvecs')
    if x.shape!=(1000000,128) or learn.shape!=(100000,128) or allq.shape!=(10000,128):
        raise ValueError('unexpected canonical shapes')
    gt = np.memmap(args.data/'sift_groundtruth.ivecs', dtype='<i4', mode='r', shape=(10000,101))[:,1:]
    qids = np.random.default_rng(20260929).permutation(10000)[:1000][:args.queries]
    queries = allq[qids]
    print('Shared IVF, candidate routes and learning-file PCA', flush=True)
    centers, labels = train_faiss(x,1024,12345,100000)
    quantizer = faiss.IndexFlatL2(128); quantizer.add(centers)
    routes = quantizer.search(queries,64)[1]
    np.savez(args.out/'shared-candidates.npz',query_ids=qids,lists=routes,centers=centers)
    refs = [reference(x,labels,q,routes[i]) for i,q in enumerate(queries)]
    basis = Basis.fit(learn,'pca'); del learn; basis.save(args.out/'pca-basis.npz')
    projected = np.lib.format.open_memmap(args.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768): projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush()
    physical = args.work/'geopack'
    original = freeze_layout(x,centers,labels,physical,layout='geopack',pack_dims=16)
    layouts = [('geopack',physical,original)]
    path=args.work/'old-linepool';t=time.monotonic()
    meta=pool_layout(physical,path,x,projected,dims=64)
    meta['build_seconds']=time.monotonic()-t;layouts.append(('old-linepool',path,meta))
    for eps in (0.,.05):
        name='compact-strict' if eps==0 else 'compact-relax5'
        path=args.work/name;t=time.monotonic()
        meta=compact_layout(physical,path,x,projected,dims=64,relaxation=eps,passes=1)
        meta['build_seconds']=time.monotonic()-t;layouts.append((name,path,meta))
        save(args.out/(name+'-layout.json'),meta)
        shutil.copyfile(path/'compact-metrics.npz',args.out/(name+'-checks.npz'))
    builds=[]; results=[]; geometric=[]; comparisons=0; audits=0
    for lname,layout,layoutmeta in layouts:
        idx=Index(layout);reader=MemoryReplay(layout/'vectors.pages')
        metrics=layout_metrics(idx,x,projected,64)
        np.savez(args.out/(lname+'-geometry.npz'),**metrics)
        geo=dict(layout=lname,payload_sha256=layoutmeta['layout_payload_sha256'],
                 statistics={k:dict(mean=float(v.mean()),median=float(np.median(v)),p95=float(np.quantile(v,.95))) for k,v in metrics.items()},
                 refinement=layoutmeta.get('refinement'),build_seconds=layoutmeta.get('build_seconds'))
        geometric.append(geo)
        # Construct offline diagnostics now, but do not pass them to search.
        diagnostics=offline_layout(idx,x,queries,routes,refs)
        offline=[dict(query_id=int(qids[i]),candidate_pages=len(d['pages']),
                      final_ivf_tau=float(np.sqrt(d['tau_squared'])),near_pages=d['near_pages']) for i,d in enumerate(diagnostics)]
        save(args.out/(lname+'-offline-layout.json'),dict(scope='offline common-radius only',observations=offline))

        def evaluate(name,index,mode='combined',window=64):
            nonlocal comparisons,audits
            records=[];checked=0
            for qi,q in enumerate(queries):
                audit=[] if qi<2 else None
                ids,stat=search(index,q,reader,nprobe=64,filtering=mode,window_pages=window,
                                preassigned_lists=routes[qi],audit_sink=audit)
                np.testing.assert_array_equal(ids,refs[qi])
                if audit: checked+=audit_rejections(index,x,q,audit)
                diag=diagnostics[qi];true=diag['near_pages']['1.0']
                if mode=='none': admitted=stat['candidate_pages'];fp=admitted-true
                else: admitted,fp=final_admissions(index,q,routes[qi],diag)
                if stat['read_pages']<admitted:raise AssertionError('online plan below final-radius admission count')
                records.append(dict(query_id=int(qids[qi]),recall_at_10=len(set(map(int,ids))&set(map(int,gt[qids[qi],:10])))/10.,
                                    true_pages_at_final_radius=true,final_radius_admitted_pages=admitted,
                                    final_radius_false_positive_pages=fp,
                                    online_extra_above_final_radius=stat['read_pages']-admitted,
                                    near_pages_radius_1_1=diag['near_pages']['1.1'],near_pages_radius_1_25=diag['near_pages']['1.25'],
                                    saved_fraction=1-stat['read_pages']/stat['candidate_pages'],**stat))
            tag=lname+'-'+name
            with (args.out/(tag+'.csv')).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
            row=dict(layout=lname,name=name,queries=len(queries),exact_matches=len(queries),audited_rejections=checked,
                     geometry_bytes_per_page=0 if mode=='none' else index.meta['summary_bytes']/index.meta['n_pages'],
                     directory_array_bytes=index.meta['directory_array_bytes'],
                     means={k:float(np.mean([r[k] for r in records])) for k in records[0] if k!='query_id'},
                     median_saved_fraction=float(np.median([r['saved_fraction'] for r in records])),csv=tag+'.csv')
            results.append(row);comparisons+=len(queries);audits+=checked
            save(args.out/'partial-results.json',results)
            print(json.dumps(dict(name=tag,pages=row['means']['read_pages'],true_pages=row['means']['true_pages_at_final_radius'],
                                  false_positive_pages=row['means']['final_radius_false_positive_pages'],extents=row['means']['read_requests'])),flush=True)

        evaluate('none-whole-list',idx,mode='none',window=0);del idx
        for dims,bits in [(18,8),(64,4)]:
            name=f'independent-d{dims}-b{bits}';out=args.work/(lname+'-'+name)
            meta=summarize_dependencies(layout,out,basis,dims=dims,scheme='independent',bits=bits,projected_by_id=projected)
            builds.append(dict(layout=lname,name=name,**meta))
            idx=DependentIndex(layout,out);evaluate(name,idx);del idx
        out=args.work/(lname+'-line64')
        meta=summarize_lines(layout,out,basis,projected,dims=64,precision='uint8')
        builds.append(dict(layout=lname,name='line64',**meta))
        for bound in ('ball','shell'):
            idx=LineIndex(layout,out,bound=bound);evaluate('line64-'+bound,idx);del idx
        reader.close()
        if digest(layout/'vectors.pages')!=layoutmeta['layout_payload_sha256']:raise AssertionError('payload mutated')
    report=dict(dataset=dataset,independent_queries=len(queries),query_ids=qids.tolist(),query_split='development',
                nlist=1024,nprobe=64,k=10,seed=12345,source_layout_sha256=original['layout_payload_sha256'],
                pca_basis_sha256=basis.fingerprint(),candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                configuration_query_matches=comparisons,audited_rejection_decisions=audits,
                layouts=geometric,builds=builds,results=results,elapsed_seconds=time.monotonic()-started,
                maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                numpy=np.__version__,faiss=faiss.__version__,python=platform.python_version(),
                storage_latency_valid=False,production_speedup_valid=False,
                limitations=['64 reused development queries, one dataset and seed','RAM replay, not physical SSD traffic',
                             'local balanced proposals in 64-vector pools, not globally shortest segments',
                             'diameter is only a proxy for query locality','least-squares line refit, not minimax line fit',
                             'offline final radius never used by online search','engineering floating-point guards','no CLIP result'])
    save(args.out/'compact-results.json',report)
    print(f'PASS {comparisons} comparisons; {audits} rejection decisions audited',flush=True)

if __name__=='__main__':main()
