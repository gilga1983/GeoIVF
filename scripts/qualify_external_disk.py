#!/usr/bin/env python3
"""Pinned released DiskANN3 versus GeoIVF, disjoint development and held-out sets.

DiskANN search/training code is unmodified. No claim of equal hard RSS caps:
PQ codes and disabled node caching are explicit; runtime overhead is separate.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import faiss
from geoivf.index import vectors,train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis,digest_file
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.execution import RollingPooled
from geoivf.search import search
from scripts.qualify_sift1m import reference
from scripts.qualify_speed import save,csv_write


def fbin(path,array):
    a=np.asarray(array,dtype='<f4',order='C')
    with Path(path).open('wb') as f:
        np.asarray(a.shape,dtype='<u4').tofile(f);a.tofile(f)


def gtbin(path,ids,x,q):
    ids=np.asarray(ids,dtype='<u4',order='C')
    d=np.sum((x[ids].astype(np.float64)-q[:,None,:])**2,axis=2).astype('<f4')
    with Path(path).open('wb') as f:
        np.asarray(ids.shape,dtype='<u4').tofile(f);ids.tofile(f);d.tofile(f)


def result_rows(obj):
    found=[]
    if isinstance(obj,dict):
        if 'search_l' in obj and 'mean_latency' in obj and 'recall' in obj:found.append(obj)
        else:
            for v in obj.values():found.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj:found.extend(result_rows(v))
    return found


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--queries',type=int,default=256);ap.add_argument('--development',type=int,default=128)
    ap.add_argument('--repeats',type=int,default=3);a=ap.parse_args()
    if not 1<=a.development<=1000 or not 1<=a.queries<=9000 or a.repeats<1:ap.error('invalid split')
    for p in (a.work,a.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    a.work=a.work.resolve();a.out=a.out.resolve();a.binary=a.binary.resolve()
    lockpath=Path.home()/'.cache/geoivf/speed-device.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(a)


def run(a):
    faiss.omp_set_num_threads(1);started=time.monotonic()
    if digest_file(a.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':raise ValueError('wrong corpus')
    x=vectors(a.data/'sift_base.fvecs');allq=vectors(a.data/'sift_query.fvecs');learn=vectors(a.data/'sift_learn.fvecs')
    gt=np.memmap(a.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    perm=np.random.default_rng(20260929).permutation(10000)
    dev=perm[:a.development];held=perm[1000:1000+a.queries]
    assert not set(dev)&set(held)
    save(a.out/'splits.json',dict(development=dev.tolist(),heldout=held.tolist(),seed=20260929))
    fbin(a.work/'base.fbin',x)
    for name,ids in [('development',dev),('heldout',held)]:
        fbin(a.work/(name+'.fbin'),allq[ids]);gtbin(a.work/(name+'.gt'),gt[ids,:100],x,allq[ids])
    prefix=str(a.work/'diskann-index')
    def disk_run(tag,source,split,ls,beam):
        phase=dict(queries=str(a.work/(split+'.fbin')),groundtruth=str(a.work/(split+'.gt')),
            search_list=ls,beam_width=beam,recall_at=10,num_threads=1,is_flat_search=False,
            distance='squared_l2',vector_filters_file=None,num_nodes_to_cache=0,search_io_limit=None,post_processor=None)
        cfg=dict(search_directories=[str(a.work)],jobs=[dict(type='disk-index',content=dict(source=source,search_phase=phase))])
        inp=a.out/(tag+'-input.json');out=a.out/(tag+'-output.json');save(inp,cfg)
        with (a.out/(tag+'.log')).open('w') as log:
            subprocess.run([str(a.binary),'run','--input-file',str(inp),'--output-file',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=result_rows(json.loads(out.read_text()))
        if not rows:raise ValueError('no upstream search measurements in '+str(out))
        for r in rows:r.update(beam=beam,split=split,source_output=out.name)
        print(tag,rows,flush=True);return rows
    build=dict(**{'disk-index-source':'Build'},data_type='float32',data=str(a.work/'base.fbin'),
        distance='squared_l2',dim=128,max_degree=64,l_build=100,num_threads=4,
        build_ram_limit_gb=6.0,num_pq_chunks=64,quantization_type='FP',save_path=prefix)
    load=dict(**{'disk-index-source':'Load'},data_type='float32',load_path=prefix)
    grid=[20,40,60,80,120,200,400]
    disk_dev=disk_run('disk-build-dev',build,'development',grid,4)
    disk_dev+=disk_run('disk-dev-beam8',load,'development',grid,8)
    # Upstream calculate_recall reports percentages; never silently assume scale.
    save(a.out/'disk-development.json',disk_dev)
    eligible=[r for r in disk_dev if float(r['recall'])>=99.0]
    if not eligible:raise ValueError('no DiskANN configuration reaches 99% development recall')
    best_disk=min(eligible,key=lambda r:float(r['mean_latency']))
    print('Build unchanged GeoIVF geometry',flush=True)
    centers,labels=train_faiss(x,1024,12345,100000)
    basis=Basis.fit(learn,'pca');del learn
    projected=np.lib.format.open_memmap(a.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
    for s in range(0,len(x),32768):projected[s:s+32768]=basis.project(x[s:s+32768])
    projected.flush();layout=a.work/'geopack'
    meta=freeze_layout(x,centers,labels,layout,layout='geopack',pack_dims=16)
    side=a.work/'summary';summary=summarize_dependencies(layout,side,basis,dims=64,scheme='independent',bits=8,projected_by_id=projected)
    certify_cells(layout,side,projected,side/'cells.json');del projected
    index=CellIndex(layout,side,shape='ball')
    # Single-thread query latency, same one-core affinity for both systems after build.
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    dev_rows=[]
    reader=RollingPooled(layout/'vectors.pages',direct=True,uring=True,depth=16)
    try:
        for npb in [8,16,32,64,96,128]:
            for qid in dev:
                q=allq[qid];t=time.perf_counter_ns()
                ids,stats=search(index,q,reader,nprobe=npb,filtering='combined',selection='prepared',scan='simd',window_pages=256,gap_pages=2)
                ms=(time.perf_counter_ns()-t)/1e6
                recall=len(set(map(int,ids))&set(map(int,gt[qid,:10])))/10
                dev_rows.append(dict(query_id=int(qid),nprobe=npb,wall_ms=ms,recall=recall,**stats))
        csv_write(a.out/'geo-development.csv',dev_rows)
        dev_stats=[dict(nprobe=n,mean_ms=float(np.mean([r['wall_ms'] for r in dev_rows if r['nprobe']==n])),recall=float(np.mean([r['recall'] for r in dev_rows if r['nprobe']==n]))) for n in [8,16,32,64,96,128]]
        eligible=[r for r in dev_stats if r['recall']>=.99]
        if not eligible:raise ValueError('no GeoIVF configuration reaches target on development')
        best_geo=min(eligible,key=lambda r:r['mean_ms'])
        save(a.out/'frozen-selection.json',dict(target_recall=.99,diskann=best_disk,geoivf=best_geo,
             chosen_before_heldout=True,geo_schedule=dict(window=256,gap=2,queue_depth=16,bits=8),disk_node_cache=0,disk_pq_bytes_per_vector=64))
        # Once frozen, only these configurations are measured on held-out queries.
        quantizer=faiss.IndexFlatL2(128);quantizer.add(centers)
        routes=quantizer.search(allq[held],best_geo['nprobe'])[1]
        oracle=[reference(x,labels,allq[qid],routes[i]) for i,qid in enumerate(held)]
        rows=[];disk_test=[]
        rng=np.random.default_rng(71541)
        for rep in range(a.repeats):
            # Alternate arm order between rounds; full external programs are separate blocks.
            def geo_round():
                for i in rng.permutation(len(held)):
                    qid=int(held[i]);t=time.perf_counter_ns()
                    ids,stats=search(index,allq[qid],reader,nprobe=best_geo['nprobe'],filtering='combined',selection='prepared',scan='simd',window_pages=256,gap_pages=2)
                    ms=(time.perf_counter_ns()-t)/1e6
                    np.testing.assert_array_equal(ids,oracle[i])
                    rec=len(set(map(int,ids))&set(map(int,gt[qid,:10])))/10
                    rows.append(dict(round=rep,query_id=qid,wall_ms=ms,recall=rec,**stats))
                csv_write(a.out/'geo-heldout.csv',rows)
            def disk_round():disk_test.extend(disk_run('disk-heldout-'+str(rep),load,'heldout',[int(best_disk['search_l'])],int(best_disk['beam'])))
            if rep%2:geo_round();disk_round()
            else:disk_round();geo_round()
        save(a.out/'external-disk-results.json',dict(geo_development=dev_stats,disk_development=disk_dev,
            disk_heldout=disk_test,geo_heldout=dict(queries=len(held),repeats=a.repeats,mean_ms=float(np.mean([r['wall_ms'] for r in rows])),median_ms=float(np.median([r['wall_ms'] for r in rows])),p95_ms=float(np.quantile([r['wall_ms'] for r in rows],.95)),recall=float(np.mean([r['recall'] for r in rows])),means={k:float(np.mean([r[k] for r in rows])) for k in ['read_pages','read_bytes','read_requests','filter_us','io_us','distance_us']}),
            source_layout=meta,geo_summary=summary,disk_files={p.name:p.stat().st_size for p in a.work.glob('diskann-index*') if p.is_file()},
            cpu_affinity=[allowed[0]],heldout_configuration_frozen=True,elapsed_seconds=time.monotonic()-started,
            limitations=['one dataset/seed and shared machine','no equal hard RSS cap or independent physical-I/O instrumentation','upstream aggregate recall uses its tie-aware metric; Geo uses strict supplied top10 IDs','DiskANN query timer and Geo full application timer not identical scope','no upstream search code modified','fixed RAM code budget does not include every runtime allocation','DiskANN dev calibration precedes process pinning; heldout both pinned','sequential blocks per round, not per-query interleaving'] ))
    finally:reader.close();os.sched_setaffinity(0,allowed)

if __name__=='__main__':main()
