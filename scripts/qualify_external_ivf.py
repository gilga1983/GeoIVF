#!/usr/bin/env python3
"""Released in-memory IVF algorithms versus GeoIVF RAM replay on disjoint queries.

This intentionally does NOT label in-memory CLIP as an SSD engine. The common
coarse centroids isolate execution/routing improvements without query leakage.
"""
from __future__ import annotations
import argparse, ctypes as ct, fcntl, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import faiss
from geoivf.index import vectors,train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis,digest_file
from geoivf.dependent import summarize_dependencies
from geoivf.cells import CellIndex,certify_cells
from geoivf.io import MemoryReplay
from geoivf.search import search
from scripts.qualify_sift1m import reference
from scripts.qualify_speed import save,csv_write

class Upstream:
    def __init__(self,path,kind,x,centers):
        self.lib=ct.CDLL(str(path));self.error=ct.create_string_buffer(2048)
        self.lib.gu_build.argtypes=[ct.c_int,ct.c_void_p,ct.c_longlong,ct.c_uint,ct.c_void_p,ct.c_uint,ct.c_void_p,ct.c_size_t]
        self.lib.gu_build.restype=ct.c_void_p
        self.lib.gu_search.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint,ct.c_uint,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_size_t]
        self.lib.gu_search.restype=ct.c_int
        self.lib.gu_save.argtypes=[ct.c_void_p,ct.c_char_p,ct.c_void_p,ct.c_size_t];self.lib.gu_save.restype=ct.c_int
        self.lib.gu_free.argtypes=[ct.c_void_p];self.lib.gu_free.restype=None
        self.handle=self.lib.gu_build(kind,x.ctypes.data,len(x),x.shape[1],centers.ctypes.data,len(centers),self.error,2048)
        if not self.handle:raise RuntimeError(self.error.value.decode())
        self.ids=np.empty(10,np.int64);self.distances=np.empty(10,np.float32)
    def search(self,q,nprobe):
        rc=self.lib.gu_search(self.handle,q.ctypes.data,nprobe,10,self.ids.ctypes.data,self.distances.ctypes.data,self.error,2048)
        if rc:raise RuntimeError(self.error.value.decode())
        return self.ids
    def save(self,path):
        if self.lib.gu_save(self.handle,str(path).encode(),self.error,2048):raise RuntimeError(self.error.value.decode())
    def close(self):
        if self.handle:self.lib.gu_free(self.handle);self.handle=None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--library',type=Path,default=Path('build/external-clip/libgeoivf_upstream.so'))
    ap.add_argument('--queries',type=int,default=256);ap.add_argument('--development',type=int,default=128);ap.add_argument('--repeats',type=int,default=5)
    a=ap.parse_args()
    if not 1<=a.development<=1000 or not 1<=a.queries<=9000 or a.repeats<1:ap.error('invalid scope')
    for p in (a.work,a.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    lock=Path.home()/'.cache/geoivf/speed-device.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('w') as f:fcntl.flock(f,fcntl.LOCK_EX);run(a)


def run(a):
    faiss.omp_set_num_threads(1);start=time.monotonic()
    if digest_file(a.data/'sift_base.fvecs')!='21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816':raise ValueError('wrong dataset')
    x=vectors(a.data/'sift_base.fvecs');allq=vectors(a.data/'sift_query.fvecs');learn=vectors(a.data/'sift_learn.fvecs')
    gt=np.memmap(a.data/'sift_groundtruth.ivecs',dtype='<i4',mode='r',shape=(10000,101))[:,1:]
    perm=np.random.default_rng(20260929).permutation(10000);dev=perm[:a.development];held=perm[1000:1000+a.queries]
    assert not set(dev)&set(held)
    save(a.out/'splits.json',dict(development=dev.tolist(),heldout=held.tolist(),split_seed=20260929))
    centers,labels=train_faiss(x,1024,12345,100000);np.save(a.out/'common-centroids.npy',centers)
    external={};builds=[]
    try:
        for name,kind in [('faiss-vendored',0),('ivf-clip',1),('hivf-clip',2)]:
            print('Build released '+name,flush=True);t=time.monotonic()
            ext=Upstream(a.library.resolve(),kind,x,centers);external[name]=ext
            ext.save(a.out/(name+'-trained.bin'))
            builds.append(dict(method=name,seconds=time.monotonic()-t,common_coarse_centroids=True))
        # Current packaged Faiss is an extra control, not the version linked by CLIP.
        qz=faiss.IndexFlatL2(128);qz.add(centers)
        current=faiss.IndexIVFFlat(qz,128,1024);current.is_trained=True;current.add(x)
        basis=Basis.fit(learn,'pca');del learn
        z=np.lib.format.open_memmap(a.work/'projection.npy',mode='w+',dtype=np.float64,shape=x.shape)
        for s in range(0,len(x),32768):z[s:s+32768]=basis.project(x[s:s+32768])
        z.flush();layout=a.work/'geopack';lm=freeze_layout(x,centers,labels,layout,layout='geopack',pack_dims=16)
        side=a.work/'summary';sm=summarize_dependencies(layout,side,basis,dims=64,scheme='independent',bits=8,projected_by_id=z)
        certify_cells(layout,side,z,side/'cells.json');del z
        geo=CellIndex(layout,side,shape='ball');reader=MemoryReplay(layout/'vectors.pages')
        names=list(external)+['faiss-current','geoivf-replay']
        def one(name,q,npb):
            if name in external:return external[name].search(q,npb)
            if name=='faiss-current':current.nprobe=npb;return current.search(q[None,:],10)[1][0]
            return search(geo,q,reader,nprobe=npb,selection='prepared',scan='simd',window_pages=256,gap_pages=2)[0]
        allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
        rows=[];rng=np.random.default_rng(654)
        grid=[8,16,32,64,96,128,192,256]
        for npb in grid:
            for qid in rng.permutation(dev):
                for name in rng.permutation(names):
                    t=time.perf_counter_ns();ids=one(name,allq[qid],npb);ms=(time.perf_counter_ns()-t)/1e6
                    recall=len(set(map(int,ids))&set(map(int,gt[qid,:10])))/10
                    rows.append(dict(method=name,query_id=int(qid),nprobe=npb,wall_ms=ms,recall=recall))
            csv_write(a.out/'development.csv',rows)
        dev_summary=[];selected={}
        for name in names:
            for npb in grid:
                rr=[r for r in rows if r['method']==name and r['nprobe']==npb]
                dev_summary.append(dict(method=name,nprobe=npb,mean_ms=float(np.mean([r['wall_ms'] for r in rr])),recall=float(np.mean([r['recall'] for r in rr]))))
            eligible=[r for r in dev_summary if r['method']==name and r['recall']>=.99]
            if not eligible:raise ValueError('target not reached on development: '+name)
            selected[name]=min(eligible,key=lambda r:r['mean_ms'])['nprobe']
        save(a.out/'frozen-selection.json',dict(target=.99,selected_nprobe=selected,development=dev_summary,selected_before_heldout=True))
        # Fixed-nprobe comparison additionally tests transparent pruning on identical IVF candidates.
        routes=qz.search(allq[held],64)[1];oracle=[reference(x,labels,allq[qid],routes[i]) for i,qid in enumerate(held)]
        fixed=[];timed=[]
        for i,qid in enumerate(held):
            for name in names:
                ids=one(name,allq[qid],64).copy()
                fixed.append(dict(method=name,query_id=int(qid),same_candidate_answer=bool(np.array_equal(ids,oracle[i])),same_id_set=bool(set(ids)==set(oracle[i])),recall=len(set(ids)&set(gt[qid,:10]))/10))
        csv_write(a.out/'same-nprobe-checks.csv',fixed)
        for rep in range(a.repeats):
            for qid in rng.permutation(held):
                for name in rng.permutation(names):
                    npb=selected[name];t=time.perf_counter_ns();ids=one(name,allq[qid],npb);ms=(time.perf_counter_ns()-t)/1e6
                    timed.append(dict(method=name,round=rep,query_id=int(qid),nprobe=npb,wall_ms=ms,recall=len(set(map(int,ids))&set(map(int,gt[qid,:10])))/10))
            csv_write(a.out/'heldout.csv',timed)
            print('Held-out round',rep+1,flush=True)
        summary=[]
        for name in names:
            rr=[r for r in timed if r['method']==name];wall=np.array([r['wall_ms'] for r in rr])
            summary.append(dict(method=name,nprobe=selected[name],mean_ms=float(wall.mean()),median_ms=float(np.median(wall)),p95_ms=float(np.quantile(wall,.95)),recall=float(np.mean([r['recall'] for r in rr])),distinct_queries=len(held),rounds=a.repeats))
        save(a.out/'external-ivf-results.json',dict(results=summary,development=dev_summary,builds=builds,
            geo_layout=lm,geo_summary=sm,faiss_current=faiss.__version__,elapsed_seconds=time.monotonic()-start,
            cpu_affinity=[allowed[0]],common_centroids_sha256=hashlib.sha256(centers.tobytes()).hexdigest(),
            same_nprobe_summary={name:dict(exact_order_matches=sum(r['same_candidate_answer'] for r in fixed if r['method']==name),same_id_sets=sum(r['same_id_set'] for r in fixed if r['method']==name),queries=len(held)) for name in names},
            limitations=['in-memory upstream algorithms versus Geo RAM replay, NOT SSD speed comparison','full-vector resident baselines use more vector RAM','all indexes/data/oracle resident together; RSS is not per-system deployment RAM','shared machine, single pinned thread, no isolated frequency control','common external IVF centroids rather than each authors default build recipe','HIVF builds hierarchy with released API on shared leaf centroids','C++ released kernels and Python coordinator have different call overhead','one nlist, one dataset/seed, not full SOTA search','strict top10 ID recall may differ at ties','parameters frozen after development, no retuning on heldout']))
        reader.close();os.sched_setaffinity(0,allowed)
        for r in summary:print(json.dumps(r),flush=True)
    finally:
        for ext in external.values():ext.close()

if __name__=='__main__':main()
