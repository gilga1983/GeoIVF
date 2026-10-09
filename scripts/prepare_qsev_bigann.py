#!/usr/bin/env python3
"""Generate strong independently reproduced QSEV entry pools for BigANN-L2.

Train full-data L2 k-means and map its centroids through a high-probe
FAISS IVF-Flat approximate nearest-neighbor index over all 10M base
vectors. A medoid is always included. This is an independently implemented
entry-selector component, NOT full original DiskANN++ page layout.
"""
from __future__ import annotations
import argparse
import json
import struct
import time
from pathlib import Path
import faiss
import numpy as np
from prepare_diskannpp_qsev import read_medoid,locate_disk_index,write_pool

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--base",type=Path,required=True)
    a.add_argument("--index-prefix",type=Path,required=True)
    a.add_argument("--out-dir",type=Path,required=True)
    a.add_argument("--counts",default="200,800,3200")
    a.add_argument("--train-size",type=int,default=100000)
    a.add_argument("--seed",type=int,default=12345)
    a.add_argument("--threads",type=int,default=4)
    args=a.parse_args()
    faiss.omp_set_num_threads(args.threads)
    with args.base.open("rb") as fi: rows,dim=struct.unpack("<II",fi.read(8))
    if (rows,dim)!=(10_000_000,128):raise ValueError(f"Wrong BigANN corpus {(rows,dim)}")
    if args.base.stat().st_size!=8+rows*dim:raise ValueError("truncated BigANN u8bin")
    base=np.memmap(args.base,dtype=np.uint8,mode="r",offset=8,shape=(rows,dim))
    rng=np.random.default_rng(args.seed)
    sampled=rng.choice(rows,args.train_size,replace=False)
    train=np.array(base[sampled],dtype=np.float32,order="C")
    medoid=read_medoid(locate_disk_index(args.index_prefix),rows,dim)
    klist=tuple(int(x) for x in args.counts.split(","))
    if min(klist)<2 or max(klist)>4096:raise ValueError("unsupported QSEV pool capacities")

    # Offline L2 IVF maps k-means centroid queries to graph vertices.
    # Relatively generous nprobe=64 avoids a deliberately weak baseline.
    nlist=512
    index=faiss.IndexIVFFlat(faiss.IndexFlatL2(dim),dim,nlist,faiss.METRIC_L2)
    start=time.monotonic()
    print("Training offline IVF-Flat for QSEV centroid-to-data lookup",flush=True)
    index.train(train)
    index.nprobe=64
    for begin in range(0,rows,32768):
        end=min(rows,begin+32768)
        index.add(np.asarray(base[begin:end],dtype=np.float32,order="C"))
        if begin%524288==0:
            print(f"IVF mapped {end:,}/{rows:,}",flush=True)
    if index.ntotal!=rows:raise AssertionError("IVF lost base points")
    indexed_seconds=time.monotonic()-start
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for n in klist:
        print(f"Training squared-L2 QSEV centroid pool for {n} entries",flush=True)
        km=faiss.Kmeans(dim,n-1,niter=8,nredo=1,seed=args.seed,
                        verbose=False,min_points_per_centroid=10)
        t0=time.monotonic();km.train(train)
        centers=np.asarray(km.centroids,dtype=np.float32,order="C")
        center_train_s=time.monotonic()-t0
        dists,nearest=index.search(centers,1)
        if nearest.shape!=(n-1,1) or np.any(nearest<0):
            raise RuntimeError("IVF mapping failed for QSEV centroid")
        ids=np.concatenate([np.array([medoid],dtype=np.uint32),
                            nearest[:,0].astype(np.uint32)],axis=0)
        coords=np.asarray(base[ids.astype(np.int64)],dtype=np.float32,order="C")
        f=args.out_dir/f"qsev-{n}.bin"
        write_pool(f,ids,coords)
        expected=16+n*(4+dim*4)
        assert f.stat().st_size==expected
        result={"metric":"squared_l2","dataset":"BigANN-10M",
                "policy":"QSEV full-precision nearest entry from a centroid pool",
                "scope":"independent DiskANN++ QSEV entry component only",
                "centroid_mapping":"FAISS IVF-Flat, 512 inverted lists, nprobe=64, all 10M vertices indexed",
                "count":n,"clusters":n-1,"bytes":expected,
                "unique_ids":int(len(np.unique(ids))),"base_points":rows,
                "train_queries":int(len(train)),"medoid":medoid,
                "clustering_seconds":center_train_s,"ivf_index_build_seconds":indexed_seconds,
                "seed":args.seed,"IVF_mapping_mean_squared_distance":float(dists[:,0].mean())}
        (args.out_dir/f"qsev-{n}.manifest.json").write_text(json.dumps(result,indent=2)+"\n")
        manifests.append(result)
        print(f"BIGANN_QSEV_READY entries={n} bytes={expected}",flush=True)
    (args.out_dir/"qsev-bigann.manifest.json").write_text(json.dumps(manifests,indent=2)+"\n")
if __name__=="__main__": main()
