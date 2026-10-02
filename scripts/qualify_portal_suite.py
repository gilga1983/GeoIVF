#!/usr/bin/env python3
"""Fixed-policy multi-dataset test of tiny centroid portals as DiskANN starts.

No per-dataset tuning: every dataset runs medoid, one-portal/cell np1, np8 and
np32 at DiskANN L=60, beam=8. Angular datasets are row-normalized and searched
with squared L2, which preserves cosine-neighbor ordering for nonzero vectors.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import faiss, h5py, numpy as np
from geoivf.index import train_faiss
from scripts.qualify_speed import save

PINNED_DISKANN="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
FIXED_NPROBES=(1,8,32)
FIXED_L=60
FIXED_BEAM=8
NLIST=1024
K=10
DATASET_SPECS={
    "glove-200-angular":dict(dim=200,distance=("angular",),normalize=True),
    "nytimes-256-angular":dict(dim=256,distance=("angular",),normalize=True),
    "fashion-mnist-784-euclidean":dict(dim=784,distance=("euclidean",),normalize=False),
    "gist-960-euclidean":dict(dim=960,distance=("euclidean",),normalize=False),
    "yahoo-minilm-384-normalized":dict(dim=384,distance=("any","cosine","angular"),normalize=True),
    "coco-nomic-768-normalized":dict(dim=768,distance=("any","cosine","angular"),normalize=True),
    "imagenet-clip-512-normalized":dict(dim=512,distance=("any","cosine","angular"),normalize=True),
}

def fbin(path,array):
    a=np.asarray(array,dtype=np.float32,order="C")
    with Path(path).open("wb") as f:
        np.asarray(a.shape,dtype="<u4").tofile(f)
        for s in range(0,len(a),32768):
            np.asarray(a[s:s+32768],dtype="<f4",order="C").tofile(f)

def gtbin(path,ids,x,q):
    ids=np.asarray(ids,dtype="<u4",order="C")
    with Path(path).open("wb") as f:
        np.asarray(ids.shape,dtype="<u4").tofile(f)
        ids.tofile(f)
        for s in range(0,len(ids),512):
            block=ids[s:s+512]
            qq=q[s:s+512].astype(np.float64)
            d=np.sum((x[block].astype(np.float64)-qq[:,None,:])**2,axis=2).astype("<f4")
            d.tofile(f)

def normalize_rows(a):
    a=np.asarray(a,dtype=np.float32,order="C")
    norms=np.linalg.norm(a.astype(np.float64),axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms<=0): raise ValueError("angular dataset contains zero/nonfinite vector")
    return np.asarray(a/norms[:,None],dtype=np.float32,order="C")

def load_dataset(path,spec,seed):
    with h5py.File(path,"r") as f:
        x=np.asarray(f["train"][:],dtype=np.float32,order="C")
        tests=f["test"]
        neigh=f["neighbors"]
        distance=f.attrs.get("distance","")
        if isinstance(distance,bytes):distance=distance.decode()
        if x.ndim!=2 or x.shape[1]!=spec["dim"] or distance not in spec["distance"]: raise ValueError("dataset schema mismatch")
        rng=np.random.default_rng(seed)
        perm=rng.permutation(tests.shape[0])
        if len(perm)<384: raise ValueError("need at least 384 benchmark queries")
        dev_ids=perm[:128]; held_ids=perm[128:384]
        dev=np.asarray(tests[dev_ids],dtype=np.float32,order="C")
        held=np.asarray(tests[held_ids],dtype=np.float32,order="C")
        dev_gt=np.asarray(neigh[dev_ids,:100],dtype=np.int64)
        held_gt=np.asarray(neigh[held_ids,:100],dtype=np.int64)
    if spec["normalize"]:
        x=normalize_rows(x);dev=normalize_rows(dev);held=normalize_rows(held)
    for gt in (dev_gt,held_gt):
        if gt.min()<0 or gt.max()>=len(x):raise ValueError("ground truth out of range")
    return x,dev,held,dev_gt,held_gt,dev_ids,held_ids

def build_portal_table(x,centers,labels):
    order=np.argsort(labels,kind="stable");counts=np.bincount(labels,minlength=len(centers));cuts=np.r_[0,np.cumsum(counts)]
    if np.any(counts==0):raise ValueError("empty IVF cell")
    ids=np.empty(len(centers),dtype=np.uint32);vecs=np.empty_like(centers,dtype=np.float32)
    for li in range(len(centers)):
        members=order[cuts[li]:cuts[li+1]]
        pts=np.asarray(x[members],dtype=np.float32)
        d=pts.astype(np.float64)-centers[li].astype(np.float64)
        pick=int(np.argmin(np.einsum("ij,ij->i",d,d)))
        ids[li]=np.uint32(members[pick]);vecs[li]=pts[pick]
    return ids,vecs

def portal_seeds(router,portal_ids,portal_vecs,queries,nprobe):
    rows=np.empty((len(queries),1),dtype=np.uint32);times=[]
    router.search(np.ascontiguousarray(queries[:1]),nprobe)
    for i,q in enumerate(queries):
        t=time.perf_counter_ns()
        _,cells=router.search(np.ascontiguousarray(q[None]),nprobe)
        cs=cells[0];pv=portal_vecs[cs]
        delta=pv.astype(np.float64)-q.astype(np.float64)
        rows[i,0]=portal_ids[cs[int(np.argmin(np.einsum("ij,ij->i",delta,delta)))]]
        times.append((time.perf_counter_ns()-t)/1e6)
    return rows,dict(mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)))

def result_rows(obj):
    found=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj and "recall" in obj:found.append(obj)
        else:
            for v in obj.values():found.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj:found.extend(result_rows(v))
    return found

def disk_run(binary,work,out,tag,split,source,seed_file=None):
    phase=dict(queries=str(work/f"{split}.fbin"),groundtruth=str(work/f"{split}.gt"),
        search_list=[FIXED_L],beam_width=FIXED_BEAM,recall_at=K,num_threads=1,is_flat_search=False,
        distance="squared_l2",vector_filters_file=None,num_nodes_to_cache=None,search_io_limit=None,post_processor=None)
    cfg=dict(search_directories=[str(work)],jobs=[dict(type="disk-index",content=dict(source=source,search_phase=phase))])
    inp=out/f"{tag}-input.json";output=out/f"{tag}-output.json";save(inp,cfg)
    env=os.environ.copy()
    if seed_file is None:env.pop("DISKANN_START_POINTS_FILE",None)
    else:env["DISKANN_START_POINTS_FILE"]=str(Path(seed_file).resolve())
    with (out/f"{tag}.log").open("w") as log:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(output)],
            stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    rows=result_rows(json.loads(output.read_text()))
    if len(rows)!=1:raise ValueError(f"expected one DiskANN row, got {len(rows)}")
    return dict(rows[0],source_output=output.name)

def summarize(rows):
    result={}
    names=["medoid"]+[f"portal-np{x}" for x in FIXED_NPROBES]
    for name in names:
        rr=[r for r in rows if r["method"]==name]
        result[name]=dict(
            rounds=len(rr),recall_percent=float(np.mean([r["recall"] for r in rr])),
            mean_diskann_us=float(np.mean([r["mean_latency"] for r in rr])),
            mean_ios=float(np.mean([r["mean_ios"] for r in rr])),
            mean_io_us=float(np.mean([r["mean_io_time"] for r in rr])),
            mean_cpu_us=float(np.mean([r["mean_cpu_time"] for r in rr])),
            mean_comparisons=float(np.mean([r["mean_comparisons"] for r in rr])),
            mean_hops=float(np.mean([r["mean_hops"] for r in rr])),
            route_mean_ms=float(np.mean([r["route_mean_ms"] for r in rr])),
            composed_mean_ms=float(np.mean([r["composed_mean_ms"] for r in rr])),
        )
    base=result["medoid"]
    for name,s in result.items():
        if name=="medoid":continue
        s["io_reduction_fraction_vs_medoid"]=1-s["mean_ios"]/base["mean_ios"]
        s["comparison_reduction_fraction_vs_medoid"]=1-s["mean_comparisons"]/base["mean_comparisons"]
        s["recall_delta_points_vs_medoid"]=s["recall_percent"]-base["recall_percent"]
    return result

def run(a):
    started=time.monotonic();faiss.omp_set_num_threads(1)
    spec=DATASET_SPECS[a.dataset]
    seed=int.from_bytes(hashlib.sha256(a.dataset.encode()).digest()[:8],"little")
    x,dev,held,dev_gt,held_gt,dev_ids,held_ids=load_dataset(a.data,spec,seed)
    save(a.out/"cohorts.json",dict(dataset=a.dataset,seed=seed,development_ids=dev_ids.tolist(),heldout_ids=held_ids.tolist(),
        policy_frozen=True,nprobe_arms=list(FIXED_NPROBES),l=FIXED_L,beam=FIXED_BEAM))

    print(f"{a.dataset}: {len(x)} x {x.shape[1]}; build fixed 1024-cell router",flush=True)
    t=time.perf_counter();centers,labels=train_faiss(x,NLIST,12345,min(100000,len(x)));ivf_build_s=time.perf_counter()-t
    t=time.perf_counter();portal_ids,portal_vecs=build_portal_table(x,centers,labels);portal_build_s=time.perf_counter()-t
    router=faiss.IndexFlatL2(x.shape[1]);router.add(np.ascontiguousarray(centers))
    np.savez(a.out/"portal-table.npz",centers=centers,portal_ids=portal_ids,portal_vectors=portal_vecs)

    fbin(a.work/"base.fbin",x)
    for name,q,gt in (("development",dev,dev_gt),("heldout",held,held_gt)):
        fbin(a.work/f"{name}.fbin",q);gtbin(a.work/f"{name}.gt",gt,x,q)

    seeds={};route_stats={}
    for split,q in (("development",dev),("heldout",held)):
        seeds[split]={};route_stats[split]={}
        for npb in FIXED_NPROBES:
            rows,st=portal_seeds(router,portal_ids,portal_vecs,q,npb)
            path=a.out/f"{split}-portal-np{npb}.ubin";fbin(path,rows)
            seeds[split][npb]=path;route_stats[split][npb]=st

    prefix=str(a.work/"diskann-index")
    pq_chunks=min(64,x.shape[1])
    build=dict(**{"disk-index-source":"Build"},data_type="float32",data=str(a.work/"base.fbin"),
        distance="squared_l2",dim=int(x.shape[1]),max_degree=64,l_build=100,num_threads=4,
        build_ram_limit_gb=8.0,num_pq_chunks=pq_chunks,quantization_type="FP",save_path=prefix)
    load=dict(**{"disk-index-source":"Load"},data_type="float32",load_path=prefix)

    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[0]})
    rows=[]
    try:
        for rep in range(a.repeats):
            arms=["medoid"]+[f"portal-np{x}" for x in FIXED_NPROBES]
            if rep%2:arms=list(reversed(arms))
            for arm in arms:
                split="heldout"
                if arm=="medoid":
                    r=disk_run(a.binary,a.work,a.out,f"{split}-{arm}-r{rep}",split,build if rep==0 else load)
                    route_ms=0.0
                else:
                    npb=int(arm.split("np")[1]);r=disk_run(a.binary,a.work,a.out,f"{split}-{arm}-r{rep}",split,load,seeds[split][npb])
                    route_ms=route_stats[split][npb]["mean_ms"]
                rows.append(dict(round=rep,method=arm,route_mean_ms=route_ms,
                    composed_mean_ms=route_ms+float(r["mean_latency"])/1000.0,**r))
        save(a.out/"heldout-rows.json",rows)
        summary=summarize(rows)
        extra_bytes=int(centers.nbytes+portal_vecs.nbytes+portal_ids.nbytes)
        save(a.out/"portal-suite-result.json",dict(
            dataset=a.dataset,dimension=int(x.shape[1]),train_rows=len(x),accepted_source_distances=list(spec["distance"]),
            normalized_for_l2=bool(spec["normalize"]),diskann_revision=PINNED_DISKANN,
            fixed_policy=dict(nlist=NLIST,nprobes=list(FIXED_NPROBES),one_portal_per_cell=True,l=FIXED_L,beam=FIXED_BEAM,
                max_degree=64,l_build=100,pq_chunks=pq_chunks,num_nodes_to_cache=None),
            portal_extra_bytes=extra_bytes,portal_extra_mib=extra_bytes/(1<<20),
            ivf_build_seconds=ivf_build_s,portal_build_seconds=portal_build_s,
            route_stats=route_stats,heldout_summary=summary,cpu_affinity=[allowed[0]],
            disk_files={p.name:p.stat().st_size for p in a.work.glob("diskann-index*") if p.is_file()},
            elapsed_seconds=time.monotonic()-started,
            limitations=[
                "No per-dataset policy tuning; fixed nprobe arms 1/8/32 and L60/beam8.",
                "Portal0 is geometric nearest-to-centroid, not graph-trained.",
                "Angular data are unit-normalized then searched by squared L2; this preserves cosine ordering for nonzero vectors.",
                "DiskANN mean_ios is provider vertex-load work, not independently measured NVMe commands.",
                "Routing timing is Python/Faiss and added diagnostically to DiskANN internal timing, not integrated in-process latency.",
                "One index build per dataset on a shared host with node cache disabled.",
            ]))
        print(json.dumps(summary,indent=2),flush=True)
    finally:os.sched_setaffinity(0,allowed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset",choices=sorted(DATASET_SPECS));ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--binary",type=Path,required=True);ap.add_argument("--work",type=Path,required=True);ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--repeats",type=int,default=3);a=ap.parse_args()
    if a.repeats<3:ap.error("repeats must be >=3")
    for p in (a.work,a.out):
        if p.exists() and any(p.iterdir()):raise FileExistsError(p)
        p.mkdir(parents=True,exist_ok=True)
    a.data,a.binary,a.work,a.out=map(Path.resolve,(a.data,a.binary,a.work,a.out))
    lockpath=Path.home()/".cache/geoivf/speed-device.lock";lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run(a)
if __name__=="__main__":main()
