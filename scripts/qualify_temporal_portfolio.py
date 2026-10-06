#!/usr/bin/env python3
"""Evaluate partitioned and additive temporal NavHints portfolios."""
from __future__ import annotations
import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS,BEAM,K=4,8,10
LS=(10,20,40,80,160)
HEARTBEAT=30
TIMEOUT=900

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2)+"\n")

def fbin_shape(path):
    with Path(path).open("rb") as f: raw=f.read(8)
    rows,dim=struct.unpack("<II",raw)
    if Path(path).stat().st_size!=8+rows*dim*4: raise ValueError("bad fbin")
    return rows,dim

def suffix_fbin(src,dst,start):
    rows,dim=fbin_shape(src)
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8+start*dim*4); fout.write(struct.pack("<II",rows-start,dim))
        remaining=(rows-start)*dim*4
        while remaining:
            b=fin.read(min(16<<20,remaining))
            if not b: raise ValueError("truncated fbin")
            fout.write(b); remaining-=len(b)

def result_rows(obj):
    out=[]
    if isinstance(obj,dict):
        if "search_l" in obj and "mean_latency" in obj: out.append(obj)
        else:
            for v in obj.values(): out.extend(result_rows(v))
    elif isinstance(obj,list):
        for v in obj: out.extend(result_rows(v))
    return out

def payload(path):
    path=Path(path)
    manifest=json.loads(path.with_suffix(path.suffix+".manifest.json").read_text())
    return path.stat().st_size+int(manifest["nlist"])*(64+4)

def run(binary,out,tag,queries,gt,prefix,primary,primary_probe,primary_limit,banks):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
            "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
            "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
            "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"
    save(inp,cfg)
    env=os.environ.copy()
    for name in (
        "DISKANN_SKIP_RECALL","DISKANN_STATIC_CACHE_IDS_FILE","DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE","DISKANN_HINT_IVF_MAX_STARTS","DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_PROGRESSIVE_HINT_TOPK","DISKANN_GLOBAL_START_IDS_FILE","DISKANN_START_POINTS_FILE",
        "DISKANN_QSEV_FILE","DISKANN_IP_PORTAL_ROUTER_FILE","DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT","DISKANN_WAYPOINT_CACHE_FILE","DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_TEMPORAL_HINT_IVF1_FILE","DISKANN_TEMPORAL_HINT_IVF1_NPROBE",
        "DISKANN_TEMPORAL_HINT_IVF1_LIMIT","DISKANN_TEMPORAL_HINT_IVF1_HOP",
        "DISKANN_TEMPORAL_HINT_IVF2_FILE","DISKANN_TEMPORAL_HINT_IVF2_NPROBE",
        "DISKANN_TEMPORAL_HINT_IVF2_LIMIT","DISKANN_TEMPORAL_HINT_IVF2_HOP",
        "DISKANN_TEMPORAL_HINT_IVF3_FILE","DISKANN_TEMPORAL_HINT_IVF3_NPROBE",
        "DISKANN_TEMPORAL_HINT_IVF3_LIMIT","DISKANN_TEMPORAL_HINT_IVF3_HOP",
    ): env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(primary)
    env["DISKANN_HINT_IVF_NPROBE"]=str(primary_probe)
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    env["DISKANN_PROGRESSIVE_HINTS"]="1"
    env["DISKANN_PROGRESSIVE_HINT_TOPK"]=str(primary_limit)
    for i,(path,nprobe,limit,hop) in enumerate(banks,1):
        env[f"DISKANN_TEMPORAL_HINT_IVF{i}_FILE"]=str(path)
        env[f"DISKANN_TEMPORAL_HINT_IVF{i}_NPROBE"]=str(nprobe)
        env[f"DISKANN_TEMPORAL_HINT_IVF{i}_LIMIT"]=str(limit)
        env[f"DISKANN_TEMPORAL_HINT_IVF{i}_HOP"]=str(hop)
    cmd=[str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    started=time.monotonic()
    with logp.open("w") as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        while True:
            try: rc=p.wait(timeout=HEARTBEAT); break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-started
                print(f"heartbeat {tag} {elapsed:.0f}s",flush=True)
                if elapsed>=TIMEOUT:
                    p.terminate()
                    try:p.wait(timeout=15)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                    raise TimeoutError(tag)
        if rc: raise subprocess.CalledProcessError(rc,cmd)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS): raise ValueError(f"{tag}: unexpected Ls")
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rr=[next(r for r in rows if int(r["search_l"])==l) for rows in reps]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rr])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rr])),
            "median_latency_us":float(np.median([float(r["mean_latency"]) for r in rr])),
            "median_cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rr])),
            "mean_comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rr])),
        }
    return out

def points(summary):
    out=[]; best=-1e99
    for l in LS:
        row=summary[str(l)]; rec=float(row["recall_percent"])
        if rec+1e-9>=best:
            out.append((rec,row)); best=max(best,rec)
    return out

def interp(summary,target,field):
    pts=points(summary)
    if target<pts[0][0] or target>pts[-1][0]: return None
    for (lr,lo),(hr,hi) in zip(pts,pts[1:]):
        if lr<=target<=hr:
            if hr<=lr+1e-12:return float(hi[field])
            a=(target-lr)/(hr-lr)
            return float(lo[field])+a*(float(hi[field])-float(lo[field]))
    return float(pts[-1][1][field]) if abs(target-pts[-1][0])<1e-9 else None

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt","index-prefix","canonical","general8",
              "partition-s8","partition-s16","partition-s24",
              "additive-s8","additive-s16","additive-s24","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    a=ap.parse_args()
    for n in ("binary","queries","gt","index_prefix","canonical","general8",
              "partition_s8","partition_s16","partition_s24",
              "additive_s8","additive_s16","additive_s24","work","out"):
        setattr(a,n,getattr(a,n).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if fbin_shape(a.queries)!=(10000,768): raise ValueError("unexpected workload")
    held=a.work/"heldout.fbin"; suffix_fbin(a.queries,held,5000)

    arms={
        "canonical":(a.canonical,8,16,[]),
        "partition-all":(a.general8,4,8,[
            (a.partition_s8,2,4,0),(a.partition_s16,1,2,0),(a.partition_s24,1,2,0)]),
        "partition-staged":(a.general8,4,8,[
            (a.partition_s8,2,4,8),(a.partition_s16,1,2,16),(a.partition_s24,1,2,24)]),
        "additive-k16":(a.canonical,8,8,[
            (a.additive_s8,2,4,8),(a.additive_s16,1,2,16),(a.additive_s24,1,2,24)]),
        "additive-k24":(a.canonical,8,16,[
            (a.additive_s8,2,4,8),(a.additive_s16,1,2,16),(a.additive_s24,1,2,24)]),
    }
    runs={m:[] for m in arms}
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            print(f"waiting for speed-device lock: {lockp}",flush=True)
            fcntl.flock(lock,fcntl.LOCK_EX)
            print("acquired speed-device lock",flush=True)
            methods=tuple(arms)
            for rep in range(a.reps):
                order=methods[rep%len(methods):]+methods[:rep%len(methods)]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for m in order:
                    rr=run(a.binary,a.out,f"r{rep}-{m}",held,a.gt,a.index_prefix,*arms[m])
                    runs[m].append(rr); save(a.out/"runs.partial.json",runs)
                    print(f"finished rep={rep} {m}: L10 recall={float(rr[0]['recall']):.3f} io={float(rr[0]['mean_ios']):.3f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in arms}
    memory={
        "canonical":payload(a.canonical),
        "partition-all":payload(a.general8)+payload(a.partition_s8)+payload(a.partition_s16)+payload(a.partition_s24),
        "partition-staged":payload(a.general8)+payload(a.partition_s8)+payload(a.partition_s16)+payload(a.partition_s24),
        "additive-k16":payload(a.canonical)+payload(a.additive_s8)+payload(a.additive_s16)+payload(a.additive_s24),
        "additive-k24":payload(a.canonical)+payload(a.additive_s8)+payload(a.additive_s16)+payload(a.additive_s24),
    }
    fixed={}
    lo=max(summary[m]["10"]["recall_percent"] for m in arms)
    hi=min(summary[m]["160"]["recall_percent"] for m in arms)
    for target in (35.0,45.0,55.0,65.0):
        if not(lo<=target<=hi):continue
        item={}
        for m in arms:
            item[m]={
                "mean_ios":interp(summary[m],target,"mean_ios"),
                "latency_us":interp(summary[m],target,"median_latency_us"),
                "runtime_payload_bytes":memory[m],
            }
        base=item["canonical"]
        for m in arms:
            if m=="canonical":continue
            item[m]["io_change_vs_canonical_percent"]=100*(item[m]["mean_ios"]/base["mean_ios"]-1)
            item[m]["latency_change_vs_canonical_percent"]=100*(item[m]["latency_us"]/base["latency_us"]-1)
        fixed[str(target)]=item

    result={
        "workload":"MedRAG-Zipf heldout 5000",
        "arms":{
            "canonical":"16K general, 512 cells/probe8, retained K=16",
            "partition-all":"8K general +4K/2K/2K disjoint stage specialists, 256/128/64/64 cells and 4/2/1/1 probes; quotas 8/4/2/2; all active",
            "partition-staged":"same equal-resource portfolio; specialists activate at hops 8/16/24",
            "additive-k16":"full 16K general plus 8K disjoint specialists; quotas 8/4/2/2, staged activation",
            "additive-k24":"same 24K persistent portfolio; quotas 16/4/2/2, staged activation",
        },
        "memory_payload_bytes":memory,
        "evaluation":{"Ls":list(LS),"beam":BEAM,"threads":THREADS,"repetitions":a.reps},
        "summary":summary,"fixed_recall":fixed,
    }
    save(a.out/"temporal-portfolio.json",result)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
