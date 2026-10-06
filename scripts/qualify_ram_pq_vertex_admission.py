#!/usr/bin/env python3
"""Admission test: does a 16 MiB RAM PQ graph add value after free vertex hints?"""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("navhint","vertex_s2","ram16","vertex_s2_ram16")

def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2)+"\n")

def fbin_shape(p):
    with p.open("rb") as f: r,d=struct.unpack("<II",f.read(8))
    return r,d

def suffix_fbin(src,dst,start):
    r,d=fbin_shape(src)
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*d*4); fo.write(struct.pack("<II",r-start,d))
        rem=(r-start)*d*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)

def suffix_gt(src,dst,start):
    raw=src.read_bytes(); r,k=struct.unpack("<II",raw[:8])
    with dst.open("wb") as f:
        f.write(struct.pack("<II",r-start,k)); f.write(raw[8+start*k*4:])

def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o: out.append(o)
        else:
            for v in o.values(): out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o: out.extend(result_rows(v))
    return out

def run_one(binary,out,tag,queries,gt,index_prefix,ivf,graph,method):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(index_prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"; output=out/f"{tag}.output.json"; logp=out/f"{tag}.log"; save(inp,cfg)
    env=os.environ.copy()
    for name in list(env):
        if name.startswith("DISKANN_"): env.pop(name,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf); env["DISKANN_HINT_IVF_NPROBE"]="8"; env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    if "vertex_s2" in method: env["DISKANN_VERTEX_HINT_VARIANT"]="1"
    if "ram16" in method:
        env["DISKANN_RAM_PQ_GRAPH"]=str(graph); env["DISKANN_RAM_PQ_BUDGET"]="16"; env["DISKANN_RAM_PQ_SEEDS"]="2"
    with logp.open("w") as log:
        subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(output)],stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    rows=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
    if [int(r["search_l"]) for r in rows]!=list(LS): raise ValueError("bad L grid")
    return rows

def aggregate(reps):
    out={}
    for l in LS:
        rs=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
            "recall_percent":float(np.mean([float(r["recall"]) for r in rs])),
            "mean_ios":float(np.mean([float(r["mean_ios"]) for r in rs])),
            "latency_us":float(np.median([float(r["mean_latency"]) for r in rs])),
            "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in rs])),
            "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in rs])),
        }
    return out

def monotone(s):
    out=[]; best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best: out.append((r,s[str(l)])); best=max(best,r)
    return out

def interp(s,t,field):
    p=monotone(s)
    if t<p[0][0] or t>p[-1][0]: return None
    for (a,ra),(b,rb) in zip(p,p[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[field])+x*(float(rb[field])-float(ra[field]))
    return float(p[-1][1][field])

def delta(a,b):
    return 100*(a/b-1)

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","ram-graph","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=5)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","ram_graph","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True); args.out.mkdir(parents=True,exist_ok=True)
    held=args.work/"held.fbin"; gt=args.work/"held.gt"; suffix_fbin(args.queries,held,9000); suffix_gt(args.gt5000,gt,4000)

    runs={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0)); os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS); order=METHODS[shift:]+METHODS[:shift]
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                rr=run_one(args.binary,args.out,f"r{rep}-{m}",held,gt,args.index_prefix,args.ivf_16k,args.ram_graph,m)
                runs[m].append(rr); save(args.out/"runs.partial.json",runs)
                r20=next(x for x in rr if int(x["search_l"])==20)
                print(f"{m} L20 recall={float(r20['recall']):.3f} io={float(r20['mean_ios']):.3f} lat={float(r20['mean_latency']):.1f}",flush=True)
    finally: os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}
    fixed={}
    for t in (35.,45.,55.,65.,72.,75.):
        fixed[str(t)]={}
        for m in METHODS:
            io=interp(summary[m],t,"mean_ios"); lat=interp(summary[m],t,"latency_us"); cpu=interp(summary[m],t,"cpu_us")
            if io is not None: fixed[str(t)][m]={"io":io,"latency":lat,"cpu":cpu}
        if all(m in fixed[str(t)] for m in METHODS):
            n=fixed[str(t)]["navhint"]; v=fixed[str(t)]["vertex_s2"]; r=fixed[str(t)]["ram16"]; c=fixed[str(t)]["vertex_s2_ram16"]
            fixed[str(t)]["deltas"]={
                "vertex_vs_nav":{"io_pct":delta(v["io"],n["io"]),"lat_pct":delta(v["latency"],n["latency"])},
                "ram_vs_nav":{"io_pct":delta(r["io"],n["io"]),"lat_pct":delta(r["latency"],n["latency"])},
                "combo_vs_nav":{"io_pct":delta(c["io"],n["io"]),"lat_pct":delta(c["latency"],n["latency"])},
                "ram_increment_on_vertex":{"io_pct":delta(c["io"],v["io"]),"lat_pct":delta(c["latency"],v["latency"])},
            }

    incr=[]
    for t in ("35.0","45.0","55.0","65.0","72.0","75.0"):
        if t in fixed and "deltas" in fixed[t]:
            d=fixed[t]["deltas"]["ram_increment_on_vertex"]
            incr.append({"recall":float(t),"io_pct":d["io_pct"],"lat_pct":d["lat_pct"]})
    result={
        "policy":"same enriched graph for all arms; canonical bootstrap; optional support-2 vertex hints; optional 16 MiB d4/b16/s2 RAM PQ walk",
        "ram_graph_bytes":args.ram_graph.stat().st_size,
        "evaluation":{"Ls":list(LS),"reps":args.reps},
        "summary":summary,"fixed_recall":fixed,"ram_increment_on_vertex":incr,
        "mean_ram_increment_on_vertex":{
            "io_pct":float(np.mean([x["io_pct"] for x in incr])),
            "lat_pct":float(np.mean([x["lat_pct"] for x in incr])),
        }
    }
    save(args.out/"ram-pq-vertex-admission.json",result); print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
