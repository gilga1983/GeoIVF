#!/usr/bin/env python3
"""Evaluate from-cold causal online hub learning plus seed-only value cache.

Every method replays 5,000 chronologically ordered queries and measures only
the final 1,000. All methods use exactly the same integrated Rust binary and
request timing, including cache selection and update CPU; hub disk mutations
remain logical and their modeled write counts are reported separately.
"""
from __future__ import annotations

import argparse,fcntl,json,os,re,struct,subprocess
from pathlib import Path

import numpy as np

THREADS=4
BEAM=8
K=10
LS=(10,20,40,80,160,320)
METHODS=("vertex","onlinehub10","onlinecache512","onlinehub10_cache512")
HUB_RE=re.compile(
 r"ONLINE_HUB_STATS L=(\d+) active_hubs=(\d+) learned=(\d+) "
 r"duplicates=(\d+) full=(\d+) batch_writes=(\d+) dirty_pages=(\d+) "
 r"dirty_entries=(\d+) writes_if_final_flush=(\d+)"
)
CACHE_RE=re.compile(
 r"ONLINE_VALUE_STATS L=(\d+) capacity=(\d+) occupancy=(\d+) "
 r"inserts=(\d+) duplicate_skips=(\d+) evictions=(\d+)"
)


def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2)+"\n")


def shape(p):
    with p.open("rb") as f:return struct.unpack("<II",f.read(8))


def range_fbin(src,dst,start,count):
    rows,dim=shape(src)
    if start<0 or count<=0 or start+count>rows:
        raise ValueError("bad query range")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4);fo.write(struct.pack("<II",count,dim))
        remaining=count*dim*4
        while remaining:
            b=fi.read(min(16<<20,remaining))
            if not b:raise ValueError("truncated query file")
            fo.write(b);remaining-=len(b)


def result_rows(o):
    out=[]
    if isinstance(o,dict):
        if "search_l" in o and "mean_latency" in o:out.append(o)
        else:
            for v in o.values():out.extend(result_rows(v))
    elif isinstance(o,list):
        for v in o:out.extend(result_rows(v))
    return out


def run(binary,out,tag,q,gt,prefix,ivf,hub,cache):
    cfg={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp=out/f"{tag}.input.json"
    op=out/f"{tag}.output.json"
    log=out/f"{tag}.log"
    save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):env.pop(n,None)
    env["DISKANN_HINT_IVF_FILE"]=str(ivf)
    env["DISKANN_HINT_IVF_NPROBE"]="8"
    env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    env["DISKANN_VERTEX_HINT_VARIANT"]="1"
    env["DISKANN_ONLINE_HUB_REPLAY"]="1"
    env["DISKANN_ONLINE_HUB_CAPACITY"]=str(hub)
    env["DISKANN_ONLINE_HUB_WARMUP"]="4000"
    env["DISKANN_ONLINE_HUB_FLUSH_BATCH"]="10"
    env["DISKANN_COMBINED_VALUE_CACHE_CAPACITY"]=str(cache)
    with log.open("w") as lf:
        subprocess.run(
          [str(binary),"run","--input-file",str(inp),"--output-file",str(op)],
          stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)

    rr=sorted(result_rows(json.loads(op.read_text())),key=lambda r:int(r["search_l"]))
    if [int(x["search_l"]) for x in rr]!=list(LS):
        raise ValueError(f"{tag}: unexpected L grid")

    txt=log.read_text()
    hub_stats={}
    cache_stats={}
    for m in HUB_RE.finditer(txt):
        hub_stats[m.group(1)]={
          "active_hubs":int(m.group(2)),
          "learned":int(m.group(3)),
          "duplicates":int(m.group(4)),
          "full":int(m.group(5)),
          "batch_writes":int(m.group(6)),
          "dirty_pages":int(m.group(7)),
          "dirty_entries":int(m.group(8)),
          "writes_if_final_flush":int(m.group(9)),
        }
    for m in CACHE_RE.finditer(txt):
        cache_stats[m.group(1)]={
          "capacity":int(m.group(2)),
          "occupancy":int(m.group(3)),
          "inserts":int(m.group(4)),
          "duplicate_skips":int(m.group(5)),
          "evictions":int(m.group(6)),
        }
    if len(hub_stats)!=len(LS) or len(cache_stats)!=len(LS):
        raise ValueError(f"{tag}: missing hub/cache accounting rows")
    return rr,{"hubs":hub_stats,"cache":cache_stats}


def aggregate(reps):
    out={}
    for l in LS:
        a=[r for rr in reps for r in rr if int(r["search_l"])==l]
        out[str(l)]={
          "recall_percent":float(np.mean([float(r["recall"]) for r in a])),
          "mean_ios":float(np.mean([float(r["mean_ios"]) for r in a])),
          "latency_us":float(np.median([float(r["mean_latency"]) for r in a])),
          "cpu_us":float(np.median([float(r["mean_cpu_time"]) for r in a])),
          "comparisons":float(np.mean([float(r["mean_comparisons"]) for r in a])),
        }
    return out


def aggregate_diag(reps):
    result={}
    for subsystem in ("hubs","cache"):
        result[subsystem]={}
        for l in LS:
            values=[x[subsystem][str(l)] for x in reps]
            result[subsystem][str(l)]={
              k:float(np.mean([v[k] for v in values])) for k in values[0]
            }
    return result


def mono(s):
    out=[];best=-1e99
    for l in LS:
        r=s[str(l)]["recall_percent"]
        if r+1e-9>=best:
            out.append((r,s[str(l)]))
            best=max(best,r)
    return out


def interp(s,t,field):
    p=mono(s)
    if t<p[0][0] or t>p[-1][0]:return None
    for (a,ra),(b,rb) in zip(p,p[1:]):
        if a<=t<=b:
            x=0 if b<=a+1e-12 else (t-a)/(b-a)
            return float(ra[field])+x*(float(rb[field])-float(ra[field]))
    return float(p[-1][1][field])


def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","gt5000","index-prefix","ivf-16k","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--reps",type=int,default=3)
    args=ap.parse_args()
    for n in ("binary","queries","gt5000","index_prefix","ivf_16k","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True)
    args.out.mkdir(parents=True,exist_ok=True)
    q=args.work/"replay5000.fbin"
    range_fbin(args.queries,q,5000,5000)

    cfg={
      "vertex":(0,0),
      "onlinehub10":(10,0),
      "onlinecache512":(0,512),
      "onlinehub10_cache512":(10,512),
    }
    runs={m:[] for m in METHODS}
    diagnostics={m:[] for m in METHODS}
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"
    lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
      with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for rep in range(args.reps):
            shift=rep%len(METHODS)
            order=list(METHODS[shift:]+METHODS[:shift])
            print(f"rep={rep} order={' '.join(order)}",flush=True)
            for m in order:
                hub,cache=cfg[m]
                rr,diag=run(args.binary,args.out,f"r{rep}-{m}",q,args.gt5000,
                  args.index_prefix,args.ivf_16k,hub,cache)
                runs[m].append(rr)
                diagnostics[m].append(diag)
                r40=next(x for x in rr if int(x["search_l"])==40)
                print(
                  f"{m} L40 recall={float(r40['recall']):.3f} "
                  f"io={float(r40['mean_ios']):.2f} "
                  f"lat={float(r40['mean_latency']):.1f} "
                  f"cpu={float(r40['mean_cpu_time']):.1f}",
                  flush=True)
    finally:
      os.sched_setaffinity(0,set(allowed))

    summary={m:aggregate(runs[m]) for m in METHODS}
    diag={m:aggregate_diag(diagnostics[m]) for m in METHODS}
    base=summary["vertex"]
    targets=(35.,45.,55.,65.,72.,75.)
    fixed={}
    for t in targets:
        bio=interp(base,t,"mean_ios")
        bl=interp(base,t,"latency_us")
        if bio is None:continue
        fixed[str(t)]={}
        for m in METHODS:
            io=interp(summary[m],t,"mean_ios")
            la=interp(summary[m],t,"latency_us")
            cpu=interp(summary[m],t,"cpu_us")
            if io is None:continue
            fixed[str(t)][m]={
              "io_change_percent_vs_vertex":100*(io/bio-1),
              "latency_change_percent_vs_vertex":100*(la/bl-1),
              "cpu_us":cpu,
            }

    incremental=[]
    for t in targets:
        hio=interp(summary["onlinehub10"],t,"mean_ios")
        cio=interp(summary["onlinehub10_cache512"],t,"mean_ios")
        hla=interp(summary["onlinehub10"],t,"latency_us")
        cla=interp(summary["onlinehub10_cache512"],t,"latency_us")
        hcpu=interp(summary["onlinehub10"],t,"cpu_us")
        ccpu=interp(summary["onlinehub10_cache512"],t,"cpu_us")
        if None in (hio,cio,hla,cla,hcpu,ccpu):continue
        incremental.append({
          "recall":t,
          "incremental_io_change_percent_vs_onlinehub10":100*(cio/hio-1),
          "incremental_latency_change_percent_vs_onlinehub10":100*(cla/hla-1),
          "incremental_cpu_us":ccpu-hcpu,
        })

    ranking=[]
    for m in METHODS[1:]:
        values=[fixed[str(t)][m] for t in targets if str(t) in fixed and m in fixed[str(t)]]
        ranking.append({
          "method":m,
          "mean_io_change_percent_vs_vertex":float(np.mean([x["io_change_percent_vs_vertex"] for x in values])),
          "mean_latency_change_percent_vs_vertex":float(np.mean([x["latency_change_percent_vs_vertex"] for x in values])),
        })
    ranking.sort(key=lambda x:x["mean_io_change_percent_vs_vertex"])

    result={
      "policy":"causal from-cold online hub winners + resident PQ-scored SkipDup value-ID cache; both online state stores begin empty",
      "runtime":{"cache_capacity_ids":512,"cache_ram_bytes_for_ids":2048,
        "hub_winner_limit":10,"hub_page_data":"causal logical overlay","physical_hub_writes":"counted, not executed",
        "query_timing":"includes search and RAM update bookkeeping"},
      "evaluation":{"Ls":list(LS),"reps":args.reps,"replay_queries":5000,"warmup":4000,"measured":1000},
      "summary":summary,
      "diagnostics":diag,
      "fixed_recall":fixed,
      "cache_incremental_over_online_hub":incremental,
      "ranking":ranking,
    }
    save(args.out/"online-combined.json",result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
