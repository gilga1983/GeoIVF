#!/usr/bin/env python3
"""Collect only the deployed canonical 16K NavHint hub for a query range."""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np

THREADS=4
BEAM=8
L=10

def save(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+"\n")

def shape(p):
    with p.open("rb") as f:return struct.unpack("<II",f.read(8))

def range_fbin(src,dst,start,count):
    rows,dim=shape(src)
    if start<0 or count<=0 or start+count>rows:raise ValueError("bad range")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4);fo.write(struct.pack("<II",count,dim))
        rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b:raise ValueError("truncated")
            fo.write(b);rem-=len(b)

def fake_gt(p,rows):
    with p.open("wb") as f:
        f.write(struct.pack("<II",rows,1));np.zeros(rows,dtype="<u4").tofile(f)

def main():
    ap=argparse.ArgumentParser()
    for n in ("binary","queries","index-prefix","start-ivf","work","out"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--start-row",type=int,default=5000)
    ap.add_argument("--rows",type=int,default=4000)
    args=ap.parse_args()
    for n in ("binary","queries","index_prefix","start_ivf","work","out"):
        setattr(args,n,getattr(args,n).resolve())
    args.work.mkdir(parents=True,exist_ok=True);args.out.mkdir(parents=True,exist_ok=True)
    q=args.work/"hub-range.fbin";gt=args.work/"fake.gt";range_fbin(args.queries,q,args.start_row,args.rows);fake_gt(gt,args.rows)
    trace=args.out/"hubs.jsonl";inp=args.out/"hub-trace.input.json";op=args.out/"hub-trace.output.json"
    cfg={"search_directories":[str(args.work)],"jobs":[{"type":"disk-index","content":{
      "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(args.index_prefix)},
      "search_phase":{"queries":str(q),"groundtruth":str(gt),"search_list":[L],"beam_width":BEAM,
        "recall_at":1,"num_threads":THREADS,"is_flat_search":False,"distance":"inner_product",
        "vector_filters_file":None,"num_nodes_to_cache":None,"search_io_limit":None,"post_processor":None}}}]}
    save(inp,cfg)
    env=os.environ.copy()
    for n in list(env):
        if n.startswith("DISKANN_"):env.pop(n,None)
    env["DISKANN_SKIP_RECALL"]="1";env["DISKANN_TRACE_FILE"]=str(trace)
    env["DISKANN_HINT_IVF_FILE"]=str(args.start_ivf);env["DISKANN_HINT_IVF_NPROBE"]="8";env["DISKANN_HINT_IVF_MAX_STARTS"]="1"
    lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
    with lockp.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (args.out/"hub-trace.log").open("w") as lf:
            subprocess.run([str(args.binary),"run","--input-file",str(inp),"--output-file",str(op)],
              stdout=lf,stderr=subprocess.STDOUT,env=env,check=True)
    # With a single L the trace patch may emit either the requested base path or .L10.
    actual=trace if trace.is_file() else args.out/"hubs.L10.jsonl"
    rec=[json.loads(x) for x in actual.read_text().splitlines() if x.strip()]
    if len(rec)!=args.rows:raise ValueError(f"expected {args.rows}, got {len(rec)}")
    hubs=[]
    for i,r in enumerate(rec):
        if int(r["query"])!=i or not r["ids"]:raise ValueError("bad trace")
        hubs.append(int(r["ids"][0]))
    # Normalize to a stable file name expected by downstream builder.
    with trace.open("w") as f:
        for i,(r,h) in enumerate(zip(rec,hubs)):
            f.write(json.dumps({"query":i,"ids":[h]})+"\n")
    vals=np.asarray(hubs,dtype=np.uint32)
    summary={"query_start":args.start_row,"queries":args.rows,"L":L,
      "unique_deployed_hubs":int(len(np.unique(vals))),
      "mean_queries_per_active_hub":float(args.rows/len(np.unique(vals)))}
    save(args.out/"hub-summary.json",summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":main()
