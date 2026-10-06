#!/usr/bin/env python3
"""Build rolling PQ-key semantic-cache entries from ACTUAL prior DiskANN results."""
from __future__ import annotations
import argparse,json,struct,time
from pathlib import Path
import numpy as np
from analyze_pq_semantic_filter import fbin_memmap, load_pq, lut_for_query
from compare_semantic_cache_query_keys import encode_queries_pq, score_pq_cache

WARM0=5000
EVAL0=9000
N=1000

def read_ids(path:Path):
    with path.open("rb") as f:
        r,k=struct.unpack("<II",f.read(8))
        a=np.fromfile(f,dtype="<u4",count=r*k)
    if a.size!=r*k: raise ValueError("truncated result dump")
    return a.reshape(r,k)

def trace_starts(path:Path):
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if len(rows)!=N: raise ValueError("trace rows")
    out=np.empty(N,dtype=np.uint32)
    for i,r in enumerate(rows):
        if int(r["query"])!=i or not r["ids"]: raise ValueError("bad trace")
        out[i]=int(r["ids"][0])
    return out

def write_rows(path:Path,a):
    a=np.asarray(a,dtype="<u4")
    with path.open("wb") as f:
        f.write(struct.pack("<II",a.shape[0],a.shape[1])); a.tofile(f)

def main():
    ap=argparse.ArgumentParser()
    for n in ("queries","pq-pivots","pq-codes","results-l80","results-l160","out-dir"):
        ap.add_argument("--"+n,type=Path,required=True)
    ap.add_argument("--capacities",default="512,2048")
    args=ap.parse_args()
    caps=sorted({int(x) for x in args.capacities.split(",") if x.strip()})
    q=fbin_memmap(args.queries.resolve())
    piv,offs,_=load_pq(args.pq_pivots.resolve(),args.pq_codes.resolve())
    qwarm=np.asarray(q[WARM0:],dtype=np.float32)
    qeval=np.asarray(q[EVAL0:EVAL0+N],dtype=np.float32)
    codes=encode_queries_pq(qwarm,piv,offs)
    prod={"80":read_ids(args.results_l80.resolve()),"160":read_ids(args.results_l160.resolve())}
    for L,a in prod.items():
        if a.shape!=(5000,10): raise ValueError(f"L{L} producer shape {a.shape}")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    manifest={"policy":"rolling PQ-key semantic cache using actual prior DiskANN top-10 outputs","capacities":{}}

    for cap in caps:
        nearest=np.empty(N,dtype=np.int32)
        t0=time.perf_counter()
        for ei,qq in enumerate(qeval):
            rel=(EVAL0+ei)-WARM0
            lo=max(0,rel-cap); hi=rel
            scores=score_pq_cache(qq,codes[lo:hi],piv,offs)
            nearest[ei]=lo+int(np.argmax(scores))
        scan=time.perf_counter()-t0
        cmeta={"capacity":cap,"pq_key_plus_anchor_bytes":cap*(codes.shape[1]+4),
               "python_scan_us_per_query":1e6*scan/N,"sources":{}}
        for L,a in prod.items():
            rows=np.empty((N,10),dtype=np.uint32)
            duplicate_rows=0
            for ei,ri in enumerate(nearest):
                ids=[int(x) for x in a[int(ri)]]
                # Keep producer's best returned ID as anchor; de-duplicate siblings.
                uniq=[]
                seen=set()
                for x in ids:
                    if x in seen: continue
                    seen.add(x); uniq.append(x)
                if len(uniq)<10:
                    duplicate_rows+=1
                    # Fill from remaining producer row impossible if duplicate-heavy; pad anchor
                    # is not allowed by search parser, so fail closed.
                    raise RuntimeError("producer returned duplicate IDs")
                rows[ei]=np.asarray(uniq[:10],dtype=np.uint32)
            p=args.out_dir/f"actual-L{L}-c{cap}.bin"; write_rows(p,rows)
            cmeta["sources"][L]={"file":p.name,"duplicate_rows":duplicate_rows}
        manifest["capacities"][str(cap)]=cmeta
        print(json.dumps({"capacity":cap,"scan_us":cmeta["python_scan_us_per_query"],
                          "bytes":cmeta["pq_key_plus_anchor_bytes"]}),flush=True)

    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=="__main__": main()
