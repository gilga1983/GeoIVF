#!/usr/bin/env python3
"""Measure NavHints *RAM-only Core* alongside native BigANN-10M systems.

Use the same indexed 10M uint8/L2 corpus, five thousand held-out queries,
first 4K queries for causal online-state warm-up, final 1K measured.
Each invocation starts fresh online state. Reports L-grid and source
counters, never converts another method's graph I/O into physical bytes.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import qualify_public_causal_fill2 as q

q.LS=(10,20,40,80,160,320,640)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",type=Path,required=True)
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--index-prefix",type=Path,required=True)
    ap.add_argument("--ivf",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--rep",type=int,required=True)
    a=ap.parse_args()
    m=json.loads(a.manifest.read_text())
    assert m["dataset"]=="bigann-10M"
    assert m["data_type"]=="uint8" and m["metric"]=="squared_l2"
    assert a.rep in (0,1,2)
    a.out.mkdir(parents=True,exist_ok=True)
    cfg={"ivf":True,"experience":True,"cache":512,"hub":0,
         "sample":2,"fill":False}
    queries=Path(m["files"]["heldout5000"])
    groundtruth=Path(m["files"]["heldout5000_gt"])
    rows,stats=q.run(a.binary,a.out,f"core-rep{a.rep}",
                     queries,groundtruth,a.index_prefix,
                     m["data_type"],m["metric"],a.ivf,cfg)
    assert len(rows)==len(q.LS),rows
    assert stats is not None and set(stats)==set(map(str,q.LS))
    for row in rows:
        assert float(row["recall"])>=0
    result={"system":"NavHints RAM-only Core","rep":a.rep,
            "dataset":"BigANN-10M","K":10,"beam_width":8,
            "search_threads":4,"warmup_queries":4000,"measured_queries":1000,
            "persistent_graph_updates":0,
            "auxiliary_routing_payload_bytes":102940,
            "rows":rows,"experience_stats":stats}
    outfile=a.out/f"core-rep{a.rep}-summary.json"
    q.save(outfile,result)
    for x in rows:
        print(f"CORE_NATIVE_EPOCH rep={a.rep} L={x['search_l']} "
              f"recall={x['recall']} reads={x['mean_ios']} "
              f"latency_us={x['mean_latency']}",flush=True)
    print("CORE_NATIVE_EPOCH_SUCCESS",flush=True)

if __name__=="__main__":main()
