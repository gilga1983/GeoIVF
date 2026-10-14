#!/usr/bin/env python3
"""Fail-closed native LAANN/NavHints comparison at matched Recall@10.

Every arm must use the same frozen native PageANN-reordered ID ground truth,
the same list of L values, 4K warm/1K measurement, and one in-flight query.
Numbers are descriptive within one deterministic replay, NOT confidence
intervals over independent populations.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def truth(path):
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(8), dtype="<u4")
        if len(head) != 2:
            raise ValueError("Bad ground truth header")
        n, k = map(int, head)
        values = np.frombuffer(f.read(n * k * 4), dtype="<u4")
    if n != 5000 or k < 10 or values.size != n * k:
        raise ValueError("Ground truth must match the 5K native-ID replay")
    return values.reshape(n, k)[:, :10]


def read_one(csv_file, gt):
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 5000:
        raise ValueError(f"{csv_file}: expected exactly 5K causally ordered requests")
    ls = {int(r["L"]) for r in rows}
    if len(ls) != 1:
        raise ValueError(f"{csv_file}: one search L per independent process required")
    L = ls.pop()
    expected = ["L", "query_idx", "reads", "latency_us", "result_ids"]
    if set(rows[0]) != set(expected):
        raise ValueError(f"{csv_file}: malformed record columns")
    for i, row in enumerate(rows):
        if int(row["query_idx"]) != i:
            raise ValueError(f"{csv_file}: non-causal reordered request #{i}")
    measured = rows[4000:]
    ios, times, hit = [], [], []
    for i, row in enumerate(measured):
        result = [int(v) for v in row["result_ids"].split("|")]
        if len(result) != 10:
            raise ValueError(f"{csv_file}: returned result count must be 10")
        io, us = float(row["reads"]), float(row["latency_us"])
        if not math.isfinite(io) or not math.isfinite(us) or io < 0 or us <= 0:
            raise ValueError(f"{csv_file}: non-finite or negative cost")
        ios.append(io)
        times.append(us)
        # All IDs were translated by the ORIGINAL PageANN/LAANN remapper.
        hit.append(len(set(result) & set(map(int,gt[4000+i])))/10.)
    return {"L":L,"queries":1000,"recall":100*np.mean(hit),
            "reads":float(np.mean(ios)),"service_us":float(np.mean(times)),
            "p50_us":float(np.percentile(times,50)),
            "p95_us":float(np.percentile(times,95)),
            "p99_us":float(np.percentile(times,99))}


def read_group(path, gt):
    files=sorted(path.glob("L*.csv"))
    if not files:
        raise ValueError(f"No per-width LAANN records under {path}")
    out=[read_one(file,gt) for file in files]
    if len(out)!=len(set(p["L"] for p in out)):
        raise ValueError(f"{path}: duplicate search widths")
    return sorted(out,key=lambda p:p["L"])


def interpolate(points, recall, col):
    seq=sorted(points,key=lambda x:(x["recall"],x["L"]))
    seq=[p for i,p in enumerate(seq) if i==0 or p["recall"]>seq[i-1]["recall"]+1e-8]
    if len(seq)<2 or recall<seq[0]["recall"]-1e-8 or recall>seq[-1]["recall"]+1e-8:
        return None
    for a in seq:
        if abs(a["recall"]-recall)<1e-8:return a[col]
    for a,b in zip(seq,seq[1:]):
        if a["recall"]<recall<b["recall"]:
            t=(recall-a["recall"])/(b["recall"]-a["recall"])
            return a[col]+t*(b[col]-a[col])
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth",type=Path,required=True)
    p.add_argument("--results-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    gt=truth(a.truth)
    arms={n:read_group(a.results_root/n,gt) for n in ("baseline","junction","core")}
    Lsets=[set(p["L"] for p in x) for x in arms.values()]
    if len(set(map(frozenset,Lsets)))!=1:
        raise ValueError("All arms must use identical tested search widths")
    # Reference anchors are ACTUAL native LAANN+NavHints achieved recalls,
    # never synthesized grid points or extrapolated regions.
    anchors=[]
    seen_recalls=[]
    for item in arms["core"]:
        recall=item["recall"]
        # Multiple L settings can achieve exactly the same recall. A matched
        # recall value is ONE target, not one vote for every redundant L.
        # Keep the first (lowest-L) occurrence on a plateau.
        if any(abs(recall - x) < 1e-7 for x in seen_recalls):
            continue
        at={name:{
            metric:interpolate(points,recall,metric)
            for metric in ("reads","service_us")}
            for name,points in arms.items()}
        if any(None in m.values() for m in at.values()):
            continue
        if at["baseline"]["reads"] <= 0 or at["junction"]["reads"] <= 0:
            continue
        seen_recalls.append(recall)
        anchors.append({
            "recall":recall,
            "L_core":item["L"],
            "baseline":at["baseline"],"junction":at["junction"],"core":at["core"],
            "read_saved_vs_baseline_pct":100*(1-at["core"]["reads"]/at["baseline"]["reads"]),
            "read_saved_vs_junction_pct":100*(1-at["core"]["reads"]/at["junction"]["reads"]),
            "latency_saved_vs_baseline_pct":100*(1-at["core"]["service_us"]/at["baseline"]["service_us"]),
            "latency_saved_vs_junction_pct":100*(1-at["core"]["service_us"]/at["junction"]["service_us"])
        })
    if len(anchors)<2:
        raise ValueError("Less than two jointly reachable measured native recall targets")
    summary={
        "system":"native LAANN with its original PQ memory, nav graph, page graph and I/O pipeline",
        "measurement":"real SSD provider page reads and feedback-inclusive per-request latency",
        "single_query_in_flight":True,
        "warm_queries":4000,"measure_queries":1000,
        "anchor_policy":"NavHints-measured, jointly reachable recall targets, no extrapolation",
        "matched_targets":len(anchors),
        "read_saving_vs_baseline_pct":float(np.mean([x["read_saved_vs_baseline_pct"] for x in anchors])),
        "read_saving_vs_junction_pct":float(np.mean([x["read_saved_vs_junction_pct"] for x in anchors])),
        "latency_saving_vs_baseline_pct":float(np.mean([x["latency_saved_vs_baseline_pct"] for x in anchors])),
        "latency_saving_vs_junction_pct":float(np.mean([x["latency_saved_vs_junction_pct"] for x in anchors])),
        "limitations":"One deterministic replay per arm and width; numbers are not independent-workload CIs."
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({"summary":summary,"per_L":arms,"anchors":anchors},indent=2)+"\n")
    print("NH_LAANN_MATCHED_RECALL_PASS",json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
