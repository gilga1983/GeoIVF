#!/usr/bin/env python3
"""Hard-check exact route identity before comparing Starling fast/Core timing.

Always fail CI if the optimized scorer changes a selected ID, PQ distance,
probe rank, logical page/read route, native cache counter, or Recall@10.
No suffix tuning or raw cross-system throughput comparison permitted.
"""
import csv
import json
from pathlib import Path
import re
import statistics
import sys

REPS=range(3)
LS=(12,20,40,80)
MODES=("baseline","recent512","fast_recent512","core16k_recent512","fast_core")
QUERY_COUNT=5000
FIELDS_TO_MATCH=("recent_id","recent_dist","recent_rank","recent_injected",
                 "junction_id","junction_dist","junction_rank","junction_injected",
                 "recent_popped","junction_popped",
                 "recent_page_read","junction_page_read",
                 "oracle_id","oracle_injected",
                 "ios","cache_hits","total_pages",
                 "p0","p1","p2","p3","p4","p5","p6","p7")
SOURCE_EXPECTED={(rep,mode,L) for rep in REPS for mode in MODES for L in LS}

def get_rows(root):
    d=root/"diagnostics"
    seen={}
    for f in d.glob("rep*-*/L*.csv"):
        m=re.fullmatch(r"rep(\d+)-(.+)",f.parent.name)
        lm=re.fullmatch(r"L(\d+)\.csv",f.name)
        if m is None or lm is None:raise ValueError(f"unexpected diagnostic filename {f}")
        key=(int(m.group(1)),m.group(2),int(lm.group(1)))
        if key in seen:raise ValueError(f"duplicate {key}")
        with f.open(newline="") as reader:
            rows=list(csv.DictReader(reader))
        if len(rows)!=QUERY_COUNT:raise ValueError(f"{key} has {len(rows)} records")
        if [int(q["query"]) for q in rows]!=list(range(QUERY_COUNT)):
            raise ValueError(f"{key} causal measured suffix misaligned")
        seen[key]=rows
    if set(seen)!=SOURCE_EXPECTED:
        raise ValueError(f"missing {sorted(SOURCE_EXPECTED-set(seen))}; unexpected {sorted(set(seen)-SOURCE_EXPECTED)}")
    return seen

def native_recall(root):
    rows={}
    for rep,mode,L in SOURCE_EXPECTED:
        p=root/f"starling-rep{rep}-{mode}-summary.txt"
        if not p.exists():raise ValueError(f"missing native recall table {p}")
        match=[q.split() for q in p.read_text().splitlines()
               if q.strip() and q.split()[0]==str(L)]
        if len(match)!=1 or len(match[0])<10:
            raise ValueError(f"missing/duplicate native L={L} in {p}")
        rows[rep,mode,L]=float(match[0][-1])
    return rows

def average(rows,key,mult=1.0):
    return statistics.mean(float(q[key])*mult for q in rows)

def analyze(root):
    rows=get_rows(root)
    recalls=native_recall(root)
    matched=0
    verified={}
    for left,right in (
        ("recent512","fast_recent512"),
        ("core16k_recent512","fast_core"),
    ):
        for rep in REPS:
            for L in LS:
                classic=rows[rep,left,L]
                fast=rows[rep,right,L]
                for i,(a,b) in enumerate(zip(classic,fast)):
                    for k in FIELDS_TO_MATCH:
                        if k not in a or k not in b:
                            raise ValueError(f"missing route audit field {k}")
                        if a[k]!=b[k]:
                            raise ValueError(f"CHANGED_NAVIGATION {left}/{right} "
                                             f"rep={rep} L={L} q={i} key={k}: "
                                             f"classic={a[k]} vs fast={b[k]}")
                    matched+=1
                if recalls[rep,left,L]!=recalls[rep,right,L]:
                    raise ValueError(f"CHANGED_RECALL rep={rep} L={L} "
                                     f"{left}={recalls[rep,left,L]} {right}={recalls[rep,right,L]}")
                verified[f"rep{rep}/L{L}/{left}_vs_{right}"]={
                    "paired_rows":len(classic),
                    "route_and_PQ_selection_bitwise_same":True,
                    "recall10_pct":recalls[rep,left,L],
                    "legacy_hint_scoring_us":round(
                        average(classic,"recent_ns",.001)
                        +average(classic,"junction_ns",.001),3),
                    "fast_hint_scoring_us":round(
                        average(fast,"recent_ns",.001)
                        +average(fast,"junction_ns",.001),3),
                }
    out={"metadata":{
         "dataset":"Coveo chronological production search 31950x50",
         "protocol":"original pinned Starling, identical disk graph, 5000 distinct static training,"
                    " 20000 causal online warm, final 5000 measured, 3 rotated repetitions",
         "strictness":"bitwise-identical selected candidates/dists; route IO and first 8 native physical page IDs,"
                      " identical recall at every L and repetition",
         "fast_changes":"query-local packed PQ batch gather, one lookup per coarse/Recent512, per-cell"
                        " pooled scores, lower_bound top 8 in place of repeated sort",
         "no_new_persistent_RAM":True,
         "extra_ram":"per-query ephemeral PQ-code+dists vectors, at most 512*n_chunks bytes + 512 floats,"
                     " allocated inside timed page_search and freed after query",
         "metric":"original native Starling logical I/Os and per-query total_us, not SSD block metrics",
         "cross_system_comparisons":"none"},
         "validated_route_pairs":matched,"checks":verified,
         "arms":{},"paired":{}}
    for L in LS:
        for mode in MODES:
            rr=[row for rep in REPS for row in rows[rep,mode,L]]
            nrec=statistics.mean(recalls[rep,mode,L] for rep in REPS)
            out["arms"][f"L{L}/{mode}"]={
                "n_measured":len(rr),
                "recall10_percent":round(nrec,4),
                "logical_ios":round(average(rr,"ios"),5),
                "mean_latency_us":round(average(rr,"total_us"),3),
                "recent_scoring_us":round(average(rr,"recent_ns",.001),3),
                "junction_scoring_us":round(average(rr,"junction_ns",.001),3),
                "total_scoring_us":round(
                    average(rr,"recent_ns",.001)+average(rr,"junction_ns",.001),3)
            }
        for classic,fast in [("recent512","fast_recent512"),("core16k_recent512","fast_core")]:
            a=out["arms"][f"L{L}/{classic}"]
            b=out["arms"][f"L{L}/{fast}"]
            base=out["arms"][f"L{L}/baseline"]
            out["paired"][f"L{L}/{classic}_vs_{fast}"]={
                "same_IO_and_recall":True,
                "scoring_cpu_reduction_pct":round(100*(1-b["total_scoring_us"]/a["total_scoring_us"]),3),
                "latency_reduction_pct":round(100*(1-b["mean_latency_us"]/a["mean_latency_us"]),3),
                "faster_by_us":round(a["mean_latency_us"]-b["mean_latency_us"],3),
                "native_IO_reduction_pct":round(100*(1-b["logical_ios"]/base["logical_ios"]),3),
                "native_latency_improvement_pct":round(
                    100*(1-b["mean_latency_us"]/base["mean_latency_us"]),3),
            }
    return out

def markdown(out):
    lines=[
        "# Starling native Core packed-PQ speedup: Coveo causal workload",
        "",
        "Three self-hosted repetitions, original native Starling index, 31,950 × 50"
        " normalized product vectors; 5K static / 20K causal warm / 5K eval.",
        "",
        f"**Verified navigation-equivalent paired queries: {out['validated_route_pairs']:,}.** "
        "Identical selected IDs/PQ dists, native logical I/O counts, first eight pages,"
        " page-route counters, and Recall@10 at each search width/repetition.",
        "",
        "| L | Mode | I/Os/q | Recall@10 | Latency (µs) | Scoring (µs) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for L in LS:
        for mode in MODES:
            r=out["arms"][f"L{L}/{mode}"]
            lines.append(f"| {L} | {mode} | {r['logical_ios']:.3f} | "
                         f"{r['recall10_percent']:.2f}% | "
                         f"{r['mean_latency_us']:.1f} | "
                         f"{r['total_scoring_us']:.1f} |")
    lines+=["","## Optimization effect (classic vs identical-decision fast)", ""]
    for L in LS:
        for c,f in [("recent512","fast_recent512"),("core16k_recent512","fast_core")]:
            p=out["paired"][f"L{L}/{c}_vs_{f}"]
            lines.append(f"- L{L} {c} => {f}: scoring CPU {p['scoring_cpu_reduction_pct']:+.2f}% "
                         f"reduction; end-to-end time {p['latency_reduction_pct']:+.2f}% "
                         f"reduction; {p['native_IO_reduction_pct']:.2f}% fewer logical I/Os "
                         f"than native Starling; end-to-end benefit over native "
                         f"{p['native_latency_improvement_pct']:+.2f}%.")
    lines+=["",
            "## Restrictions",
            "All speedups here are native *per-query* latency, not derived from whole-run"
            " throughput; original Starling's memory navigator is present in all arms."
            " These are equal-L and exact-matched-recall for the fast/legacy pairs.",
            "No persistent vector codes or changes to NavHints' score/tie/insertion"
            " policy. The fast selector uses per-query scratch and preserves all routes.",
            "No physical block-device counter claims; no author-original Starling source"
            " files are committed. Do not fold into the approved manuscript without review.",
            ""]
    return "\n".join(lines)

def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: summarize_starling_fast_core.py DIAGNOSTICS OUT_JSON OUT_MD")
    root=Path(sys.argv[1]).resolve().parent
    out=analyze(root)
    Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+"\n")
    report=markdown(out)
    Path(sys.argv[3]).write_text(report)
    print("STARLING_FAST_CORE_ROUTES_AND_RECALL_IDENTICAL",flush=True)
    print(report,flush=True)

if __name__=="__main__":main()
