#!/usr/bin/env python3
"""Strict final-Core vs author-native Starling three-arm causal SSD analysis.

Usage: summarize_starling_final_core.py DIAG_FOLDER OUTPUT_JSON OUTPUT_MD

This is a *changed-navigation* comparison. It must not use the old exact-ID
identity criterion; instead it checks query count/order, native recall,
logical native read counts, same L and causal protocol. All metrics use native
Starling counters; three reps are repeats of one query stream.
"""
import csv
import json
from pathlib import Path
import re
import statistics
import sys

MODES=("baseline","fast_core","final_core")
REPS=range(3)

def path_config(artifact):
    if "bigann" in artifact.name.lower():
        return (20,40,80,160),1000,"BigANN-10M 128d uint8 L2",4000
    if "coveo" in artifact.name.lower():
        return (12,20,40,80),5000,"Coveo chronological 31,950 products 50d",20000
    raise ValueError(f"unsupported native dataset evidence {artifact}")

def read_points(artifact):
    Ls,N,dataset,warm=path_config(artifact)
    readings={}
    recalls={}
    for rep in REPS:
        for mode in MODES:
            summary=artifact/f"starling-rep{rep}-{mode}-summary.txt"
            if not summary.is_file():
                raise FileNotFoundError(summary)
            t=summary.read_text().splitlines()
            for L in Ls:
                lines=[x.split() for x in t if x.strip() and x.split()[0]==str(L)]
                if len(lines)!=1 or len(lines[0])<10:
                    raise ValueError(f"invalid or missing native Recall10 row: {summary} L={L}")
                rec=float(lines[0][-1])
                if not 0<=rec<=100:raise ValueError("invalid Recall@10 percent")
                file=artifact/"diagnostics"/f"rep{rep}-{mode}"/f"L{L}.csv"
                if not file.is_file():raise FileNotFoundError(file)
                with file.open(newline="") as f:
                    rows=list(csv.DictReader(f))
                if len(rows)!=N or [int(q["query"]) for q in rows]!=list(range(N)):
                    raise ValueError(f"wrong measured causal query sequence: {file}")
                rr={}
                for key in ("ios","total_us","cpu_us","cache_hits"):
                    vals=[float(x[key]) for x in rows]
                    if min(vals)<0:raise ValueError(f"negative {key}")
                    rr[key]=statistics.mean(vals)
                rr["recall"]=rec
                readings[f"r{rep}-{mode}-L{L}"]=rr
                recalls[rep,mode,L]=rec
    return Ls,N,dataset,warm,readings

def curve(rows,rep,mode,Ls):
    pts=sorted(
        ((rows[f"r{rep}-{mode}-L{l}"]["recall"],
          rows[f"r{rep}-{mode}-L{l}"]["ios"],l,
          rows[f"r{rep}-{mode}-L{l}"]["total_us"])
         for l in Ls),key=lambda z:(z[0],z[1])
    )
    # Remove same-recall ties; never extrapolate.
    out=[]
    for rec,io,l,lat in pts:
        if out and abs(rec-out[-1][0])<1e-9:
            if io<out[-1][1]:out[-1]=(rec,io,l,lat)
        else:out.append((rec,io,l,lat))
    return out

def matched(points,target):
    if target<points[0][0]-1e-9 or target>points[-1][0]+1e-9:return None
    for rec,io,l,lat in points:
        if abs(rec-target)<1e-9:
            return {"reads":io,"latency_us":lat,"low_L":l,"high_L":l}
    for (r0,i0,l0,t0),(r1,i1,l1,t1) in zip(points,points[1:]):
        if r0<target<r1:
            x=(target-r0)/(r1-r0)
            return {"reads":i0+x*(i1-i0),"latency_us":t0+x*(t1-t0),
                    "low_L":l0,"high_L":l1}
    return None

def analyze(artifact):
    Ls,N,dataset,warm,raw=read_points(artifact)
    by={}
    for mode in MODES:
        by[mode]={}
        for L in Ls:
            points=[raw[f"r{rep}-{mode}-L{L}"] for rep in REPS]
            by[mode][str(L)]={k:statistics.mean(p[k] for p in points)
                for k in ("ios","total_us","cpu_us","cache_hits","recall")}
    anchors=[]
    for L in Ls:
        # Baseline anchor is independently recomputed per replay.
        per_rep=[]
        for rep in REPS:
            target=raw[f"r{rep}-baseline-L{L}"]["recall"]
            old=matched(curve(raw,rep,"fast_core",Ls),target)
            new=matched(curve(raw,rep,"final_core",Ls),target)
            baseline=matched(curve(raw,rep,"baseline",Ls),target)
            if old is None or new is None or baseline is None:continue
            per_rep.append({
                "rep":rep,"recall_target":target,
                "baseline":baseline,"control":old,"core":new,
                "core_saves_vs_native":100*(1-new["reads"]/baseline["reads"]),
                "core_saves_vs_control":100*(1-new["reads"]/old["reads"]),
                "core_latency_saves_vs_native":100*(1-new["latency_us"]/baseline["latency_us"]),
            })
        anchors.append({"baseline_L":L,"paired_reps":len(per_rep),
                        "available":len(per_rep)==len(REPS),
                        "reps":per_rep,
                        "core_saves_vs_native_percent":statistics.mean(
                         x["core_saves_vs_native"] for x in per_rep) if per_rep else None,
                        "core_saves_vs_control_percent":statistics.mean(
                         x["core_saves_vs_control"] for x in per_rep) if per_rep else None})
    if not any(q["available"] for q in anchors):
        raise RuntimeError("no common recall anchor across native Starling arms")
    return {
        "dataset":dataset,"native_engine":"original pinned Starling",
        "protocol":{"static_training":5000,"online_warmup":warm,"measured":N,
                    "repetitions":3,"L":list(Ls),
                    "run_order":"rotated and same native SSD graph/index",
                    "baseline":"original Starling navigator enabled"},
        "modes":{"baseline":"unchanged native Starling",
                 "fast_core":"compact-history 16K / Recent512 / one start",
                 "final_core":"train-only 14,464 ID native junctions / Recent2048 / eight PQ starts"},
        "measured_replay_count":len(raw),
        "intra_native_methods":by,
        "raw_runs":raw,
        "matched_recall_vs_native_baseline":anchors,
        "warnings":[
          "3 repeats share one frozen query sequence, not population-independent sampling",
          "native Starling logical IO != device physical block counters",
          "Starling original memory navigator and page layout remain intact",
          "per-query search latency may exclude post-query FIFO update",
          "do not compute across-engine speed ratios"
        ]
    }

def markdown(data):
    lines=["# Native original Starling: NavHints selected Core on "+data["dataset"],"",
           "Original native Starling index and navigator remain enabled; all arms use",
           "the same causal history/query/index and three rotated repetitions.","",
           "| L | Mode | Recall@10 | Reads/q | Latency us |",
           "|---:|---|---:|---:|---:|"]
    for L in data["protocol"]["L"]:
        for name in MODES:
            x=data["intra_native_methods"][name][str(L)]
            lines.append(f"| {L} | {name} | {x['recall']:.3f}% | {x['ios']:.3f} | {x['total_us']:.2f} |")
    lines+=["","## Matched native baseline Recall@10 (no extrapolation)","",
            "| Baseline L | Reachable in 3/3 replays | Core IO saved vs native | Core saved vs compact control |",
            "|---:|:---:|---:|---:|"]
    for v in data["matched_recall_vs_native_baseline"]:
        lines.append(f"| {v['baseline_L']} | {'yes' if v['available'] else 'no'} | "
          f"{v['core_saves_vs_native_percent'] if v['core_saves_vs_native_percent'] is not None else float('nan'):+.3f}% | "
          f"{v['core_saves_vs_control_percent'] if v['core_saves_vs_control_percent'] is not None else float('nan'):+.3f}% |")
    lines+=["","Native read and latency counters only; one deterministic workload, no independent uncertainty estimate.",""]
    return "\n".join(lines)

def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: summarize_starling_final_core.py DIAGNOSTICS OUT_JSON OUT_MD")
    artifact=Path(sys.argv[1]).resolve().parent
    out=analyze(artifact)
    Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    Path(sys.argv[3]).write_text(markdown(out))
    print("STARLING_FINAL_CORE_NATIVE_CAUSAL_REPORT_VALIDATED",flush=True)
    print(markdown(out),flush=True)

if __name__=="__main__":main()
