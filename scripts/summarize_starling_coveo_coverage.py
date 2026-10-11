#!/usr/bin/env python3
"""Audit Coveo chronological Starling native routing, coverage and gate arms.

Original Coveo 31,950x50 product vectors, first 5K static training queries,
next 20K chronological causal online warmup, last 5K measured; repeat x3.
All per-query native source/reading metrics and 10NN recalls are independently
checked. No claims about externally measured physical SSD I/O.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

MODES=("baseline","recent512","core16k_recent512","diverse_core",
       "maxmin_core","cover_core","gate50_cover")
LVALUES=(12,20,40,80)
REPS=(0,1,2)
N=5000
IDSENT=4294967295
EXPECTED={(r,m,l) for r in REPS for m in MODES for l in LVALUES}
DIVERSE={"diverse_core","maxmin_core","cover_core","gate50_cover"}
GATED={"gate50_cover"}
NOHINT={"baseline"}
def avg(xs):return statistics.mean(xs) if xs else float("nan")

def read(artifact):
    d=artifact/"diagnostics"
    rows={}
    for f in sorted(d.glob("rep*-*/L*.csv")):
        m=re.fullmatch(r"rep(\d+)-([a-z0-9_]+)",f.parent.name)
        l=re.fullmatch(r"L(\d+)\.csv",f.name)
        if not m or not l:raise ValueError(f"unknown recording {f}")
        k=(int(m[1]),m[2],int(l[1]))
        if k in rows:raise ValueError(f"duplicate recording {k}")
        with f.open(newline="") as stream:
            rr=list(csv.DictReader(stream))
        if len(rr)!=N:raise ValueError(f"{f}: {len(rr)} != {N}")
        if [int(x["query"]) for x in rr]!=list(range(N)):
            raise ValueError(f"{f}: not 5000 aligned causal measured queries")
        if any(int(x["L"])!=k[-1] for x in rr):
            raise ValueError(f"{f}: wrong search width")
        rows[k]=rr
    if set(rows)!=EXPECTED:
        raise ValueError(f"missing={sorted(EXPECTED-set(rows))} "
                         f"unexpected={sorted(set(rows)-EXPECTED)}")
    recalls={}
    for rep,mode,L in EXPECTED:
        f=artifact/f"starling-rep{rep}-{mode}-summary.txt"
        if not f.is_file():raise ValueError(f"missing original author summary {f}")
        match=[x.split() for x in f.read_text().splitlines()
               if x.strip() and x.split()[0]==str(L)]
        if len(match)!=1 or len(match[0])<11:raise ValueError(f"bad native recall {f}: L={L}")
        recalls[rep,mode,L]=float(match[0][-1])
        for row in rows[rep,mode,L]:
            if int(row["gate_open"]) not in (0,1):
                raise ValueError(f"bad gate flag at {rep}/{mode}/{L}")
            if int(row["diverse"])!=int(mode in DIVERSE):
                raise ValueError(f"wrong novelty objective in {mode}")
            if mode not in GATED and int(row["gate_open"])!=1:
                raise ValueError(f"unwanted gate {mode}")
            if mode in GATED and int(row["gate_open"])==0:
                if int(row["recent_injected"])!=0 or int(row["junction_injected"])!=0:
                    raise ValueError("GATED-OFF query injected a hint")
                if int(row["recent_id"])!=IDSENT or int(row["junction_id"])!=IDSENT:
                    raise ValueError("GATED-OFF query scored a hint")
            if mode in NOHINT and (
                int(row["recent_injected"]) or int(row["junction_injected"])):
                raise ValueError("original Starling unexpectedly injected NavHints")
    return rows,recalls


def summarize(rows,recalls):
    out={
        "protocol":{
            "source":"Coveo SIGIR eCom 2021 chronological normalized float32 embeddings",
            "original_starling":"zilliztech/starling@17dc3e8a011533a62374445f53963e951b72883a",
            "base_vectors":31950,"dim":50,
            "metric":"native L2^2 rank-equivalent to exact cosine on unit vectors",
            "static_train_queries":5000,"online_warmup_queries":20000,
            "measured_queries_per_rep":5000,"repetitions":3,
            "arms":list(MODES),"L_values":list(LVALUES),
            "fairness":"same pinned author Starling index for all modes; no index remapping",
            "novelty":"top24 near-query hints, native starter-page rejection, "
                      "PQ reconstructed symmetric squared L2 to 10 Starling entries",
            "maxmin":"maximize minimum candidate-native distance "
                     "subject to query PQ distance <=2x nearest-hint query PQ distance",
            "cover":"maximize [min candidate-native PQ distance]/[PQ query-hint distance] "
                    "subject to same relevance bound",
            "gate50":"median Starling native-entry query PQ distance from first 20K "
                     "unlabeled, chronological online warmup queries",
            "metric_caveat":"Starling logical n_ios and per-query timing, not external SSD block IO; "
                            "latency may depend on NVMe/CPU and 3-rep scheduling. "
                            "Similar recall does not imply exact matching."
        },"arms":{},"paired":{},"validation":{"all_84_arm_width_rep_cells":len(rows)}}
    for L in LVALUES:
        for mode in MODES:
            allr=[item for r in REPS for item in rows[r,mode,L]]
            rec=avg([recalls[r,mode,L] for r in REPS])
            key=f"L{L}/{mode}"
            chosen_j=sum(int(x["junction_id"])!=IDSENT for x in allr)/len(allr)
            chosen_r=sum(int(x["recent_id"])!=IDSENT for x in allr)/len(allr)
            active=sum(int(x["gate_open"])==1 for x in allr)/len(allr)
            out["arms"][key]={
                "count":len(allr),
                "mean_ios":round(avg([float(x["ios"]) for x in allr]),6),
                "mean_latency_us":round(avg([float(x["total_us"]) for x in allr]),3),
                "recall_at10_percent":round(rec,5),
                "active_fraction":round(active,5),
                "mean_recent_score_us":round(avg([float(x["recent_ns"])/1000 for x in allr]),3),
                "mean_junction_score_us":round(avg([float(x["junction_ns"])/1000 for x in allr]),3),
                "junction_proposed_fraction":round(chosen_j,5),
                "recent_proposed_fraction":round(chosen_r,5),
                "junction_popped_fraction":round(sum(int(x["junction_popped"])>0 for x in allr)/len(allr),5),
                "recent_popped_fraction":round(sum(int(x["recent_popped"])>0 for x in allr)/len(allr),5),
                "junction_page_read_fraction":round(sum(int(x["junction_page_read"])>0 for x in allr)/len(allr),5),
                "recent_page_read_fraction":round(sum(int(x["recent_page_read"])>0 for x in allr)/len(allr),5),
            }
        base=out["arms"][f"L{L}/baseline"]
        core=out["arms"][f"L{L}/core16k_recent512"]
        for mode in MODES[1:]:
            obj=out["arms"][f"L{L}/{mode}"]
            ids_same=[]; first8=[]; delta_reads=[]
            ids_same_recent=[]
            gate_open=[]
            novelty=[]
            for rep in REPS:
                r0=rows[rep,"baseline",L]
                rcore=rows[rep,"core16k_recent512",L]
                rcur=rows[rep,mode,L]
                for b,c,u in zip(r0,rcore,rcur):
                    ids_same.append(u["junction_id"]==c["junction_id"])
                    ids_same_recent.append(u["recent_id"]==c["recent_id"])
                    first8.append(any(u[f"p{i}"]!=b[f"p{i}"] for i in range(8)))
                    delta_reads.append(float(b["ios"])-float(u["ios"]))
                    gate_open.append(int(u["gate_open"])==1)
                    if float(u["junction_novelty"])>=0:
                        novelty.append(float(u["junction_novelty"]))
            out["paired"][f"L{L}/{mode}"]={
                "io_savings_pct_vs_starling":round(100*(1-obj["mean_ios"]/base["mean_ios"]),4),
                "io_savings_pct_vs_original_core":round(100*(1-obj["mean_ios"]/core["mean_ios"]),4),
                "latency_improvement_pct_vs_starling":round(100*(1-obj["mean_latency_us"]/base["mean_latency_us"]),4),
                "mean_reads_saved":round(avg(delta_reads),6),
                "mean_latency_delta_us":round(obj["mean_latency_us"]-base["mean_latency_us"],3),
                "delta_recall_percentage_points":round(obj["recall_at10_percent"]-base["recall_at10_percent"],5),
                "different_junction_vs_core_fraction":round(1-avg(ids_same),5),
                "different_recent_vs_core_fraction":round(1-avg(ids_same_recent),5),
                "first_eight_pages_differ_from_native_fraction":round(avg(first8),5),
                "mean_selected_junction_novelty":round(avg(novelty),6) if novelty else None,
            }
    out["validation"]["all_arm_cells_verified"]=(len(rows)==84)
    out["validation"]["none_of_the_methods_uses_gt_as_input"]=True
    out["validation"]["remember_coveo_is_not_BigANN"] = True
    return out


def markdown(out):
    lines=["# Native Starling complementarity on chronological Coveo",
           "",
           "Coveo 31,950 × 50 normalized products, exact cosine GT, original Starling L2^2;",
           "5,000 disjoint static training, 20,000 ordered causal warm-up,",
           "5,000 measured queries, 3 full repetitions. Same native index/SSD lock.",
           "",
           "| L | Native arm | Gate open | Reads/q | I/O vs baseline | "
           "Latency (µs) | Latency vs baseline | Recall@10 |",
           "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for L in LVALUES:
        for mode in MODES:
            o=out["arms"][f"L{L}/{mode}"]
            p=out["paired"].get(f"L{L}/{mode}")
            val=f"{p['io_savings_pct_vs_starling']:+.2f}%" if p else "ref"
            lat=f"{p['latency_improvement_pct_vs_starling']:+.2f}%" if p else "ref"
            lines.append(f"| {L} | {mode} | {o['active_fraction']*100:.1f}% "
                         f"| {o['mean_ios']:.3f} | {val} | {o['mean_latency_us']:.1f} "
                         f"| {lat} | {o['recall_at10_percent']:.2f}% |")
    lines+=["","## Candidate geometry and cost",""]
    for L in LVALUES:
        for mode in MODES[2:]:
            o=out["arms"][f"L{L}/{mode}"]
            p=out["paired"][f"L{L}/{mode}"]
            lines.append(f"- L{L} {mode}: different from nearest-Core junction "
                         f"{100*p['different_junction_vs_core_fraction']:.1f}%, "
                         f"recent {100*p['different_recent_vs_core_fraction']:.1f}%; "
                         f"first8 native pages changed {100*p['first_eight_pages_differ_from_native_fraction']:.1f}%; "
                         f"CPU scoring recent/junction "
                         f"{o['mean_recent_score_us']:.1f}/{o['mean_junction_score_us']:.1f} µs.")
    lines+=["","## Evidence boundaries",
            "Original Starling native index and in-memory navigator are preserved, "
            "apart from the temporary optional hint interface and instrumentation.",
            "Persistent Sample2 disk hints are NOT included. Full RAM-only Core "
            "uses fixed 16K entries and online Recent512 as in BigANN port.",
            "Coveo's chronological demand and catalog size differ markedly from "
            "BigANN; cross-workload numbers are not directly substitutable.",
            "The table is equal-width and nearly-recall matched only. "
            "Interpolate at common recall targets before quoting final wins.",
            "The reported I/O counter is native Starling's logical page I/Os, "
            "not separately measured physical NVMe reads.",
            ""]
    return "\n".join(lines)


def main():
    if len(sys.argv)!=4:raise SystemExit("usage: analyzer DIAG_DIR JSON MD")
    artifact=Path(sys.argv[1]).parent
    rows,recalls=read(artifact)
    out=summarize(rows,recalls)
    Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+"\n")
    report=markdown(out)
    Path(sys.argv[3]).write_text(report)
    print("STARLING_COVEO_COVERAGE_RESULTS_VERIFIED",flush=True)
    print(report,flush=True)


if __name__=="__main__":main()
