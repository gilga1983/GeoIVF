#!/usr/bin/env python3
"""Validate six-arm native Starling per-query comparison and render a report.

Do not substitute comparisons against arbitrary search widths for recall-
matched or physical-I/O claims. All runs are the same native Starling graph,
same heldout suffix and temporary 16K + Recent512 NavHints state.
"""
from collections import defaultdict
import csv
import json
import re
import statistics
from pathlib import Path
import sys

REPS = range(3)
WIDTHS = (20,40,80)
MODES = ("baseline","core16k_recent512","diverse_core","gate50_core",
         "gate50_diverse","gate25_diverse")
EXPECTED = {(rep,mode,L) for rep in REPS for mode in MODES for L in WIDTHS}
QUERY_COUNT = 1000
BAD_ID = 4294967295


def read_csv(dir):
    all_rows={}
    for f in dir.glob("rep*-*/L*.csv"):
        m=re.fullmatch(r"rep(\d+)-(.+)",f.parent.name)
        n=re.fullmatch(r"L(\d+)\.csv",f.name)
        if not m or not n: raise ValueError(f"unexpected result filename {f}")
        key=(int(m[1]),m[2],int(n[1]))
        if key in all_rows: raise ValueError(f"duplicated result {key}")
        with f.open(newline="") as h:
            rows=list(csv.DictReader(h))
        if len(rows)!=QUERY_COUNT:
            raise ValueError(f"truncated {f}: {len(rows)}")
        if [int(row["query"]) for row in rows]!=list(range(QUERY_COUNT)):
            raise ValueError(f"unordered heldout suffix {f}")
        if any(int(row["L"])!=key[2] for row in rows):
            raise ValueError(f"mismatched search width {f}")
        if any(int(row["gate_open"]) not in (0,1) for row in rows):
            raise ValueError(f"malformed gate {f}")
        all_rows[key]=rows
    if set(all_rows)!=EXPECTED:
        raise ValueError(f"incorrect arm set missing={sorted(EXPECTED-set(all_rows))} "
                         f"extra={sorted(set(all_rows)-EXPECTED)}")
    for rep in REPS:
        for L in WIDTHS:
            baseline=all_rows[rep,"baseline",L]
            core=all_rows[rep,"core16k_recent512",L]
            if not all(int(row["gate_open"])==1 for row in baseline+core):
                raise ValueError("ungated reference unexpectedly gated")
            for mode in MODES[2:]:
                rows=all_rows[rep,mode,L]
                diverse="diverse" in mode
                if any(int(r["diverse"])!=int(diverse) for r in rows):
                    raise ValueError(f"novelty status mismatch {mode}")
                if mode.startswith("gate"):
                    for row in rows:
                        if int(row["gate_open"])==0 and (
                            int(row["recent_injected"])!=0 or
                            int(row["junction_injected"])!=0 or
                            int(row["recent_id"])!=BAD_ID or
                            int(row["junction_id"])!=BAD_ID):
                            raise ValueError("a gated query still injected/scored hints")
    return all_rows


def read_recall(art):
    out={}
    for rep in REPS:
        for mode in MODES:
            p=art/f"starling-rep{rep}-{mode}-summary.txt"
            if not p.is_file():
                raise ValueError(f"missing author native recall evidence: {p}")
            for line in p.read_text().splitlines():
                cols=line.split()
                if len(cols)<10 or cols[0] not in ("20","40","80"):
                    continue
                L=int(cols[0])
                out[rep,mode,L]=float(cols[-1])
    if set(out)!=EXPECTED:
        raise ValueError(f"incomplete native Starling recall rows {len(out)}")
    return out


def avg(items,key):
    return sum(float(r[key]) for r in items)/len(items)


def summary(rows,recalls):
    result={
      "protocol":{
        "system":"original zilliztech/starling native, pinned 17dc3e8",
        "dataset":"BigANN-10M, original uint8 base and GT",
        "query_training":"5K disjoint, junctions selected by Starling traversal",
        "causal_replay":"4K heldout warmup, last 1K measured, 3 rotated repetitions",
        "search_widths":WIDTHS,"modes":MODES,
        "gate_training":"unlabeled quantiles of native entry best-PQ-distance "
                        "from first 4K heldout warmups; no GT",
        "novelty":"top 24 PQ(query,hint) entries; reject native 4KiB pages; "
                  "min reconstructed PQ pairwise L2^2 >=0.35 * "
                  "PQ(query,best native start) L2^2",
        "memory":"identical 16K IDs and Recent512 online state; diversity uses "
                 "10 transient reconstructed PQ starts and two transient top24 lists. "
                 "Full native Starling memory navigator unchanged.",
        "fairness":"same original native graph and recall sweep, fixed search width "
                   "in tabulation; distinguish near-matched recall.",
        "measurement":"native Starling logical n_ios, per-query total_us; "
                      "NOT SSD hardware physical-read counters",
      },
      "arms":{},"paired":{},"validation":{}
    }
    for L in WIDTHS:
        for mode in MODES:
            rs=[r for rep in REPS for r in rows[rep,mode,L]]
            rec=statistics.mean(recalls[rep,mode,L] for rep in REPS)
            key=f"L{L}/{mode}"
            gated=mode.startswith("gate")
            active=sum(int(r["gate_open"]) for r in rs)/len(rs)
            added=sum(int(r["recent_injected"])+int(r["junction_injected"])
                      for r in rs)/len(rs)
            novelty=[float(r["junction_novelty"])/max(1.,float(r["native_best"]))
                     for r in rs if float(r["junction_novelty"])>=0.]
            result["arms"][key]={
              "count":len(rs),"recall10_percent":round(rec,4),
              "mean_ios":round(avg(rs,"ios"),5),
              "mean_latency_us":round(avg(rs,"total_us"),3),
              "mean_recent_us":round(avg(rs,"recent_ns")/1e3,3),
              "mean_junction_us":round(avg(rs,"junction_ns")/1e3,3),
              "active_fraction":round(active,5),
              "mean_injected_hints_per_query":round(added,5),
              "recent_pop_fraction":round(
                  sum(int(r["recent_popped"])>0 for r in rs)/len(rs),5),
              "junction_pop_fraction":round(
                  sum(int(r["junction_popped"])>0 for r in rs)/len(rs),5),
              "mean_junction_novelty_to_native_qdist":round(
                  statistics.mean(novelty),4) if novelty else None,
              "gated":gated
            }
        base=result["arms"][f"L{L}/baseline"]
        for mode in MODES[1:]:
            variant=result["arms"][f"L{L}/{mode}"]
            dif=variant["recall10_percent"]-base["recall10_percent"]
            key=f"L{L}/{mode}"
            pages_changed=0
            reads_changed=0
            for rep in REPS:
                base_rows=rows[rep,"baseline",L]
                meth=rows[rep,mode,L]
                for a,b in zip(base_rows,meth):
                    if any(a[f"p{i}"]!=b[f"p{i}"] for i in range(8)):
                        pages_changed+=1
                    reads_changed+=(a["ios"]!=b["ios"])
            result["paired"][key]={
              "mean_avoided_ios":round(base["mean_ios"]-variant["mean_ios"],5),
              "reads_reduction_percent":round(
                  100*(1-variant["mean_ios"]/base["mean_ios"]),4),
              "mean_latency_delta_us":round(
                  variant["mean_latency_us"]-base["mean_latency_us"],3),
              "latency_improvement_percent":round(
                  100*(1-variant["mean_latency_us"]/base["mean_latency_us"]),4),
              "recall10_delta_percentage_points":round(dif,5),
              "first8_page_path_changed_fraction":round(pages_changed/3000,5),
              "query_read_count_changed_fraction":round(reads_changed/3000,5),
            }
    result["validation"]["arms_complete"] = len(rows)==len(EXPECTED)
    result["validation"]["all_self_consistent"] = True
    result["validation"]["experimental_control"] = "Original Starling source and graph, "
    result["validation"]["experimental_control"] += "identical resource accounting and 4K warmup"
    return result


def md(result):
    out=["# Starling: selective activation and PQ-geometric diversity",
         "",
         "Original pinned Starling, BigANN-10M; 3 repetitions; 4K causal "
         "warmup and 1K measured; no GT used for routing or gates.",
         "",
         "The reported native I/Os are per-query logical page reads. "
         "All figures below are equal-search-width, not interpolated recall matches.",
         "",
         "| L | Arm | Active | I/Os/q | vs baseline | Latency (us) | "
         "Delta latency | Recall@10 |",
         "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for L in WIDTHS:
        for mode in MODES:
            a=result["arms"][f"L{L}/{mode}"]
            comp=result["paired"].get(f"L{L}/{mode}")
            reduction=f"{comp['reads_reduction_percent']:+.2f}%" if comp else "ref"
            latdelta=f"{comp['mean_latency_delta_us']:+.1f}" if comp else "ref"
            out.append(f"| {L} | {mode} | {100*a['active_fraction']:.1f}% | "
                       f"{a['mean_ios']:.3f} | {reduction} | "
                       f"{a['mean_latency_us']:.1f} | {latdelta} | "
                       f"{a['recall10_percent']:.2f}% |")
    out+=["","## Entry and CPU mechanisms",""]
    for L in WIDTHS:
        out.append(f"### L={L}")
        for mode in MODES[1:]:
            a=result["arms"][f"L{L}/{mode}"]
            b=result["paired"][f"L{L}/{mode}"]
            out.append(
                f"- {mode}: active {100*a['active_fraction']:.1f}%, "
                f"mean entries injected {a['mean_injected_hints_per_query']:.2f}, "
                f"recent/junction score time {a['mean_recent_us']:.1f}/"
                f"{a['mean_junction_us']:.1f} µs; "
                f"first 8 pages changed {100*b['first8_page_path_changed_fraction']:.1f}%, "
                f"recall delta {b['recall10_delta_percentage_points']:+.3f} pp.")
    out+=["","## Evidence boundaries",
          "Diversity rejects the pages containing native Starling starts and "
          "uses symmetric reconstructed PQ distances, not graph distance. "
          "The 0.35 novelty threshold, top-24 candidates and 50/25% gate policies "
          "are declared experimental choices, not tuned on the measured suffix.",
          "Only transient PQ scratch grows; Starling's existing much larger "
          "in-memory navigator is still present. RAM-only Core and Full Sample2 "
          "must not be conflated. Do not claim this is a same-RAM alternative "
          "to Starling or a comprehensive whole-system comparison.",
          "Cache warmth, same SSD lock and three-order rotation affect latency. "
          "Quantify recall-matched interpolation and noise before paper use.",
          ""]
    return "\n".join(out)


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: analyzer DIAG_DIR OUT_JSON OUT_MD")
    diag=Path(sys.argv[1])
    rows=read_csv(diag)
    recall=read_recall(diag.parent)
    result=summary(rows,recall)
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
    text=md(result)
    Path(sys.argv[3]).write_text(text)
    print("STARLING_DIVERSE_SIX_ARM_VALIDATED",flush=True)
    print(text,flush=True)


if __name__=="__main__":main()
