#!/usr/bin/env python3
"""Regenerate final NavHints NavHints paper charts only from completed SSD artifacts.

Accept four explicit GitHub Actions ZIPs, fail closed on missing provenance or
method state, and write measured per-L recall/frontier and matched-recall
comparison CSVs. Use only baseline, junction-only and NavHints results.
Use on a local machine or self-hosted runner; no web traffic or hosted compute.
"""
from __future__ import annotations
import argparse,csv,json,math,statistics,zipfile
from pathlib import Path

METHODS=("baseline","ivf","core")
INPUTS=("coveo","pubmed","text2image","bigann")
NAMES=("Coveo","PubMed1M","Text2Image-10M","BigANN-10M")
JSON_NAMES={
 "coveo":"results/coveo-final-core.json",
 "pubmed":"results/public-final-core.json",
 "text2image":"results/public-final-core.json",
 "bigann":"results/public-final-core.json"
}

def read_zip(zip_path:Path,key:str):
    if not zip_path.is_file():raise FileNotFoundError(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        desired=JSON_NAMES[key]
        found=[x for x in z.namelist() if x==desired or x.endswith("/"+desired)]
        if len(found)!=1:
            raise ValueError(f"{key}: expected precisely one {desired}; saw {found}")
        result=json.loads(z.read(found[0]))
    if not all(q in result["summary"] for q in METHODS):
        raise ValueError(f"{key}: missing one or more final NavHints methods")
    metadata=result.get("controller",{})
    if metadata.get("cache_ids",metadata.get("recent_result_hints"))!=2048:
        raise ValueError(f"{key}: not final NavHints's 2048 destinations: {metadata}")
    if metadata.get("additional_starts",metadata.get("extra_ranked_starts"))!=8:
        raise ValueError(f"{key}: not final NavHints's eight extra starts: {metadata}")
    B=metadata.get("entry_ids",metadata.get("landmark_ids"))
    if B is None or not 0<B<=14464:
        raise ValueError(f"{key}: final NavHints junction budget violated: {metadata}")
    return result

def frontier(s):
    pts=sorted(
        [{"L":int(L),"recall":float(v["recall_percent"]),
          "reads":float(v["mean_ios"]),"latency":float(v["latency_us"])}
         for L,v in s.items()],key=lambda p:(p["recall"],p["reads"])
    )
    out=[]
    for p in pts:
        if any(not math.isfinite(p[k]) for k in ("recall","reads","latency")):
            raise ValueError("non-finite SSD result")
        if not 0<=p["recall"]<=100 or p["reads"]<0 or p["latency"]<0:
            raise ValueError("negative cost or bad recall")
        if out and p["recall"] <= out[-1]["recall"]+1e-8:
            if p["reads"]<out[-1]["reads"]:out[-1]=p
        else:out.append(p)
    return out

def interpolate(curve,t):
    if not curve or t<curve[0]["recall"]-1e-8 or t>curve[-1]["recall"]+1e-8:
        return None
    for v in curve:
        if abs(v["recall"]-t)<1e-8:
            return {"reads":v["reads"],"latency":v["latency"],"low":v["L"],"high":v["L"]}
    for a,b in zip(curve,curve[1:]):
        if a["recall"]<t<b["recall"]:
            x=(t-a["recall"])/(b["recall"]-a["recall"])
            return {"reads":a["reads"]+x*(b["reads"]-a["reads"]),
                    "latency":a["latency"]+x*(b["latency"]-a["latency"]),
                    "low":a["L"],"high":b["L"]}
    return None

def save_csv(path,columns,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in INPUTS:ap.add_argument("--"+key,type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("figures/data"))
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    details={}
    curves={}
    # Read and validate ALL four artifacts before writing ANY paper CSV.
    for idx,key in enumerate(INPUTS):
        result=read_zip(getattr(a,key),key)
        by={m:frontier(result["summary"][m]) for m in METHODS}
        targets=sorted(result["summary"]["core"].items(),key=lambda x:int(x[0]))
        rows=[]
        for l,entry in targets:
            recall=float(entry["recall_percent"])
            at={m:interpolate(by[m],recall) for m in METHODS}
            if any(v is None for v in at.values()):continue
            if at["baseline"]["reads"]<=0:raise ValueError("bad DiskANN baseline")
            B,J,C=(at[m]["reads"] for m in METHODS)
            rows.append({
                "L":int(l),"recall":recall,
                "junction":100*(1-J/B),
                "navhints":100*(1-C/B),
                "baseline_io":B,"junction_io":J,"navhints_io":C,
            })
        if len(rows)<3:
            raise ValueError(f"{key}: only {len(rows)} jointly reachable anchors; no extrapolation")
        records.append({
            "dataset_id":idx,"dataset":NAMES[idx],
            **{col:round(statistics.mean(q[col] for q in rows),4)
               for col in ("junction","navhints")},
            "anchors":len(rows),"recall_min":round(min(q["recall"] for q in rows),4),
            "recall_max":round(max(q["recall"] for q in rows),4)
        })
        details[key]={"anchors":rows,"controller":result["controller"],
                      "split":result.get("split",result.get("query_split"))}
        curves[key]=by
    save_csv(a.output/"eval_components.csv",
             ["dataset_id","dataset","junction","navhints",
              "anchors","recall_min","recall_max"],records)
    # Export the measured search-list frontiers for all four Figure 6 panels.
    for key in INPUTS:
        for method in METHODS:
            name={"baseline":"baseline","ivf":"junction","core":"core","full":"full"}[method]
            arr=curves[key][method]
            save_csv(a.output/f"eval_frontier_{key}_{name}.csv",
                     ["recall","ssd_reads","latency_us"],
                     [{"recall":round(p["recall"],6),"ssd_reads":round(p["reads"],6),
                       "latency_us":round(p["latency"],4)} for p in arr])
    (a.output/"eval_final_core_manifest.json").write_text(
       json.dumps({"source":"NavHints versus baseline and junction-only routing at NavHints-achieved matched-recall targets",
                   "dataset_evidence":details,"components":records},indent=2)+"\n")
    print("FINAL_CORE_FOUR_DATASET_PAPER_PLOTS_VALIDATED")
    for r in records:print(json.dumps(r))

if __name__=="__main__":main()
