#!/usr/bin/env python3
"""Recover completed BigANN 10M memory-sweep measurements without rerunning SSD jobs.

Original archived Actions #37983319473 completed every arm over three
rotated repetitions; its *legacy PubMed-specific* postprocessing crashed on
a missing L=12, after writing results/runs.partial.json. This script reads
that archived JSON directly, keeps one fixed and explicitly common recall
anchor set (NavHints Core L={20,40,80,160,320}), and never extrapolates.

Example:
  python3 scripts/recover_bigann_memory_sweep.py \
    --artifact /path/to/navhints-bigann-memory.zip --output-dir /tmp/bigann-report
"""
from __future__ import annotations
import argparse,csv,json,statistics
from pathlib import Path
from zipfile import ZipFile

LS=(10,20,40,80,160,320,640)
ANCHORS=(20,40,80,160,320)
METHODS=("baseline","navhints-core","hot-256","hot-1024","hot-4096",
         "qsev-200","qsev-800","qsev-3200","cat-7-192","cat-8-96",
         "cat-7-800","cat-8-396","cat-7-3200","cat-8-1600")
MEMORY={"navhints-core":102940,"baseline":0}
MEMORY.update({f"hot-{n}":384*n for n in (256,1024,4096)})
MEMORY.update({f"qsev-{n}":16+(128*4+4)*n for n in (200,800,3200)})
MEMORY.update({f"cat-{h}-{cap}":h*128*4+(1<<h)*cap*4
               for h,caps in ((7,(192,800,3200)),(8,(96,396,1600)))
               for cap in caps})

def load(path:Path)->dict:
    if path.suffix.lower()==".zip":
        with ZipFile(path) as z: return json.loads(z.read("results/runs.partial.json"))
    return json.loads(path.read_text())

def aggregate(arms:dict)->dict:
    if set(arms)!=set(METHODS):raise ValueError("Unexpected method set, refusing posthoc exclusion")
    result={}
    for method in METHODS:
        reps=arms[method]
        if len(reps)!=3:raise ValueError(f"{method}: need three completed repetitions")
        result[method]={}
        for L in LS:
            points=[]
            for rep in reps:
                rows=[r for r in rep if int(r["search_l"])==L]
                if len(rows)!=1: raise ValueError(f"{method}: missing/duplicated L={L}")
                points.extend(rows)
            result[method][str(L)]={
                "recall":statistics.mean(float(r["recall"]) for r in points),
                "reads":statistics.mean(float(r["mean_ios"]) for r in points),
                "latency_us":statistics.median(float(r["mean_latency"]) for r in points),
                "cache_hit_percent":statistics.mean(float(r.get("cache_hit_percentage",0)) for r in points)
            }
    for n in (256,1024,4096):
        cache=result[f"hot-{n}"]["160"]
        base=result["baseline"]["160"]
        if cache["cache_hit_percent"]<=0.05 or cache["reads"]>=base["reads"]:
            raise ValueError(f"hot-{n}: real cache-hit or provider-I/O saving not verified")
    return result

def interpolate(curve:dict, target:float, key:str)->float|None:
    items=[]; best=-1.e30
    for L in LS:
        p=curve[str(L)]
        if p["recall"]+1.e-9>=best:
            items.append((p["recall"],p));best=max(best,p["recall"])
    if not items or target<items[0][0] or target>items[-1][0]:
        return None
    for (a,x),(b,y) in zip(items,items[1:]):
        if a<=target<=b:
            w=0.0 if b<=a+1.e-12 else (target-a)/(b-a)
            return x[key]+w*(y[key]-x[key])
    return items[-1][1][key]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    summary=aggregate(load(args.artifact))
    anchor_rows=[]
    methods=[]
    for name in METHODS:
        entries=[]
        for L in ANCHORS:
            nav=summary["navhints-core"][str(L)]
            target=nav["recall"]
            io=interpolate(summary[name],target,"reads")
            latency=interpolate(summary[name],target,"latency_us")
            if io is None or latency is None:
                raise ValueError(f"{name}: missing common recall target {target:.5f}%")
            item={"method":name,"anchor_L":L,"target_recall_percent":target,
                  "core_reads":nav["reads"],"reference_reads":io,
                  "core_io_saving_percent":100*(1-nav["reads"]/io),
                  "core_latency_us":nav["latency_us"],"reference_latency_us":latency,
                  "core_latency_saving_percent":100*(1-nav["latency_us"]/latency)}
            entries.append(item);anchor_rows.append(item)
        methods.append({
            "method":name,"aux_memory_bytes":MEMORY[name],"anchor_count":len(entries),
            "core_read_saving_percent":statistics.mean(x["core_io_saving_percent"] for x in entries),
            "core_latency_saving_percent":statistics.mean(x["core_latency_saving_percent"] for x in entries),
            "io_saving_range_percent":[min(x["core_io_saving_percent"] for x in entries),
                                       max(x["core_io_saving_percent"] for x in entries)],
        })
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={
        "source":"GeoIVF run 37983319473 artifact 11645367589; three measured repetitions recovered from runs.partial.json",
        "dataset":"BigANN-10M uint8/squared L2, first 5K static training, next 4K causal warm-up, last 1K measured",
        "methodological_boundary":"independently reproduced QSEV and Catapult entry components; no full original author systems",
        "status":"physical benchmark complete; legacy PubMed-specific postprocessor failed; this is analysis-only recovery",
        "anchors":[summary["navhints-core"][str(L)]["recall"] for L in ANCHORS],
        "same_graph":True,"repetitions":3,"reference_memory_payload_before_container_metadata":True,
        "results":methods,"per_anchor":anchor_rows,
    }
    (args.output_dir/"bigann-memory-recovered.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output_dir/"bigann-memory-recovered.csv").open("w",newline="") as f:
        fields=("method","aux_memory_bytes","anchor_count","core_read_saving_percent","core_latency_saving_percent")
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows({k:v[k] for k in fields} for v in methods)
    print("BIGANN_MEMORY_THREE_REPETITIONS_RECOVERED",flush=True)
    for row in methods:
        print(f"{row['method']}: anchors={row['anchor_count']} "
              f"RAM={row['aux_memory_bytes']}B "
              f"Core SSD-read saving={row['core_read_saving_percent']:+.2f}%")
if __name__=="__main__": main()
