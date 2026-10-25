#!/usr/bin/env python3
"""Regenerate NavHints Core warm-up, locality, shift and persistence CSVs.

Requires only successful frozen-SSD experiment artifacts from the chosen
B=14,464/C=2,048/k=8 Core. Fail closed on absent or invalid inputs. Legacy
small-history results remain untouched if any campaign artifact is missing.
"""
import argparse,csv,json,math,zipfile
from pathlib import Path

FILES={
 "warmup":"warmup/final-core-warmup.json",
 "locality":"locality/final-core-locality.json",
 "shift":"shift/final-core-shift.json",
 "writes":"results/final-core-write-sweep.json",
}

def load(path,inside):
    if not path.is_file():raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as z:
        found=[n for n in z.namelist() if n==inside or n.endswith("/"+inside)]
        if len(found)!=1:raise ValueError(f"{path} requires one {inside}; saw {found}")
        return json.loads(z.read(found[0]))

def emit(path,cols,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        w=csv.DictWriter(file,fieldnames=cols,lineterminator="\n")
        w.writeheader();w.writerows(rows)

def numerical(x,label):
    if x is None or not math.isfinite(float(x)):
        raise ValueError(f"missing numerical result: {label}")
    return round(float(x),5)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in FILES:p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("figures/data"))
    a=p.parse_args()
    data={k:load(getattr(a,k),v) for k,v in FILES.items()}

    warm=sorted(data["warmup"]["history"],key=lambda r:r["history_queries"])
    if set(int(r["history_queries"]) for r in warm)!={0,250,500,1000,2000,3000,4000}:
        raise ValueError("incomplete original warm-up experiment grid")
    warmrows=[{"history":int(r["history_queries"]),
               "core":numerical(r["core_vs_ivf"]["mean_io_saving_percent"],"Core warmup"),
               "full":numerical(r["sample2_vs_ivf"]["mean_io_saving_percent"],"Full warmup")}
              for r in warm]

    pools=sorted(data["locality"]["pools"],key=lambda r:r["mean_requests_per_query"])
    if set(int(r["working_set_queries"]) for r in pools)!={5000,2000,1000,500,100}:
        raise ValueError("incomplete 5-pool locality sweep")
    localrows=[{"working_set":int(r["working_set_queries"]),
               "repeats":numerical(r["mean_requests_per_query"],"repeats"),
               "core":numerical(r["core_vs_ivf"]["io_saving_percent"],"Core locality"),
               "full":numerical(r["sample2_vs_ivf"]["io_saving_percent"],"Full locality")}
              for r in pools]

    shift=data["shift"]["groups"]
    if len(shift)<2:raise ValueError("missing near/far train-support comparison")
    shiftrows=[{
      "group":r["group"],
      "queries":int(r["queries"]),
      "mean_support_similarity":numerical(r["mean_support_similarity"],"train support"),
      "core_vs_junction":numerical(r["core_vs_ivf"]["io_saving_percent"],"Core shift"),
      "full_vs_junction":numerical(r["sample2_vs_ivf"]["io_saving_percent"],"Full shift")
    } for r in shift]

    wdata=data["writes"]
    if wdata["controller"].get("B")!=14464 or wdata["controller"].get("C")!=2048 \
        or wdata["controller"].get("k")!=8:
        raise ValueError("write frontier does not match final Core")
    cmp=wdata["comparison"]
    rows=[]
    for label,method,denom in [
        ("Core","core",0),("Sample1","sample1",1),
        ("Sample2","sample2",2),("Sample4","sample4",4),
        ("Sample10","sample10",10),
        ("Sample2 without empty-slot seeding","nofill2",2)]:
        e=cmp[method]
        rows.append({
            "method":label,"admission_denominator":denom,
            "n_anchors":int(e["matched_anchor_count"]),
            "reads_saved_vs_junction_percent":numerical(
                e["reads_saved_vs_junction_percent"],"write savings"),
            "reads_saved_vs_core_percent":numerical(
                e["reads_saved_vs_core_percent"],"write increment"),
            "L160_eval_logical_writes_per_query":numerical(
                e["measured_L160_logical_writes_per_query"],"writes")
        })
    if len({r["n_anchors"] for r in rows})!=1 or rows[0]["n_anchors"]<3:
        raise ValueError("inconsistent common-recall read/write frontier anchors")
    emit(a.output/"eval_warmup.csv",["history","core","full"],warmrows)
    emit(a.output/"eval_locality.csv",["working_set","repeats","core","full"],localrows)
    emit(a.output/"eval_final_core_shift.csv",
         ["group","queries","mean_support_similarity","core_vs_junction","full_vs_junction"],shiftrows)
    emit(a.output/"eval_final_core_persistence.csv",
         ["method","admission_denominator","n_anchors",
          "reads_saved_vs_junction_percent","reads_saved_vs_core_percent",
          "L160_eval_logical_writes_per_query"],rows)
    print("FINAL_CORE_SENSITIVITY_PLOTS_VALIDATED")
    print(json.dumps({"warmup":warmrows,"locality":localrows,"shift":shiftrows,"write_frontier":rows},indent=2))

if __name__=="__main__":main()
