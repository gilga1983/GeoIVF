#!/usr/bin/env python3
"""Exploratory high-recall mechanism diagnosis: NavHints top-K vs Catapult bucket cap.

Results are fixed-L read, recall, CPU evidence. The baseline and treatments
share one pinned SSD graph, dataset, query suffix and PQ. Catapult arms start
from the SAME strictly causal pre-history snapshot for a given L, then
update the LRU during the 4k causal warm-up and 1k measured suffix. NavHints
starts with the same learned 16k entries and empty Recent512 for all K values.

No statistical claims from one seed / one ordering; the run is mechanism
diagnosis, not a substitute for the 3x3 accepted head-to-head.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time

from qualify_navhints_catapult_headtohead import (
    run_one, slice_fbin, shape, save, rows_of, snapshot_entries,
    TRAIN, WARM, MEASURE, DIM, K, BEAM, SEARCH_THREADS,
)
from qualify_frozen_paper_competitors import slice_gt

LS = (20, 40, 80, 160, 320)
NAV_KS = (1, 2, 4, 8, 10)
CAT_CAPS = (1, 4, 16, "all")
CAT_SEED = 1


def baseline_run(binary, queries, gt, index, L, out):
    inp = out / f"baseline-L{L}.input.json"
    op = out / f"baseline-L{L}.output.json"
    log = out / f"baseline-L{L}.log"
    save(inp, {
        "search_directories": [str(queries.parent)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": str(index),
                },
                "search_phase": {
                    "queries": str(queries),
                    "groundtruth": str(gt),
                    "search_list": [L],
                    "beam_width": BEAM, "recall_at": K,
                    "num_threads": SEARCH_THREADS,
                    "is_flat_search": False, "distance": "inner_product",
                    "vector_filters_file": None, "num_nodes_to_cache": None,
                    "search_io_limit": None, "post_processor": None,
                },
            },
        }],
    })
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISKANN_")}
    with log.open("w") as stream:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(op)],
            env=env, stdout=stream, stderr=subprocess.STDOUT,
            check=True, timeout=1800,
        )
    found = list(rows_of(json.loads(op.read_text())))
    if len(found) != 1 or int(found[0]["search_l"]) != L:
        raise RuntimeError("baseline: missing L")
    o = found[0]
    return {"reads": float(o["mean_ios"]), "recall": float(o["recall"]),
            "latency_us": float(o["mean_latency"]),
            "cpu_us": float(o["mean_cpu_time"])}


def frontier(rows):
    p = sorted(({"L": L, **v} for L, v in rows.items()), key=lambda o: (o["recall"], o["reads"]))
    out = []
    for row in p:
        if out and abs(row["recall"]-out[-1]["recall"]) < 1e-6:
            if row["reads"] < out[-1]["reads"]:
                out[-1] = row
        else:
            out.append(row)
    return out


def interp(pts, target):
    if target < pts[0]["recall"]-1e-6 or target > pts[-1]["recall"]+1e-6:
        return None
    for a in pts:
        if abs(a["recall"]-target) < 1e-6:
            return {"reads": a["reads"], "lower_L": a["L"], "upper_L": a["L"]}
    for a,b in zip(pts,pts[1:]):
        if a["recall"] <= target <= b["recall"]:
            frac=(target-a["recall"])/(b["recall"]-a["recall"])
            return {"reads":a["reads"]+frac*(b["reads"]-a["reads"]),
                    "lower_L":a["L"],"upper_L":b["L"]}
    return None


def report(result):
    by_arm={}
    for L in LS:
        for arm,row in result["results"][str(L)].items():
            by_arm.setdefault(arm,{})[L]=row
    original = {L:result["results"][str(L)]["nav-k1"] for L in LS}
    results=[]
    for L in LS:
        original_row=original[L]
        target=original_row["recall"]
        old_reads=original_row["reads"]
        a={"L":L,"nav_k1_recall":target,"nav_k1_reads":old_reads,"arms":{}}
        for arm,pts in sorted(by_arm.items()):
            same_L=pts[L]
            matched=interp(frontier(pts),target)
            a["arms"][arm]={
                "same_L_read_difference":same_L["reads"]-old_reads,
                "same_L_recall_difference":same_L["recall"]-target,
                "fixed_L_read_saving_percent":100*(1-same_L["reads"]/old_reads),
                "matched_recall":None if matched is None else {
                    **matched,
                    "nav_k1_reads_saved_percent":100*(1-old_reads/matched["reads"]),
                }
            }
        results.append(a)
    return results


def main():
    ap=argparse.ArgumentParser()
    for q in ("nav-bin","catapult-bin","queries","train-gt","heldout-gt",
              "index-prefix","ivf","work","out"):
        ap.add_argument("--"+q,type=Path,required=True)
    a=ap.parse_args()
    for q in ("nav_bin","catapult_bin","queries","train_gt","heldout_gt",
              "index_prefix","ivf","work","out"):
        setattr(a,q,getattr(a,q).resolve())
    if shape(a.queries)!=(TRAIN+WARM+MEASURE,DIM):
        raise RuntimeError("wrong shared frozen query stream")
    for item in (a.nav_bin,a.catapult_bin,a.queries,a.train_gt,a.heldout_gt,a.ivf):
        if not item.is_file():
            raise FileNotFoundError(item)
    a.work.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    query_train=slice_fbin(a.queries,a.work/"train5000.fbin",0,TRAIN)
    query_held=slice_fbin(a.queries,a.work/"heldout5000.fbin",TRAIN,WARM+MEASURE)
    query_test=slice_fbin(query_held,a.work/"test1000.fbin",WARM,MEASURE)
    truth_test=a.work/"test1000.gt"
    slice_gt(a.heldout_gt,truth_test,WARM,MEASURE)

    # Identical auxiliary routing state regardless of number of injected IDs.
    manifest=json.loads(
        a.ivf.with_suffix(a.ivf.suffix+".manifest.json").read_text()
    )
    nav_memory=a.ivf.stat().st_size+int(manifest["nlist"])*(64+4)+512*4
    cat_memory=(1<<8)*80*4+8*DIM*4
    if nav_memory != 102940 or cat_memory != 106496:
        raise RuntimeError(f"unexpected RAM contract: nav {nav_memory}, cat {cat_memory}")

    allowed=sorted(os.sched_getaffinity(0))
    if len(allowed)<SEARCH_THREADS:
        raise RuntimeError("need 4 CPU cores")
    os.sched_setaffinity(0,set(allowed[:SEARCH_THREADS]))
    results={}
    training={}
    try:
        for i,L in enumerate(LS):
            name=f"train-cat-L{L}-s{CAT_SEED}"
            snapshot=a.work/(name+".snapshot")
            run_one(
                a.catapult_bin,name,query_train,a.train_gt,a.index_prefix,L,a.out,
                mode="catapult",seed=CAT_SEED,snapshot_dump=snapshot,train=True,
            )
            sz=snapshot_entries(snapshot)
            if not 0<sz <= 256*80:
                raise RuntimeError("unexpected Catapult snapshot occupancy")
            training[str(L)]={"entries":sz,"sha256":hashlib.sha256(snapshot.read_bytes()).hexdigest()}
            save(a.out/"training.partial.json",training)
            arms=[f"nav-k{k}" for k in NAV_KS]+[
                f"cat-cap{c}" for c in CAT_CAPS
            ]+["baseline"]
            arms=arms[i%len(arms):]+arms[:i%len(arms)]
            results[str(L)]={}
            for arm in arms:
                print(f"DIAGNOSTIC L={L} arm={arm}",flush=True)
                name=f"diag-L{L}-{arm}"
                if arm=="baseline":
                    data=baseline_run(a.catapult_bin,query_test,truth_test,a.index_prefix,L,a.out)
                elif arm.startswith("nav-k"):
                    k=int(arm.split("nav-k")[1])
                    data=run_one(
                        a.nav_bin,name,query_held,a.heldout_gt,
                        a.index_prefix,L,a.out,mode="nav",ivf=a.ivf,
                        extra_env={"DISKANN_DIAG_NAV_RECENT_STARTS":k},
                    )
                else:
                    cap=arm.split("cat-cap")[1]
                    env={} if cap=="all" else {"DISKANN_DIAG_CATAPULT_BUCKET_STARTS":cap}
                    data=run_one(
                        a.catapult_bin,name,query_held,a.heldout_gt,
                        a.index_prefix,L,a.out,mode="catapult",seed=CAT_SEED,
                        snapshot_load=snapshot,extra_env=env,
                    )
                results[str(L)][arm]=data
                save(a.out/"results.partial.json",results)
    finally:
        os.sched_setaffinity(0,set(allowed))
    out={
        "status":"exploratory mechanism ablation, NOT publication-grade repetitions",
        "graph":"pinned Microsoft DiskANN SSD, PubMed1M MedCPT, IP, K=10, beam=8",
        "protocol":"same 5k static-history/4k causal warm-up/1k measured",
        "catapult_training":"identical default-full-seed Catapult LRU snapshot per L; online updates after warm-up",
        "navhints":"static 16k plus recent512, same PQ scoring of 512 IDs for each K",
        "catapult":"one representative seed, original full candidate list vs newest 1/4/16 bucket entries",
        "repetitions":1,
        "catapult_seed":CAT_SEED,
        "widths":list(LS),
        "nav_ks":list(NAV_KS),
        "cat_bucket_caps":list(CAT_CAPS),
        "state_payload_bytes":{"navhints":nav_memory,"catapult":cat_memory},
        "training_snapshots":training,
        "results":results,
    }
    out["matched_recall_diagnostics"]=report(out)
    save(a.out/"highrecall-selector-diagnostics.json",out)
    print("NAVHINTS_CATAPULT_SELECTOR_ABLATIONS_COMPLETE",flush=True)
    for item in out["matched_recall_diagnostics"]:
        print(f"L={item['L']} nav_rec={item['nav_k1_recall']:.2f} nav_reads={item['nav_k1_reads']:.2f}",flush=True)
        for name,x in item["arms"].items():
            print(f"  {name:11} same_L_delta_reads={x['same_L_read_difference']:+.3f} "
                  f"same_L_delta_recall={x['same_L_recall_difference']:+.3f}",flush=True)


if __name__=="__main__":
    main()
