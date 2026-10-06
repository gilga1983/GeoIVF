#!/usr/bin/env python3
"""Paper memory frontier for progressive NavHints with retained runner-ups."""
from __future__ import annotations

import argparse, fcntl, json, os, struct, subprocess, time
from pathlib import Path
import numpy as np

THREADS, BEAM, K = 4, 8, 10
PQ_CODE_BYTES = 64
NAV_LS = (10, 20, 40, 80, 160)
BASE_LS = (10,12,14,16,18,20,22,24,28,32,36,40,44,48,56,64,72,80,88,96,112,128,144,160,176)
SPECS = (
    ("b2048-c64-p1", 2048, 64, 1),
    ("b4096-c128-p2", 4096, 128, 2),
    ("b8192-c256-p4", 8192, 256, 4),
    ("b16000-c512-p8", 16000, 512, 8),
)
RUN_TIMEOUT, HEARTBEAT = 600, 30

def save(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")

def shape(path):
    path = Path(path)
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated fbin: {path}")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"bad fbin size: {path}")
    return rows, dim

def suffix_fbin(src, dst, start):
    rows, dim = shape(src)
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remaining = (rows - start) * dim * 4
        while remaining:
            block = fin.read(min(16 << 20, remaining))
            if not block:
                raise ValueError("truncated fbin")
            fout.write(block)
            remaining -= len(block)

def result_rows(obj):
    out = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            out.append(obj)
        else:
            for v in obj.values():
                out.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(result_rows(v))
    return out

def run(binary, out, tag, queries, gt, prefix, ls, ivf=None, probe=None):
    cfg = {"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
        "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(prefix)},
        "search_phase":{"queries":str(queries),"groundtruth":str(gt),"search_list":list(ls),
        "beam_width":BEAM,"recall_at":K,"num_threads":THREADS,"is_flat_search":False,
        "distance":"inner_product","vector_filters_file":None,"num_nodes_to_cache":None,
        "search_io_limit":None,"post_processor":None}}}]}
    inp, output, logp = out/f"{tag}.input.json", out/f"{tag}.output.json", out/f"{tag}.log"
    save(inp, cfg)
    env = os.environ.copy()
    for name in ("DISKANN_SKIP_RECALL","DISKANN_STATIC_CACHE_IDS_FILE","DISKANN_HINT_IVF_FILE",
        "DISKANN_HINT_IVF_NPROBE","DISKANN_HINT_IVF_MAX_STARTS","DISKANN_PROGRESSIVE_HINTS",
        "DISKANN_PROGRESSIVE_HINT_TOPK","DISKANN_GLOBAL_START_IDS_FILE",
        "DISKANN_START_POINTS_FILE","DISKANN_QSEV_FILE","DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE","DISKANN_PAPER_CATAPULT","DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"):
        env.pop(name, None)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = str(probe)
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"
        env["DISKANN_PROGRESSIVE_HINTS"] = "1"
        env["DISKANN_PROGRESSIVE_HINT_TOPK"] = "16"
    cmd = [str(binary),"run","--input-file",str(inp),"--output-file",str(output)]
    started = time.monotonic()
    print(f"starting {tag} Ls={list(ls)}", flush=True)
    with logp.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        while True:
            try:
                rc = proc.wait(timeout=HEARTBEAT)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                load = os.getloadavg()
                print(f"heartbeat {tag} pid={proc.pid} elapsed={elapsed:.0f}s log={logp.stat().st_size}B load1={load[0]:.2f}", flush=True)
                if elapsed >= RUN_TIMEOUT:
                    proc.terminate()
                    try: proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
                    raise TimeoutError(f"{tag} exceeded {RUN_TIMEOUT}s")
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    rows = result_rows(json.loads(output.read_text()))
    by_l = {int(r["search_l"]):dict(r) for r in rows}
    if set(by_l) != set(ls):
        raise ValueError(f"{tag}: expected {list(ls)}, got {sorted(by_l)}")
    print(f"completed {tag} in {time.monotonic()-started:.1f}s", flush=True)
    return by_l

def mean(rows, key): return float(np.mean([float(r[key]) for r in rows]))
def median(rows, key): return float(np.median([float(r[key]) for r in rows]))

def summarize(rows, method, ls, payload):
    out = {}
    for l in ls:
        rr = [r for r in rows if r["method"]==method and r["L"]==l]
        out[str(l)] = {
            "rounds":len(rr),"runtime_total_payload_bytes":int(payload),
            "recall_at_10_percent":mean(rr,"recall"),"mean_ios":mean(rr,"mean_ios"),
            "median_qps":median(rr,"qps"),"median_latency_us":median(rr,"mean_latency"),
            "median_io_us":median(rr,"mean_io_time"),"median_cpu_us":median(rr,"mean_cpu_time"),
            "median_pq_preprocess_us":median(rr,"mean_pq_preprocess_time"),
            "mean_hops":mean(rr,"mean_hops"),"mean_comparisons":mean(rr,"mean_comparisons")}
    return out

def points(summary):
    out, best = [], -1e99
    for l, row in sorted((int(k),v) for k,v in summary.items()):
        rec = float(row["recall_at_10_percent"])
        if rec + 1e-9 >= best:
            out.append((l,rec,row)); best=max(best,rec)
    return out

def interp(summary, target, field, clamp=False):
    pts = points(summary)
    if target < pts[0][1]-1e-9:
        return float(pts[0][2][field]) if clamp else None
    if target > pts[-1][1]+1e-9:
        return float(pts[-1][2][field]) if clamp else None
    if target <= pts[0][1]: return float(pts[0][2][field])
    for lo, hi in zip(pts, pts[1:]):
        if lo[1] <= target <= hi[1]:
            if hi[1] <= lo[1]+1e-12: return float(hi[2][field])
            a=(target-lo[1])/(hi[1]-lo[1])
            return float(lo[2][field])+a*(float(hi[2][field])-float(lo[2][field]))
    return float(pts[-1][2][field])

def main():
    ap=argparse.ArgumentParser()
    for arg in ("binary","queries","gt","index-prefix","ivf-2048","ivf-4096","ivf-8192","ivf-16000","work","out"):
        ap.add_argument("--"+arg, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    a=ap.parse_args()
    for name in ("binary","queries","gt","index_prefix","ivf_2048","ivf_4096","ivf_8192","ivf_16000","work","out"):
        setattr(a,name,getattr(a,name).resolve())
    a.work.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if shape(a.queries)!=(10000,768): raise ValueError("unexpected query workload")
    held=a.work/"heldout.fbin"; suffix_fbin(a.queries,held,5000)

    ivfs={2048:a.ivf_2048,4096:a.ivf_4096,8192:a.ivf_8192,16000:a.ivf_16000}
    nlists={2048:64,4096:128,8192:256,16000:512}
    probes={2048:1,4096:2,8192:4,16000:8}
    payloads, manifests = {}, {}
    for budget, path in ivfs.items():
        mp=path.with_suffix(path.suffix+".manifest.json")
        manifest=json.loads(mp.read_text())
        if int(manifest["landmark_ids"])!=budget or int(manifest["nlist"])!=nlists[budget]:
            raise ValueError(f"manifest mismatch: {path}")
        manifests[str(budget)]=manifest
        payloads[budget]=path.stat().st_size+nlists[budget]*(PQ_CODE_BYTES+4)

    methods=["baseline"]+[s[0] for s in SPECS]
    rows=[]
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,set(allowed[:THREADS]))
    lockp=Path.home()/".cache/geoivf/speed-device.lock"; lockp.parent.mkdir(parents=True,exist_ok=True)
    try:
        with lockp.open("w") as lock:
            print(f"waiting for speed-device lock: {lockp}",flush=True); fcntl.flock(lock,fcntl.LOCK_EX)
            print("acquired speed-device lock",flush=True)
            for rep in range(a.reps):
                shift=rep%len(methods); order=methods[shift:]+methods[:shift]
                print(f"rep {rep}: {' '.join(order)}",flush=True)
                for method in order:
                    if method=="baseline":
                        ls, kw=BASE_LS, {}
                    else:
                        _,budget,_,probe=next(x for x in SPECS if x[0]==method)
                        ls,kw=NAV_LS,{"ivf":ivfs[budget],"probe":probe}
                    rr=run(a.binary,a.out,f"r{rep}-{method}",held,a.gt,a.index_prefix,ls,**kw)
                    for l,row in rr.items(): rows.append({"rep":rep,"method":method,"L":l,**row})
                    save(a.out/"rows.partial.json",rows)
                    r0=rr[min(rr)]
                    print(f"finished rep={rep} method={method} L{min(rr)} recall={float(r0['recall']):.3f} ios={float(r0['mean_ios']):.3f}",flush=True)
    finally:
        os.sched_setaffinity(0,set(allowed))
    save(a.out/"rows.json",rows)

    summary={"baseline":summarize(rows,"baseline",BASE_LS,0)}
    for name,budget,_,_ in SPECS:
        summary[name]=summarize(rows,name,NAV_LS,payloads[budget])

    matched={}
    for name,budget,_,_ in SPECS:
        matched[name]={}
        for l in NAV_LS:
            nav=summary[name][str(l)]; target=nav["recall_at_10_percent"]
            bio=interp(summary["baseline"],target,"mean_ios",True)
            blat=interp(summary["baseline"],target,"median_latency_us",True)
            matched[name][str(l)]={
                "recall_percent":target,"runtime_total_payload_bytes":payloads[budget],
                "navhints_mean_ios":nav["mean_ios"],"baseline_interpolated_mean_ios":bio,
                "navhints_latency_us":nav["median_latency_us"],"baseline_interpolated_latency_us":blat,
                "io_saving_percent":100*(1-nav["mean_ios"]/bio),
                "latency_saving_percent":100*(1-nav["median_latency_us"]/blat)}

    fixed={}
    for target in (25.0,35.0,45.0,55.0,65.0):
        bio=interp(summary["baseline"],target,"mean_ios")
        blat=interp(summary["baseline"],target,"median_latency_us")
        item={"baseline_mean_ios":bio,"baseline_latency_us":blat,"variants":{}}
        for name,budget,_,_ in SPECS:
            nio=interp(summary[name],target,"mean_ios"); nlat=interp(summary[name],target,"median_latency_us")
            if nio is None or nlat is None:
                item["variants"][name]={"available":False,"runtime_total_payload_bytes":payloads[budget]}
            else:
                item["variants"][name]={"available":True,"runtime_total_payload_bytes":payloads[budget],
                    "mean_ios":nio,"latency_us":nlat,
                    "io_saving_percent":100*(1-nio/bio),"latency_saving_percent":100*(1-nlat/blat)}
        fixed[str(target)]=item

    result={
        "workload":"MedRAG-Zipf heldout 5000, exact PubMed1M IP top-16 ground truth",
        "training":"ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "policy":"progressive retained shortlist, K_h=16, one start plus at most one runner-up admission after each native beam",
        "design":{"variants":[{"name":n,"hints":b,"nlist":c,"nprobe":p} for n,b,c,p in SPECS],
            "principle":"about 32 hints per cell and about 1/64 of cells probed"},
        "evaluation":{"K":K,"navhints_Ls":list(NAV_LS),"baseline_Ls":list(BASE_LS),
            "beam":BEAM,"threads":THREADS,"repetitions":a.reps},
        "runtime_payload_bytes":{str(b):payloads[b] for _,b,_,_ in SPECS},
        "ivf_manifests":manifests,"summary":summary,
        "matched_baseline_by_nav_point":matched,"fixed_recall_frontier":fixed}
    save(a.out/"paper-memory-frontier.json",result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=="__main__": main()
