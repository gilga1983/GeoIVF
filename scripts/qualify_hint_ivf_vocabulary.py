#!/usr/bin/env python3
"""Evaluate whether indexed NavHints benefits from a larger learned vocabulary."""
from __future__ import annotations

import argparse, fcntl, json, os, struct, subprocess
from pathlib import Path
import numpy as np

THREADS = 4
IO_BEAM = 8
K = 1
PROBES = (8, 16, 32)
BUDGETS = (16000, 20000)


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path):
    path = Path(path)
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def suffix_fbin(src, dst, start):
    rows, dim = fbin_shape(src)
    with Path(src).open("rb") as fin, Path(dst).open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", rows - start, dim))
        remaining = (rows - start) * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)


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


def run_one(binary, out, tag, queries, gt, index_prefix, *, starts=None, ivf=None, probe=None):
    phase = {
        "queries": str(queries), "groundtruth": str(gt), "search_list": [K],
        "beam_width": IO_BEAM, "recall_at": K, "num_threads": THREADS,
        "is_flat_search": False, "distance": "inner_product",
        "vector_filters_file": None, "num_nodes_to_cache": None,
        "search_io_limit": None, "post_processor": None,
    }
    cfg = {"search_directories": [str(out)], "jobs": [{
        "type": "disk-index", "content": {
            "source": {"disk-index-source": "Load", "data_type": "float32",
                       "load_path": str(index_prefix)},
            "search_phase": phase,
        }
    }]}
    inp = Path(out) / f"{tag}-input.json"
    output = Path(out) / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_GLOBAL_START_IDS_FILE", "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE", "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT", "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE", "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
        "DISKANN_HINT_IVF_FILE", "DISKANN_HINT_IVF_NPROBE",
        "DISKANN_HINT_IVF_MAX_STARTS",
    ):
        env.pop(name, None)
    if starts is not None:
        env["DISKANN_GLOBAL_START_IDS_FILE"] = str(starts)
    if ivf is not None:
        env["DISKANN_HINT_IVF_FILE"] = str(ivf)
        env["DISKANN_HINT_IVF_NPROBE"] = str(probe)
        env["DISKANN_HINT_IVF_MAX_STARTS"] = "1"

    with (Path(out) / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
        )
    rr = result_rows(json.loads(output.read_text()))
    if len(rr) != 1 or float(rr[0]["recall"]) < 0:
        raise ValueError(f"{tag}: invalid result")
    return dict(rr[0])


def avg(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


def med(rows, key):
    return float(np.median([float(r[key]) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--landmark-root", type=Path, required=True)
    ap.add_argument("--ivf-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    for name in ("binary","queries","gt","index_prefix","landmark_root","ivf_root","work","out"):
        setattr(args, name, getattr(args, name).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if fbin_shape(args.queries) != (10000, 768):
        raise ValueError("unexpected query workload")
    held = args.work / "heldout.fbin"
    suffix_fbin(args.queries, held, 5000)

    landmarks = {b: args.landmark_root / f"landmarks-b{b}.bin" for b in BUDGETS}
    ivfs = {b: args.ivf_root / f"hints-b{b}-nlist512-spherical.bin" for b in BUDGETS}
    for p in [*landmarks.values(), *ivfs.values()]:
        if not p.is_file():
            raise FileNotFoundError(p)

    methods = ["baseline-medoid", "flat-b16000", "flat-b20000"]
    for b in BUDGETS:
        methods.extend(f"ivf-b{b}-p{p}" for p in PROBES)

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    rows = []
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for rep in range(args.reps):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for method in order:
                    kw = {}
                    if method.startswith("flat-b"):
                        b = int(method.split("b")[1])
                        kw["starts"] = landmarks[b]
                    elif method.startswith("ivf-b"):
                        left, rawp = method.rsplit("-p", 1)
                        b = int(left.split("-b")[1])
                        kw["ivf"] = ivfs[b]
                        kw["probe"] = int(rawp)
                    row = run_one(args.binary, args.out, f"r{rep}-{method}",
                                  held, args.gt, args.index_prefix, **kw)
                    rows.append({"rep": rep, "method": method, **row})
                    save(args.out / "rows.partial.json", rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", rows)
    manifests = {
        str(b): json.loads(
            ivfs[b].with_suffix(ivfs[b].suffix + ".manifest.json").read_text()
        ) for b in BUDGETS
    }
    summary = {}
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        state = 0
        est = None
        if method.startswith("flat-b"):
            b = int(method.split("b")[1])
            state = landmarks[b].stat().st_size
            est = b
        elif method.startswith("ivf-b"):
            left, rawp = method.rsplit("-p", 1)
            b = int(left.split("-b")[1])
            p = int(rawp)
            state = ivfs[b].stat().st_size
            m = manifests[str(b)]
            est = float(m["nlist"] + p * m["bucket_children_mean"])
        summary[method] = {
            "rounds": len(rr), "state_bytes": int(state),
            "estimated_router_pq_scores": est,
            "recall_percent": avg(rr, "recall"),
            "mean_ios": avg(rr, "mean_ios"),
            "median_qps": med(rr, "qps"),
            "median_latency_us": med(rr, "mean_latency"),
            "median_cpu_us": med(rr, "mean_cpu_time"),
        }

    result = {
        "workload": "MedRAG-Zipf heldout 5000, exact PubMed1M IP ground truth",
        "training": "ordinary DiskANN medoid teacher L=4 on first 5000 queries",
        "online_search": {"K":1,"L":1,"beam":IO_BEAM,"threads":THREADS},
        "budgets": list(BUDGETS), "nlist": 512, "probes": list(PROBES),
        "manifests": manifests, "summary": summary,
    }
    save(args.out / "hint-ivf-vocabulary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
