#!/usr/bin/env python3
"""Extend ONLY BigANN native BFS graph-cache measurements to L=640.

Keeps the selected (B,C,k)=(14464,2048,8) NavHints reference and the
previous 3x measured BFS results frozen. Recovers the fifth Table 8
matched-recall target (~99.97%) without extrapolating. All runs use
the pinned native DiskANN benchmark with a source-aligned graph-cache patch,
the identical 1K heldout measured queries, and real SSD-provider read counts.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import subprocess

CAPACITIES = (268, 1072, 4289)
SEARCH_LIST = (640,)
REPS = 3
BUDGET = 102940
TARGET_WIDTHS = ("20", "40", "80", "160", "320")

def write_json(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, indent=2, sort_keys=True) + "\n")

def slice_xbin(src, dst, start, count):
    with src.open("rb") as fin:
        n, dim = struct.unpack("<II", fin.read(8))
        if (n, dim) != (5000, 128):
            raise ValueError(f"Unexpected BigANN heldout query shape {(n, dim)}")
        fin.seek(8 + start * dim)
        chunk = fin.read(count * dim)
        if len(chunk) != count * dim:
            raise ValueError("Truncated uint8 heldout query array")
    with dst.open("wb") as fout:
        fout.write(struct.pack("<II", count, dim))
        fout.write(chunk)

def slice_gt(src, dst, start, count):
    with src.open("rb") as f:
        n, k = struct.unpack("<II", f.read(8))
    if n != 5000 or k < 10:
        raise ValueError(f"Unexpected BigANN ground truth shape {(n, k)}")
    row_bytes = 4 * k
    size = src.stat().st_size
    has_dist = size == 8 + 2 * n * row_bytes
    if size not in (8 + n * row_bytes, 8 + 2 * n * row_bytes):
        raise ValueError("Unexpected BigANN GT byte length")
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<II", count, k))
        fin.seek(8 + start * row_bytes)
        ids = fin.read(count * row_bytes)
        if len(ids) != count * row_bytes:
            raise ValueError("Truncated BigANN GT IDs")
        fout.write(ids)
        if has_dist:
            fin.seek(8 + n * row_bytes + start * row_bytes)
            d = fin.read(count * row_bytes)
            if len(d) != count * row_bytes:
                raise ValueError("Truncated BigANN GT distances")
            fout.write(d)

def result_rows(value):
    if isinstance(value, dict):
        if "search_l" in value and "mean_latency" in value:
            return [value]
        return [r for v in value.values() for r in result_rows(v)]
    if isinstance(value, list):
        return [r for v in value for r in result_rows(v)]
    return []

def interpolate(curve, recall):
    rows = sorted((float(p["recall_percent"]), float(p["mean_ios"])) for p in curve.values())
    if not rows or recall < rows[0][0] - 1e-6 or recall > rows[-1][0] + 1e-6:
        return None
    for r, v in rows:
        if abs(r - recall) < 1e-6:
            return v
    for (r0, c0), (r1, c1) in zip(rows, rows[1:]):
        if r0 < recall < r1 and r1 > r0:
            return c0 + (recall - r0) * (c1 - c0) / (r1 - r0)
    return None

def run(binary, index, queries, gt, out, rep, arm):
    cap = int(arm.split("-")[1]) if arm.startswith("bfs-") else None
    prefix = f"r{rep}-{arm}-L640"
    job = {
        "search_directories": [str(out)],
        "jobs": [{"type": "disk-index", "content": {
            "source": {"disk-index-source": "Load", "data_type": "uint8",
                       "load_path": str(index)},
            "search_phase": {
                "queries": str(queries), "groundtruth": str(gt),
                "search_list": [640], "beam_width": 8, "recall_at": 10,
                "num_threads": 4, "is_flat_search": False,
                "distance": "squared_l2", "vector_filters_file": None,
                "num_nodes_to_cache": cap, "search_io_limit": None,
                "post_processor": None
            }
        }}]
    }
    inp, output, log = (out / f"{prefix}.{ext}" for ext in
                        ("input.json", "output.json", "log"))
    write_json(inp, job)
    env = {k: v for k, v in os.environ.items() if not k.startswith("DISKANN_")}
    with log.open("w") as f:
        subprocess.run([str(binary), "run", "--input-file", str(inp),
                        "--output-file", str(output)], env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1 or int(rows[0]["search_l"]) != 640:
        raise ValueError(f"Incomplete L640 results in {output}")
    x = rows[0]
    if cap is not None:
        if float(x.get("cache_hit_percentage", 0.0)) <= 0:
            raise ValueError(f"{prefix}: native BFS graph cache has no hits")
        if not 0 <= float(x["mean_ios"]) <= float(x["mean_hops"]) + 1e-5:
            raise ValueError(f"{prefix}: invalid uncached SSD-provider read count")
    if not 0 <= float(x["recall"]) <= 100.000001:
        raise ValueError(f"{prefix}: invalid recall")
    return x

def main():
    ap = argparse.ArgumentParser()
    for k in ("binary", "index-prefix", "queries", "gt", "original-json", "out"):
        ap.add_argument("--" + k, required=True, type=Path)
    ap.add_argument("--ivf", required=True, type=Path)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads(a.original_json.read_text())
    if original.get("dataset") != "bigann-10M" or original.get("reps") != REPS:
        raise ValueError("Wrong original selected-core BigANN artifact")
    if original.get("selected_core") != {
        "B": 14464, "C": 2048, "k": 8,
        "regions": 512, "persistent": False
    }:
        raise ValueError("Not the frozen selected NavHints configuration")
    if original["method_budgets"]["core_payload_bytes"] != BUDGET:
        raise ValueError("Wrong routing payload in original")
    if tuple(original["method_budgets"]["bfs_nodes"]) != CAPACITIES:
        raise ValueError("Inconsistent original BigANN BFS cache capacities")
    if a.ivf.stat().st_size + 512 * (64 + 4) + 2048 * 4 != BUDGET:
        raise ValueError("Mismatching BigANN hint payload, dataset may differ")
    # The frozen source artifact's original BigANN IVF has this SHA256.
    if hashlib.sha256(a.ivf.read_bytes()).hexdigest() != (
            "76149d8cdeda5ed78d952d62a247589fd88e8439c9c018ab5199020084a1716e"):
        raise ValueError("BigANN IVF identity differs from the original run")
    query = out / "measured1000.input"
    gt = out / "measured1000.gt"
    slice_xbin(a.queries, query, 4000, 1000)
    slice_gt(a.gt, gt, 4000, 1000)

    targets = [float(original["summary"]["core"][L]["recall_percent"])
               for L in TARGET_WIDTHS]
    if not (94.0 < targets[0] < 94.1 and 99.96 < targets[-1] < 99.99):
        raise ValueError("Unexpected selected-Core anchor grid")
    for cap in CAPACITIES:
        k = f"bfs-{cap}"
        prev = original["matched_core_anchor"][k]
        if prev["count"] != 4:
            raise ValueError(f"Unexpected historical frontier for {k}: {prev}")
        if original["summary"][k]["320"]["recall_percent"] >= targets[-1]:
            raise ValueError("High-recall extension is not needed")

    arms = ["baseline"] + [f"bfs-{cap}" for cap in CAPACITIES]
    data = {m: [] for m in arms}
    affinity = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, set(affinity[:4]))
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for rep in range(REPS):
            order = arms[rep % len(arms):] + arms[:rep % len(arms)]
            for arm in order:
                row = run(a.binary, a.index_prefix, query, gt, out, rep, arm)
                data[arm].append(row)
                write_json(out / "l640-partial-raw.json", data)

    means = {arm: {
        "recall_percent": statistics.mean(float(x["recall"]) for x in arr),
        "mean_ios": statistics.mean(float(x["mean_ios"]) for x in arr),
        "mean_hops": statistics.mean(float(x["mean_hops"]) for x in arr),
        "cache_hit_percentage": statistics.mean(
            float(x.get("cache_hit_percentage", 0)) for x in arr),
    } for arm, arr in data.items()}
    comparison = {}
    for cap in CAPACITIES:
        arm = f"bfs-{cap}"
        curve = dict(original["summary"][arm])
        if "640" in curve:
            raise ValueError("Unexpected repeated L640 in frozen curve")
        curve["640"] = means[arm]
        details = []
        for r in targets:
            base = interpolate(original["summary"]["core"], r)
            alt = interpolate(curve, r)
            if base is None or alt is None:
                raise ValueError(f"{arm}: unsupported original target {r:.6f}")
            details.append({"recall": r, "core_reads": base,
                            "bfs_reads": alt, "navhints_saving_pct": 100 * (1 - base / alt)})
        common = statistics.mean(v["navhints_saving_pct"] for v in details)
        comparison[arm] = {
            "five_targets": details,
            "five_target_mean_saving_pct": common,
            "previous_four_target_mean_saving_pct":
                original["matched_core_anchor"][arm]["saving_pct"]
        }
    write_json(out / "bigann-bfs-five-target-results.json", {
        "source_archive": "GeoIVF run 38164299742, artifact 11711862390",
        "source_commit": "e6fd9a32255706aed5ae37743083c81445aeff8c",
        "core": "(B,C,k)=(14464,2048,8)",
        "metric": "mean logical SSD-provider reads/query at matched Recall@10",
        "repetitions": REPS, "L_added": SEARCH_LIST,
        "capacity_nodes": CAPACITIES, "replayed_queries": 1000,
        "original_anchor_recalls": targets, "new_L640": means,
        "matched": comparison
    })
    print("BIGANN_BFS_L640_COMPLETE", flush=True)
    for arm, row in comparison.items():
        print(arm, f"five targets: {row['five_target_mean_saving_pct']:.3f}% "
              f"vs four targets: {row['previous_four_target_mean_saving_pct']:.3f}%",
              flush=True)

if __name__ == "__main__":
    main()
