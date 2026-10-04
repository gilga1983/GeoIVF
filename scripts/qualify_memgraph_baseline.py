#!/usr/bin/env python3
"""Compare official Starling MemGraph starts against learned regional waypoints.

Starling navigation is executed separately with four threads and its selected
original vertex IDs are then supplied to the identical pinned DiskANN search.
For each MemGraph arm:
  * logical SSD I/O, recall, hops, comparisons come from the DiskANN phase;
  * conservative end-to-end mean latency is Starling nav mean + DiskANN mean;
  * conservative end-to-end QPS serializes the two 4-thread batch phases:
      1 / (1/nav_qps + 1/disk_qps).

A one-repetition screen evaluates all mem_L values. The best comparable-recall
arm per sampling mode is then confirmed for two more repetitions alongside ours.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np

THREADS = 4
IO_BEAM = 8
K = 1


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def fbin_shape(path: Path):
    with path.open("rb") as f:
        rows, dim = struct.unpack("<II", f.read(8))
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError("bad fbin size")
    return rows, dim


def suffix_fbin(src: Path, dst: Path, start: int):
    rows, dim = fbin_shape(src)
    n = rows - start
    with src.open("rb") as fin, dst.open("wb") as fout:
        fin.seek(8 + start * dim * 4)
        fout.write(struct.pack("<II", n, dim))
        remaining = n * dim * 4
        while remaining:
            b = fin.read(min(16 << 20, remaining))
            if not b:
                raise ValueError("truncated fbin")
            fout.write(b)
            remaining -= len(b)
    return n, dim


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj:
            found.append(obj)
        else:
            for v in obj.values():
                found.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(result_rows(v))
    return found


def run_one(
    binary: Path,
    out: Path,
    tag: str,
    queries: Path,
    gt: Path,
    index_prefix: Path,
    *,
    starts: Path | None = None,
    router: Path | None = None,
    waypoint: Path | None = None,
):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [K],
        "beam_width": IO_BEAM,
        "recall_at": K,
        "num_threads": THREADS,
        "is_flat_search": False,
        "distance": "inner_product",
        "vector_filters_file": None,
        "num_nodes_to_cache": None,
        "search_io_limit": None,
        "post_processor": None,
    }
    cfg = {
        "search_directories": [str(out)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load",
                    "data_type": "float32",
                    "load_path": str(index_prefix),
                },
                "search_phase": phase,
            },
        }],
    }
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_SKIP_RECALL", None)
    for name in (
        "DISKANN_START_POINTS_FILE",
        "DISKANN_IP_PORTAL_ROUTER_FILE",
        "DISKANN_IP_PORTAL_NPROBE",
        "DISKANN_PAPER_CATAPULT",
        "DISKANN_CATAPULT_HASHES",
        "DISKANN_CATAPULT_CAPACITY",
        "DISKANN_CATAPULT_SEED",
        "DISKANN_QSEV_FILE",
        "DISKANN_WAYPOINT_CACHE_FILE",
        "DISKANN_WAYPOINT_MAX_IDS_PER_QUERY",
    ):
        env.pop(name, None)
    if starts is not None:
        env["DISKANN_START_POINTS_FILE"] = str(starts)
    if router is not None:
        env["DISKANN_IP_PORTAL_ROUTER_FILE"] = str(router)
        env["DISKANN_IP_PORTAL_NPROBE"] = "32"
    if waypoint is not None:
        env["DISKANN_WAYPOINT_CACHE_FILE"] = str(waypoint)
        env["DISKANN_WAYPOINT_MAX_IDS_PER_QUERY"] = "40"

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

    rr = result_rows(json.loads(output.read_text()))
    if len(rr) != 1:
        raise ValueError(f"{tag}: expected one result row, got {len(rr)}")
    row = dict(rr[0])
    if float(row["recall"]) < 0:
        raise ValueError(f"{tag}: recall missing")
    return row


def memgraph_metrics(row, nav):
    disk_qps = float(row["qps"])
    nav_qps = float(nav["nav_qps"])
    combined_qps = 1.0 / (1.0 / nav_qps + 1.0 / disk_qps)
    return {
        "recall_percent": float(row["recall"]),
        "mean_ios": float(row["mean_ios"]),
        "disk_qps": disk_qps,
        "nav_qps": nav_qps,
        "combined_qps": combined_qps,
        "disk_mean_latency_us": float(row["mean_latency"]),
        "nav_mean_latency_us": float(nav["nav_mean_latency_us"]),
        "combined_mean_latency_us": (
            float(row["mean_latency"]) + float(nav["nav_mean_latency_us"])
        ),
        "mean_hops": float(row["mean_hops"]),
        "mean_comparisons": float(row["mean_comparisons"]),
    }


def ours_metrics(row):
    return {
        "recall_percent": float(row["recall"]),
        "mean_ios": float(row["mean_ios"]),
        "combined_qps": float(row["qps"]),
        "combined_mean_latency_us": float(row["mean_latency"]),
        "mean_hops": float(row["mean_hops"]),
        "mean_comparisons": float(row["mean_comparisons"]),
    }


def mean_rows(rows):
    keys = rows[0].keys()
    out = {}
    for key in keys:
        vals = [r[key] for r in rows]
        if all(isinstance(v, (int, float)) for v in vals):
            out[key] = float(np.mean(vals))
    return out


def choose_arm(screen, mode, target_recall):
    candidates = [
        (name, x) for name, x in screen.items()
        if name.startswith(mode + "-L")
    ]
    eligible = [
        (name, x) for name, x in candidates
        if x["recall_percent"] >= target_recall - 0.25
    ]
    pool = eligible if eligible else candidates
    return max(
        pool,
        key=lambda kv: (
            kv[1]["combined_qps"],
            -abs(kv[1]["recall_percent"] - target_recall),
            -kv[1]["mean_ios"],
        ),
    )[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-binary", type=Path, required=True)
    ap.add_argument("--waypoint-binary", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--index-prefix", type=Path, required=True)
    ap.add_argument("--router", type=Path, required=True)
    ap.add_argument("--waypoint", type=Path, required=True)
    ap.add_argument("--memgraph-manifest", action="append", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=5000)
    args = ap.parse_args()

    for name in (
        "start_binary", "waypoint_binary", "queries", "gt", "index_prefix",
        "router", "waypoint", "work", "out",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.memgraph_manifest = [p.resolve() for p in args.memgraph_manifest]
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, dim = fbin_shape(args.queries)
    if rows != 10000 or dim != 768 or args.train_rows != 5000:
        raise ValueError("expected frozen 10000x768 workload with 5000/5000 split")
    held = args.work / "heldout.fbin"
    nheld, _ = suffix_fbin(args.queries, held, args.train_rows)
    if nheld != 5000:
        raise AssertionError("heldout size mismatch")

    manifests = [json.loads(p.read_text()) for p in args.memgraph_manifest]
    modes = {m["mode"]: m for m in manifests}
    if set(modes) != {"frequency", "uniform"}:
        raise ValueError(f"expected frequency and uniform manifests, got {set(modes)}")

    arms = {}
    for mode, m in modes.items():
        for raw_l, nav in m["results"].items():
            name = f"{mode}-L{raw_l}"
            starts = Path(nav["start_file"]).resolve()
            if not starts.is_file():
                raise FileNotFoundError(starts)
            arms[name] = {
                "mode": mode,
                "mem_L": int(raw_l),
                "starts": starts,
                "nav": nav,
                "sample_count": int(m["sample_count"]),
                "memgraph_bytes": int(m["deployed_index_bytes_on_disk"]),
            }

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < THREADS:
        raise RuntimeError("fewer than four CPUs")
    os.sched_setaffinity(0, set(allowed[:THREADS]))
    lock_path = Path.home() / ".cache/geoivf/speed-device.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = []
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            # Screen every MemGraph effort once, with ours in the same device epoch.
            ours = run_one(
                args.waypoint_binary, args.out, "screen-ours-512",
                held, args.gt, args.index_prefix,
                router=args.router, waypoint=args.waypoint,
            )
            raw_rows.append({"phase": "screen", "rep": 0, "method": "ours-512", **ours})
            screen = {"ours-512": ours_metrics(ours)}

            for name in sorted(arms):
                spec = arms[name]
                row = run_one(
                    args.start_binary, args.out, f"screen-{name}",
                    held, args.gt, args.index_prefix, starts=spec["starts"],
                )
                raw_rows.append({"phase": "screen", "rep": 0, "method": name, **row})
                screen[name] = memgraph_metrics(row, spec["nav"])
                save(args.out / "rows.partial.json", raw_rows)

            target_recall = screen["ours-512"]["recall_percent"]
            selected = {
                mode: choose_arm(screen, mode, target_recall)
                for mode in ("frequency", "uniform")
            }

            # Two extra repetitions for selected MemGraph arms and ours.
            confirmations = {
                "ours-512": [screen["ours-512"]],
                selected["frequency"]: [screen[selected["frequency"]]],
                selected["uniform"]: [screen[selected["uniform"]]],
            }
            methods = ["ours-512", selected["frequency"], selected["uniform"]]
            for rep in (1, 2):
                shift = rep % len(methods)
                order = methods[shift:] + methods[:shift]
                for name in order:
                    if name == "ours-512":
                        row = run_one(
                            args.waypoint_binary, args.out, f"confirm-r{rep}-{name}",
                            held, args.gt, args.index_prefix,
                            router=args.router, waypoint=args.waypoint,
                        )
                        metrics = ours_metrics(row)
                    else:
                        spec = arms[name]
                        row = run_one(
                            args.start_binary, args.out, f"confirm-r{rep}-{name}",
                            held, args.gt, args.index_prefix, starts=spec["starts"],
                        )
                        metrics = memgraph_metrics(row, spec["nav"])
                    raw_rows.append({"phase": "confirm", "rep": rep, "method": name, **row})
                    confirmations[name].append(metrics)
                    save(args.out / "rows.partial.json", raw_rows)
    finally:
        os.sched_setaffinity(0, set(allowed))

    save(args.out / "rows.json", raw_rows)
    confirmed = {name: mean_rows(rr) for name, rr in confirmations.items()}

    result = {
        "workload": "MedRAG-Zipf heldout 5000 vs exact PubMed1M IP ground truth",
        "comparison": "same DiskANN SSD graph; only in-RAM navigation differs",
        "search": {"k": 1, "L": 1, "beam": IO_BEAM, "threads": THREADS},
        "memgraph_implementation": "official zilliztech/starling Vamana memory index",
        "memgraph_combination_rule": (
            "sequential conservative accounting: total batch time = Starling "
            "navigation batch time + DiskANN SSD-search batch time"
        ),
        "screen": screen,
        "selected": selected,
        "confirmed_three_run_means": confirmed,
        "memgraph_manifests": modes,
    }
    save(args.out / "memgraph-baseline.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
