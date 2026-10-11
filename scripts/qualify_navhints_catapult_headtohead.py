#!/usr/bin/env python3
"""Head-to-head: RAM-only NavHints Core vs author-aligned CatapultDB-on-DiskANN.

One frozen PubMed1M graph, PQ, SSD, and query split. NavHints trains its
16K directory on disjoint first-5K history; Catapult's 8-bit/80-entry LRU
also observes first 5K chronologically. BOTH then observe heldout 4K warm-up
and 1K measured requests in strict completion order, retaining online
updates during measurement. Search engine and K/L/beam are held constant.

Comparisons interpolate the actual Recall@10/cost frontiers separately
per Catapult random seed and repetition. Never extrapolate recall.
The result is a controlled routing-mechanism comparison, not the author's
in-memory engine versus SSD search.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import time
from pathlib import Path

K = 10
DIM = 768
BEAM = 8
SEARCH_THREADS = 4  # internal search context; queries dispatched in causal sequence
LS = (10, 12, 20, 24, 40, 80, 160, 320)
SEEDS = (0, 1, 2)
HASH_BITS = 8
CATAPULT_CAPACITY = 80
WARM = 4000
MEASURE = 1000
TRAIN = 5000
PQ_CACHE = 512


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def shape(path):
    with path.open("rb") as f:
        head = f.read(8)
    if len(head) != 8:
        raise ValueError(f"invalid fbin header {path}")
    n, d = struct.unpack("<II", head)
    if path.stat().st_size != 8 + n * d * 4:
        raise ValueError(f"bad fbin length {path}")
    return n, d


def slice_fbin(src, dst, start, count):
    n, dim = shape(src)
    if not 0 <= start < n or start + count > n:
        raise ValueError("query split exceeds frozen stream")
    with src.open("rb") as fi, dst.open("wb") as fo:
        fi.seek(8 + start * dim * 4)
        fo.write(struct.pack("<II", count, dim))
        left = count * dim * 4
        while left:
            b = fi.read(min(left, 16 << 20))
            if not b:
                raise ValueError("truncated fbin data")
            fo.write(b)
            left -= len(b)
    return dst


def rows_of(o):
    if isinstance(o, list):
        for x in o:
            yield from rows_of(x)
    elif isinstance(o, dict):
        if "search_l" in o and "mean_ios" in o:
            yield o
        else:
            for x in o.values():
                yield from rows_of(x)


def run_one(binary, label, queries, gt, index, L, out, *,
            mode, seed=None, snapshot_load=None, snapshot_dump=None, train=False, ivf=None):
    phase = {
        "queries": str(queries),
        "groundtruth": str(gt),
        "search_list": [L],
        "beam_width": BEAM,
        "recall_at": K,
        "num_threads": 1 if train else SEARCH_THREADS,
        "is_flat_search": False,
        "distance": "inner_product",
        "vector_filters_file": None,
        "num_nodes_to_cache": None,
        "search_io_limit": None,
        "post_processor": None,
    }
    inp = out / f"{label}.input.json"
    result = out / f"{label}.output.json"
    log = out / f"{label}.log"
    save(inp, {
        "search_directories": [str(queries.parent)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Load", "data_type": "float32",
                    "load_path": str(index),
                },
                "search_phase": phase,
            },
        }],
    })
    env = os.environ.copy()
    for k in list(env):
        if k.startswith("DISKANN_"):
            env.pop(k, None)

    if train:
        env["DISKANN_SKIP_RECALL"] = "1"
    if mode == "nav":
        if ivf is None or train:
            raise ValueError("NavHints requires static directory, no online training")
        env.update({
            "DISKANN_HINT_IVF_FILE": str(ivf),
            "DISKANN_HINT_IVF_NPROBE": "8",
            "DISKANN_HINT_IVF_MAX_STARTS": "1",
            "DISKANN_EXPERIENCE_REPLAY": "1",
            "DISKANN_EXPERIENCE_WARMUP": str(WARM),
            "DISKANN_EXPERIENCE_CACHE_CAPACITY": str(PQ_CACHE),
            "DISKANN_EXPERIENCE_HUB_CAPACITY": "0",  # strictly write-free Core
        })
    elif mode == "catapult":
        if seed is None:
            raise ValueError("Catapult random-hyperplane seed required")
        env.update({
            "DISKANN_PAPER_CATAPULT": "1",
            "DISKANN_CATAPULT_HASHES": str(HASH_BITS),
            "DISKANN_CATAPULT_CAPACITY": str(CATAPULT_CAPACITY),
            "DISKANN_CATAPULT_SEED": str(seed),
            "DISKANN_CATAPULT_CAUSAL_REPLAY": "1",
            "DISKANN_CATAPULT_CAUSAL_WARMUP": "0" if train else str(WARM),
        })
        if snapshot_load is not None:
            env["DISKANN_CATAPULT_SNAPSHOT_LOAD"] = str(snapshot_load)
        if snapshot_dump is not None:
            env["DISKANN_CATAPULT_SNAPSHOT_DUMP"] = str(snapshot_dump)
        # Deliberately NO DISKANN_CATAPULT_FREEZE: online learning is causal
        # during measured requests, like NavHints.
    else:
        raise ValueError(f"unknown method {mode}")

    started = time.monotonic()
    with log.open("w") as lf:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp),
             "--output-file", str(result)],
            env=env, stdout=lf, stderr=subprocess.STDOUT,
            timeout=2400, check=True,
        )
    rr = list(rows_of(json.loads(result.read_text())))
    if len(rr) != 1 or int(rr[0]["search_l"]) != L:
        raise RuntimeError(f"{label}: missing native L={L} result: {rr}")
    row = rr[0]
    if train:
        if float(row["recall"]) != -1.0:
            raise RuntimeError(f"{label}: training erroneously scored recall")
        if not snapshot_dump or not snapshot_dump.is_file():
            raise RuntimeError(f"{label}: missing Catapult trained LRU snapshot")
    else:
        if not 0.0 <= float(row["recall"]) <= 100.0:
            raise RuntimeError(f"{label}: invalid Recall@10")
        info = log.read_text()
        if mode == "catapult":
            token = f"CATAPULT_CAUSAL_STATS L={L} warmup={WARM} measured={MEASURE}"
            if token not in info:
                raise RuntimeError(f"{label}: missing causal suffix proof")
        else:
            marker = re.search(r"EXPERIENCE_STATS L=\d+ .*?hub_capacity=(\d+).*?eval_writes=(\d+)", info)
            if marker is None or tuple(map(int, marker.groups())) != (0, 0):
                raise RuntimeError(f"{label}: Core accidentally enabled persistence")
    print(f"PASS {label}: recall={row['recall']} reads={row['mean_ios']:.6f} "
          f"latency_us={row['mean_latency']:.2f} seconds={time.monotonic()-started:.1f}",
          flush=True)
    return {
        "recall": float(row["recall"]),
        "reads": float(row["mean_ios"]),
        "latency_us": float(row["mean_latency"]),
        "cpu_us": float(row.get("mean_cpu_time", 0.0)),
        "io_us": float(row.get("mean_io_time", 0.0)),
        "qps": float(row.get("qps", 0.0)),
    }


def snapshot_entries(path):
    raw = path.read_bytes()
    if len(raw) < 32 or raw[:8] != b"GICAT001":
        raise ValueError(f"invalid Catapult snapshot {path}")
    return struct.unpack("<I", raw[28:32])[0]


def frontier(one_curve):
    """Measured, non-extrapolated increasing-recall frontier; identical recall
    ties retain lower-cost result. No interpolation through dominated points.
    """
    pts = sorted(
        [{"L": L, **one_curve[str(L)]} for L in LS],
        key=lambda v: (v["recall"], v["reads"]),
    )
    out = []
    for p in pts:
        if out and p["recall"] <= out[-1]["recall"] + 1e-8:
            if p["reads"] < out[-1]["reads"]:
                out[-1] = p
        else:
            out.append(p)
    return out


def interp(points, recall):
    if recall < points[0]["recall"] - 1e-8 or recall > points[-1]["recall"] + 1e-8:
        return None
    for p in points:
        if abs(p["recall"] - recall) < 1e-8:
            return {"lower_L": p["L"], "upper_L": p["L"],
                    "reads": p["reads"], "latency_us": p["latency_us"]}
    for a, b in zip(points, points[1:]):
        if a["recall"] < recall < b["recall"]:
            x = (recall - a["recall"]) / (b["recall"] - a["recall"])
            return {
                "lower_L": a["L"], "upper_L": b["L"],
                "reads": a["reads"] + x * (b["reads"] - a["reads"]),
                "latency_us": a["latency_us"] + x * (b["latency_us"] - a["latency_us"]),
            }
    return None


def summarize(runs, reps):
    nav_by_rep = []
    cat_by_rep_seed = []
    for rep in range(reps):
        nav = {str(L): runs[f"eval-r{rep}-L{L}-nav"] for L in LS}
        nav_by_rep.append(nav)
        for seed in SEEDS:
            cat_by_rep_seed.append((rep, seed, {
                str(L): runs[f"eval-r{rep}-L{L}-cat-s{seed}"] for L in LS
            }))
    # All same queries give deterministic recall on repeated runs, but
    # take the mean rather than assuming byte-identical outcomes.
    anchors = []
    for L in LS:
        target = statistics.mean(nav[str(L)]["recall"] for nav in nav_by_rep)
        nav_reads = statistics.mean(nav[str(L)]["reads"] for nav in nav_by_rep)
        nav_latency = statistics.mean(nav[str(L)]["latency_us"] for nav in nav_by_rep)
        matched = []
        for rep, seed, one in cat_by_rep_seed:
            x = interp(frontier(one), target)
            if x is None:
                break
            matched.append({"rep": rep, "seed": seed, **x})
        if len(matched) != len(cat_by_rep_seed):
            anchors.append({
                "nav_L": L, "recall_at_10": target, "available": False,
                "reason": "Not attainable within every Catapult seed/repetition; no extrapolation",
            })
            continue
        cat_reads = statistics.mean(x["reads"] for x in matched)
        cat_latency = statistics.mean(x["latency_us"] for x in matched)
        rep_ratios = []
        for rep in range(reps):
            per_rep = [x for x in matched if x["rep"] == rep]
            c_read = statistics.mean(x["reads"] for x in per_rep)
            n_read = nav_by_rep[rep][str(L)]["reads"]
            rep_ratios.append(100 * (1.0 - n_read / c_read))
        anchors.append({
            "nav_L": L, "recall_at_10": target, "available": True,
            "nav_reads_per_query": nav_reads,
            "catapult_reads_per_query": cat_reads,
            "nav_latency_us": nav_latency,
            "catapult_latency_us": cat_latency,
            "nav_reads_saved_percent": 100 * (1.0 - nav_reads / cat_reads),
            "nav_latency_saved_percent": 100 * (1.0 - nav_latency / cat_latency),
            "paired_repetition_reads_saved_percent": rep_ratios,
            "catapult_seed_repetition_interpolants": matched,
        })
    qualified = [a for a in anchors if a["available"]]
    return {
        "matched_recall": anchors,
        "qualifying_anchors": len(qualified),
        "equal_anchor_mean_reads_saved_percent": (
            statistics.mean(x["nav_reads_saved_percent"] for x in qualified)
            if qualified else None
        ),
        "equal_anchor_mean_latency_saved_percent": (
            statistics.mean(x["nav_latency_saved_percent"] for x in qualified)
            if qualified else None
        ),
        "caveat": "Three SSD replays use the same deterministic query stream, not independent query samples. No inferential confidence interval is claimed.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("nav-bin", "catapult-bin", "queries", "train-gt", "heldout-gt",
                "index-prefix", "ivf", "work", "out"):
        ap.add_argument("--" + arg, type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    for key in ("nav_bin", "catapult_bin", "queries", "train_gt",
                "heldout_gt", "index_prefix", "ivf", "work", "out"):
        setattr(a, key, getattr(a, key).resolve())
    if a.reps != 3:
        raise ValueError("publication trial requires exactly 3 order-rotated repetitions")
    if shape(a.queries) != (TRAIN + WARM + MEASURE, DIM):
        raise ValueError("require exactly 10k frozen 768-dimensional PubMed queries")
    if not a.train_gt.is_file() or not a.heldout_gt.is_file():
        raise FileNotFoundError("both disjoint GT inputs required")
    if not (a.nav_bin.is_file() and a.catapult_bin.is_file() and a.ivf.is_file()):
        raise FileNotFoundError("native binaries or routing map unavailable")
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    prefix = slice_fbin(a.queries, a.work / "train5000.fbin", 0, TRAIN)
    held = slice_fbin(a.queries, a.work / "heldout5000.fbin", TRAIN, WARM + MEASURE)

    # This routes on the same pinned graph, compressed PQ, SSD and requests.
    # The 16K directory is trained ONLY using the disjoint 5K history.
    manifest_path = a.ivf.with_suffix(a.ivf.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    nlist = int(manifest["nlist"])
    nav_bytes = a.ivf.stat().st_size + nlist * (64 + 4) + PQ_CACHE * 4
    cat_bytes = (1 << HASH_BITS) * CATAPULT_CAPACITY * 4 + HASH_BITS * DIM * 4
    if nav_bytes <= 0 or cat_bytes < nav_bytes:
        raise ValueError(f"unsafe budget accounting: nav={nav_bytes} cat={cat_bytes}")
    if not 0.9 <= nav_bytes / cat_bytes <= 1.05:
        raise ValueError(f"not comparable auxiliary RAM: nav={nav_bytes} cat={cat_bytes}")

    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < SEARCH_THREADS:
        raise RuntimeError("self-hosted benchmark needs four available CPUs")
    os.sched_setaffinity(0, set(allowed[:SEARCH_THREADS]))
    original_cores = allowed
    snapshots = {}
    results = {}
    try:
        # Static/offline history is EXACTLY the first 5K for each policy.
        # For Catapult we build a fresh causal snapshot for every (L, seed),
        # rather than letting an entry trained with another L skew the result.
        for li, L in enumerate(LS):
            for seed in SEEDS:
                label = f"train-L{L}-s{seed}"
                dest = a.work / (label + ".snapshot")
                run_one(a.catapult_bin, label, prefix, a.train_gt,
                        a.index_prefix, L, a.out,
                        mode="catapult", seed=seed,
                        snapshot_dump=dest, train=True)
                occupied = snapshot_entries(dest)
                if not (0 < occupied <= (1 << HASH_BITS) * CATAPULT_CAPACITY):
                    raise RuntimeError(f"invalid bucket occupancy for {label}: {occupied}")
                snapshots[label] = {
                    "resident_entries": occupied,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                }
                save(a.out / "snapshots.partial.json", snapshots)
        # Rotate arm order within each L and repetition to reduce SSD drift.
        for rep in range(a.reps):
            grid = list(LS)
            grid = grid[rep:] + grid[:rep]
            for ix, L in enumerate(grid):
                arms = ["nav"] + [f"cat-s{s}" for s in SEEDS]
                shift = (rep + ix) % len(arms)
                arms = arms[shift:] + arms[:shift]
                print(f"REP={rep} L={L} rotated arms={','.join(arms)}", flush=True)
                for arm in arms:
                    name = f"eval-r{rep}-L{L}-{arm}"
                    if arm == "nav":
                        row = run_one(a.nav_bin, name, held, a.heldout_gt,
                                      a.index_prefix, L, a.out,
                                      mode="nav", ivf=a.ivf)
                    else:
                        seed = int(arm.rsplit("s", 1)[-1])
                        row = run_one(
                            a.catapult_bin, name, held, a.heldout_gt,
                            a.index_prefix, L, a.out, mode="catapult",
                            seed=seed,
                            snapshot_load=a.work / f"train-L{L}-s{seed}.snapshot",
                        )
                    results[name] = row
                    save(a.out / "results.partial.json", results)
    finally:
        os.sched_setaffinity(0, set(original_cores))

    out = {
        "experiment": "author-aligned CatapultDB adapted to DiskANN vs RAM-only NavHints Core",
        "source_kind": "same pinned Microsoft DiskANN SSD backend; NOT author's native in-memory graph",
        "population": "PubMed1M MedCPT + 10K MedRAG-Zipf queries, 768d inner product",
        "query_split": {"static_history": 5000, "causal_warmup": WARM,
                        "measured_suffix": MEASURE},
        "causality": "Both arms strictly sequential per query, feedback applied only after each completion; both continue updating during the 1K measurement.",
        "architecture": {"K": K, "beam": BEAM, "configured_search_threads": SEARCH_THREADS,
                         "causal_query_dispatch_threads": 1,
                         "widths": list(LS), "seeds": list(SEEDS),
                         "repetitions": a.reps},
        "memory": {
            "nav_auxiliary_bytes": nav_bytes,
            "catapult_auxiliary_bytes": cat_bytes,
            "catapult_id_bucket_payload": (1 << HASH_BITS) * CATAPULT_CAPACITY * 4,
            "catapult_hash_plane_payload": HASH_BITS * DIM * 4,
            "excludes_allocator_lock_and_container_metadata": True,
        },
        "ivf_sha256": hashlib.sha256(a.ivf.read_bytes()).hexdigest(),
        "snapshots": snapshots,
        "results": results,
        "summary": summarize(results, a.reps),
    }
    if out["summary"]["qualifying_anchors"] < 3:
        raise RuntimeError("Insufficient common recall anchors for defensible comparison")
    save(a.out / "navhints-vs-author-aligned-catapult.json", out)
    print("NAVHINTS_CATAPULT_AUTHOR_ALIGNED_CAUSAL_HEADTOHEAD_PASS", flush=True)
    print(json.dumps(out["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
