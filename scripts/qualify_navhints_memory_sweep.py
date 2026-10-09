#!/usr/bin/env python3
"""Memory-budget sensitivity: hot graph cache, QSEV entries and CatapultDB.

Run on the dedicated self-hosted SSD runner. The same PubMed/MedCPT frozen
index, first 5K static training requests, 4K held-out online warm-up, final
1K measurement requests, Recall@10 and L grid are used for every method.

This is a diagnostic sensitivity sweep, not a tuning set: we report *all*
configurations, including both hash granularities for Catapult. The fixed
NavHints Sample2 controller is rerun for matched-recall reference.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

import qualify_final_catapult as catlib

THREADS = 4
K = 10
BEAM = 8
DIM = 768
MAX_DEGREE = 64
LS = (10, 12, 20, 22, 24, 40, 44, 80, 88, 160, 176, 320)
ANCHORS = (10, 20, 40, 80, 160, 320)
MEMORY_TIERS = (1, 2, 4, 8, 16)
HOT_N = (30, 60, 120, 240, 480)
QSEV_N = (32, 64, 128, 256, 512)
CAT_HASHES = (7, 8)
# Capacities chosen to keep reserved Catapult payload close to the
# nominal 1x, 2x, 4x, 8x and 16x RAM levels, not to merely double slots.
CAT_CAP = {7: (160, 360, 760, 1600, 3200),
           8: (80, 180, 380, 780, 1580)}
SEEDS = (0, 1, 2)


def cache_file_check(path: Path, n: int) -> None:
    raw = path.read_bytes()
    if len(raw) != 16 + 4*n or raw[:8] != b"GIDST001":
        raise ValueError(f"bad hot-cache file {path}")
    if struct.unpack_from("<II", raw, 8) != (n, 0):
        raise ValueError(f"bad hot-cache header {path}")
    ids = struct.unpack_from(f"<{n}I", raw, 16)
    if len(set(ids)) != n:
        raise ValueError(f"duplicate cached vertex ID in {path}")


def run(binary: Path, folder: Path, name: str, queries: Path, gt: Path,
        index_prefix: Path, envextra: dict[str, str], *, cache_nodes=None,
        list_sizes=LS, skip_recall=False, verbose=False) -> list[dict]:
    folder.mkdir(parents=True, exist_ok=True)
    cfg = {
        "search_directories": [str(folder)],
        "jobs": [{"type": "disk-index", "content": {
            "source": {"disk-index-source": "Load", "data_type": "float32",
                       "load_path": str(index_prefix)},
            "search_phase": {
                "queries": str(queries), "groundtruth": str(gt),
                "search_list": list(list_sizes), "beam_width": BEAM,
                "recall_at": K, "num_threads": THREADS,
                "is_flat_search": False, "distance": "inner_product",
                "vector_filters_file": None, "num_nodes_to_cache": cache_nodes,
                "search_io_limit": None, "post_processor": None,
            },
        }}],
    }
    inp = folder / f"{name}.input.json"
    out = folder / f"{name}.output.json"
    log = folder / f"{name}.log"
    catlib.save(inp, cfg)

    env = os.environ.copy()
    for k in list(env):
        if k.startswith("DISKANN_"):
            env.pop(k, None)
    env.update({k: str(v) for k, v in envextra.items()})
    if skip_recall:
        env["DISKANN_SKIP_RECALL"] = "1"
    t0 = time.monotonic()
    with log.open("w") as fp:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(out)],
            stdout=fp, stderr=subprocess.STDOUT, check=True, env=env, timeout=3600,
        )
    records = sorted(catlib.result_rows(json.loads(out.read_text())),
                     key=lambda row: int(row["search_l"]))
    if [int(row["search_l"]) for row in records] != list(list_sizes):
        raise ValueError(f"{name}: incomplete L sweep")
    if name.startswith("r") and not skip_recall:
        if any(not np.isfinite(float(row["mean_ios"])) for row in records):
            raise ValueError(f"{name}: nonfinite I/O")
    if name.startswith("r") and not skip_recall and ("hot-" in name or "bfs-" in name):
        logfile = log.read_text()
        if "hot-" in name:
            n = int(name.rsplit("hot-", 1)[1])
            expected = f"NAVHINTS_CACHE_READY mode=static_ids requested={n} loaded={n}"
        else:
            n = int(name.rsplit("bfs-", 1)[1])
            expected = f"NAVHINTS_CACHE_MODE=native_bfs count={n}"
        if expected not in logfile:
            raise RuntimeError(f"{name}: cache failed to initialize: {expected}")
        hit = [float(row["cache_hit_percentage"]) for row in records]
        if max(hit) < 0.05:
            raise RuntimeError(f"{name}: no measured cache hits; hit percentages={hit}")
        print(f"CACHE_VERIFIED {name}: hit@L160={hit[list(list_sizes).index(160)]:.3f}%",
              flush=True)
    print(f"{name}: {time.monotonic()-t0:.1f}s", flush=True)
    if verbose:
        print(f"{name}: L160={next((x for x in records if x['search_l']==160),None)}",
              flush=True)
    return records


def stats(rows: list[list[dict]]) -> dict:
    summary = {}
    for l in LS:
        series = [row for rr in rows for row in rr if int(row["search_l"]) == l]
        if len(series) != len(rows):
            raise ValueError(f"L={l}: lost a replicate")
        summary[str(l)] = {
            "reps": len(series),
            "recall_percent": float(np.mean([float(x["recall"]) for x in series])),
            "mean_ios": float(np.mean([float(x["mean_ios"]) for x in series])),
            "latency_us": float(np.median([float(x["mean_latency"]) for x in series])),
            "median_qps": float(np.median([float(x["qps"]) for x in series])),
            "mean_hops": float(np.mean([float(x["mean_hops"]) for x in series])),
            "cache_hit_percentage": float(np.mean([
                float(x.get("cache_hit_percentage", 0)) for x in series
            ])),
            "catapult_usage_percentage": float(np.mean([
                float(x.get("catapult_usage_percentage", 0)) for x in series
            ])),
            "mean_catapult_starts": float(np.mean([
                float(x.get("mean_catapult_starts", 0)) for x in series
            ])),
        }
    return summary


def match_method(variant: dict, nav: dict) -> list[dict]:
    result = []
    for l in ANCHORS:
        target = float(nav[str(l)]["recall_percent"])
        io = catlib.interp(variant, target, "mean_ios")
        latency = catlib.interp(variant, target, "latency_us")
        row = {"anchor_L": l, "recall_percent": target, "available": io is not None}
        if io is not None and latency is not None:
            anchor_read = float(nav[str(l)]["mean_ios"])
            anchor_lat = float(nav[str(l)]["latency_us"])
            row.update(competitor_ios=io, competitor_latency_us=latency,
                       navhints_io_saving_percent=100*(1-anchor_read/io),
                       navhints_latency_saving_percent=100*(1-anchor_lat/latency))
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for a in ("hot-binary", "qsev-binary", "cat-binary", "nav-binary",
              "queries", "heldout-gt", "train-gt", "index-prefix", "ivf",
              "hot-dir", "qsev-dir", "work", "out"):
        parser.add_argument("--"+a, type=Path, required=True)
    parser.add_argument("--reps", type=int, default=3)
    a = parser.parse_args()
    if not 1 <= a.reps <= 3:
        raise ValueError("only one to three deterministic seeds available")
    for attr in ("hot_binary", "qsev_binary", "cat_binary", "nav_binary",
                 "queries", "heldout_gt", "train_gt", "index_prefix",
                 "ivf", "hot_dir", "qsev_dir", "work", "out"):
        setattr(a, attr, getattr(a, attr).resolve())
    if catlib.fshape(a.queries) != (10000, DIM):
        raise ValueError("expected 10K x 768 original PubMed/MedCPT query replay")
    a.work.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)

    trainq = a.work / "train5000.fbin"
    heldq = a.work / "heldout5000.fbin"
    warmq = a.work / "warm4000.fbin"
    evalq = a.work / "eval1000.fbin"
    warmgt = a.work / "warm4000.gt"
    evalgt = a.work / "eval1000.gt"
    catlib.slice_fbin(a.queries, trainq, 0, 5000)
    catlib.slice_fbin(a.queries, heldq, 5000, 5000)
    catlib.slice_fbin(heldq, warmq, 0, 4000)
    catlib.slice_fbin(heldq, evalq, 4000, 1000)
    catlib.slice_gt(a.heldout_gt, warmgt, 0, 4000)
    catlib.slice_gt(a.heldout_gt, evalgt, 4000, 1000)

    cat_variants = [
        (h, CAT_CAP[h][i], MEMORY_TIERS[i]) for i in range(len(MEMORY_TIERS))
        for h in CAT_HASHES
    ]
    cat_names = [f"cat-{h}-{cap}" for h, cap, _ in cat_variants]
    hot_names = [f"hot-{n}" for n in HOT_N]
    qsev_names = [f"qsev-{n}" for n in QSEV_N]
    methods = (["baseline", "bfs-30", "navhints-full"] +
               hot_names + qsev_names + cat_names)
    results = {m: [] for m in methods}
    snapshots = {}
    snapshot_meta = []
    mem = {}

    for i, n in enumerate(HOT_N):
        f = a.hot_dir / f"hot-cache-n{n}.bin"
        cache_file_check(f, n)
        mem[f"hot-{n}"] = {
            "tier": MEMORY_TIERS[i], "method": "workload_hot_cache",
            "nodes": n, "aux_bytes_estimated": n*(DIM*4 + MAX_DEGREE*4),
            "accounting": "full vector + max-degree edge-ID bound; excludes container overhead"
        }
    mem["bfs-30"] = {**mem["hot-30"], "method": "native_BFS_cache"}
    for i, n in enumerate(QSEV_N):
        f = a.qsev_dir / f"qsev-{n}.bin"
        expect = 16 + n*(DIM*4+4)
        if f.stat().st_size != expect:
            raise ValueError(f"{f}: expected {expect} bytes, got {f.stat().st_size}")
        mem[f"qsev-{n}"] = {
            "tier": MEMORY_TIERS[i], "method": "DiskANNpp_QSEV_entry",
            "entries": n, "aux_bytes_estimated": expect,
            "accounting": "measured on-disk representation of resident entry IDs and vectors"
        }
    for h, cap, tier in cat_variants:
        name = f"cat-{h}-{cap}"
        mem[name] = {
            "tier": tier, "method": "CatapultDB",
            "hash_bits": h, "bucket_capacity": cap,
            "aux_bytes_estimated": h*DIM*4 + (1 << h)*cap*4,
            "accounting": "reserved bucket IDs plus projection hyperplanes; excludes metadata"
        }
    ivfm = json.loads(a.ivf.with_suffix(a.ivf.suffix+".manifest.json").read_text())
    static = a.ivf.stat().st_size + int(ivfm["nlist"])*(64+4)
    navram = static + 512*4
    mem["navhints-full"] = {
        "tier": 1, "method": "NavHints_sample2", "aux_bytes_estimated": navram,
        "accounting": "published packed directory + 512 IDs; optional exits stored on SSD"
    }
    mem["baseline"] = {"tier": 0, "method": "DiskANN",
                       "aux_bytes_estimated": 0}

    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < THREADS:
        raise RuntimeError(f"need {THREADS} CPUs")
    previous = set(affinity)
    os.sched_setaffinity(0, set(affinity[:THREADS]))
    lock = Path.home() / ".cache/geoivf/speed-device.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("w") as lockfile:
            print(f"waiting for exclusive SSD lock: {lock}", flush=True)
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            print("acquired exclusive SSD lock", flush=True)
            for h, cap, _ in cat_variants:
                name = f"cat-{h}-{cap}"
                for seed in SEEDS[:a.reps]:
                    env = {
                        "DISKANN_PAPER_CATAPULT": "1",
                        "DISKANN_CATAPULT_HASHES": str(h),
                        "DISKANN_CATAPULT_CAPACITY": str(cap),
                        "DISKANN_CATAPULT_SEED": str(seed),
                    }
                    trained = a.work / f"{name}-s{seed}-train.snapshot"
                    warmed = a.work / f"{name}-s{seed}-warm.snapshot"
                    run(a.cat_binary, a.out, f"train-{name}-s{seed}",
                        trainq, a.train_gt, a.index_prefix,
                        {**env, "DISKANN_CATAPULT_SNAPSHOT_DUMP": str(trained)},
                        list_sizes=(10,), skip_recall=True)
                    run(a.cat_binary, a.out, f"warm-{name}-s{seed}",
                        warmq, warmgt, a.index_prefix,
                        {**env, "DISKANN_CATAPULT_SNAPSHOT_LOAD": str(trained),
                         "DISKANN_CATAPULT_SNAPSHOT_DUMP": str(warmed)},
                        list_sizes=(10,), skip_recall=True)
                    snapshots[(name, seed)] = warmed
                    info = {"name": name, "seed": seed,
                            "train_entries": catlib.snapshot_entries(trained),
                            "warm_entries": catlib.snapshot_entries(warmed),
                            "warm_snapshot_bytes": warmed.stat().st_size}
                    snapshot_meta.append(info)
                    print(f"SNAPSHOT {info}", flush=True)
            catlib.save(a.out / "snapshot-state.json", snapshot_meta)

            for rep in range(a.reps):
                # Rotate method order to limit time- or thermal-position bias.
                order = methods[(rep*7) % len(methods):] + methods[:(rep*7) % len(methods)]
                print(f"rep={rep}, order={order}", flush=True)
                for name in order:
                    prefix = f"r{rep}-{name}"
                    if name == "navhints-full":
                        env = {
                            "DISKANN_HINT_IVF_FILE": str(a.ivf),
                            "DISKANN_HINT_IVF_NPROBE": "8",
                            "DISKANN_HINT_IVF_MAX_STARTS": "1",
                            "DISKANN_EXPERIENCE_REPLAY": "1",
                            "DISKANN_EXPERIENCE_WARMUP": "4000",
                            "DISKANN_EXPERIENCE_CACHE_CAPACITY": "512",
                            "DISKANN_EXPERIENCE_HUB_CAPACITY": "10",
                            "DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR": "2",
                            "DISKANN_EXPERIENCE_FILL_FROM_CACHE": "1",
                        }
                        rr = run(a.nav_binary, a.out, prefix, heldq,
                                 a.heldout_gt, a.index_prefix, env)
                    else:
                        env, cfgcache = {}, None
                        binary = a.hot_binary
                        if name.startswith("bfs-"):
                            cfgcache = int(name.split("-")[1])
                        elif name.startswith("hot-"):
                            env["DISKANN_STATIC_CACHE_IDS_FILE"] = str(
                                a.hot_dir / f"hot-cache-n{name.split('-')[1]}.bin")
                        elif name.startswith("qsev-"):
                            binary = a.qsev_binary
                            env["DISKANN_QSEV_FILE"] = str(
                                a.qsev_dir / f"qsev-{name.split('-')[1]}.bin")
                        elif name.startswith("cat-"):
                            binary = a.cat_binary
                            bits, capacity = map(int, name.split("-")[1:])
                            env.update({
                                "DISKANN_PAPER_CATAPULT": "1",
                                "DISKANN_CATAPULT_HASHES": str(bits),
                                "DISKANN_CATAPULT_CAPACITY": str(capacity),
                                "DISKANN_CATAPULT_SEED": str(SEEDS[rep]),
                                "DISKANN_CATAPULT_SNAPSHOT_LOAD":
                                    str(snapshots[(name, SEEDS[rep])]),
                            })
                        if name.startswith("cat-"):
                            # Same storage-warm-up queries; frozen Catapult
                            # state avoids changing the causal snapshot.
                            run(binary, a.out, prefix+"-storagewarm",
                                warmq, warmgt, a.index_prefix,
                                {**env, "DISKANN_CATAPULT_FREEZE": "1"},
                                list_sizes=LS, skip_recall=True)
                        else:
                            run(binary, a.out, prefix+"-storagewarm",
                                warmq, warmgt, a.index_prefix, env,
                                cache_nodes=cfgcache, list_sizes=LS,
                                skip_recall=True)
                        rr = run(binary, a.out, prefix, evalq, evalgt,
                                 a.index_prefix, env, cache_nodes=cfgcache)
                    results[name].append(rr)
                    catlib.save(a.out / "runs.partial.json", results)
                print(f"completed replicate {rep+1}/{a.reps}", flush=True)
    finally:
        os.sched_setaffinity(0, previous)

    agg = {name: stats(results[name]) for name in methods}
    baseline = agg["baseline"]
    for name in ["bfs-30"] + hot_names:
        for l in LS:
            rows = agg[name][str(l)]
            b = baseline[str(l)]
            if abs(rows["recall_percent"]-b["recall_percent"]) > 0.06:
                raise AssertionError(
                    f"{name} L{l}: cache changed recall {rows['recall_percent']} "
                    f"vs {b['recall_percent']}"
                )
        if agg[name]["160"]["cache_hit_percentage"] <= 0.05:
            raise AssertionError(f"{name}: no verified hits at L=160")
        if agg[name]["160"]["mean_ios"] >= baseline["160"]["mean_ios"] - 0.001:
            raise AssertionError(f"{name}: cache did not reduce physical graph-record reads")

    matched = {}
    for name in methods:
        points = match_method(agg[name], agg["navhints-full"])
        eligible = [p for p in points if p["available"]]
        matched[name] = {
            "per_anchor": points,
            "common_anchor_count": len(eligible),
            "mean_navhints_io_saving_percent":
                float(np.mean([p["navhints_io_saving_percent"] for p in eligible]))
                if eligible else None,
            "mean_navhints_latency_saving_percent":
                float(np.mean([p["navhints_latency_saving_percent"] for p in eligible]))
                if eligible else None,
        }
    out = {
        "workload": "PubMed1M/MedCPT: first 5K static training; 4K online warm-up; final 1K measured",
        "diskann_version": "fcf90534174cf29c78c9f13b4cccf1fcabff85f5",
        "protocol": {
            "repetitions": a.reps, "K": K, "beam": BEAM, "threads": THREADS,
            "Ls": list(LS), "navhints_anchor_Ls": list(ANCHORS),
            "query_order_rotated": True, "same_graph_pq_ssd": True,
            "storage_lock": str(lock),
            "fixed_controller": "NavHints Sample2 (p=1/2), 16K entry + Recent512 + H10",
            "controller_query_side_payload_bytes": navram,
            "cache_patch_accounting": "actual uncached graph-record reads per query; hits exclude disk I/O",
            "catapult_online_history": "5K static + 4K warm, online continues in final 1K",
            "catapult_hash_seeds": list(SEEDS[:a.reps]),
            "baseline_guarantee": "native BFS and workload-selected hot cache must show actual avoided reads at unchanged recall",
            "disclaimer": "larger competitor configurations are sensitivity results, not prospectively tuned hyperparameters"
        },
        "memory": mem, "snapshots": snapshot_meta,
        "summary": agg, "matched_at_navhints": matched,
    }
    catlib.save(a.out / "navhints-memory-sweep.json", out)
    rows = []
    for name in methods:
        x = matched[name]
        rows.append({
            "method": name,
            "tier": mem[name].get("tier"),
            "aux_payload_bytes": mem[name].get("aux_bytes_estimated"),
            "anchors": x["common_anchor_count"],
            "navhints_io_saving_percent": x["mean_navhints_io_saving_percent"],
            "navhints_latency_saving_percent": x["mean_navhints_latency_saving_percent"],
            "io_at_L160": agg[name]["160"]["mean_ios"],
            "cache_hit_percent_at_L160": agg[name]["160"]["cache_hit_percentage"],
            "catapult_starts_at_L160": agg[name]["160"]["mean_catapult_starts"],
        })
    catlib.save(a.out / "memory-sweep-summary.json", rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
