#!/usr/bin/env python3
"""Qualification of navigation-trained DiskANN portals.

The online mechanism is identical to the integrated geometric-portal experiment:
1024 IVF centroids route each query to one existing graph vertex, and DiskANN
search proceeds unchanged from that start. Only the offline choice of the portal
vertex changes.

Training uses the fixed development split. For each coarse cell we consider the
eight database vectors nearest its centroid, run real DiskANN searches from
those candidates for development queries that place the cell in their nearest
32 coarse cells, and record per-query I/O plus recall. Two diagnostic tables are
learned:
  * nav-safe: minimum-I/O candidate among those within 0.25 hit/query of the
    best observed recall for that cell;
  * nav-io: minimum-I/O candidate without a recall constraint.
Cells with fewer than two development observations fall back to the geometric
nearest-to-centroid portal. Held-out evaluation is untouched.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import faiss
import numpy as np

from geoivf.index import train_faiss
from scripts.qualify_portal_integrated import (
    CANONICAL,
    PINNED_DISKANN,
    disk_run,
    write_router,
)
from scripts.qualify_portal_suite import (
    DATASET_SPECS,
    FIXED_BEAM,
    FIXED_LS,
    FIXED_NPROBES,
    K,
    NLIST,
    fbin,
    gtbin,
    load_dataset,
    result_rows,
    ubin,
)
from scripts.qualify_speed import save

CANDIDATES_PER_CELL = 8
TRAIN_NPROBE = 32
TRAIN_L = 60
MIN_CELL_OBSERVATIONS = 2
SAFE_HIT_TOLERANCE = 0.25


def build_candidate_table(x: np.ndarray, centers: np.ndarray, labels: np.ndarray):
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=len(centers))
    cuts = np.r_[0, np.cumsum(counts)]
    if np.any(counts == 0):
        raise ValueError("empty IVF cell")

    ids = np.empty((len(centers), CANDIDATES_PER_CELL), dtype=np.uint32)
    for cell in range(len(centers)):
        members = order[cuts[cell] : cuts[cell + 1]]
        pts = np.asarray(x[members], dtype=np.float32)
        delta = pts - centers[cell]
        d2 = np.einsum("ij,ij->i", delta, delta)
        take = min(CANDIDATES_PER_CELL, len(members))
        nearest = np.argpartition(d2, take - 1)[:take]
        nearest = nearest[np.argsort(d2[nearest], kind="stable")]
        chosen = members[nearest].astype(np.uint32, copy=False)
        if take < CANDIDATES_PER_CELL:
            chosen = np.resize(chosen, CANDIDATES_PER_CELL)
        ids[cell] = chosen
    return ids


def write_training_batch(
    work: Path,
    out: Path,
    router,
    dev: np.ndarray,
    dev_gt: np.ndarray,
    candidate_ids: np.ndarray,
    x: np.ndarray,
):
    _, cells = router.search(np.ascontiguousarray(dev), TRAIN_NPROBE)
    q_idx = np.repeat(np.arange(len(dev), dtype=np.int32), TRAIN_NPROBE * CANDIDATES_PER_CELL)
    cell_idx = np.repeat(cells.reshape(-1), CANDIDATES_PER_CELL).astype(np.int32)
    cand_idx = np.tile(
        np.arange(CANDIDATES_PER_CELL, dtype=np.int16),
        len(dev) * TRAIN_NPROBE,
    )

    queries = np.ascontiguousarray(dev[q_idx], dtype=np.float32)
    truth = np.ascontiguousarray(dev_gt[q_idx], dtype=np.int64)
    seeds = candidate_ids[cell_idx, cand_idx][:, None]

    fbin(work / "navigation-train.fbin", queries)
    gtbin(work / "navigation-train.gt", truth, x, queries)
    ubin(out / "navigation-train-seeds.ubin", seeds)
    np.savez(
        out / "navigation-train-map.npz",
        query_index=q_idx,
        cell_index=cell_idx,
        candidate_index=cand_idx,
    )
    return q_idx, cell_idx, cand_idx, out / "navigation-train-seeds.ubin"


def training_run(
    binary: Path,
    work: Path,
    out: Path,
    source: dict,
    seed_file: Path,
):
    phase = dict(
        queries=str(work / "navigation-train.fbin"),
        groundtruth=str(work / "navigation-train.gt"),
        search_list=[TRAIN_L],
        beam_width=FIXED_BEAM,
        recall_at=K,
        num_threads=1,
        is_flat_search=False,
        distance="squared_l2",
        vector_filters_file=None,
        num_nodes_to_cache=None,
        search_io_limit=None,
        post_processor=None,
    )
    cfg = dict(
        search_directories=[str(work)],
        jobs=[dict(type="disk-index", content=dict(source=source, search_phase=phase))],
    )
    inp = out / "navigation-train-input.json"
    output = out / "navigation-train-output.json"
    audit = out / "navigation-train-audit.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_PORTAL_ROUTER_FILE", None)
    env.pop("DISKANN_PORTAL_NPROBE", None)
    env["DISKANN_START_POINTS_FILE"] = str(seed_file.resolve())
    env["DISKANN_QUERY_AUDIT_FILE"] = str(audit.resolve())

    with (out / "navigation-train.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )

    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1 or int(rows[0]["search_l"]) != TRAIN_L:
        raise ValueError("unexpected navigation training aggregate result")
    obj = json.loads(audit.read_text())
    if int(obj["search_l"]) != TRAIN_L:
        raise ValueError("unexpected navigation audit L")
    return rows[0], obj["rows"]


def learn_portals(
    audit_rows,
    q_idx: np.ndarray,
    cell_idx: np.ndarray,
    cand_idx: np.ndarray,
    dev_gt: np.ndarray,
    candidate_ids: np.ndarray,
):
    if len(audit_rows) != len(q_idx):
        raise ValueError("audit/training-map row mismatch")

    truth10 = [set(map(int, row[:K])) for row in dev_gt]
    shape = (len(candidate_ids), CANDIDATES_PER_CELL)
    io_sum = np.zeros(shape, dtype=np.float64)
    hit_sum = np.zeros(shape, dtype=np.float64)
    hop_sum = np.zeros(shape, dtype=np.float64)
    cmp_sum = np.zeros(shape, dtype=np.float64)
    obs = np.zeros(shape, dtype=np.int32)

    for row_no, row in enumerate(audit_rows):
        cell = int(cell_idx[row_no])
        cand = int(cand_idx[row_no])
        qi = int(q_idx[row_no])
        hits = sum(int(v) in truth10[qi] for v in row["result_ids"])
        io_sum[cell, cand] += float(row["total_io_operations"])
        hit_sum[cell, cand] += float(hits)
        hop_sum[cell, cand] += float(row["search_hops"])
        cmp_sum[cell, cand] += float(row["total_comparisons"])
        obs[cell, cand] += 1

    geometric = candidate_ids[:, 0].copy()
    nav_safe = geometric.copy()
    nav_io = geometric.copy()
    cells = []
    trained = 0
    changed_safe = 0
    changed_io = 0

    for cell in range(len(candidate_ids)):
        counts = obs[cell]
        if counts.min() < MIN_CELL_OBSERVATIONS:
            continue
        trained += 1
        mean_io = io_sum[cell] / counts
        mean_hits = hit_sum[cell] / counts
        mean_hops = hop_sum[cell] / counts
        mean_cmp = cmp_sum[cell] / counts

        io_pick = min(
            range(CANDIDATES_PER_CELL),
            key=lambda j: (mean_io[j], -mean_hits[j], mean_hops[j], mean_cmp[j], j),
        )
        best_hits = float(np.max(mean_hits))
        eligible = [
            j
            for j in range(CANDIDATES_PER_CELL)
            if mean_hits[j] >= best_hits - SAFE_HIT_TOLERANCE
        ]
        safe_pick = min(
            eligible,
            key=lambda j: (mean_io[j], -mean_hits[j], mean_hops[j], mean_cmp[j], j),
        )

        nav_io[cell] = candidate_ids[cell, io_pick]
        nav_safe[cell] = candidate_ids[cell, safe_pick]
        changed_io += int(io_pick != 0)
        changed_safe += int(safe_pick != 0)
        cells.append(
            dict(
                cell=cell,
                observations=int(counts[0]),
                geometric_id=int(candidate_ids[cell, 0]),
                safe_id=int(nav_safe[cell]),
                io_id=int(nav_io[cell]),
                safe_candidate=int(safe_pick),
                io_candidate=int(io_pick),
                candidate_ids=[int(v) for v in candidate_ids[cell]],
                mean_ios=[float(v) for v in mean_io],
                mean_hits_at_10=[float(v) for v in mean_hits],
                mean_hops=[float(v) for v in mean_hops],
                mean_comparisons=[float(v) for v in mean_cmp],
            )
        )

    report = dict(
        training_l=TRAIN_L,
        training_nprobe=TRAIN_NPROBE,
        candidates_per_cell=CANDIDATES_PER_CELL,
        min_cell_observations=MIN_CELL_OBSERVATIONS,
        safe_hit_tolerance=SAFE_HIT_TOLERANCE,
        trained_cells=trained,
        total_cells=len(candidate_ids),
        changed_safe=changed_safe,
        changed_io=changed_io,
        cells=cells,
    )
    return geometric, nav_safe, nav_io, report


def summarize(rows):
    methods = ["medoid"] + [
        f"{kind}-np{npb}"
        for kind in ("geo", "nav-safe", "nav-io")
        for npb in FIXED_NPROBES
    ]
    result = {}
    for method in methods:
        result[method] = {}
        for l in FIXED_LS:
            rr = [
                r
                for r in rows
                if r["method"] == method and int(r["search_l"]) == l
            ]
            if not rr:
                raise ValueError(f"missing {method} L={l}")
            result[method][str(l)] = dict(
                rounds=len(rr),
                recall_percent=float(np.mean([r["recall"] for r in rr])),
                mean_end_to_end_us=float(np.mean([r["mean_latency"] for r in rr])),
                p95_end_to_end_us=float(
                    np.mean(
                        [
                            float(str(r["p95_latency"]).removesuffix("us"))
                            for r in rr
                        ]
                    )
                ),
                mean_route_us=float(
                    np.mean([r.get("mean_route_latency", 0.0) for r in rr])
                ),
                mean_ios=float(np.mean([r["mean_ios"] for r in rr])),
                mean_io_us=float(np.mean([r["mean_io_time"] for r in rr])),
                mean_cpu_us=float(np.mean([r["mean_cpu_time"] for r in rr])),
                mean_comparisons=float(np.mean([r["mean_comparisons"] for r in rr])),
                mean_hops=float(np.mean([r["mean_hops"] for r in rr])),
                qps=float(np.mean([r["qps"] for r in rr])),
            )

    for l in FIXED_LS:
        base = result["medoid"][str(l)]
        for method in methods[1:]:
            arm = result[method][str(l)]
            arm["latency_reduction_fraction_vs_medoid_same_l"] = (
                1.0 - arm["mean_end_to_end_us"] / base["mean_end_to_end_us"]
            )
            arm["io_reduction_fraction_vs_medoid_same_l"] = (
                1.0 - arm["mean_ios"] / base["mean_ios"]
            )
            arm["recall_delta_points_vs_medoid_same_l"] = (
                arm["recall_percent"] - base["recall_percent"]
            )

        for kind in ("nav-safe", "nav-io"):
            for npb in FIXED_NPROBES:
                arm = result[f"{kind}-np{npb}"][str(l)]
                geo = result[f"geo-np{npb}"][str(l)]
                arm["latency_reduction_fraction_vs_geo_same_l"] = (
                    1.0 - arm["mean_end_to_end_us"] / geo["mean_end_to_end_us"]
                )
                arm["io_reduction_fraction_vs_geo_same_l"] = (
                    1.0 - arm["mean_ios"] / geo["mean_ios"]
                )
                arm["recall_delta_points_vs_geo_same_l"] = (
                    arm["recall_percent"] - geo["recall_percent"]
                )
    return result


def run(a):
    started = time.monotonic()
    faiss.omp_set_num_threads(1)
    spec = DATASET_SPECS[a.dataset]
    seed = int.from_bytes(hashlib.sha256(a.dataset.encode()).digest()[:8], "little")
    x, dev, held, dev_gt, held_gt, dev_ids, held_ids = load_dataset(a.data, spec, seed)

    save(
        a.out / "cohorts.json",
        dict(
            dataset=a.dataset,
            seed=seed,
            development_ids=dev_ids.tolist(),
            heldout_ids=held_ids.tolist(),
            identical_to_fixed_portal_suite=True,
            heldout_never_used_for_training=True,
            nprobe_arms=list(FIXED_NPROBES),
            l_sweep=list(FIXED_LS),
            beam=FIXED_BEAM,
        ),
    )

    print(f"{a.dataset}: build frozen {NLIST}-cell router and candidate portals", flush=True)
    t = time.perf_counter()
    centers, labels = train_faiss(x, NLIST, 12345, min(100000, len(x)))
    ivf_build_s = time.perf_counter() - t

    t = time.perf_counter()
    candidate_ids = build_candidate_table(x, centers, labels)
    candidate_build_s = time.perf_counter() - t
    router = faiss.IndexFlatL2(x.shape[1])
    router.add(np.ascontiguousarray(centers))

    fbin(a.work / "base.fbin", x)
    fbin(a.work / "heldout.fbin", held)
    gtbin(a.work / "heldout.gt", held_gt, x, held)

    q_idx, cell_idx, cand_idx, seed_file = write_training_batch(
        a.work, a.out, router, dev, dev_gt, candidate_ids, x
    )

    prefix = str(a.work / "diskann-index")
    pq_chunks = min(64, x.shape[1])
    build = dict(
        **{"disk-index-source": "Build"},
        data_type="float32",
        data=str(a.work / "base.fbin"),
        distance="squared_l2",
        dim=int(x.shape[1]),
        max_degree=64,
        l_build=100,
        num_threads=4,
        build_ram_limit_gb=8.0,
        num_pq_chunks=pq_chunks,
        quantization_type="FP",
        save_path=prefix,
    )
    load = dict(
        **{"disk-index-source": "Load"},
        data_type="float32",
        load_path=prefix,
    )

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    try:
        print(
            f"{a.dataset}: train portal choices with {len(q_idx)} seeded DiskANN searches",
            flush=True,
        )
        train_aggregate, audit_rows = training_run(
            a.binary, a.work, a.out, build, seed_file
        )
        geometric, nav_safe, nav_io, training_report = learn_portals(
            audit_rows, q_idx, cell_idx, cand_idx, dev_gt, candidate_ids
        )
        training_report["aggregate"] = train_aggregate
        save(a.out / "navigation-training.json", training_report)

        tables = dict(
            geo=geometric,
            **{"nav-safe": nav_safe, "nav-io": nav_io},
        )
        router_files = {}
        for name, ids in tables.items():
            vecs = np.ascontiguousarray(x[ids.astype(np.int64)], dtype=np.float32)
            path = a.out / f"{name}-router.bin"
            write_router(path, centers, ids, vecs)
            router_files[name] = path
            np.savez(
                a.out / f"{name}-portal-table.npz",
                centers=centers,
                portal_ids=ids,
                portal_vectors=vecs,
            )

        rows = []
        arms = ["medoid"] + [
            f"{kind}-np{npb}"
            for kind in ("geo", "nav-safe", "nav-io")
            for npb in FIXED_NPROBES
        ]
        for rep in range(a.repeats):
            order = arms if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                if arm == "medoid":
                    rr = disk_run(
                        a.binary,
                        a.work,
                        a.out,
                        f"heldout-{arm}-r{rep}",
                        load,
                    )
                else:
                    kind, nptext = arm.rsplit("-np", 1)
                    rr = disk_run(
                        a.binary,
                        a.work,
                        a.out,
                        f"heldout-{arm}-r{rep}",
                        load,
                        router_files[kind],
                        int(nptext),
                    )
                for r in rr:
                    rows.append(dict(round=rep, method=arm, **r))

        save(a.out / "heldout-rows.json", rows)
        summary = summarize(rows)
        per_router_bytes = router_files["geo"].stat().st_size - 16
        save(
            a.out / "navigation-portal-result.json",
            dict(
                dataset=a.dataset,
                dimension=int(x.shape[1]),
                train_rows=len(x),
                diskann_revision=PINNED_DISKANN,
                router_execution="inside DiskANN timed query path",
                training=dict(
                    development_queries=len(dev),
                    expanded_seeded_searches=len(q_idx),
                    training_nprobe=TRAIN_NPROBE,
                    training_l=TRAIN_L,
                    candidates_per_cell=CANDIDATES_PER_CELL,
                    min_cell_observations=MIN_CELL_OBSERVATIONS,
                    safe_hit_tolerance=SAFE_HIT_TOLERANCE,
                    trained_cells=training_report["trained_cells"],
                    changed_safe=training_report["changed_safe"],
                    changed_io=training_report["changed_io"],
                ),
                fixed_policy=dict(
                    nlist=NLIST,
                    nprobes=list(FIXED_NPROBES),
                    one_portal_per_cell=True,
                    l_sweep=list(FIXED_LS),
                    beam=FIXED_BEAM,
                    max_degree=64,
                    l_build=100,
                    pq_chunks=pq_chunks,
                    num_nodes_to_cache=None,
                ),
                per_deployed_router_extra_bytes=per_router_bytes,
                per_deployed_router_extra_mib=per_router_bytes / (1 << 20),
                ivf_build_seconds=ivf_build_s,
                candidate_build_seconds=candidate_build_s,
                heldout_summary=summary,
                cpu_affinity=[allowed[0]],
                elapsed_seconds=time.monotonic() - started,
                limitations=[
                    "Navigation training considers only the eight database vectors nearest each centroid.",
                    "Training uses 128 development queries and L=60; held-out queries are never used for portal selection.",
                    "Cells with fewer than two development observations fall back to the geometric portal.",
                    "nav-safe allows at most 0.25 hit/query below the best development candidate before minimizing I/O.",
                    "One build per dataset on a shared host; node cache disabled.",
                    "DiskANN provider mean_ios is not an independently measured physical NVMe command count.",
                ],
            ),
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        os.sched_setaffinity(0, allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=CANONICAL)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.repeats < 3:
        ap.error("repeats must be >=3")
    for p in (a.work, a.out):
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    a.data, a.binary, a.work, a.out = map(
        Path.resolve, (a.data, a.binary, a.work, a.out)
    )

    lockpath = Path.home() / ".cache/geoivf/speed-device.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(a)


if __name__ == "__main__":
    main()
