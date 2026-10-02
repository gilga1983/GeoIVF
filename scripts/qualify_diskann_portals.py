#!/usr/bin/env python3
"""Tiny centroid-portal routing as a DiskANN graph-start carrier.

The DiskANN graph/disk image/PQ/traversal/reranking are unchanged. A tiny RAM
router selects exactly ONE query-specific start vertex. Development tunes only
the portal router. Held-out queries are fresh and never used for selection.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import vectors, train_faiss
from geoivf.projections import digest_file
from scripts.qualify_speed import save

PINNED_DISKANN = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
NPROBES = (1, 2, 4, 8, 16, 32)
PORTALS_PER_CELL = (1, 2, 4, 8)
FIXED_L = 60
FIXED_BEAM = 8
TARGET_RECALL_PERCENT = 99.0
MAX_PORTALS = max(PORTALS_PER_CELL)


def fbin(path, array, dtype="<f4"):
    a = np.asarray(array, dtype=dtype, order="C")
    with Path(path).open("wb") as f:
        np.asarray(a.shape, dtype="<u4").tofile(f)
        a.tofile(f)


def gtbin(path, ids, x, q):
    ids = np.asarray(ids, dtype="<u4", order="C")
    d = np.sum((x[ids].astype(np.float64) - q[:, None, :]) ** 2, axis=2).astype("<f4")
    with Path(path).open("wb") as f:
        np.asarray(ids.shape, dtype="<u4").tofile(f)
        ids.tofile(f)
        d.tofile(f)


def result_rows(obj):
    found = []
    if isinstance(obj, dict):
        if "search_l" in obj and "mean_latency" in obj and "recall" in obj:
            found.append(obj)
        else:
            for v in obj.values():
                found.extend(result_rows(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(result_rows(v))
    return found


def disk_run(binary, work, out, tag, split, source, seed_file=None):
    phase = dict(
        queries=str(work / f"{split}.fbin"),
        groundtruth=str(work / f"{split}.gt"),
        search_list=[FIXED_L],
        beam_width=FIXED_BEAM,
        recall_at=10,
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
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)
    env = os.environ.copy()
    if seed_file is None:
        env.pop("DISKANN_START_POINTS_FILE", None)
    else:
        env["DISKANN_START_POINTS_FILE"] = str(Path(seed_file).resolve())
    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log, stderr=subprocess.STDOUT, env=env, check=True,
        )
    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != 1:
        raise ValueError(f"expected one fixed-L row from {tag}, got {len(rows)}")
    return dict(rows[0], source_output=output.name)


def build_portals(x, centers, labels, max_portals=MAX_PORTALS):
    """Greedy farthest-point portals inside each coarse IVF cell.

    Portal 0 is the indexed vector closest to the coarse centroid. Additional
    portals maximize distance to the already chosen set, spreading actual graph
    vertices across the cell. This construction uses no query data.
    """
    t0 = time.perf_counter()
    nlist = len(centers)
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=nlist)
    if np.min(counts) < 1:
        raise ValueError("coarse IVF produced an empty cell")
    cuts = np.r_[0, np.cumsum(counts)]
    invalid = np.iinfo(np.uint32).max
    ids_out = np.full((nlist, max_portals), invalid, dtype=np.uint32)
    vec_out = np.full((nlist, max_portals, x.shape[1]), np.nan, dtype=np.float32)
    cover = np.full((nlist, max_portals), np.nan, dtype=np.float64)

    for li in range(nlist):
        ids = order[cuts[li]:cuts[li + 1]]
        pts = np.asarray(x[ids], dtype=np.float32)
        c = centers[li].astype(np.float64)
        d2c = np.sum((pts.astype(np.float64) - c) ** 2, axis=1)
        available = min(max_portals, len(ids))
        chosen = []
        first = int(np.argmin(d2c))
        chosen.append(first)
        delta = pts.astype(np.float64) - pts[first].astype(np.float64)
        min_d2 = np.einsum("ij,ij->i", delta, delta)
        min_d2[first] = -1.0

        for j in range(available):
            if j:
                nxt = int(np.argmax(min_d2))
                if nxt in chosen:
                    raise AssertionError("duplicate farthest-point portal")
                chosen.append(nxt)
                delta = pts.astype(np.float64) - pts[nxt].astype(np.float64)
                d2 = np.einsum("ij,ij->i", delta, delta)
                min_d2 = np.minimum(min_d2, d2)
                min_d2[chosen] = -1.0
            idx = chosen[j]
            ids_out[li, j] = np.uint32(ids[idx])
            vec_out[li, j] = pts[idx]
            remain = min_d2[min_d2 >= 0]
            cover[li, j] = float(np.sqrt(remain.max())) if len(remain) else 0.0

    valid_ids = ids_out[ids_out != np.iinfo(np.uint32).max]
    if len(np.unique(valid_ids)) != len(valid_ids):
        raise AssertionError("a portal ID appeared in more than one cell/slot")
    return ids_out, vec_out, cover, time.perf_counter() - t0


def portal_seed_rows(router, portal_ids, portal_vectors, queries, gt10, nprobe, portals_per_cell):
    if not 1 <= portals_per_cell <= portal_ids.shape[1]:
        raise ValueError("invalid portal count")
    # IndexFlatL2 is exact over the 1024 coarse centroids; request the desired cells directly.
    router.search(np.ascontiguousarray(queries[:1]), nprobe)
    rows = np.empty((len(queries), 1), dtype=np.uint32)
    elapsed = []
    any_gt10 = []
    seed_recall10 = []
    for qi, q in enumerate(queries):
        t = time.perf_counter_ns()
        _, cells = router.search(np.ascontiguousarray(q[None], dtype=np.float32), nprobe)
        cells = cells[0]
        ids = portal_ids[cells, :portals_per_cell].reshape(-1)
        vecs = portal_vectors[cells, :portals_per_cell].reshape(-1, q.shape[0])
        valid = ids != np.iinfo(np.uint32).max
        if not np.any(valid):
            raise AssertionError("selected coarse cells contain no portal")
        ids = ids[valid]
        vecs = vecs[valid]
        diff = vecs.astype(np.float64) - q.astype(np.float64)
        d2 = np.einsum("ij,ij->i", diff, diff)
        pick = int(np.argmin(d2))
        rows[qi, 0] = ids[pick]
        elapsed.append((time.perf_counter_ns() - t) / 1e6)
        overlap = int(rows[qi, 0]) in set(map(int, gt10[qi]))
        any_gt10.append(overlap)
        seed_recall10.append(0.1 if overlap else 0.0)
    return rows, dict(
        mean_ms=float(np.mean(elapsed)),
        median_ms=float(np.median(elapsed)),
        p95_ms=float(np.quantile(elapsed, .95)),
        any_global_top10_fraction=float(np.mean(any_gt10)),
        mean_global_top10_fraction=float(np.mean(seed_recall10)),
    )


def train_ivfpq_control(x, learn, seed=12345):
    quantizer = faiss.IndexFlatL2(x.shape[1])
    index = faiss.IndexIVFPQ(quantizer, x.shape[1], 1024, 64, 8)
    index.cp.niter = 20
    index.cp.seed = seed
    index.pq.cp.niter = 20
    index.pq.cp.seed = seed
    t = time.perf_counter()
    index.train(np.ascontiguousarray(learn, dtype=np.float32))
    train_s = time.perf_counter() - t
    t = time.perf_counter()
    for start in range(0, len(x), 65536):
        index.add(np.ascontiguousarray(x[start:start + 65536], dtype=np.float32))
    add_s = time.perf_counter() - t
    return index, train_s, add_s


def ivfpq_seed_rows(index, queries, gt10):
    index.nprobe = 2  # frozen from the preceding seeded-DiskANN campaign
    index.search(np.ascontiguousarray(queries[:1]), 1)
    rows = np.empty((len(queries), 1), dtype=np.uint32)
    elapsed, hit = [], []
    for i, q in enumerate(queries):
        t = time.perf_counter_ns()
        _, ids = index.search(np.ascontiguousarray(q[None], dtype=np.float32), 1)
        elapsed.append((time.perf_counter_ns() - t) / 1e6)
        if ids[0, 0] < 0:
            raise AssertionError("invalid IVF-PQ seed")
        rows[i, 0] = np.uint32(ids[0, 0])
        hit.append(int(rows[i, 0]) in set(map(int, gt10[i])))
    return rows, dict(
        mean_ms=float(np.mean(elapsed)),
        median_ms=float(np.median(elapsed)),
        p95_ms=float(np.quantile(elapsed, .95)),
        any_global_top10_fraction=float(np.mean(hit)),
        mean_global_top10_fraction=float(np.mean(hit)) / 10.0,
    )


def deployed_portal_bytes(centers, portals_per_cell, d=128):
    # Coarse centroids + selected portal FP32 vectors + u32 IDs.
    return int(centers.nbytes + len(centers) * portals_per_cell * (d * 4 + 4))


def run(a):
    faiss.omp_set_num_threads(1)
    started = time.monotonic()
    if digest_file(a.data / "sift_base.fvecs") != "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816":
        raise ValueError("unexpected SIFT1M base corpus")
    x = vectors(a.data / "sift_base.fvecs")
    learn = vectors(a.data / "sift_learn.fvecs")
    allq = vectors(a.data / "sift_query.fvecs")
    gt = np.memmap(a.data / "sift_groundtruth.ivecs", dtype="<i4", mode="r", shape=(10000, 101))[:, 1:]
    if (x.shape, learn.shape, allq.shape) != ((1000000, 128), (100000, 128), (10000, 128)):
        raise ValueError("unexpected SIFT1M shapes")

    perm = np.random.default_rng(20260929).permutation(10000)
    dev_ids = perm[2408:2408 + a.development]
    held_ids = perm[2536:2536 + a.heldout]
    if set(dev_ids) & set(perm[:2408]) or set(held_ids) & set(perm[:2536]):
        raise AssertionError("query cohort overlap")
    save(a.out / "splits.json", dict(
        development=dev_ids.tolist(), heldout=held_ids.tolist(),
        development_slice=[2408, 2408 + a.development],
        heldout_slice=[2536, 2536 + a.heldout],
        excludes_previous_prefix=2408, seed=20260929,
    ))

    fbin(a.work / "base.fbin", x)
    for name, ids in (("development", dev_ids), ("heldout", held_ids)):
        fbin(a.work / f"{name}.fbin", allq[ids])
        gtbin(a.work / f"{name}.gt", gt[ids, :100], x, allq[ids])

    print("Build tiny coarse-IVF portal router", flush=True)
    t = time.perf_counter()
    centers, labels = train_faiss(x, 1024, 12345, 100000)
    coarse_build_s = time.perf_counter() - t
    portal_ids, portal_vectors, cover, portal_build_s = build_portals(x, centers, labels)
    router = faiss.IndexFlatL2(128)
    router.add(np.ascontiguousarray(centers, dtype=np.float32))
    np.savez(
        a.out / "portal-table.npz",
        centers=centers, portal_ids=portal_ids, portal_vectors=portal_vectors,
        cover_radius=cover,
    )

    seed_dir = a.out / "seeds"
    seed_dir.mkdir()
    portal_meta = {"development": {}, "heldout": {}}
    portal_files = {"development": {}, "heldout": {}}
    for split, ids in (("development", dev_ids), ("heldout", held_ids)):
        q = allq[ids]
        gt10 = gt[ids, :10]
        for nprobe in NPROBES:
            for m in PORTALS_PER_CELL:
                key = f"np{nprobe}-m{m}"
                rows, stats = portal_seed_rows(router, portal_ids, portal_vectors, q, gt10, nprobe, m)
                path = seed_dir / f"{split}-{key}.ubin"
                fbin(path, rows, dtype="<u4")
                portal_files[split][key] = path
                portal_meta[split][key] = dict(
                    nprobe=nprobe, portals_per_cell=m, seed_count=1,
                    deployed_bytes=deployed_portal_bytes(centers, m),
                    deployed_mib=deployed_portal_bytes(centers, m) / (1 << 20),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    **stats,
                )

    # Frozen control from the preceding campaign, evaluated on the same new cohorts.
    print("Build previous 69 MiB IVF-PQ seeder as a same-cohort control", flush=True)
    ivfpq, ivfpq_train_s, ivfpq_add_s = train_ivfpq_control(x, learn)
    ivfpq_files, ivfpq_meta = {}, {}
    for split, ids in (("development", dev_ids), ("heldout", held_ids)):
        rows, stats = ivfpq_seed_rows(ivfpq, allq[ids], gt[ids, :10])
        path = seed_dir / f"{split}-ivfpq-np2-r1.ubin"
        fbin(path, rows, dtype="<u4")
        ivfpq_files[split] = path
        ivfpq_meta[split] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **stats)

    save(a.out / "portal-generation.json", dict(
        coarse_build_seconds=coarse_build_s,
        portal_build_seconds=portal_build_s,
        portal_counts=list(PORTALS_PER_CELL),
        nprobes=list(NPROBES),
        max_portal_table_bytes=deployed_portal_bytes(centers, MAX_PORTALS),
        max_portal_table_mib=deployed_portal_bytes(centers, MAX_PORTALS) / (1 << 20),
        mean_cover_radius_by_count={
            str(m): float(np.nanmean(cover[:, m - 1])) for m in PORTALS_PER_CELL
        },
        portal_configs=portal_meta,
        control_ivfpq=dict(
            nlist=1024, chunks=64, nbits=8, nprobe=2, seed_count=1,
            train_seconds=ivfpq_train_s, add_seconds=ivfpq_add_s,
            core_bytes_estimate=int(len(x) * 64 + len(x) * 8 + 1024 * 128 * 4 + 64 * 256 * 2 * 4),
            configs=ivfpq_meta,
        ),
    ))

    prefix = str(a.work / "diskann-index")
    build = dict(
        **{"disk-index-source": "Build"}, data_type="float32", data=str(a.work / "base.fbin"),
        distance="squared_l2", dim=128, max_degree=64, l_build=100, num_threads=4,
        build_ram_limit_gb=6.0, num_pq_chunks=64, quantization_type="FP", save_path=prefix,
    )
    load = dict(**{"disk-index-source": "Load"}, data_type="float32", load_path=prefix)

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    try:
        print("Build DiskANN once and calibrate portal router", flush=True)
        baseline_dev = disk_run(a.binary, a.work, a.out, "dev-medoid", "development", build)
        dev_rows = [dict(
            method="medoid", seed_config=None, seed_mean_ms=0.0,
            composed_mean_ms=float(baseline_dev["mean_latency"]) / 1000.0,
            deployed_extra_bytes=0, **baseline_dev,
        )]
        for nprobe in NPROBES:
            for m in PORTALS_PER_CELL:
                key = f"np{nprobe}-m{m}"
                print("development portal", key, flush=True)
                row = disk_run(a.binary, a.work, a.out, f"dev-{key}", "development", load, portal_files["development"][key])
                sm = portal_meta["development"][key]["mean_ms"]
                dev_rows.append(dict(
                    method="portal", seed_config=key, nprobe=nprobe, portals_per_cell=m,
                    seed_mean_ms=sm,
                    composed_mean_ms=sm + float(row["mean_latency"]) / 1000.0,
                    deployed_extra_bytes=deployed_portal_bytes(centers, m),
                    **row,
                ))
        ivfrow = disk_run(
            a.binary, a.work, a.out, "dev-ivfpq-control", "development", load,
            ivfpq_files["development"],
        )
        dev_rows.append(dict(
            method="ivfpq-control", seed_config="ivfpq-np2-r1", nprobe=2, portals_per_cell=0,
            seed_mean_ms=ivfpq_meta["development"]["mean_ms"],
            composed_mean_ms=ivfpq_meta["development"]["mean_ms"] + float(ivfrow["mean_latency"]) / 1000.0,
            deployed_extra_bytes=int(len(x) * 64 + len(x) * 8 + 1024 * 128 * 4 + 64 * 256 * 2 * 4),
            **ivfrow,
        ))
        save(a.out / "development-results.json", dev_rows)

        eligible = [r for r in dev_rows if r["method"] == "portal" and float(r["recall"]) >= TARGET_RECALL_PERCENT]
        if not eligible:
            raise ValueError("no tiny portal configuration reaches 99% development recall")
        best_io = min(eligible, key=lambda r: (float(r["mean_ios"]), r["deployed_extra_bytes"], float(r["mean_latency"]), r["seed_config"]))
        best_graph = min(eligible, key=lambda r: (float(r["mean_latency"]), float(r["mean_ios"]), r["deployed_extra_bytes"], r["seed_config"]))
        best_composed = min(eligible, key=lambda r: (float(r["composed_mean_ms"]), float(r["mean_ios"]), r["deployed_extra_bytes"], r["seed_config"]))
        best_memory = min(eligible, key=lambda r: (r["deployed_extra_bytes"], float(r["mean_ios"]), float(r["mean_latency"]), r["seed_config"]))
        selected = list(dict.fromkeys([
            "np1-m1", best_memory["seed_config"], best_io["seed_config"],
            best_graph["seed_config"], best_composed["seed_config"],
        ]))
        frozen = dict(
            target_recall_percent=TARGET_RECALL_PERCENT,
            fixed_l=FIXED_L, fixed_beam=FIXED_BEAM,
            selected_portal_configs=selected,
            selection_objectives=dict(
                smallest_eligible=best_memory["seed_config"],
                best_io=best_io["seed_config"],
                best_graph_latency=best_graph["seed_config"],
                best_composed_estimate=best_composed["seed_config"],
            ),
            fixed_ivfpq_control="ivfpq-np2-r1",
            chosen_before_heldout=True,
            diskann_node_cache=None,
        )
        save(a.out / "frozen-selection.json", frozen)
        print("FROZEN", json.dumps(frozen), flush=True)

        held_rows = []
        for rep in range(a.repeats):
            arms = [("medoid", None)] + [("portal", k) for k in selected] + [("ivfpq-control", "ivfpq-np2-r1")]
            if rep % 2:
                arms = list(reversed(arms))
            for method, key in arms:
                if method == "medoid":
                    row = disk_run(a.binary, a.work, a.out, f"held-medoid-r{rep}", "heldout", load)
                    held_rows.append(dict(
                        round=rep, method=method, seed_config=None, seed_mean_ms=0.0,
                        composed_mean_ms=float(row["mean_latency"]) / 1000.0,
                        deployed_extra_bytes=0, **row,
                    ))
                elif method == "portal":
                    meta = portal_meta["heldout"][key]
                    row = disk_run(a.binary, a.work, a.out, f"held-{key}-r{rep}", "heldout", load, portal_files["heldout"][key])
                    held_rows.append(dict(
                        round=rep, method=method, seed_config=key,
                        nprobe=meta["nprobe"], portals_per_cell=meta["portals_per_cell"],
                        seed_mean_ms=meta["mean_ms"],
                        composed_mean_ms=meta["mean_ms"] + float(row["mean_latency"]) / 1000.0,
                        deployed_extra_bytes=meta["deployed_bytes"], **row,
                    ))
                else:
                    meta = ivfpq_meta["heldout"]
                    row = disk_run(a.binary, a.work, a.out, f"held-ivfpq-r{rep}", "heldout", load, ivfpq_files["heldout"])
                    held_rows.append(dict(
                        round=rep, method=method, seed_config=key, nprobe=2, portals_per_cell=0,
                        seed_mean_ms=meta["mean_ms"],
                        composed_mean_ms=meta["mean_ms"] + float(row["mean_latency"]) / 1000.0,
                        deployed_extra_bytes=int(len(x) * 64 + len(x) * 8 + 1024 * 128 * 4 + 64 * 256 * 2 * 4),
                        **row,
                    ))
        save(a.out / "heldout-results.json", held_rows)

        names = [("medoid", None)] + [("portal", k) for k in selected] + [("ivfpq-control", "ivfpq-np2-r1")]
        summary = {}
        for method, key in names:
            rr = [r for r in held_rows if r["method"] == method and r.get("seed_config") == key]
            label = "medoid" if method == "medoid" else key
            summary[label] = dict(
                method=method, rounds=len(rr),
                recall_percent=float(np.mean([r["recall"] for r in rr])),
                mean_diskann_us=float(np.mean([r["mean_latency"] for r in rr])),
                mean_ios=float(np.mean([r["mean_ios"] for r in rr])),
                mean_io_us=float(np.mean([r["mean_io_time"] for r in rr])),
                mean_cpu_us=float(np.mean([r["mean_cpu_time"] for r in rr])),
                mean_comparisons=float(np.mean([r["mean_comparisons"] for r in rr])),
                mean_hops=float(np.mean([r["mean_hops"] for r in rr])),
                seed_mean_ms=float(np.mean([r["seed_mean_ms"] for r in rr])),
                composed_mean_ms=float(np.mean([r["composed_mean_ms"] for r in rr])),
                deployed_extra_bytes=int(rr[0]["deployed_extra_bytes"]),
                deployed_extra_mib=float(rr[0]["deployed_extra_bytes"]) / (1 << 20),
            )
        baseline = summary["medoid"]
        for key, s in summary.items():
            if key == "medoid":
                continue
            s["io_reduction_fraction_vs_medoid"] = 1 - s["mean_ios"] / baseline["mean_ios"]
            s["comparison_reduction_fraction_vs_medoid"] = 1 - s["mean_comparisons"] / baseline["mean_comparisons"]
            s["diskann_latency_reduction_fraction_vs_medoid"] = 1 - s["mean_diskann_us"] / baseline["mean_diskann_us"]

        save(a.out / "portal-diskann-results.json", dict(
            diskann_revision=PINNED_DISKANN,
            protocol=dict(
                fixed_l=FIXED_L, fixed_beam=FIXED_BEAM,
                target_recall_percent=TARGET_RECALL_PERCENT,
                num_nodes_to_cache=None, max_degree=64, l_build=100, pq_chunks=64,
                one_seed_per_query=True,
            ),
            portal_router=dict(
                nlist=1024, portal_counts=list(PORTALS_PER_CELL), nprobes=list(NPROBES),
                coarse_build_seconds=coarse_build_s, portal_build_seconds=portal_build_s,
                max_table_bytes=deployed_portal_bytes(centers, MAX_PORTALS),
                construction="portal0 closest to cell centroid; later portals greedy farthest-point in full 128D",
            ),
            frozen_selection=frozen,
            development=dev_rows,
            heldout_summary=summary,
            disk_files={p.name: p.stat().st_size for p in a.work.glob("diskann-index*") if p.is_file()},
            cpu_affinity=[allowed[0]],
            elapsed_seconds=time.monotonic() - started,
            limitations=[
                "Portal routing uses a separate coarse IVF table and cached FP32 portal vectors; it does not reuse DiskANN PQ codes.",
                "Portal construction is geometry-only and does not optimize graph reachability or I/O directly.",
                "Composed latency adds separately measured Python/Faiss portal time and DiskANN internal timing; it is diagnostic, not integrated in-process latency.",
                "DiskANN mean_ios is its provider vertex-load counter, not independently measured physical NVMe commands.",
                "L=60/beam8 is frozen from the prior DiskANN study; only the portal routing policy is tuned.",
                "One SIFT1M build/seed on a shared host; node cache disabled; corpus fits host RAM.",
                "No GeoPack physical layout or certified completion is part of this stage.",
            ],
        ))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        os.sched_setaffinity(0, allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path.home() / ".cache/geoivf/datasets/sift1m")
    ap.add_argument("--development", type=int, default=128)
    ap.add_argument("--heldout", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    if a.development != 128 or a.heldout != 256 or a.repeats < 3:
        ap.error("canonical portal experiment requires development=128, heldout=256, repeats>=3")
    for p in (a.work, a.out):
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    a.work, a.out, a.binary = a.work.resolve(), a.out.resolve(), a.binary.resolve()
    lockpath = Path.home() / ".cache/geoivf/speed-device.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(a)

if __name__ == "__main__":
    main()
