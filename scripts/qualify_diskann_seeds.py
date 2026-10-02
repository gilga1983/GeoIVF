#!/usr/bin/env python3
"""Test IVF-PQ-selected graph start points on an otherwise unchanged DiskANN3.

DiskANN graph, disk layout, PQ, traversal and exact reranking remain upstream.
The experiment changes only query-specific graph entry vertices. Development
selects an I/O-saving seed policy; a fresh held-out cohort is never retuned.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import vectors
from geoivf.projections import digest_file
from scripts.qualify_speed import save

PINNED_DISKANN = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
NPROBES = (1, 2, 4, 8, 16)
SEED_COUNTS = (1, 2, 4, 8)
FIXED_L = 60
FIXED_BEAM = 8
TARGET_RECALL_PERCENT = 99.0


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


def train_seeder(x, learn, seed=12345):
    faiss.omp_set_num_threads(1)
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
    if index.ntotal != len(x):
        raise AssertionError("IVF-PQ seeder did not index every vector")
    return index, train_s, add_s


def seed_rows(index, queries, gt10, nprobe, count):
    index.nprobe = int(nprobe)
    times = []
    rows = np.empty((len(queries), count), dtype=np.uint32)
    any_gt10 = []
    seed_recall10 = []
    # Warm the exact path that will be timed one query at a time.
    index.search(np.ascontiguousarray(queries[:1]), count)
    for i, q in enumerate(queries):
        t = time.perf_counter_ns()
        _, ids = index.search(np.ascontiguousarray(q[None], dtype=np.float32), count)
        times.append((time.perf_counter_ns() - t) / 1e6)
        ids = ids[0]
        if np.any(ids < 0) or len(np.unique(ids)) != count:
            raise AssertionError("invalid or duplicate IVF-PQ seed IDs")
        rows[i] = ids.astype(np.uint32)
        overlap = len(set(map(int, ids)) & set(map(int, gt10[i])))
        any_gt10.append(overlap > 0)
        seed_recall10.append(overlap / 10)
    return rows, dict(
        mean_ms=float(np.mean(times)),
        median_ms=float(np.median(times)),
        p95_ms=float(np.quantile(times, .95)),
        any_global_top10_fraction=float(np.mean(any_gt10)),
        mean_global_top10_fraction=float(np.mean(seed_recall10)),
    )


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


def core_seed_bytes(n, d=128, nlist=1024, chunks=64):
    # IVF-PQ payload + 64-bit IDs + coarse centroids + 256 two-D codewords/chunk.
    return int(n * 64 + n * 8 + nlist * d * 4 + chunks * 256 * (d // chunks) * 4)


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
    dev_ids = perm[2024:2024 + a.development]
    held_ids = perm[2152:2152 + a.heldout]
    previously_exposed = set(perm[:2024])
    if set(dev_ids) & previously_exposed or set(held_ids) & (previously_exposed | set(dev_ids)):
        raise AssertionError("query cohort overlap")
    save(a.out / "splits.json", dict(
        development=dev_ids.tolist(), heldout=held_ids.tolist(),
        development_slice=[2024, 2024 + a.development],
        heldout_slice=[2152, 2152 + a.heldout], seed=20260929,
        excludes_previous_prefix=2024,
    ))

    fbin(a.work / "base.fbin", x)
    for name, ids in (("development", dev_ids), ("heldout", held_ids)):
        fbin(a.work / f"{name}.fbin", allq[ids])
        gtbin(a.work / f"{name}.gt", gt[ids, :100], x, allq[ids])

    print("Train independent one-thread Faiss IVF-PQ seed index", flush=True)
    seeder, seeder_train_s, seeder_add_s = train_seeder(x, learn)
    pq_centroids = faiss.vector_to_array(seeder.pq.centroids)
    pq_hash = hashlib.sha256(pq_centroids.tobytes()).hexdigest()
    seed_root = a.out / "seeds"
    seed_root.mkdir()
    seed_meta = {"development": {}, "heldout": {}}
    seed_files = {"development": {}, "heldout": {}}
    for split, ids in (("development", dev_ids), ("heldout", held_ids)):
        q = allq[ids]
        gt10 = gt[ids, :10]
        for nprobe in NPROBES:
            for count in SEED_COUNTS:
                rows, timing = seed_rows(seeder, q, gt10, nprobe, count)
                path = seed_root / f"{split}-np{nprobe}-r{count}.ubin"
                fbin(path, rows, dtype="<u4")
                key = f"np{nprobe}-r{count}"
                seed_files[split][key] = path
                seed_meta[split][key] = dict(
                    nprobe=nprobe, seed_count=count, rows=len(rows),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **timing
                )
    save(a.out / "seed-generation.json", dict(
        train_seconds=seeder_train_s, add_seconds=seeder_add_s,
        faiss=faiss.__version__, nlist=1024, chunks=64, nbits=8,
        core_bytes_estimate=core_seed_bytes(len(x)),
        core_mib_estimate=core_seed_bytes(len(x)) / (1 << 20),
        pq_centroids_sha256=pq_hash,
        note="Core estimate excludes Faiss allocator/list/runtime overhead; DiskANN already stores separate PQ codes.",
        configs=seed_meta,
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
        print("Build DiskANN once; run unchanged medoid baseline", flush=True)
        baseline_dev = disk_run(a.binary, a.work, a.out, "dev-baseline", "development", build)
        dev_rows = [dict(method="medoid", seed_config=None, seed_mean_ms=0.0,
                         composed_mean_ms=float(baseline_dev["mean_latency"]) / 1000.0,
                         **baseline_dev)]
        for nprobe in NPROBES:
            for count in SEED_COUNTS:
                key = f"np{nprobe}-r{count}"
                print("development seeded", key, flush=True)
                row = disk_run(
                    a.binary, a.work, a.out, f"dev-{key}", "development", load,
                    seed_files["development"][key],
                )
                sm = seed_meta["development"][key]["mean_ms"]
                dev_rows.append(dict(
                    method="ivfpq-seeded", seed_config=key, nprobe=nprobe, seed_count=count,
                    seed_mean_ms=sm, composed_mean_ms=sm + float(row["mean_latency"]) / 1000.0,
                    **row,
                ))
        save(a.out / "development-results.json", dev_rows)

        eligible = [r for r in dev_rows if r["method"] == "ivfpq-seeded" and float(r["recall"]) >= TARGET_RECALL_PERCENT]
        if not eligible:
            raise ValueError("no seeded configuration reaches 99% development recall")
        best_io = min(eligible, key=lambda r: (float(r["mean_ios"]), float(r["mean_latency"]), r["seed_config"]))
        best_graph = min(eligible, key=lambda r: (float(r["mean_latency"]), float(r["mean_ios"]), r["seed_config"]))
        best_composed = min(eligible, key=lambda r: (float(r["composed_mean_ms"]), float(r["mean_ios"]), r["seed_config"]))
        selected_keys = list(dict.fromkeys([best_io["seed_config"], best_graph["seed_config"], best_composed["seed_config"]]))
        frozen = dict(
            target_recall_percent=TARGET_RECALL_PERCENT, fixed_l=FIXED_L, fixed_beam=FIXED_BEAM,
            selection_objectives=dict(best_io=best_io["seed_config"], best_graph_latency=best_graph["seed_config"],
                                      best_composed_estimate=best_composed["seed_config"]),
            selected_seed_configs=selected_keys, chosen_before_heldout=True,
            diskann_node_cache=None,
        )
        save(a.out / "frozen-selection.json", frozen)
        print("FROZEN", json.dumps(frozen), flush=True)

        held_rows = []
        for rep in range(a.repeats):
            row = disk_run(a.binary, a.work, a.out, f"held-baseline-r{rep}", "heldout", load)
            held_rows.append(dict(round=rep, method="medoid", seed_config=None, seed_mean_ms=0.0,
                                  composed_mean_ms=float(row["mean_latency"]) / 1000.0, **row))
            for key in selected_keys:
                nprobe = int(key.split("-")[0][2:])
                count = int(key.split("-")[1][1:])
                row = disk_run(
                    a.binary, a.work, a.out, f"held-{key}-r{rep}", "heldout", load,
                    seed_files["heldout"][key],
                )
                sm = seed_meta["heldout"][key]["mean_ms"]
                held_rows.append(dict(
                    round=rep, method="ivfpq-seeded", seed_config=key, nprobe=nprobe, seed_count=count,
                    seed_mean_ms=sm, composed_mean_ms=sm + float(row["mean_latency"]) / 1000.0,
                    **row,
                ))
        save(a.out / "heldout-results.json", held_rows)

        summary = {}
        for name in ["medoid"] + selected_keys:
            rr = [r for r in held_rows if (r["method"] == "medoid" and name == "medoid") or r.get("seed_config") == name]
            summary[name] = dict(
                rounds=len(rr),
                recall_percent=float(np.mean([r["recall"] for r in rr])),
                mean_diskann_us=float(np.mean([r["mean_latency"] for r in rr])),
                mean_ios=float(np.mean([r["mean_ios"] for r in rr])),
                mean_io_us=float(np.mean([r["mean_io_time"] for r in rr])),
                mean_cpu_us=float(np.mean([r["mean_cpu_time"] for r in rr])),
                mean_comparisons=float(np.mean([r["mean_comparisons"] for r in rr])),
                mean_hops=float(np.mean([r["mean_hops"] for r in rr])),
                seed_mean_ms=float(np.mean([r["seed_mean_ms"] for r in rr])),
                composed_mean_ms=float(np.mean([r["composed_mean_ms"] for r in rr])),
            )
        baseline = summary["medoid"]
        for key in selected_keys:
            s = summary[key]
            s["io_reduction_fraction_vs_medoid"] = 1 - s["mean_ios"] / baseline["mean_ios"]
            s["diskann_latency_reduction_fraction_vs_medoid"] = 1 - s["mean_diskann_us"] / baseline["mean_diskann_us"]
        save(a.out / "seeded-diskann-results.json", dict(
            diskann_revision=PINNED_DISKANN,
            protocol=dict(fixed_l=FIXED_L, fixed_beam=FIXED_BEAM, target_recall_percent=TARGET_RECALL_PERCENT,
                          num_nodes_to_cache=None, max_degree=64, l_build=100, pq_chunks=64),
            seeder=dict(nlist=1024, chunks=64, nbits=8, train_seconds=seeder_train_s, add_seconds=seeder_add_s,
                        core_bytes_estimate=core_seed_bytes(len(x)), pq_centroids_sha256=pq_hash),
            development=dev_rows, frozen_selection=frozen, heldout_summary=summary,
            disk_files={p.name: p.stat().st_size for p in a.work.glob("diskann-index*") if p.is_file()},
            cpu_affinity=[allowed[0]], elapsed_seconds=time.monotonic() - started,
            limitations=[
                "Seed generator duplicates roughly 64-byte/vector PQ codes plus IVF IDs; no RAM reuse with DiskANN yet.",
                "Composed latency adds separately measured Python/Faiss seeding and DiskANN internal query timing; it is diagnostic, not an integrated in-process latency.",
                "DiskANN mean_ios is its provider vertex-load counter, not an independently measured NVMe-command count.",
                "Fixed L=60/beam8 comes from the prior released-baseline study; only seed policy is tuned here.",
                "One SIFT1M index/seed on a shared host; node cache disabled; dataset fits host RAM.",
                "No GeoPack layout change or certification path is part of this carrier experiment.",
            ],
        ))
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
        ap.error("canonical experiment requires development=128, heldout=256, repeats>=3")
    for p in (a.work, a.out):
        if p.exists() and any(p.iterdir()):
            raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    a.work = a.work.resolve()
    a.out = a.out.resolve()
    a.binary = a.binary.resolve()
    lockpath = Path.home() / ".cache/geoivf/speed-device.lock"
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(a)

if __name__ == "__main__":
    main()
