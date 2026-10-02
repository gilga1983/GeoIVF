#!/usr/bin/env python3
"""End-to-end qualification of the in-process DiskANN portal router.

The router is not precomputed per query. Python writes only the static tiny
centroid/portal table. DiskANN selects the portal inside the timed query path.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, struct, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np

from scripts.qualify_portal_suite import (
    DATASET_SPECS,
    FIXED_BEAM,
    FIXED_LS,
    FIXED_NPROBES,
    K,
    NLIST,
    build_portal_table,
    fbin,
    gtbin,
    load_dataset,
    result_rows,
)
from geoivf.index import train_faiss
from scripts.qualify_speed import save

PINNED_DISKANN = "fcf90534174cf29c78c9f13b4cccf1fcabff85f5"
CANONICAL = (
    "yahoo-minilm-384-normalized",
    "imagenet-clip-512-normalized",
    "coco-nomic-768-normalized",
)


def write_router(path: Path, centers: np.ndarray, portal_ids: np.ndarray, portal_vecs: np.ndarray):
    centers = np.asarray(centers, dtype="<f4", order="C")
    portal_ids = np.asarray(portal_ids, dtype="<u4", order="C")
    portal_vecs = np.asarray(portal_vecs, dtype="<f4", order="C")
    if centers.shape != portal_vecs.shape or portal_ids.shape != (len(centers),):
        raise ValueError("portal-router shape mismatch")
    if not np.isfinite(centers).all() or not np.isfinite(portal_vecs).all():
        raise ValueError("portal-router contains nonfinite coordinates")
    with path.open("wb") as f:
        f.write(b"GIPORT01")
        f.write(struct.pack("<II", centers.shape[0], centers.shape[1]))
        centers.tofile(f)
        portal_ids.tofile(f)
        portal_vecs.tofile(f)
    expected = 16 + centers.size * 4 + portal_ids.size * 4 + portal_vecs.size * 4
    if path.stat().st_size != expected:
        raise AssertionError("portal-router byte count mismatch")


def disk_run(binary: Path, work: Path, out: Path, tag: str, source: dict,
             router_file: Path | None = None, nprobe: int | None = None):
    phase = dict(
        queries=str(work / "heldout.fbin"),
        groundtruth=str(work / "heldout.gt"),
        search_list=list(FIXED_LS),
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
    inp = out / f"{tag}-input.json"
    output = out / f"{tag}-output.json"
    save(inp, cfg)

    env = os.environ.copy()
    env.pop("DISKANN_START_POINTS_FILE", None)
    env.pop("DISKANN_PORTAL_ROUTER_FILE", None)
    env.pop("DISKANN_PORTAL_NPROBE", None)
    if router_file is not None:
        if nprobe not in FIXED_NPROBES:
            raise ValueError("integrated router nprobe must be a frozen suite arm")
        env["DISKANN_PORTAL_ROUTER_FILE"] = str(router_file.resolve())
        env["DISKANN_PORTAL_NPROBE"] = str(nprobe)
    elif nprobe is not None:
        raise ValueError("nprobe without router table")

    with (out / f"{tag}.log").open("w") as log:
        subprocess.run(
            [str(binary), "run", "--input-file", str(inp), "--output-file", str(output)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    rows = result_rows(json.loads(output.read_text()))
    if len(rows) != len(FIXED_LS):
        raise ValueError(f"expected {len(FIXED_LS)} DiskANN rows, got {len(rows)}")
    by_l = {int(r["search_l"]): dict(r, source_output=output.name) for r in rows}
    if sorted(by_l) != sorted(FIXED_LS):
        raise ValueError(f"unexpected L values: {sorted(by_l)}")
    return [by_l[l] for l in FIXED_LS]


def summarize(rows):
    methods = ["medoid"] + [f"portal-np{x}" for x in FIXED_NPROBES]
    result = {}
    for method in methods:
        result[method] = {}
        for l in FIXED_LS:
            rr = [r for r in rows if r["method"] == method and int(r["search_l"]) == l]
            if not rr:
                raise ValueError(f"missing {method} L={l}")
            result[method][str(l)] = dict(
                rounds=len(rr),
                recall_percent=float(np.mean([r["recall"] for r in rr])),
                mean_end_to_end_us=float(np.mean([r["mean_latency"] for r in rr])),
                p95_end_to_end_us=float(np.mean([
                    float(str(r["p95_latency"]).removesuffix("us")) for r in rr
                ])),
                mean_route_us=float(np.mean([r.get("mean_route_latency", 0.0) for r in rr])),
                mean_ios=float(np.mean([r["mean_ios"] for r in rr])),
                mean_io_us=float(np.mean([r["mean_io_time"] for r in rr])),
                mean_cpu_us=float(np.mean([r["mean_cpu_time"] for r in rr])),
                mean_comparisons=float(np.mean([r["mean_comparisons"] for r in rr])),
                mean_hops=float(np.mean([r["mean_hops"] for r in rr])),
                qps=float(np.mean([r["qps"] for r in rr])),
            )

    for l in FIXED_LS:
        b = result["medoid"][str(l)]
        for method in methods[1:]:
            a = result[method][str(l)]
            a["latency_reduction_fraction_vs_medoid_same_l"] = (
                1.0 - a["mean_end_to_end_us"] / b["mean_end_to_end_us"]
            )
            a["io_reduction_fraction_vs_medoid_same_l"] = 1.0 - a["mean_ios"] / b["mean_ios"]
            a["comparison_reduction_fraction_vs_medoid_same_l"] = (
                1.0 - a["mean_comparisons"] / b["mean_comparisons"]
            )
            a["recall_delta_points_vs_medoid_same_l"] = (
                a["recall_percent"] - b["recall_percent"]
            )
            a["qps_improvement_fraction_vs_medoid_same_l"] = a["qps"] / b["qps"] - 1.0
    return result


def run(a):
    started = time.monotonic()
    faiss.omp_set_num_threads(1)
    spec = DATASET_SPECS[a.dataset]
    seed = int.from_bytes(hashlib.sha256(a.dataset.encode()).digest()[:8], "little")
    x, dev, held, dev_gt, held_gt, dev_ids, held_ids = load_dataset(a.data, spec, seed)
    del dev, dev_gt

    save(
        a.out / "cohorts.json",
        dict(
            dataset=a.dataset,
            seed=seed,
            development_ids=dev_ids.tolist(),
            heldout_ids=held_ids.tolist(),
            identical_to_fixed_portal_suite=True,
            nprobe_arms=list(FIXED_NPROBES),
            l_sweep=list(FIXED_LS),
            beam=FIXED_BEAM,
        ),
    )

    print(f"{a.dataset}: build same frozen 1024-cell portal table", flush=True)
    t = time.perf_counter()
    centers, labels = train_faiss(x, NLIST, 12345, min(100000, len(x)))
    ivf_build_s = time.perf_counter() - t
    t = time.perf_counter()
    portal_ids, portal_vecs = build_portal_table(x, centers, labels)
    portal_build_s = time.perf_counter() - t

    router_file = a.out / "portal-router.bin"
    write_router(router_file, centers, portal_ids, portal_vecs)
    np.savez(
        a.out / "portal-table-audit.npz",
        centers=centers,
        portal_ids=portal_ids,
        portal_vectors=portal_vecs,
    )

    fbin(a.work / "base.fbin", x)
    fbin(a.work / "heldout.fbin", held)
    gtbin(a.work / "heldout.gt", held_gt, x, held)

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
    load = dict(**{"disk-index-source": "Load"}, data_type="float32", load_path=prefix)

    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    rows = []
    try:
        for rep in range(a.repeats):
            arms = ["medoid"] + [f"portal-np{x}" for x in FIXED_NPROBES]
            if rep % 2:
                arms.reverse()
            for arm in arms:
                if arm == "medoid":
                    rr = disk_run(
                        a.binary,
                        a.work,
                        a.out,
                        f"heldout-{arm}-r{rep}",
                        build if rep == 0 else load,
                    )
                else:
                    npb = int(arm.split("np")[1])
                    rr = disk_run(
                        a.binary,
                        a.work,
                        a.out,
                        f"heldout-{arm}-r{rep}",
                        load,
                        router_file,
                        npb,
                    )
                for r in rr:
                    rows.append(dict(round=rep, method=arm, **r))

        save(a.out / "heldout-rows.json", rows)
        summary = summarize(rows)
        extra_bytes = router_file.stat().st_size - 16
        save(
            a.out / "integrated-portal-result.json",
            dict(
                dataset=a.dataset,
                dimension=int(x.shape[1]),
                train_rows=len(x),
                diskann_revision=PINNED_DISKANN,
                router_execution="inside DiskANN timed query path",
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
                portal_extra_bytes=extra_bytes,
                portal_extra_mib=extra_bytes / (1 << 20),
                ivf_build_seconds=ivf_build_s,
                portal_build_seconds=portal_build_s,
                heldout_summary=summary,
                cpu_affinity=[allowed[0]],
                elapsed_seconds=time.monotonic() - started,
                limitations=[
                    "One build per dataset on a shared host; node cache disabled.",
                    "Portal policy is geometric nearest-to-centroid with frozen nprobe arms 1/8/32.",
                    "DiskANN provider mean_ios is not an independently measured physical NVMe command count.",
                    "The router uses a compact scalar Rust implementation over 1024 centroids; no Faiss call occurs at query time.",
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
