#!/usr/bin/env python3
"""Prepare the real Coveo SIGIR eCom 2021 search stream as a NavHints ANN workload.

Input files come from the original research release:
  * search_train.csv
  * sku_to_content.csv

The release documents search query vectors and catalog description vectors as
compatible dense representations. This converter:
  * keeps catalog rows with a valid description vector;
  * keeps search events with a valid query vector;
  * sorts events by the provided server timestamp (stable by source row);
  * L2-normalizes product and query vectors so inner product == cosine;
  * writes DiskANN-compatible .fbin arrays;
  * preserves timestamp/session/source-row metadata in JSONL;
  * records exact-repeat and session/locality diagnostics in a manifest.

It deliberately does not use clicked products as ANN ground truth. ANN recall is
defined against exact nearest neighbors in the released embedding space.
Clicks remain side metadata for later behavioral analyses.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_DIM = 50


def parse_vector(raw: str, *, expected_dim: int | None = None) -> np.ndarray | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        x = np.asarray(ast.literal_eval(raw), dtype=np.float32)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"invalid vector literal: {raw[:80]}") from e
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("vector must be finite and one-dimensional")
    if expected_dim is not None and x.size != expected_dim:
        raise ValueError(f"vector dimension {x.size}, expected {expected_dim}")
    return x


def parse_list(raw: str) -> list[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        out = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"invalid list literal: {raw[:80]}") from e
    if not isinstance(out, list):
        raise ValueError("expected list literal")
    return [str(x) for x in out]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(n)) or np.any(n <= 0):
        raise ValueError("zero/invalid vector norm")
    return (x / n).astype(np.float32, copy=False)


def write_fbin(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32, order="C")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<II", x.shape[0], x.shape[1]))
        x.astype("<f4", copy=False).tofile(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def quantiles(a: np.ndarray) -> dict[str, float]:
    if len(a) == 0:
        return {}
    return {
        str(q): float(np.quantile(a, q))
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    }


def read_catalog(path: Path) -> tuple[np.ndarray, list[str], dict[str, int], dict]:
    vecs: list[np.ndarray] = []
    skus: list[str] = []
    dropped_missing = 0
    dropped_zero = 0
    dim: int | None = None

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"product_sku_hash", "description_vector"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"catalog missing fields {required}")
        for row in reader:
            raw = row.get("description_vector", "")
            if not raw or not raw.strip():
                dropped_missing += 1
                continue
            v = parse_vector(raw, expected_dim=dim)
            if v is None:
                dropped_missing += 1
                continue
            if dim is None:
                dim = int(v.size)
            if not np.isfinite(v).all() or float(np.linalg.norm(v)) <= 0:
                dropped_zero += 1
                continue
            sku = row["product_sku_hash"]
            if not sku:
                raise ValueError("nonempty vector with empty SKU")
            skus.append(sku)
            vecs.append(v)

    if not vecs:
        raise ValueError("no valid catalog vectors")
    if len(set(skus)) != len(skus):
        raise ValueError("duplicate SKU hashes among vectorized products")

    x = normalize_rows(np.stack(vecs))
    sku_to_id = {sku: i for i, sku in enumerate(skus)}
    stats = {
        "catalog_rows_with_vectors": len(skus),
        "catalog_rows_dropped_missing_vector": dropped_missing,
        "catalog_rows_dropped_invalid_norm": dropped_zero,
        "dimension": int(x.shape[1]),
    }
    return x, skus, sku_to_id, stats


def read_queries(
    path: Path,
    *,
    dim: int,
    sku_to_id: dict[str, int],
) -> tuple[np.ndarray, list[dict], dict]:
    records: list[tuple[int, int, np.ndarray, dict]] = []
    dropped_missing = 0
    dropped_zero = 0
    clicked_total = 0
    clicked_mapped = 0
    result_total = 0
    result_mapped = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "session_id_hash",
            "query_vector",
            "server_timestamp_epoch_ms",
            "product_skus_hash",
            "clicked_skus_hash",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"search file missing fields {required}")
        for source_row, row in enumerate(reader):
            raw = row.get("query_vector", "")
            if not raw or not raw.strip():
                dropped_missing += 1
                continue
            v = parse_vector(raw, expected_dim=dim)
            if v is None:
                dropped_missing += 1
                continue
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n <= 0:
                dropped_zero += 1
                continue
            ts = int(row["server_timestamp_epoch_ms"])
            clicks = parse_list(row.get("clicked_skus_hash", ""))
            results = parse_list(row.get("product_skus_hash", ""))
            mapped_clicks = [sku_to_id[s] for s in clicks if s in sku_to_id]
            mapped_results = [sku_to_id[s] for s in results if s in sku_to_id]
            clicked_total += len(clicks)
            clicked_mapped += len(mapped_clicks)
            result_total += len(results)
            result_mapped += len(mapped_results)
            meta = {
                "timestamp_ms": ts,
                "session_id_hash": row["session_id_hash"],
                "source_row": source_row,
                "clicked_product_ids": mapped_clicks,
                "returned_product_ids": mapped_results,
            }
            records.append((ts, source_row, v, meta))

    if not records:
        raise ValueError("no valid search query vectors")

    records.sort(key=lambda x: (x[0], x[1]))
    q = normalize_rows(np.stack([x[2] for x in records]))
    meta = [x[3] for x in records]

    unique_sessions = len({m["session_id_hash"] for m in meta})
    timestamps = np.asarray([m["timestamp_ms"] for m in meta], dtype=np.int64)
    gaps = np.diff(timestamps) if len(timestamps) > 1 else np.empty(0, dtype=np.int64)

    # Exact-vector repeats after float32 parsing and normalization.
    seen: set[bytes] = set()
    repeats = 0
    repeat_distances: list[int] = []
    last_seen: dict[bytes, int] = {}
    for i, row in enumerate(q):
        key = row.tobytes()
        if key in seen:
            repeats += 1
            repeat_distances.append(i - last_seen[key])
        seen.add(key)
        last_seen[key] = i

    stats = {
        "search_events_with_vectors": len(meta),
        "search_events_dropped_missing_vector": dropped_missing,
        "search_events_dropped_invalid_norm": dropped_zero,
        "unique_sessions": unique_sessions,
        "first_timestamp_ms": int(timestamps[0]),
        "last_timestamp_ms": int(timestamps[-1]),
        "interarrival_ms_quantiles": quantiles(gaps.astype(np.float64)),
        "exact_repeated_query_vectors": repeats,
        "exact_repeat_fraction": float(repeats / len(meta)),
        "repeat_distance_events_quantiles": quantiles(np.asarray(repeat_distances, dtype=np.float64)),
        "clicked_skus_total": clicked_total,
        "clicked_skus_present_in_vector_catalog": clicked_mapped,
        "clicked_sku_catalog_coverage": float(clicked_mapped / clicked_total) if clicked_total else None,
        "returned_skus_total": result_total,
        "returned_skus_present_in_vector_catalog": result_mapped,
        "returned_sku_catalog_coverage": float(result_mapped / result_total) if result_total else None,
        "dimension": dim,
    }
    return q, meta, stats


def write_metadata(path: Path, meta: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(meta):
            f.write(json.dumps({"query": i, **row}, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-csv", type=Path, required=True)
    ap.add_argument("--catalog-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--expected-dim", type=int, default=DEFAULT_DIM)
    args = ap.parse_args()

    args.search_csv = args.search_csv.resolve()
    args.catalog_csv = args.catalog_csv.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base, skus, sku_to_id, cat_stats = read_catalog(args.catalog_csv)
    if base.shape[1] != args.expected_dim:
        raise ValueError(f"catalog dimension {base.shape[1]} != expected {args.expected_dim}")

    queries, meta, query_stats = read_queries(
        args.search_csv, dim=base.shape[1], sku_to_id=sku_to_id
    )

    base_path = args.out_dir / "coveo-products-cosine.fbin"
    query_path = args.out_dir / "coveo-search-chronological-cosine.fbin"
    meta_path = args.out_dir / "coveo-search-chronological.jsonl"
    sku_path = args.out_dir / "coveo-product-id-map.tsv"

    write_fbin(base_path, base)
    write_fbin(query_path, queries)
    write_metadata(meta_path, meta)
    with sku_path.open("w", encoding="utf-8") as f:
        for i, sku in enumerate(skus):
            f.write(f"{i}\t{sku}\n")

    manifest = {
        "dataset": "Coveo SIGIR eCom 2021 Data Challenge",
        "workload_kind": "real production e-commerce search events",
        "query_order": "ascending server_timestamp_epoch_ms, stable by source row",
        "metric": "cosine implemented as inner product after L2 normalization",
        "semantic_note": "release documents query_vector and description_vector as compatible",
        "catalog": cat_stats,
        "queries": query_stats,
        "files": {
            "base_fbin": base_path.name,
            "queries_fbin": query_path.name,
            "query_metadata_jsonl": meta_path.name,
            "product_id_map_tsv": sku_path.name,
        },
        "source_sha256": {
            "search_csv": sha256(args.search_csv),
            "catalog_csv": sha256(args.catalog_csv),
        },
        "output_sha256": {
            "base_fbin": sha256(base_path),
            "queries_fbin": sha256(query_path),
            "query_metadata_jsonl": sha256(meta_path),
            "product_id_map_tsv": sha256(sku_path),
        },
    }
    (args.out_dir / "coveo-ann.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
