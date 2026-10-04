#!/usr/bin/env python3
"""Prepare the CatapultDB paper's 1M PubMed / uniform-query workload.

Inputs are the public NCBI MedCPT PubMed embedding chunks. The first million
vectors are chunk 0 in full (977,492 rows) plus the first 22,508 rows of chunk 1.
The output base file is DiskANN's fbin format and the query stream is 150,000
768-D float32 vectors sampled uniformly from [-1, 1], matching the paper.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

DIM = 768
N = 1_000_000
CHUNK0_ROWS = 977_492
CHUNK1_ROWS_NEEDED = N - CHUNK0_ROWS
QUERY_ROWS = 150_000
QUERY_SEED = 0


def npy_header(path: Path) -> tuple[int, tuple[int, ...], np.dtype, bool]:
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path}: missing NPY magic")
        major, minor = f.read(2)
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        elif major in (2, 3):
            hlen = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError(f"{path}: unsupported NPY version {major}.{minor}")
        header = ast.literal_eval(f.read(hlen).decode("latin1").strip())
        offset = f.tell()
    return offset, tuple(header["shape"]), np.dtype(header["descr"]), bool(header["fortran_order"])


def validate_chunk(path: Path, expected_rows: int | None = None, allow_truncated_rows: int | None = None):
    offset, shape, dtype, fortran = npy_header(path)
    if len(shape) != 2 or shape[1] != DIM or dtype != np.dtype("<f4") or fortran:
        raise ValueError(f"{path}: unexpected NPY metadata shape={shape} dtype={dtype} fortran={fortran}")
    if expected_rows is not None and shape[0] != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {shape[0]}")
    available_bytes = path.stat().st_size - offset
    if allow_truncated_rows is None:
        expected_bytes = shape[0] * DIM * 4
        if available_bytes != expected_bytes:
            raise ValueError(f"{path}: expected {expected_bytes} payload bytes, got {available_bytes}")
        rows = shape[0]
    else:
        need = allow_truncated_rows * DIM * 4
        if available_bytes < need:
            raise ValueError(f"{path}: only {available_bytes} payload bytes, need {need}")
        rows = allow_truncated_rows
    return offset, rows


def copy_rows(src: Path, offset: int, rows: int, out, block_rows: int = 8192) -> None:
    mm = np.memmap(src, mode="r", dtype="<f4", offset=offset, shape=(rows, DIM), order="C")
    for s in range(0, rows, block_rows):
        np.asarray(mm[s:s + block_rows], dtype="<f4", order="C").tofile(out)


def write_base(chunk0: Path, chunk1_prefix: Path, out: Path) -> None:
    off0, rows0 = validate_chunk(chunk0, expected_rows=CHUNK0_ROWS)
    off1, rows1 = validate_chunk(chunk1_prefix, expected_rows=998_731, allow_truncated_rows=CHUNK1_ROWS_NEEDED)
    if rows0 + rows1 != N:
        raise AssertionError("first-million row count mismatch")

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.asarray([N, DIM], dtype="<u4").tofile(f)
        copy_rows(chunk0, off0, rows0, f)
        copy_rows(chunk1_prefix, off1, rows1, f)
    expected = 8 + N * DIM * 4
    if tmp.stat().st_size != expected:
        raise AssertionError(f"base byte count mismatch {tmp.stat().st_size} != {expected}")
    os.replace(tmp, out)


def write_queries(out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    rng = np.random.default_rng(QUERY_SEED)
    with tmp.open("wb") as f:
        np.asarray([QUERY_ROWS, DIM], dtype="<u4").tofile(f)
        for s in range(0, QUERY_ROWS, 4096):
            n = min(4096, QUERY_ROWS - s)
            q = rng.uniform(-1.0, 1.0, size=(n, DIM)).astype("<f4")
            q.tofile(f)
    expected = 8 + QUERY_ROWS * DIM * 4
    if tmp.stat().st_size != expected:
        raise AssertionError("query byte count mismatch")
    os.replace(tmp, out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk0", type=Path, required=True)
    ap.add_argument("--chunk1-prefix", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = args.out_dir / "pubmed-medcpt-first1m.fbin"
    queries = args.out_dir / "uniform-150k-seed0.fbin"

    if not base.exists():
        write_base(args.chunk0, args.chunk1_prefix, base)
    if not queries.exists():
        write_queries(queries)

    # The benchmark requires the GT path to resolve even when recall is explicitly skipped.
    placeholder = args.out_dir / "throughput-only.gt"
    if not placeholder.exists():
        placeholder.write_bytes(struct.pack("<II", 0, 0))

    meta = {
        "source": "NCBI MedCPT PubMed article embeddings",
        "dimension": DIM,
        "base_rows": N,
        "chunk0_rows": CHUNK0_ROWS,
        "chunk1_rows_used": CHUNK1_ROWS_NEEDED,
        "query_rows": QUERY_ROWS,
        "query_distribution": "iid uniform [-1,1]^768",
        "query_seed": QUERY_SEED,
        "base_bytes": base.stat().st_size,
        "query_bytes": queries.stat().st_size,
        "base_sha256": sha256(base),
        "queries_sha256": sha256(queries),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
