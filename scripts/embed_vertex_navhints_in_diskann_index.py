#!/usr/bin/env python3
"""Embed multiple vertex-granular NavHint variants into DiskANN node records.

Input payload GIVTX001 stores V variants x K u32 hint IDs for every graph
vertex. The entire payload for one vertex becomes DiskANN associated data.

The repacker fails unless the enlarged logical node has exactly the same
physical sector allocation and the resulting disk graph has exactly the same
file size as the original.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

PAYLOAD_MAGIC = b"GIVTX001"
BLOCK_DEFAULT = 4096


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_header(block):
    if len(block) < 104:
        raise ValueError("truncated disk graph header")
    vals = struct.unpack_from("<10Q", block, 8)
    block_size = struct.unpack_from("<Q", block, 88)[0] or BLOCK_DEFAULT
    return {
        "num_pts": int(vals[0]),
        "dims": int(vals[1]),
        "node_len": int(vals[3]),
        "nodes_per_block": int(vals[4]),
        "file_size": int(vals[8]),
        "assoc_len": int(vals[9]),
        "block_size": int(block_size),
    }


def parse_payload(path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != PAYLOAD_MAGIC:
        raise ValueError("bad vertex-hint payload")
    nvertices, nvariants, slots, min_position = struct.unpack_from("<IIII", raw, 8)
    off = 24
    variants = np.frombuffer(raw, dtype="<u4", count=nvariants, offset=off).copy()
    off += nvariants * 4
    count = nvertices * nvariants * slots
    expected = off + count * 4
    if len(raw) != expected:
        raise ValueError("vertex-hint payload size mismatch")
    data = np.frombuffer(raw, dtype="<u4", count=count, offset=off).reshape(
        nvertices, nvariants, slots
    )
    return int(nvertices), int(nvariants), int(slots), int(min_position), variants, data


import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-disk-index", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--output-disk-index", type=Path, required=True)
    args = ap.parse_args()

    nvertices, nvariants, slots, min_position, variants, payload = parse_payload(args.payload)
    with args.input_disk_index.open("rb") as f:
        first = f.read(BLOCK_DEFAULT)
    hdr = parse_header(first)

    if hdr["assoc_len"] != 0:
        raise ValueError("source graph already has associated data")
    if hdr["num_pts"] != nvertices:
        raise ValueError("graph/payload vertex mismatch")
    if hdr["file_size"] != args.input_disk_index.stat().st_size:
        raise ValueError("graph header/file size mismatch")

    assoc_bytes = nvariants * slots * 4
    block = hdr["block_size"]
    old_len = hdr["node_len"]
    new_len = old_len + assoc_bytes
    old_npb = hdr["nodes_per_block"]
    new_npb = block // new_len

    if old_npb > 0:
        old_alloc = new_alloc = 1
        if new_npb != old_npb:
            raise ValueError(f"node packing changes: {old_npb} -> {new_npb}")
    else:
        old_alloc = math.ceil(old_len / block)
        new_alloc = math.ceil(new_len / block)
        new_npb = 0
        if new_alloc != old_alloc:
            raise ValueError(f"sector allocation changes: {old_alloc} -> {new_alloc}")

    expected_blocks = (
        math.ceil(nvertices / old_npb) if old_npb > 0 else nvertices * old_alloc
    )
    if (expected_blocks + 1) * block != hdr["file_size"]:
        raise ValueError("unexpected source physical layout")

    tmp = args.output_disk_index.with_suffix(args.output_disk_index.suffix + ".tmp")
    args.output_disk_index.parent.mkdir(parents=True, exist_ok=True)

    with args.input_disk_index.open("rb") as src, tmp.open("wb") as dst:
        header = bytearray(src.read(block))
        struct.pack_into("<Q", header, 8 + 24, new_len)
        struct.pack_into("<Q", header, 8 + 32, new_npb)
        struct.pack_into("<Q", header, 8 + 72, assoc_bytes)
        dst.write(header)

        if old_npb > 0:
            vid = 0
            nblocks = math.ceil(nvertices / old_npb)
            for b in range(nblocks):
                sector = src.read(block)
                if len(sector) != block:
                    raise ValueError("truncated source graph")
                out = bytearray(block)
                for slot_idx in range(old_npb):
                    if vid >= nvertices:
                        break
                    old_off = slot_idx * old_len
                    new_off = slot_idx * new_len
                    out[new_off:new_off + old_len] = sector[old_off:old_off + old_len]
                    out[new_off + old_len:new_off + new_len] = payload[vid].tobytes(order="C")
                    vid += 1
                dst.write(out)
                if b == 0 or (b + 1) % 100000 == 0 or b + 1 == nblocks:
                    print(f"repacked_blocks={b+1}/{nblocks}", flush=True)
        else:
            physical_len = old_alloc * block
            for vid in range(nvertices):
                physical = src.read(physical_len)
                if len(physical) != physical_len:
                    raise ValueError("truncated source graph")
                out = bytearray(physical)
                out[old_len:new_len] = payload[vid].tobytes(order="C")
                dst.write(out)
                if vid == 0 or (vid + 1) % 100000 == 0 or vid + 1 == nvertices:
                    print(f"repacked_nodes={vid+1}/{nvertices}", flush=True)

        if src.read(1):
            raise ValueError("source graph has trailing bytes")

    if tmp.stat().st_size != args.input_disk_index.stat().st_size:
        raise AssertionError("physical graph size changed")
    os.replace(tmp, args.output_disk_index)

    with args.output_disk_index.open("rb") as f:
        new_hdr = parse_header(f.read(block))
    if new_hdr["node_len"] != new_len or new_hdr["assoc_len"] != assoc_bytes:
        raise AssertionError("embedded graph header mismatch")
    if new_hdr["nodes_per_block"] != new_npb:
        raise AssertionError("embedded graph packing mismatch")

    manifest = {
        "input": str(args.input_disk_index),
        "output": str(args.output_disk_index),
        "vertices": nvertices,
        "dims": hdr["dims"],
        "block_size": block,
        "old_node_len": old_len,
        "new_node_len": new_len,
        "associated_data_bytes": assoc_bytes,
        "variants": variants.tolist(),
        "variants_count": nvariants,
        "slots_per_variant": slots,
        "training_min_position": min_position,
        "old_nodes_per_block": old_npb,
        "new_nodes_per_block": new_npb,
        "old_sectors_per_node": old_alloc,
        "new_sectors_per_node": new_alloc,
        "physical_file_bytes_before": args.input_disk_index.stat().st_size,
        "physical_file_bytes_after": args.output_disk_index.stat().st_size,
        "physical_file_size_change": 0,
        "payload_source": str(args.payload),
        "input_sha256": sha256(args.input_disk_index),
        "output_sha256": sha256(args.output_disk_index),
    }
    args.output_disk_index.with_suffix(args.output_disk_index.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
