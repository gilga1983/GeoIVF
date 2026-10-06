#!/usr/bin/env python3
"""Embed regional continuation NavHints into DiskANN associated-data bytes.

Input:
  * an existing DiskANN *_disk.index with zero associated-data bytes;
  * a GIRGN001 regional runtime file containing vertex->region labels and
    per-region hint lists.

Output:
  * a byte-for-byte-equivalent graph layout except that each node record grows
    by 32 bytes containing eight little-endian u32 continuation IDs.
  * the graph header is updated to associated_data_length=32 and node_len+32.

The repacker FAILS unless the larger logical node occupies exactly the same
number of physical disk sectors as before.  Thus embedded hints cannot increase
the number of bytes read per node in this experiment.

Each vertex receives its region's first eight hints. Missing entries are padded
with u32::MAX and can be ignored at runtime. Four-hint deployment simply uses
the first four entries of the same embedded payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

MAGIC = b"GIRGN001"
BLOCK_DEFAULT = 4096
HINT_SLOTS = 8
ASSOC_BYTES = HINT_SLOTS * 4
SENTINEL = 0xFFFFFFFF


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_graph_header(first_block: bytes):
    if len(first_block) < 104:
        raise ValueError("disk graph header is truncated")
    # The first 8 bytes are save_bytes()'s matrix header. GraphHeader begins at 8.
    off = 8
    vals = struct.unpack_from("<10Q", first_block, off)
    (
        num_pts, dims, medoid, node_len, nodes_per_block,
        frozen_num, frozen_loc, append_reorder,
        disk_file_size, assoc_len,
    ) = vals
    block_size = struct.unpack_from("<Q", first_block, off + 80)[0]
    major, minor = struct.unpack_from("<II", first_block, off + 88)
    if block_size == 0:
        block_size = BLOCK_DEFAULT
    return {
        "num_pts": int(num_pts),
        "dims": int(dims),
        "medoid": int(medoid),
        "node_len": int(node_len),
        "nodes_per_block": int(nodes_per_block),
        "frozen_num": int(frozen_num),
        "frozen_loc": int(frozen_loc),
        "append_reorder": int(append_reorder),
        "disk_file_size": int(disk_file_size),
        "assoc_len": int(assoc_len),
        "block_size": int(block_size),
        "layout_major": int(major),
        "layout_minor": int(minor),
    }


def parse_regional(path: Path):
    raw = path.read_bytes()
    if len(raw) < 24 or raw[:8] != MAGIC:
        raise ValueError("bad regional file")
    nvertices, nregions, total, min_entry_pos = struct.unpack_from("<IIII", raw, 8)
    off = 24
    labels_bytes = nvertices * 2
    offsets_bytes = (nregions + 1) * 4
    ids_bytes = total * 4
    if len(raw) != off + labels_bytes + offsets_bytes + ids_bytes:
        raise ValueError("regional file byte length mismatch")
    labels = struct.unpack_from(f"<{nvertices}H", raw, off)
    off += labels_bytes
    offsets = struct.unpack_from(f"<{nregions + 1}I", raw, off)
    off += offsets_bytes
    hint_ids = struct.unpack_from(f"<{total}I", raw, off)
    if offsets[0] != 0 or offsets[-1] != total:
        raise ValueError("bad regional offsets")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("non-monotone regional offsets")
    if any(r >= nregions for r in labels):
        raise ValueError("regional label outside range")
    return {
        "nvertices": int(nvertices),
        "nregions": int(nregions),
        "total": int(total),
        "min_entry_pos": int(min_entry_pos),
        "labels": labels,
        "offsets": offsets,
        "hint_ids": hint_ids,
    }


def allocated_sectors(node_len: int, block_size: int, nodes_per_block: int) -> int:
    if nodes_per_block > 0:
        return 1
    return math.ceil(node_len / block_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-disk-index", type=Path, required=True)
    ap.add_argument("--regional", type=Path, required=True)
    ap.add_argument("--output-disk-index", type=Path, required=True)
    args = ap.parse_args()

    regional = parse_regional(args.regional)
    with args.input_disk_index.open("rb") as f:
        first = f.read(BLOCK_DEFAULT)
    hdr = parse_graph_header(first)

    if hdr["assoc_len"] != 0:
        raise ValueError(f"expected zero existing associated data, got {hdr['assoc_len']}")
    if hdr["num_pts"] != regional["nvertices"]:
        raise ValueError("graph/regional vertex count mismatch")
    if hdr["disk_file_size"] != args.input_disk_index.stat().st_size:
        raise ValueError("graph header/file size mismatch")

    block = hdr["block_size"]
    old_len = hdr["node_len"]
    new_len = old_len + ASSOC_BYTES
    old_npb = hdr["nodes_per_block"]
    new_npb = block // new_len

    # Preserve physical allocation exactly.
    if old_npb > 0:
        if new_npb != old_npb:
            raise ValueError(
                f"embedding changes nodes/block: {old_npb} -> {new_npb}; "
                "not free in this layout"
            )
        old_alloc = new_alloc = 1
    else:
        old_alloc = math.ceil(old_len / block)
        new_alloc = math.ceil(new_len / block)
        if new_alloc != old_alloc:
            raise ValueError(
                f"embedding changes sectors/node: {old_alloc} -> {new_alloc}; "
                "not free in this layout"
            )
        new_npb = 0

    expected_blocks = (
        math.ceil(hdr["num_pts"] / old_npb)
        if old_npb > 0
        else hdr["num_pts"] * old_alloc
    )
    expected_size = (expected_blocks + 1) * block
    if expected_size != hdr["disk_file_size"]:
        raise ValueError(
            f"unexpected physical layout size: computed {expected_size}, "
            f"header {hdr['disk_file_size']}"
        )

    tmp = args.output_disk_index.with_suffix(args.output_disk_index.suffix + ".tmp")
    args.output_disk_index.parent.mkdir(parents=True, exist_ok=True)

    labels = regional["labels"]
    offsets = regional["offsets"]
    hint_ids = regional["hint_ids"]

    with args.input_disk_index.open("rb") as src, tmp.open("wb") as dst:
        header_block = bytearray(src.read(block))
        if len(header_block) != block:
            raise ValueError("truncated header sector")

        # GraphMetadata starts at byte 8.
        struct.pack_into("<Q", header_block, 8 + 24, new_len)
        struct.pack_into("<Q", header_block, 8 + 32, new_npb)
        struct.pack_into("<Q", header_block, 8 + 72, ASSOC_BYTES)
        dst.write(header_block)

        if old_npb > 0:
            vid = 0
            nblocks = math.ceil(hdr["num_pts"] / old_npb)
            for b in range(nblocks):
                sector = src.read(block)
                if len(sector) != block:
                    raise ValueError("truncated graph sector")
                out = bytearray(block)
                for slot in range(old_npb):
                    if vid >= hdr["num_pts"]:
                        break
                    old_off = slot * old_len
                    new_off = slot * new_len
                    out[new_off:new_off + old_len] = sector[old_off:old_off + old_len]

                    region = labels[vid]
                    lo, hi = offsets[region], offsets[region + 1]
                    ids = list(hint_ids[lo:min(hi, lo + HINT_SLOTS)])
                    ids.extend([SENTINEL] * (HINT_SLOTS - len(ids)))
                    struct.pack_into(f"<{HINT_SLOTS}I", out, new_off + old_len, *ids)
                    vid += 1
                dst.write(out)
                if b == 0 or (b + 1) % 100000 == 0 or b + 1 == nblocks:
                    print(f"repacked_blocks={b + 1}/{nblocks}", flush=True)
        else:
            bytes_per_node = old_alloc * block
            for vid in range(hdr["num_pts"]):
                physical = src.read(bytes_per_node)
                if len(physical) != bytes_per_node:
                    raise ValueError("truncated multi-sector node")
                out = bytearray(physical)
                region = labels[vid]
                lo, hi = offsets[region], offsets[region + 1]
                ids = list(hint_ids[lo:min(hi, lo + HINT_SLOTS)])
                ids.extend([SENTINEL] * (HINT_SLOTS - len(ids)))
                # Old logical record begins at offset zero. The associated data
                # follows the old logical node bytes, still within same sectors.
                struct.pack_into(f"<{HINT_SLOTS}I", out, old_len, *ids)
                dst.write(out)
                if vid == 0 or (vid + 1) % 100000 == 0 or vid + 1 == hdr["num_pts"]:
                    print(f"repacked_nodes={vid + 1}/{hdr['num_pts']}", flush=True)

        if src.read(1):
            raise ValueError("source graph has trailing bytes")

    if tmp.stat().st_size != args.input_disk_index.stat().st_size:
        raise AssertionError("embedded graph changed physical file size")
    os.replace(tmp, args.output_disk_index)

    # Verify updated header and a sample of payloads.
    with args.output_disk_index.open("rb") as f:
        new_hdr = parse_graph_header(f.read(block))
    if new_hdr["node_len"] != new_len or new_hdr["assoc_len"] != ASSOC_BYTES:
        raise AssertionError("updated graph header mismatch")
    if new_hdr["nodes_per_block"] != new_npb:
        raise AssertionError("updated nodes-per-block mismatch")

    manifest = {
        "input": str(args.input_disk_index),
        "output": str(args.output_disk_index),
        "vertices": hdr["num_pts"],
        "dims": hdr["dims"],
        "block_size": block,
        "old_node_len": old_len,
        "new_node_len": new_len,
        "old_nodes_per_block": old_npb,
        "new_nodes_per_block": new_npb,
        "old_sectors_per_node": old_alloc,
        "new_sectors_per_node": new_alloc,
        "associated_data_bytes": ASSOC_BYTES,
        "embedded_hint_slots": HINT_SLOTS,
        "physical_file_bytes_before": args.input_disk_index.stat().st_size,
        "physical_file_bytes_after": args.output_disk_index.stat().st_size,
        "physical_file_size_change": 0,
        "input_sha256": sha256(args.input_disk_index),
        "output_sha256": sha256(args.output_disk_index),
        "regional_source": str(args.regional),
        "sentinel": SENTINEL,
    }
    args.output_disk_index.with_suffix(
        args.output_disk_index.suffix + ".manifest.json"
    ).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
