#!/usr/bin/env python3
"""Build and validate a pinned DiskANN disk index for a prepared public dataset."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--dataset-manifest", type=Path, required=True)
    ap.add_argument("--save-prefix", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--build-ram-gb", type=float, default=32.0)
    args = ap.parse_args()

    for n in ("binary", "dataset_manifest", "save_prefix", "work"):
        setattr(args, n, getattr(args, n).resolve())
    args.work.mkdir(parents=True, exist_ok=True)
    args.save_prefix.parent.mkdir(parents=True, exist_ok=True)

    m = json.loads(args.dataset_manifest.read_text())
    pq_chunks = min(64, int(m["dim"]))
    cfg = {
        "search_directories": [str(args.work)],
        "jobs": [{
            "type": "disk-index",
            "content": {
                "source": {
                    "disk-index-source": "Build",
                    "data_type": m["data_type"],
                    "data": m["files"]["base"],
                    "distance": m["metric"],
                    "dim": m["dim"],
                    "max_degree": 64,
                    "l_build": 100,
                    "num_threads": 4,
                    "build_ram_limit_gb": args.build_ram_gb,
                    "num_pq_chunks": pq_chunks,
                    "quantization_type": "FP",
                    "save_path": str(args.save_prefix),
                },
                "search_phase": {
                    "queries": m["files"]["heldout5000"],
                    "groundtruth": m["files"]["heldout5000_gt"],
                    "search_list": [10],
                    "beam_width": 8,
                    "recall_at": 10,
                    "num_threads": 4,
                    "is_flat_search": False,
                    "distance": m["metric"],
                    "vector_filters_file": None,
                    "num_nodes_to_cache": None,
                    "search_io_limit": None,
                    "post_processor": None,
                },
            },
        }],
    }
    inp = args.work / "build.input.json"
    out = args.work / "build.output.json"
    save(inp, cfg)
    with (args.work / "build.log").open("w") as log:
        subprocess.run(
            [str(args.binary), "run", "--input-file", str(inp), "--output-file", str(out)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    marker = args.save_prefix.parent / (args.save_prefix.name + ".complete.json")
    result = {
        "dataset": m["dataset"],
        "save_prefix": str(args.save_prefix),
        "build": {
            "R": 64,
            "L_build": 100,
            "threads": 4,
            "build_ram_gb": args.build_ram_gb,
            "pq_chunks": pq_chunks,
            "quantization_type": "FP",
        },
        "validation_output": str(out),
    }
    save(marker, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
