#!/usr/bin/env python3
"""Enable the Coveo-proven exact-decision PQ optimization on pinned Starling BigANN.

Applied to the temporary pinned author checkout AFTER:
 recent hints -> native junctions -> entry diagnostics -> selective diverse.
No Coveo-specific input assumptions, L2^2 BigANN 10M×128 uint8 intact.
"""
from pathlib import Path
import argparse
from patch_starling_fast_core import page, self_test, once


def header(root):
    p=root/"include/pq_flash_index.h"
    s=p.read_text()
    s=once(s,
        "    bool want_diverse = false;",
        "    bool want_diverse = false;\n"
        "    bool fast_selector = false; // optimization only, original hint IDs",
        "flag exact-policy fast PQ scoring")
    p.write_text(s)


def benchmark(root):
    p=root/"tests/search_disk_index.cpp"
    s=p.read_text()
    s=once(s,
       '        nav_mode != "gate25_diverse") {',
       '        nav_mode != "gate25_diverse" &&\n'
       '        nav_mode != "fast_core" && nav_mode != "fast_recent512") {',
       "BigANN fast policies")
    s=once(s,
       '             nav_mode == "gate25_diverse") ? &recent :',
       '             nav_mode == "gate25_diverse" || nav_mode == "fast_core" ||\n'
       '             nav_mode == "fast_recent512") ? &recent :',
       "pass same causal Recent512")
    s=once(s,
       '             nav_mode == "gate25_diverse")\n'
       '            ? native_junctions.get() : nullptr;',
       '             nav_mode == "gate25_diverse" || nav_mode == "fast_core")\n'
       '            ? native_junctions.get() : nullptr;',
       "reuse identical original Starling 16K junction directory")
    s=once(s,
       '        nav_diag[i].gate_threshold = gate_threshold;',
       '        nav_diag[i].fast_selector =\n'
       '            (nav_mode == "fast_core" || nav_mode == "fast_recent512");\n'
       '        nav_diag[i].gate_threshold = gate_threshold;',
       "activate exact fast path only on new modes")
    s=once(s,
       '            nav_mode == "gate25_diverse") {',
       '            nav_mode == "gate25_diverse" ||\n'
       '            nav_mode == "fast_core" || nav_mode == "fast_recent512") {',
       "identical FIFO update logic for all dynamic modes")
    p.write_text(s)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",nargs="?",type=Path)
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test:self_test()
    if args.source is not None:
        root=args.source.resolve()
        header(root)
        page(root)
        benchmark(root)
        print("STARLING_FAST_BIGANN_ADAPTER_READY "
              "dataset=BigANN10M exact_hint_ID=1 original_search=1",
              flush=True)
    elif not args.self_test:
        ap.error("require original Starling checkout")
if __name__=="__main__":main()
