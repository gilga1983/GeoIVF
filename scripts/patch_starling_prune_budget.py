#!/usr/bin/env python3
"""Add component removal and compact-budget modes to original-native Starling.

Run AFTER patch_starling_final_core.py on the same pinned temporary original
Starling checkout. Existing baseline and final_core are left unchanged.
Every ablation skips the removed component's entire per-query PQ scan.

Modes:
  junction_only       B=14464, C=0,   k=0
  destination_only    B=0,     C=2048,k=8
  junction_compact    B=4096,  C=0,   k=0
  destination_compact B=0,     C=512, k=2
  joint_compact       B=4096,  C=512, k=2

Uses Starling's original graph, memory navigator, SSD layout, PQ codes,
frontier and termination. Training prefix and measured suffix remain disjoint.
"""
from pathlib import Path
import argparse


def once(source: str, before: str, after: str, why: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{why}: expected one exact anchor, got {count}")
    return source.replace(before, after, 1)


def patch_header(root: Path) -> None:
    path = root / "include/pq_flash_index.h"
    source = path.read_text()
    source = once(
        source,
        "    bool final_starts = false; // B14464, Recent2048, eight scored starts",
        "    bool final_starts = false; // enable ranked destination starts\n"
        "    unsigned final_max_starts = 8; // 2 for compact Starling ablations",
        "declare per-query destination-start count")
    path.write_text(source)


def patch_search(root: Path) -> None:
    path = root / "src/page_search.cpp"
    source = path.read_text()
    first = source.index("  void PQFlashIndex<T>::page_search(\n")
    last = source.index("  void PQFlashIndex<T>::page_search_interim(\n", first)
    head, body, tail = source[:first], source[first:last], source[last:]

    # Both sorting capacity and offered entries follow the chosen k.
    # The PQ scan itself is bypassed when the recent pointer is null.
    body = once(
        body,
        "    if (navhints_debug != nullptr && navhints_debug->final_starts)\n"
        "      final_recent.reserve(8);",
        "    if (navhints_debug != nullptr && navhints_debug->final_starts)\n"
        "      final_recent.reserve(navhints_debug->final_max_starts);",
        "allocate only selected top-k destination slots")
    body = once(
        body,
        "if (final_recent.size() < 8 || offer < final_recent.back())",
        "if (final_recent.size() < navhints_debug->final_max_starts ||\n"
        "                offer < final_recent.back())",
        "bounded top-k selection")
    body = once(
        body,
        "if (final_recent.size() > 8) final_recent.pop_back();",
        "if (final_recent.size() > navhints_debug->final_max_starts)\n"
        "                final_recent.pop_back();",
        "bounded top-k eviction")
    path.write_text(head + body + tail)


def patch_benchmark(root: Path) -> None:
    path = root / "tests/search_disk_index.cpp"
    source = path.read_text()
    new_modes = (
        '        nav_mode != "junction_only" &&\n'
        '        nav_mode != "destination_only" &&\n'
        '        nav_mode != "junction_compact" &&\n'
        '        nav_mode != "destination_compact" &&\n'
        '        nav_mode != "joint_compact") {'
    )
    source = once(
        source, '        nav_mode != "final_core") {',
        '        nav_mode != "final_core" &&\n' + new_modes,
        "register five independent Starling ablation modes")

    source = once(
        source,
        '             nav_mode == "final_core") ? &recent :',
        '             nav_mode == "final_core" ||\n'
        '             nav_mode == "destination_only" ||\n'
        '             nav_mode == "destination_compact" ||\n'
        '             nav_mode == "joint_compact") ? &recent :',
        "remove all destination scoring in junction-only variants")
    source = once(
        source,
        'nav_mode == "fast_core" || nav_mode == "final_core")\n'
        '            ? native_junctions.get() : nullptr;',
        'nav_mode == "fast_core" || nav_mode == "final_core" ||\n'
        '             nav_mode == "junction_only" ||\n'
        '             nav_mode == "junction_compact" ||\n'
        '             nav_mode == "joint_compact")\n'
        '            ? native_junctions.get() : nullptr;',
        "remove all junction scoring in destination-only variants")

    source = once(
        source,
        '            (nav_mode == "fast_core" || nav_mode == "fast_recent512" ||\n'
        '             nav_mode == "final_core");\n'
        '        nav_diag[i].final_starts = (nav_mode == "final_core");',
        '            (nav_mode == "fast_core" || nav_mode == "fast_recent512" ||\n'
        '             nav_mode == "final_core" ||\n'
        '             nav_mode == "junction_only" ||\n'
        '             nav_mode == "destination_only" ||\n'
        '             nav_mode == "junction_compact" ||\n'
        '             nav_mode == "destination_compact" ||\n'
        '             nav_mode == "joint_compact");\n'
        '        nav_diag[i].final_starts =\n'
        '            (nav_mode == "final_core" ||\n'
        '             nav_mode == "destination_only" ||\n'
        '             nav_mode == "destination_compact" ||\n'
        '             nav_mode == "joint_compact");\n'
        '        nav_diag[i].final_max_starts =\n'
        '            (nav_mode == "destination_compact" ||\n'
        '             nav_mode == "joint_compact") ? 2u : 8u;',
        "enable fast PQ scorer and correct top-k for each ablation")

    source = once(
        source,
        '            nav_mode == "final_core") {',
        '            nav_mode == "final_core" ||\n'
        '            nav_mode == "destination_only" ||\n'
        '            nav_mode == "destination_compact" ||\n'
        '            nav_mode == "joint_compact") {',
        "update FIFO only for destination-enabled modes")
    source = once(
        source,
        '      const size_t nav_recent_budget =\n'
        '          nav_mode == "final_core" ? 2048 : 512;',
        '      const size_t nav_recent_budget =\n'
        '          (nav_mode == "final_core" ||\n'
        '           nav_mode == "destination_only") ? 2048 : 512;',
        "assign 512 or 2048 recent IDs")
    source = once(
        source,
        '" recent_capacity=" << (nav_mode == "final_core" ? 2048 : 512)',
        '" recent_capacity=" << ((nav_mode == "final_core" ||\n'
        '      nav_mode == "destination_only") ? 2048 : 512)',
        "log destination capacities")
    source = once(
        source,
        "(total!=16000 && total!=14464) || children+nlist!=total",
        "(total!=16000 && total!=14464 && total!=4096) || children+nlist!=total",
        "load compact 4096-ID junction directory")
    path.write_text(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    for target in (
        "include/pq_flash_index.h",
        "src/page_search.cpp",
        "tests/search_disk_index.cpp",
    ):
        if not (root / target).is_file():
            raise FileNotFoundError(root / target)
    patch_header(root)
    patch_search(root)
    patch_benchmark(root)
    print("STARLING_PRUNE_BUDGET_PATCH_PASS "
          "modes=junction_only,destination_only,junction_compact,"
          "destination_compact,joint_compact; "
          "removed_component_pq_calls=0; original_native_search=1",
          flush=True)


if __name__ == "__main__":
    main()
