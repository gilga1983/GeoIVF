#!/usr/bin/env python3
"""Port final NavHints B14464, Recent2048, eight ranked starts to original Starling.

Patch after the native Starling source-fidelity adapter and its fast packed
PQ selector. 'baseline', 'fast_core' (old control) and other modes retain
their original behavior. The ONLY new mode is 'final_core'.
No edges, page layout, memory navigator, read scheduler or termination change.

Original pinned Starling source: 17dc3e8a011533a62374445f53963e951b72883a.
Full index has its own 16K/512 training-only routing directory, while
final_core loads a separately learned 14,464/512 directory from the same
first 5K native Starling traversals. The destination history remains globally
unique FIFO, now 2,048 IDs and up to 8 PQ-ranked additions to native retset.
"""
from pathlib import Path
import argparse


def once(s, before, after, why):
    n = s.count(before)
    if n != 1:
        raise RuntimeError(f"{why}: expected one exact source anchor, got {n}")
    return s.replace(before, after, 1)


def header(root):
    f = root / "include/pq_flash_index.h"
    s = f.read_text()
    s = once(
        s,
        "    bool fast_selector = false;",
        "    bool final_starts = false; // B14464, Recent2048, eight scored starts\n"
        "    bool fast_selector = false;",
        "final-mode scorer toggle",
    )
    f.write_text(s)


def page(root):
    f = root / "src/page_search.cpp"
    s = f.read_text()
    first = s.index("  void PQFlashIndex<T>::page_search(\n")
    last = s.index("  void PQFlashIndex<T>::page_search_interim(\n", first)
    before, part, after = s[:first], s[first:last], s[last:]

    needle = """    auto nh_fast_score = [&](const unsigned *ids, size_t count) {"""
    part = once(
        part,
        needle,
        """    // Sorted best-eight after a single PQ pass over the *same* recent IDs.
    std::vector<std::pair<float,unsigned>> final_recent;
    if (navhints_debug != nullptr && navhints_debug->final_starts)
      final_recent.reserve(8);
"""
        + needle,
        "top eight selector state before fast PQ scoring",
    )

    a = part.index("    if (navhints_active && navhints_recent != nullptr")
    b = part.index("    // Two-stage Hint-IVF search:", a)
    recent = part[a:b]
    recent = once(
        recent,
        """          const float d=nh_scores[i];
          if (navhints_diverse)""",
        """          const float d=nh_scores[i];
          if (navhints_debug != nullptr && navhints_debug->final_starts) {
            const std::pair<float,unsigned> offer{d,id};
            if (final_recent.size() < 8 || offer < final_recent.back()) {
              const auto pos=std::lower_bound(
                  final_recent.begin(),final_recent.end(),offer);
              final_recent.insert(pos,offer);
              if (final_recent.size() > 8) final_recent.pop_back();
            }
          }
          if (navhints_diverse)""",
        "collect ranked eight from already scored PQ candidates",
    )

    target = """      if (found && cur_list_size < l_search &&
          (navhints_debug == nullptr || navhints_debug->inject)) {"""
    replace = """      if (navhints_debug != nullptr && navhints_debug->final_starts) {
        // Leave space for the selected learned junction if possible. Starling
        // may already supply native memory-navigator candidates.
        for (const auto &offer : final_recent) {
          if (cur_list_size + 1 >= l_search) break;
          if (visited.find(offer.second) != visited.end()) continue;
          retset[cur_list_size++] =
              Neighbor(offer.second,offer.first,true);
          visited.insert(offer.second);
          navhints_debug->recent_inserted = 1;
        }
      } else if (found && cur_list_size < l_search &&
          (navhints_debug == nullptr || navhints_debug->inject)) {"""
    recent = once(recent, target, replace, "offer up to eight scored starts")
    part = part[:a] + recent + part[b:]
    f.write_text(before + part + after)


def benchmark(root):
    f = root / "tests/search_disk_index.cpp"
    s = f.read_text()
    s = once(
        s,
        '        nav_mode != "fast_core" && nav_mode != "fast_recent512") {',
        '        nav_mode != "fast_core" && nav_mode != "fast_recent512" &&\n'
        '        nav_mode != "final_core") {',
        "register final selected Core",
    )
    s = once(
        s,
        '             nav_mode == "fast_recent512") ? &recent :',
        '             nav_mode == "fast_recent512" ||\n'
        '             nav_mode == "final_core") ? &recent :',
        "pass final Recent2048",
    )
    s = once(
        s,
        'nav_mode == "fast_core")\n            ? native_junctions.get() : nullptr;',
        'nav_mode == "fast_core" || nav_mode == "final_core")\n'
        "            ? native_junctions.get() : nullptr;",
        "load final 14K train-only junctions",
    )
    s = once(
        s,
        '            (nav_mode == "fast_core" || nav_mode == "fast_recent512");',
        '            (nav_mode == "fast_core" || nav_mode == "fast_recent512" ||\n'
        '             nav_mode == "final_core");\n'
        '        nav_diag[i].final_starts = (nav_mode == "final_core");',
        "turn on fast PQ scoring and final top-eight only in new mode",
    )
    s = once(
        s,
        '            nav_mode == "fast_core" || nav_mode == "fast_recent512") {',
        '            nav_mode == "fast_core" || nav_mode == "fast_recent512" ||\n'
        '            nav_mode == "final_core") {',
        "update unique causal history after every final query",
    )
    s = once(
        s,
        "      recent.reserve(512);",
        "      const size_t nav_recent_budget =\n"
        '          nav_mode == "final_core" ? 2048 : 512;\n'
        "      recent.reserve(nav_recent_budget);",
        "size only final-mode history to 2048",
    )
    s = once(
        s,
        "if (recent.size() < 512)",
        "if (recent.size() < nav_recent_budget)",
        "use appropriate unique FIFO capacity",
    )
    s = once(
        s,
        '" recent_capacity=512"',
        '" recent_capacity=" << (nav_mode == "final_core" ? 2048 : 512)',
        "log actual per-arm destination capacity",
    )

    # Original Starling native export remains top16K and older control
    # loads 16K; new final mode loads a separate 14,464 tree.
    s = once(
        s,
        "total!=16000 || children+nlist!=total",
        "(total!=16000 && total!=14464) || children+nlist!=total",
        "accept exactly 14,464 or 16K native trained IDs",
    )
    s = once(
        s,
        "unique.size()!=16000",
        "unique.size()!=total",
        "validate distinct medoids and children in either mode",
    )
    f.write_text(s)
    print(
        "STARLING_FINAL_CORE_PATCH_PASS modes=baseline,fast_core,final_core "
        "B=14464 C=2048 k=8 original_native_search=1",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    root = a.root.resolve()
    for p in (
        "include/pq_flash_index.h",
        "src/page_search.cpp",
        "tests/search_disk_index.cpp",
    ):
        if not (root / p).is_file():
            raise FileNotFoundError(root / p)
    header(root)
    page(root)
    benchmark(root)


if __name__ == "__main__":
    main()
