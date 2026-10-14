#!/usr/bin/env python3
"""Apply a fail-closed, source-pinned NavHints adapter to ORIGINAL native LAANN.

Changes only src/pq_flash_index_laann.cpp, a separately vendored header,
and the LAANN mode of apps/search_disk_index.cpp. PageANN remains untouched.
Search graph, page layout, native nav graph, look-ahead and I/O scheduler
remain in the original implementation.
"""
from __future__ import annotations
import argparse
from pathlib import Path

SOURCE_SHA = "58fe40feb58d0ca1a168816b36c4d4b22d3eb29c"


def once(s: str, old: str, new: str, description: str) -> str:
    count = s.count(old)
    if count != 1:
        raise RuntimeError(f"{description}: expected 1 source anchor, got {count}")
    return s.replace(old, new, 1)


def patch_index(s: str) -> str:
    s = once(s, '#include "pq_flash_index_laann.h"\n',
             '#include "pq_flash_index_laann.h"\n#include "nh_laann_state.h"\n',
             "native LAANN header")
    s = once(s, "    uint32_t non_hub_requested = 0;  // Count non-hub vectors encountered\n",
             "    uint32_t non_hub_requested = 0;  // Count non-hub vectors encountered\n"
             "    std::vector<uint32_t> nh_visit_order;  // native page-representative IDs, trace mode only\n",
             "native query-local training order")
    exact = "visitedPages.insert(pageID);"
    if s.count(exact) != 3:
        raise RuntimeError(f"native LAANN page-expansion sites changed: {s.count(exact)} != 3")
    s = s.replace(exact, exact + "\n" +
                  "                    if (navhints_laann::State::get().tracing())\n" +
                  "                        nh_visit_order.push_back(static_cast<uint32_t>(pageID * _megaNode_capacity));")

    anchor = "    // End navigation timing\n"
    code = r'''    // NavHints extension: score native reordered vector IDs using LAANN's
    // already-resident PQ array. Inject into the ORIGINAL search frontier after
    // the native in-RAM navigator, without replacing its seeds or I/O pipeline.
    // All code is inactive unless NH_LAANN_MODE selects a NavHints arm.
    auto &nh_state = navhints_laann::State::get();
    if (nh_state.active()) {
        auto nh_score_batch = [&](const std::vector<uint32_t> &ids, std::vector<float> &output) {
            output.clear();
            if (ids.empty()) return;
            std::vector<uint8_t> codes(ids.size() * _n_chunks);
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] >= _num_points) throw std::runtime_error("Invalid native LAANN NavHints ID");
                std::memcpy(codes.data() + i * _n_chunks,
                            _cached_pq_buff + static_cast<size_t>(ids[i]) * _n_chunks, _n_chunks);
            }
            output.resize(ids.size());
            diskann::pq_dist_lookup(codes.data(), ids.size(), _n_chunks, pq_dists, output.data());
        };
        const bool injected = nh_state.inject(retset, static_cast<uint32_t>(_num_points),
                                              static_cast<uint32_t>(_megaNode_capacity),
                                              nh_score_batch);
        // The newly introduced starts must participate in native convergence
        // checking rather than inherit the pre-insertion navigator snapshot.
        if (injected && use_look_ahead_search) {
            convergence_phase_start = false;
            prev_nth_id = retset.size() >= conv_rank ? retset[conv_rank - 1].id : UINT32_MAX;
        }
    }

'''
    s = once(s, anchor, code + anchor, "LAANN frontier handoff")

    anchor = "    // Save all timing statistics\n"
    finish = r'''    // Causal feedback is part of the timed request. During training only,
    // append first-expanded page representatives to a deterministic trace.
    if (nh_state.active() || nh_state.tracing()) {
        if (k_search == 0 || res_ids[0] >= _num_points)
            throw std::runtime_error("Invalid LAANN result used for NavHints feedback");
        nh_state.completed(static_cast<uint32_t>(res_ids[0]), nh_visit_order);
    }

'''
    s = once(s, anchor, finish + anchor, "post-query causal feedback")
    return s


def patch_cli(s: str) -> str:
    # Ensure we alter only the LAANN template, not PageANN's sibling benchmark.
    begin = s.find("int search_laann_disk_index(")
    end = s.find("\nint main(", begin)
    if begin < 0:
        raise RuntimeError("LAANN benchmark function not found")
    if end < 0:
        end = len(s)
    prefix, part, suffix = s[:begin], s[begin:end], s[end:]
    guard = "    omp_set_num_threads(num_threads);\n"
    part = once(part, guard, r'''    // One request in flight is essential for strictly causal history.
    // Run ONE search width per process so history starts empty for each L.
    const char *nh_mode = std::getenv("NH_LAANN_MODE");
    const char *nh_results = std::getenv("NH_LAANN_RESULTS");
    const bool nh_control = nh_mode && std::string(nh_mode) != "off";
    if ((nh_control || (nh_results && *nh_results)) && (num_threads != 1 || Lvec.size() != 1))
        throw std::runtime_error("NavHints LAANN replay requires -T 1 and exactly one -L value");
    omp_set_num_threads(num_threads);
''', "causal single-query replay guard")
    anchor = "        delete[] stats;\n"
    block = r'''        // Per-request real-SSD measurements, preserving the first 4K as
        // causal warm-up and the last 1K as the measured suffix OFFLINE.
        if (nh_results && *nh_results) {
            std::ofstream csv(nh_results, std::ios::trunc);
            if (!csv) throw std::runtime_error("Cannot create NH_LAANN_RESULTS");
            csv << "L,query_idx,reads,latency_us,result_ids\n";
            for (size_t qi = 0; qi < query_num; ++qi) {
                csv << L << ',' << qi << ',' << stats[qi].n_ios
                    << ',' << std::fixed << std::setprecision(4) << stats[qi].total_us << ',';
                for (uint32_t ri = 0; ri < recall_at; ++ri) {
                    if (ri) csv << '|';
                    csv << query_result_ids[test_id][qi * recall_at + ri];
                }
                csv << '\n';
            }
        }

'''
    part = once(part, anchor, block + anchor, "native LAANN per-request metrics")
    return prefix + part + suffix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-tree", type=Path, required=True,
                   help="Git checkout of Dingyi-Kang/PageANN at the pinned native SHA")
    a = p.parse_args()
    root = a.source_tree.resolve()
    idx = root / "src" / "pq_flash_index_laann.cpp"
    cli = root / "apps" / "search_disk_index.cpp"
    src_header = Path(__file__).parent / "nh_laann_state.h"
    dest_header = root / "src" / "nh_laann_state.h"
    if not all(f.is_file() for f in (idx, cli, src_header)):
        raise FileNotFoundError("Missing pinned native LAANN source or adapter header")
    if dest_header.exists():
        raise RuntimeError("Integration already patched; refusing reapplication")
    index_old, cli_old = idx.read_text(), cli.read_text()
    index_new = patch_index(index_old)
    cli_new = patch_cli(cli_old)
    if index_new == index_old or cli_new == cli_old:
        raise RuntimeError("Native LAANN patch yielded no changes")
    # Write only after both original-source anchors validate.
    idx.write_text(index_new)
    cli.write_text(cli_new)
    dest_header.write_text(src_header.read_text())
    print("NH_LAANN_PINNED_PATCH_PASS", str(root))


if __name__ == "__main__":
    main()
