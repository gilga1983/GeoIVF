#!/usr/bin/env python3
"""Integrate causal NavHints Recent512 into *original* pinned Starling.

Author source: zilliztech/starling@17dc3e8a011533a62374445f53963e951b72883a.

Starling's in-memory graph navigator, PQ codes, disk partition layout,
page scheduler, best-first graph search and I/O accounting remain intact.
The only search change allows a caller to pass candidate vertex IDs for
an extra PQ-scored start, using the resident PQ table and existing retset.

The test harness adds three alternative 5K causal replay modes:
  baseline   : standard Starling in-memory navigator only
  random512  : same PQ scoring budget, a static random ID pool
  recent512  : unique FIFO of the previous 512 rank-one search results
Each mode runs the first 4K heldout queries to warm online state and
reports only the last 1K. The same serial query loop is used by all arms.
Native four-thread concurrent benchmark is still available unchanged when
STARLING_NAVHINTS_EVAL is unset; its throughput is not compared to our
sequential-online measurements.

We do not change the authors' graph/index, and do not claim to implement
the optional hub persistence or static learned 16K junction directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def once(s: str, before: str, after: str, label: str) -> str:
    n=s.count(before)
    if n!=1:raise ValueError(f"{label}: expected one exact source match, found {n}")
    return s.replace(before,after,1)


def header(root:Path)->None:
    p=root/"include/pq_flash_index.h"
    s=p.read_text()
    old="""    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr);"""
    new="""    // Optional IDs are only PQ-scored entry candidates; Starling's graph,
    // in-memory navigator and native disk-page traversal are unchanged.
    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const bool use_reorder_data = false, const float use_ratio = 1.0f, QueryStats *stats = nullptr,
        const std::vector<unsigned> *navhints_recent = nullptr);"""
    p.write_text(once(s,old,new,"page_search public interface"))


def page_search(root:Path)->None:
    p=root/"src/page_search.cpp"
    s=p.read_text()
    head="  void PQFlashIndex<T>::page_search(\n"
    split="  void PQFlashIndex<T>::page_search_interim(\n"
    a=s.index(head)
    b=s.index(split,a)
    before,section,after=s[:a],s[a:b],s[b:]
    old="""      const bool use_reorder_data, const float use_ratio, QueryStats *stats) {"""
    new="""      const bool use_reorder_data, const float use_ratio, QueryStats *stats,
      const std::vector<unsigned> *navhints_recent) {"""
    section=once(section,old,new,"non-SQ page_search signature")
    old="""    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;"""
    new=r'''    // NavHints portable routing interface: score an existing, bounded
    // set of *database IDs* using Starling's already-resident PQ table.
    // The caller determines whether IDs are recent, random or absent.
    // Only the best unused candidate enters Starling's ordinary frontier.
    // Chunking is mandatory: dist_scratch/pq_coord_scratch are sized for
    // degree-limited batches, NOT for 512 IDs at once.
    if (navhints_recent != nullptr && !navhints_recent->empty()) {
      float best_hint_distance = std::numeric_limits<float>::infinity();
      unsigned best_hint_id = 0;
      bool found = false;
      constexpr size_t batch_cap = 16;
      for (size_t off = 0; off < navhints_recent->size(); off += batch_cap) {
        const size_t count =
            std::min(batch_cap, navhints_recent->size() - off);
        for (size_t i = 0; i < count; ++i) {
          if ((*navhints_recent)[off+i] >= this->num_points) {
            throw ANNException("NavHints candidate outside index",
                               -1, __FUNCSIG__, __FILE__, __LINE__);
          }
        }
        compute_pq_dists(navhints_recent->data() + off, count, dist_scratch);
        if (stats != nullptr) stats->n_cmps += count;
        for (size_t i = 0; i < count; ++i) {
          const unsigned id = (*navhints_recent)[off+i];
          if (visited.find(id) != visited.end()) continue;
          if (!found || dist_scratch[i] < best_hint_distance) {
            found = true;
            best_hint_id = id;
            best_hint_distance = dist_scratch[i];
          }
        }
      }
      if (found && cur_list_size < l_search) {
        retset[cur_list_size++] =
            Neighbor(best_hint_id, best_hint_distance, true);
        visited.insert(best_hint_id);
      }
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;'''
    section=once(section,old,new,"PQ-scored one-start injection")
    p.write_text(before+section+after)


def benchmark(root:Path)->None:
    p=root/"tests/search_disk_index.cpp"
    s=p.read_text()
    s=once(s,"#include <atomic>\n",
           "#include <atomic>\n#include <cstdlib>\n#include <unordered_set>\n",
           "causal replay include")
    old="""  uint32_t optimized_beamwidth = 2;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {"""
    new=r'''  uint32_t optimized_beamwidth = 2;

  // Opt in to a *single-query causal replay* for fair comparison of online
  // policies. Original 4-thread throughput code remains intact below.
  const char *nav_raw = std::getenv("STARLING_NAVHINTS_EVAL");
  const std::string nav_mode = nav_raw ? std::string(nav_raw) : "";
  const bool causal_replay = !nav_mode.empty();
  const size_t nav_warmup = 4000;
  if (causal_replay) {
    if (nav_mode != "baseline" && nav_mode != "recent512" &&
        nav_mode != "random512") {
      throw std::runtime_error("unsupported STARLING_NAVHINTS_EVAL mode");
    }
    if (!use_page_search || use_sq || !mem_L ||
        query_num != 5000 || recall_at != 10 || num_threads != 4) {
      throw std::runtime_error(
          "NavHints replay expects 5K heldout uint8/L2 queries, 10-NN, "
          "4 configured threads, native page search and memory navigator");
    }
    diskann::cout << "STARLING_NAVHINTS_PROTOCOL mode=" << nav_mode
                  << " warmup=" << nav_warmup
                  << " measured=" << query_num-nav_warmup
                  << " recent_capacity=512" << std::endl;
  }

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {'''
    s=once(s,old,new,"causal replay mode")
    old="""    auto                  s = std::chrono::high_resolution_clock::now();

    // Using branching outside the for loop instead of inside and
"""
    new="""    auto                  s = std::chrono::high_resolution_clock::now();
    auto measured_start = s;

    // Using branching outside the for loop instead of inside and
"""
    s=once(s,old,new,"measurement timer")
    # Preserve every line of the author's original parallel implementation.
    anchor="""    // Using branching outside the for loop instead of inside and
    // std::function/std::mem_fn for less switching and function calling overhead
    if (use_page_search) {"""
    replacement=r'''    // Original native system configuration is held fixed throughout.
    // Serial causal replay gives *all three* arms identical ordering,
    // making a 512-ID online cache meaningful and preventing future leaks.
    if (causal_replay) {
      std::vector<unsigned> recent;
      std::unordered_set<unsigned> members;
      recent.reserve(512);
      size_t fifo_cursor = 0;
      std::vector<unsigned> random_ids;
      if (nav_mode == "random512") {
        random_ids.reserve(512);
        for (uint64_t j = 0; j < 512; ++j) {
          random_ids.push_back(static_cast<unsigned>(
              (j * 2654435761ULL + 1013904223ULL) % 10000000ULL));
        }
      }
      for (size_t i = 0; i < query_num; ++i) {
        if (i == nav_warmup) {
          measured_start = std::chrono::high_resolution_clock::now();
        }
        const std::vector<unsigned> *candidates =
            nav_mode == "recent512" ? &recent :
            (nav_mode == "random512" ? &random_ids : nullptr);
        _pFlashIndex->page_search(
            query + (i * query_aligned_dim), recall_at, mem_L, L,
            query_result_ids_64.data() + (i * recall_at),
            query_result_dists[test_id].data() + (i * recall_at),
            optimized_beamwidth, search_io_limit, use_reorder_data, use_ratio,
            stats + i, candidates);

        // Causal, unique 512-ID FIFO: only *completed* results are visible.
        if (nav_mode == "recent512") {
          const unsigned winner = static_cast<unsigned>(
              query_result_ids_64[i * recall_at]);
          if (members.insert(winner).second) {
            if (recent.size() < 512) {
              recent.push_back(winner);
            } else {
              members.erase(recent[fifo_cursor]);
              recent[fifo_cursor] = winner;
              fifo_cursor = (fifo_cursor + 1) % recent.size();
            }
          }
        }
      }
    } else {
    // Using branching outside the for loop instead of inside and
    // std::function/std::mem_fn for less switching and calling overhead
    if (use_page_search) {'''
    s=once(s,anchor,replacement,"causal replay and unchanged native path")
    old="""    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (1.0 * query_num) / (1.0 * diff.count());"""
    new="""    } // end original parallel benchmark branch
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - (causal_replay ? measured_start : s);
    const size_t measured_queries = causal_replay ? query_num-nav_warmup : query_num;
    const size_t offset_queries = causal_replay ? nav_warmup : 0;
    float qps = (1.0 * measured_queries) / (1.0 * diff.count());"""
    s=once(s,old,new,"postquery measurement")
    old="""    auto mean_latency = diskann::get_mean_stats<float>(
        stats, query_num,
        [](const diskann::QueryStats& stats) { return stats.total_us; });"""
    new="""    auto mean_latency = diskann::get_mean_stats<float>(
        stats+offset_queries, measured_queries,
        [](const diskann::QueryStats& stats) { return stats.total_us; });"""
    s=once(s,old,new,"measured latency")
    s=once(s,"""    auto latency_999 = diskann::get_percentile_stats<float>(
        stats, query_num, 0.999,""",
           """    auto latency_999 = diskann::get_percentile_stats<float>(
        stats+offset_queries, measured_queries, 0.999,""",
           "measured p999")
    s=once(s,"""    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num,""",
           """    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats+offset_queries, measured_queries,""",
           "measured native IO")
    s=once(s,"""    auto mean_cpus = diskann::get_mean_stats<float>(
        stats, query_num,""",
           """    auto mean_cpus = diskann::get_mean_stats<float>(
        stats+offset_queries, measured_queries,""",
           "measured CPU")
    old="""      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);"""
    new="""      recall = diskann::calculate_recall(
          measured_queries, gt_ids + offset_queries*gt_dim,
          gt_dists != nullptr ? gt_dists + offset_queries*gt_dim : nullptr,
          gt_dim,
          query_result_ids[test_id].data() + offset_queries*recall_at,
          recall_at, recall_at);"""
    s=once(s,old,new,"measured GT suffix")
    p.write_text(s)


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    args=ap.parse_args()
    root=args.source.resolve()
    for path in ("include/pq_flash_index.h","src/page_search.cpp",
                 "tests/search_disk_index.cpp"):
        if not (root/path).is_file():
            raise SystemExit(f"Starling author source not found: {root/path}")
    header(root)
    page_search(root)
    benchmark(root)
    print("STARLING_NAVHINTS_ADAPTER_READY modes=baseline,random512,recent512 "
          "causal4k_measured1k original_graph_and_search_intact",flush=True)


if __name__=="__main__":
    main()
