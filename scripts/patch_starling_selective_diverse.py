#!/usr/bin/env python3
"""Apply diversity-aware hints and a causal quality gate to original Starling.

Apply strictly after original Starling's temporary native NavHints, 16K
junctions and entry-diagnostics patches. No changes to graph, persistent
index, author navigator or stopping rule.

Diversity: score the normal recent/junction candidates; retain the top 24
by PQ(query,candidate); choose the best candidate whose page differs from
all native 10 starting pages AND whose approximate PQ-code distance to the
nearest native starter >= 0.35 * PQ(query,best_native). No disk reads.
Activation: optional 50th/75th percentile of *native* best PQ distance
calibrated on the causal 4K-query warmup only. No recall/GT tuning.
"""
import argparse
from pathlib import Path


def once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 anchor, got {n}")
    return s.replace(old, new, 1)


def patch_header(root):
    f = root / "include/pq_flash_index.h"
    s = f.read_text()
    s = once(s, "    float native_dists[32] = {};", """    float native_dists[32] = {};
    unsigned native_ids[32] = {};
    bool activation_gate = false;
    bool gate_open = true;
    bool want_diverse = false;
    float gate_threshold = 0.f;
    float recent_novelty = -1.f;
    float junction_novelty = -1.f;""", "debug fields")
    f.write_text(s)


def patch_page(root):
    f = root / "src/page_search.cpp"
    s = f.read_text()
    first = s.index("  void PQFlashIndex<T>::page_search(\n")
    last = s.index("  void PQFlashIndex<T>::page_search_interim(\n", first)
    prefix, part, suffix = s[:first], s[first:last], s[last:]
    part = once(part, "        navhints_debug->native_dists[i] = d;",
                "        navhints_debug->native_dists[i] = d;\n"
                "        navhints_debug->native_ids[i] = retset[i].id;",
                "native IDs for geometric exclusion")
    part = once(part,
      "    // NavHints portable routing interface: score an existing, bounded",
      r'''    // Gate on a predeclared unlabeled warmup quantile of the strength
    // of Starling's own 10 native starting points. This happens BEFORE
    // scanning the 512 destination IDs or the 16K directory.
    const bool navhints_active =
        navhints_debug == nullptr || !navhints_debug->activation_gate ||
        navhints_debug->gate_threshold <= 0.f ||
        navhints_debug->native_best >= navhints_debug->gate_threshold;
    if (navhints_debug != nullptr)
      navhints_debug->gate_open = navhints_active;
    const bool navhints_diverse =
        navhints_active && navhints_debug != nullptr &&
        navhints_debug->want_diverse;

    // Only transient per-query storage, not a persistent 10M-vector cache.
    // Reconstruct the 10 native PQ starts once, then evaluate novelty only
    // for a short list of the 24 best query-scored hint candidates.
    std::vector<float> navhints_native_pq;
    if (navhints_diverse) {
      navhints_native_pq.resize(
          navhints_debug->native_count * this->data_dim);
      for (unsigned i=0; i<navhints_debug->native_count; ++i) {
        const _u64 code_offset =
            (_u64)navhints_debug->native_ids[i] * this->n_chunks;
        this->pq_table.inflate_vector(
            this->data + code_offset,
            navhints_native_pq.data() + i * this->data_dim);
      }
    }
    auto navhints_novelty = [&](unsigned candidate) -> float {
      if (!navhints_diverse) return 0.f;
      float min_pair = std::numeric_limits<float>::infinity();
      for (unsigned i=0; i<navhints_debug->native_count; ++i) {
        if (id2page_[candidate] == id2page_[navhints_debug->native_ids[i]])
          return -1.f;  // Same 4 KiB page is not an independent route.
        const float pair_d = this->pq_table.l2_distance(
            navhints_native_pq.data() + i * this->data_dim,
            this->data + (_u64)candidate * this->n_chunks);
        min_pair = std::min(min_pair, pair_d);
      }
      return min_pair;
    };
    auto navhints_top = [](
        std::vector<std::pair<float,unsigned>>& pool,
        float q_dist, unsigned id) {
      pool.emplace_back(q_dist,id);
      std::sort(pool.begin(),pool.end());
      if (pool.size()>24) pool.pop_back();
    };
    auto navhints_select = [&](
        const std::vector<std::pair<float,unsigned>>& pool,
        unsigned &chosen, float &query_d, float &novelty) -> bool {
      const float minimum =
          0.35f * std::max(1.f, navhints_debug->native_best);
      for (const auto &option : pool) {
        const float distance = navhints_novelty(option.second);
        if (distance >= minimum) {
          query_d = option.first;
          chosen = option.second;
          novelty = distance;
          return true;
        }
      }
      return false;
    };

    // NavHints portable routing interface: score an existing, bounded''',
      "cheap native-start gate and PQ-code geometry")
    part = once(part,
       "    if (navhints_recent != nullptr && !navhints_recent->empty()) {",
       "    if (navhints_active && navhints_recent != nullptr && "
       "!navhints_recent->empty()) {",
       "do not score recent hints behind closed gate")
    part = once(part,
       "      float best_hint_distance = std::numeric_limits<float>::infinity();",
       "      std::vector<std::pair<float,unsigned>> recent_pool;\n"
       "      if (navhints_diverse) recent_pool.reserve(24);\n"
       "      float best_hint_distance = std::numeric_limits<float>::infinity();",
       "top-24 recent pool")
    part = once(part,
       "          const unsigned id = (*navhints_recent)[off+i];\n"
       "          if (visited.find(id) != visited.end()) continue;",
       "          const unsigned id = (*navhints_recent)[off+i];\n"
       "          if (visited.find(id) != visited.end()) continue;\n"
       "          if (navhints_diverse)\n"
       "            navhints_top(recent_pool,dist_scratch[i],id);",
       "retain competitive recent candidates")
    part = once(part,
      "      if (found && navhints_debug != nullptr) {",
      r'''      if (navhints_diverse) {
        found = navhints_select(recent_pool, best_hint_id,
                               best_hint_distance,
                               navhints_debug->recent_novelty);
      }
      if (found && navhints_debug != nullptr) {''',
      "select novel recent entry")
    part = once(part,
       "    if (navhints_junctions != nullptr) {",
       "    if (navhints_active && navhints_junctions != nullptr) {",
       "do not score junctions behind closed gate")
    part = once(part,
       "      bool found_junction = false;",
       "      bool found_junction = false;\n"
       "      std::vector<std::pair<float,unsigned>> junction_pool;\n"
       "      if (navhints_diverse) junction_pool.reserve(24);",
       "top-24 junction pool")
    part = once(part,
       "        const unsigned medoid_id = dir.medoids[cell];",
       "        const unsigned medoid_id = dir.medoids[cell];\n"
       "        if (navhints_diverse && visited.find(medoid_id) == visited.end())\n"
       "          navhints_top(junction_pool,selected.first,medoid_id);",
       "store medoid possibilities")
    part = once(part,
       "            const unsigned id = dir.children[off+i];\n"
       "            if (visited.find(id) != visited.end()) continue;",
       "            const unsigned id = dir.children[off+i];\n"
       "            if (visited.find(id) != visited.end()) continue;\n"
       "            if (navhints_diverse)\n"
       "              navhints_top(junction_pool,dist_scratch[i],id);",
       "store diverse junction possibilities")
    part = once(part,
       "      if (found_junction && navhints_debug != nullptr) {",
       r'''      if (navhints_diverse) {
        found_junction = navhints_select(
            junction_pool, best_junction_id, best_junction_distance,
            navhints_debug->junction_novelty);
      }
      if (found_junction && navhints_debug != nullptr) {''',
       "select novel junction entry")
    f.write_text(prefix+part+suffix)


def patch_benchmark(root):
    f = root / "tests/search_disk_index.cpp"
    s = f.read_text()
    s = once(s,
       '        nav_mode != "oracle") {',
       '        nav_mode != "oracle" && nav_mode != "diverse_core" &&\n'
       '        nav_mode != "gate50_core" && nav_mode != "gate50_diverse" &&\n'
       '        nav_mode != "gate25_diverse") {',
       "declare full Core policies")
    s = once(s,
       "      size_t fifo_cursor = 0;",
       """      size_t fifo_cursor = 0;
      // This stays unlabeled: only Starling's own native-entry distances.
      std::vector<float> best_native_warmup;
      best_native_warmup.reserve(nav_warmup);
      float gate_threshold = 0.f;""",
       "causal gate-calibration data")
    s = once(s,
       """        if (i == nav_warmup) {
          measured_start = std::chrono::high_resolution_clock::now();
        }""",
       r'''        if (i == nav_warmup) {
          measured_start = std::chrono::high_resolution_clock::now();
          const bool gate50 =
              (nav_mode == "gate50_core" || nav_mode == "gate50_diverse");
          const bool gate25 = (nav_mode == "gate25_diverse");
          if (gate50 || gate25) {
            if (best_native_warmup.size() != nav_warmup)
              throw std::runtime_error("incomplete causal warmup");
            std::sort(best_native_warmup.begin(),best_native_warmup.end());
            const size_t target = gate50 ? nav_warmup/2 : 3*nav_warmup/4;
            gate_threshold = best_native_warmup[target];
            diskann::cout << "STARLING_CAUSAL_GATE mode=" << nav_mode
                          << " native_best_threshold=" << gate_threshold
                          << " quantile=" << (gate50 ? 0.5 : 0.75)
                          << " no_GT=true" << std::endl;
          }
        }''',
       "fix gate quantile before measuring suffix")
    s = once(s,
       '             nav_mode == "score_only") ? &recent :',
       '             nav_mode == "score_only" || nav_mode == "diverse_core" ||\n'
       '             nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
       '             nav_mode == "gate25_diverse") ? &recent :',
       "reuse same online set")
    s = once(s,
       '             nav_mode == "score_only")\n'
       '            ? native_junctions.get() : nullptr;',
       '             nav_mode == "score_only" || nav_mode == "diverse_core" ||\n'
       '             nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
       '             nav_mode == "gate25_diverse")\n'
       '            ? native_junctions.get() : nullptr;',
       "reuse native 16K junction data")
    s = once(s,
       '        nav_diag[i].inject = (nav_mode != "score_only");',
       r'''        nav_diag[i].inject = (nav_mode != "score_only");
        nav_diag[i].activation_gate =
            (nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||
             nav_mode == "gate25_diverse");
        nav_diag[i].want_diverse =
            (nav_mode == "diverse_core" || nav_mode == "gate50_diverse" ||
             nav_mode == "gate25_diverse");
        nav_diag[i].gate_threshold = gate_threshold;''',
       "gate config for native search")
    s = once(s,
       '            stats + i, candidates, dir, &nav_diag[i]);',
       '''            stats + i, candidates, dir, &nav_diag[i]);
        if (i < nav_warmup)
          best_native_warmup.push_back(nav_diag[i].native_best);''',
       "record only past native quality")
    s = once(s,
       '            nav_mode == "score_only") {',
       '            nav_mode == "score_only" || nav_mode == "diverse_core" ||\n'
       '            nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
       '            nav_mode == "gate25_diverse") {',
       "causal FIFO updates for all hint policies")
    s = once(s,
       'csv << "query,L,ios,total_us,cpu_us,cache_hits,native_best,";',
       'csv << "query,L,ios,total_us,cpu_us,cache_hits,native_best,"\n'
       '          << "gate_open,gate_threshold,diverse,recent_novelty,junction_novelty,";',
       "add gate/novelty columns")
    s = once(s,
       '<< m.n_cache_hits << "," << d.native_best << ","',
       '<< m.n_cache_hits << "," << d.native_best << ","\n'
       '            << (d.gate_open ? 1 : 0) << "," << d.gate_threshold << ","\n'
       '            << (d.want_diverse ? 1 : 0) << ","\n'
       '            << d.recent_novelty << "," << d.junction_novelty << ","',
       "write measured gate/novelty")
    f.write_text(s)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    root=ap.parse_args().source.resolve()
    for path in ("src/page_search.cpp","include/pq_flash_index.h",
                 "tests/search_disk_index.cpp"):
        if not (root/path).is_file():
            raise RuntimeError(f"missing author source {path}")
    patch_header(root)
    patch_page(root)
    patch_benchmark(root)
    print("STARLING_SELECTIVE_DIVERSE_ADAPTER_READY "
          "policies=core,diverse_core,gate50_core,gate50_diverse,gate25_diverse "
          "native_geometry=PQ_reconstructed top24=24 threshold=0.35 "
          "causal_warmup=4000",flush=True)


if __name__=="__main__":main()
