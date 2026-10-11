#!/usr/bin/env python3
"""Instrument the already-patched original Starling page search for diagnosis.

Apply strictly AFTER patch_starling_recent_hints.py and
patch_starling_native_junctions.py, to the temporary pinned author checkout.
No disk layout, graph, search stopping rule or original navigator is changed.
The oracle is a non-deployable GT-based diagnostic, never a competitor.
"""
from pathlib import Path
import argparse


def once(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise ValueError(f"{label}: expected one anchor, saw {n}")
    return s.replace(old, new, 1)


def patch_header(root):
    p = root / "include/pq_flash_index.h"
    s = p.read_text()
    marker = "  struct StarlingHintIvf {\n"
    definition = """  // Allocated per query by diagnostic benchmark; no global state.
  struct StarlingHintDebug {
    bool inject = true;
    unsigned oracle_id = 0xffffffffu;
    unsigned recent_id = 0xffffffffu;
    unsigned junction_id = 0xffffffffu;
    unsigned native_count = 0;
    float native_dists[32] = {};
    float native_best = -1.f;
    float recent_best = -1.f;
    float junction_best = -1.f;
    float oracle_best = -1.f;
    unsigned recent_rank = 0;
    unsigned junction_rank = 0;
    unsigned oracle_rank = 0;
    unsigned recent_inserted = 0;
    unsigned junction_inserted = 0;
    unsigned oracle_inserted = 0;
    unsigned recent_popped = 0;
    unsigned junction_popped = 0;
    unsigned oracle_popped = 0;
    unsigned recent_page_read = 0;
    unsigned junction_page_read = 0;
    unsigned oracle_page_read = 0;
    unsigned num_pages = 0;
    unsigned first_pages[8] = {};
    unsigned long long recent_ns = 0;
    unsigned long long junction_ns = 0;
    unsigned long long oracle_ns = 0;
  };
"""
    s = once(s, marker, definition + marker, "debug type")
    s = once(
        s,
        "const StarlingHintIvf *navhints_junctions = nullptr);",
        "const StarlingHintIvf *navhints_junctions = nullptr,\n"
        "      StarlingHintDebug *navhints_debug = nullptr);",
        "optional debug arg",
    )
    p.write_text(s)


def patch_page_search(root):
    p = root / "src/page_search.cpp"
    s = p.read_text()
    marker = "  void PQFlashIndex<T>::page_search(\n"
    end = "  void PQFlashIndex<T>::page_search_interim(\n"
    a = s.index(marker)
    b = s.index(end, a)
    before, q, after = s[:a], s[a:b], s[b:]
    q = once(
        q,
        "const StarlingHintIvf *navhints_junctions) {",
        "const StarlingHintIvf *navhints_junctions,\n"
        "      StarlingHintDebug *navhints_debug) {",
        "debug implementation signature",
    )
    q = once(
        q,
        "    // NavHints portable routing interface: score an existing, bounded",
        """    // Capture the *native* initial frontier before any injected hints.
    if (navhints_debug != nullptr) {
      navhints_debug->native_count = std::min((unsigned)32,cur_list_size);
      for (unsigned i=0; i<navhints_debug->native_count; ++i) {
        float d = retset[i].distance;
        navhints_debug->native_dists[i] = d;
        if (navhints_debug->native_best < 0.f ||
            d < navhints_debug->native_best)
          navhints_debug->native_best = d;
      }
    }

    // NavHints portable routing interface: score an existing, bounded""",
        "capture original Starling entries",
    )
    q = once(
        q,
        "    if (navhints_recent != nullptr && !navhints_recent->empty()) {",
        """    const auto recent_begin = std::chrono::steady_clock::now();
    if (navhints_recent != nullptr && !navhints_recent->empty()) {""",
        "recent timing start",
    )
    q = once(
        q,
        "      if (found && cur_list_size < l_search) {\n"
        "        retset[cur_list_size++] =\n"
        "            Neighbor(best_hint_id, best_hint_distance, true);",
        """      if (found && navhints_debug != nullptr) {
        navhints_debug->recent_id = best_hint_id;
        navhints_debug->recent_best = best_hint_distance;
        navhints_debug->recent_rank = 1;
        for (unsigned j=0; j<navhints_debug->native_count; ++j)
          if (navhints_debug->native_dists[j] <= best_hint_distance)
            ++navhints_debug->recent_rank;
      }
      if (found && cur_list_size < l_search &&
          (navhints_debug == nullptr || navhints_debug->inject)) {
        if (navhints_debug != nullptr) navhints_debug->recent_inserted = 1;
        retset[cur_list_size++] =
            Neighbor(best_hint_id, best_hint_distance, true);""",
        "score-only recent probe",
    )
    q = once(
        q,
        "    // Two-stage Hint-IVF search: 512 PQ-scored medoids, then",
        """    if (navhints_debug != nullptr && navhints_recent != nullptr) {
      navhints_debug->recent_ns = (unsigned long long)
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now()-recent_begin).count();
    }
    const auto junction_begin = std::chrono::steady_clock::now();

    // Two-stage Hint-IVF search: 512 PQ-scored medoids, then""",
        "recent timing end, junction start",
    )
    q = once(
        q,
        "      if (found_junction && cur_list_size < l_search) {\n"
        "        retset[cur_list_size++] =\n"
        "            Neighbor(best_junction_id, best_junction_distance, true);",
        """      if (found_junction && navhints_debug != nullptr) {
        navhints_debug->junction_id = best_junction_id;
        navhints_debug->junction_best = best_junction_distance;
        navhints_debug->junction_rank = 1;
        for (unsigned j=0; j<navhints_debug->native_count; ++j)
          if (navhints_debug->native_dists[j] <= best_junction_distance)
            ++navhints_debug->junction_rank;
      }
      if (found_junction && cur_list_size < l_search &&
          (navhints_debug == nullptr || navhints_debug->inject)) {
        if (navhints_debug != nullptr) navhints_debug->junction_inserted = 1;
        retset[cur_list_size++] =
            Neighbor(best_junction_id, best_junction_distance, true);""",
        "score-only junction probe",
    )
    q = once(
        q,
        "    std::sort(retset.begin(), retset.begin() + cur_list_size);\n\n"
        "    unsigned num_ios = 0;",
        """    if (navhints_debug != nullptr && navhints_junctions != nullptr) {
      navhints_debug->junction_ns = (unsigned long long)
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now()-junction_begin).count();
    }
    if (navhints_debug != nullptr &&
        navhints_debug->oracle_id != 0xffffffffu) {
      const auto oracle_begin = std::chrono::steady_clock::now();
      const unsigned oid = navhints_debug->oracle_id;
      if (oid >= this->num_points)
        throw ANNException("Invalid oracle ID", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
      if (visited.find(oid) == visited.end()) {
        float pd = 0.f;
        compute_pq_dists(&oid, 1, &pd);
        if (stats != nullptr) ++stats->n_cmps;
        navhints_debug->oracle_best = pd;
        navhints_debug->oracle_rank = 1;
        for (unsigned j=0; j<navhints_debug->native_count; ++j)
          if (navhints_debug->native_dists[j] <= pd)
            ++navhints_debug->oracle_rank;
        if (cur_list_size < l_search) {
          retset[cur_list_size++] = Neighbor(oid, pd, true);
          visited.insert(oid);
          navhints_debug->oracle_inserted = 1;
        }
      }
      navhints_debug->oracle_ns = (unsigned long long)
          std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now()-oracle_begin).count();
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;""",
        "oracle headroom injection",
    )
    q = once(
        q,
        "        const unsigned pid = id2page_[retset[marker].id];\n"
        "        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {",
        """        const unsigned pid = id2page_[retset[marker].id];
        if (navhints_debug != nullptr && retset[marker].flag) {
          if (retset[marker].id == navhints_debug->recent_id)
            ++navhints_debug->recent_popped;
          if (retset[marker].id == navhints_debug->junction_id)
            ++navhints_debug->junction_popped;
          if (retset[marker].id == navhints_debug->oracle_id)
            ++navhints_debug->oracle_popped;
        }
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {""",
        "count actual candidate pops",
    )
    q = once(
        q,
        "            frontier.push_back(retset[marker].id);\n"
        "            page_visited.insert(pid);",
        """            frontier.push_back(retset[marker].id);
            if (navhints_debug != nullptr) {
              if (navhints_debug->num_pages < 8)
                navhints_debug->first_pages[navhints_debug->num_pages] = pid;
              ++navhints_debug->num_pages;
              if (navhints_debug->recent_id != 0xffffffffu &&
                  pid == id2page_[navhints_debug->recent_id])
                ++navhints_debug->recent_page_read;
              if (navhints_debug->junction_id != 0xffffffffu &&
                  pid == id2page_[navhints_debug->junction_id])
                ++navhints_debug->junction_page_read;
              if (navhints_debug->oracle_id != 0xffffffffu &&
                  pid == id2page_[navhints_debug->oracle_id])
                ++navhints_debug->oracle_page_read;
            }
            page_visited.insert(pid);""",
        "capture first pages and hint page reads",
    )
    p.write_text(before+q+after)


def patch_benchmark(root):
    p = root / "tests/search_disk_index.cpp"
    s = p.read_text()
    s = once(
        s,
        'nav_mode != "random512" && nav_mode != "learned16k" &&\n'
        '        nav_mode != "core16k_recent512") {',
        'nav_mode != "random512" && nav_mode != "learned16k" &&\n'
        '        nav_mode != "core16k_recent512" && nav_mode != "score_only" &&\n'
        '        nav_mode != "oracle") {',
        "diagnostic replay modes",
    )
    s = once(
        s,
        "    auto stats = new diskann::QueryStats[query_num];",
        "    auto stats = new diskann::QueryStats[query_num];\n"
        "    std::vector<diskann::StarlingHintDebug> nav_diag(query_num);",
        "per-query stats array",
    )
    s = once(
        s,
        '            (nav_mode == "recent512" || nav_mode == "core16k_recent512")\n'
        '                ? &recent :',
        '            (nav_mode == "recent512" || nav_mode == "core16k_recent512" ||\n'
        '             nav_mode == "score_only") ? &recent :',
        "full Core score-only: online input",
    )
    s = once(
        s,
        '            (nav_mode == "learned16k" || nav_mode == "core16k_recent512")\n'
        '            ? native_junctions.get() : nullptr;',
        '            (nav_mode == "learned16k" || nav_mode == "core16k_recent512" ||\n'
        '             nav_mode == "score_only")\n'
        '            ? native_junctions.get() : nullptr;',
        "full Core score-only: trained input",
    )
    s = once(
        s,
        "        _pFlashIndex->page_search(\n"
        "            query + (i * query_aligned_dim), recall_at, mem_L, L,",
        """        nav_diag[i].inject = (nav_mode != "score_only");
        if (nav_mode == "oracle") {
          // Deliberate GT disclosure ONLY to the oracle diagnostic arm.
          if (gt_ids == nullptr || gt_dim == 0)
            throw std::runtime_error("Oracle requires heldout exact GT");
          nav_diag[i].oracle_id = gt_ids[i*gt_dim];
        }
        _pFlashIndex->page_search(
            query + (i * query_aligned_dim), recall_at, mem_L, L,""",
        "per-query oracle and score-only control",
    )
    s = once(
        s,
        "            stats + i, candidates, dir);",
        "            stats + i, candidates, dir, &nav_diag[i]);",
        "pass debug state through author native page search",
    )
    s = once(
        s,
        '        if (nav_mode == "recent512" || nav_mode == "core16k_recent512") {',
        '        if (nav_mode == "recent512" || nav_mode == "core16k_recent512" ||\n'
        '            nav_mode == "score_only") {',
        "maintain same causal FIFO for score-only",
    )
    s = once(
        s,
        "    std::chrono::duration<double> diff = e - (causal_replay ? measured_start : s);",
        """    std::chrono::duration<double> diff = e - (causal_replay ? measured_start : s);
    // Write measurements AFTER timing, one CSV per L and arm.
    // The per-query timers inside native page_search include every hint cost.
    if (causal_replay) {
      const char *diagdir = std::getenv("STARLING_NAVHINTS_DIAG_DIR");
      if (diagdir == nullptr || !*diagdir)
        throw std::runtime_error("Starling diagnosis requires DIAG_DIR");
      std::ofstream csv(std::string(diagdir) + "/L" + std::to_string(L) + ".csv");
      if (!csv) throw std::runtime_error("Cannot create diagnostic CSV");
      csv << "query,L,ios,total_us,cpu_us,cache_hits,native_best,";
      csv << "recent_id,recent_dist,recent_rank,recent_injected,recent_popped,recent_page_read,recent_ns,";
      csv << "junction_id,junction_dist,junction_rank,junction_injected,junction_popped,junction_page_read,junction_ns,";
      csv << "oracle_id,oracle_dist,oracle_rank,oracle_injected,oracle_popped,oracle_page_read,oracle_ns,";
      csv << "total_pages,p0,p1,p2,p3,p4,p5,p6,p7\\n";
      for (size_t qi=nav_warmup; qi<query_num; ++qi) {
        const auto &d = nav_diag[qi];
        const auto &m = stats[qi];
        csv << qi-nav_warmup << "," << L << "," << m.n_ios
            << "," << m.total_us << "," << m.cpu_us << ","
            << m.n_cache_hits << "," << d.native_best << ","
            << d.recent_id << "," << d.recent_best << "," << d.recent_rank
            << "," << d.recent_inserted << "," << d.recent_popped
            << "," << d.recent_page_read << "," << d.recent_ns << ","
            << d.junction_id << "," << d.junction_best << "," << d.junction_rank
            << "," << d.junction_inserted << "," << d.junction_popped
            << "," << d.junction_page_read << "," << d.junction_ns << ","
            << d.oracle_id << "," << d.oracle_best << "," << d.oracle_rank
            << "," << d.oracle_inserted << "," << d.oracle_popped
            << "," << d.oracle_page_read << "," << d.oracle_ns << ","
            << d.num_pages;
        for (unsigned j=0; j<8; ++j) csv << "," << d.first_pages[j];
        csv << "\\n";
      }
      if (!csv) throw std::runtime_error("Incomplete Starling diagnostic CSV");
      diskann::cout << "STARLING_DIAGNOSTIC_CSV mode=" << nav_mode
                    << " L=" << L << " measured=" << query_num-nav_warmup
                    << std::endl;
    }""",
        "write post-timer diagnostic evidence",
    )
    p.write_text(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    root = ap.parse_args().source.resolve()
    patch_header(root)
    patch_page_search(root)
    patch_benchmark(root)
    print("STARLING_ENTRY_DIAGNOSTIC_ADAPTER_READY modes=baseline,score_only,core16k_recent512,oracle", flush=True)


if __name__ == "__main__":
    main()
