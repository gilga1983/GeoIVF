#!/usr/bin/env python3
"""Optimize the EXISTING native Starling NavHints selector without changing policy.

Apply after patch_starling_coveo_coverage.py on a temporary author checkout.
In legacy modes the original author/port code is left intact; new fast_core
and fast_recent512 share its same 16K directory and causal 512-ID state.

The optimized path gathers the same resident PQ codes into query-local pooled
scratch (no persistent memory increase), makes one chunk-major distance pass
per 512 coarse IDs / Recent512, and uses exact pair-ordered top-nprobe insert
rather than re-sorting 8 elements for every coarse candidate. Each selected
region is PQ-scored in one batch, preserving original region order and the
strict '<' distance tie rule. Never changes score, candidate, PQ metric,
training, Starling's memory navigator, graph, paging, or termination.

Use per-query original/fast route identity as a HARD validation gate.
"""
from pathlib import Path
import argparse
import random

def once(s,old,new,why):
    n=s.count(old)
    if n!=1:
        raise RuntimeError(f"{why}: expected 1 exact anchor, found {n}")
    return s.replace(old,new,1)

def header(root):
    p=root/"include/pq_flash_index.h"
    s=p.read_text()
    s=once(s,
        "    unsigned coverage_policy = 0; // 0: weak filter, 1: maxmin, 2: coverage/cost",
        "    unsigned coverage_policy = 0; // 0: weak filter, 1: maxmin, 2: coverage/cost\n"
        "    bool fast_selector = false; // only scorer changes; exact same entry IDs",
        "select fast scorer at query call")
    p.write_text(s)

def page(root):
    p=root/"src/page_search.cpp"
    s=p.read_text()
    first=s.index("  void PQFlashIndex<T>::page_search(\n")
    end=s.index("  void PQFlashIndex<T>::page_search_interim(\n",first)
    begin,part,after=s[:first],s[first:end],s[end:]
    anchor="    // NavHints portable routing interface: score an existing, bounded"
    pre=r'''    // Query-local pooled scratch: packed PQ codes and distances are
    // rebuilt from the SAME existing IDs. No new persistent PQ representation
    // or increased hint residency. The old 16-wide scorer is still used by
    // every legacy mode, giving an executable same-index control.
    const bool nh_fast =
        navhints_debug != nullptr && navhints_debug->fast_selector;
    std::vector<_u8> nh_codes;
    std::vector<float> nh_scores;
    if (nh_fast) {
      nh_codes.reserve(512 * this->n_chunks);
      nh_scores.reserve(512);
    }
    auto nh_fast_score = [&](const unsigned *ids, size_t count) {
      if (!count) {
        nh_scores.clear();
        return;
      }
      nh_codes.resize(count * this->n_chunks);
      nh_scores.resize(count);
      for (size_t i = 0; i < count; ++i) {
        if (ids[i] >= this->num_points)
          throw ANNException("NavHints fast selector ID outside index", -1,
                             __FUNCSIG__, __FILE__, __LINE__);
        memcpy(nh_codes.data() + i * this->n_chunks,
               this->data + (_u64)ids[i] * this->n_chunks,
               this->n_chunks);
      }
      // Exactly the same chunk-major PQ LUT and operation order as native
      // compute_pq_dists; only the gather/lookup batch size changes.
      pq_flash_index_utils::pq_dist_lookup(
          nh_codes.data(), count, this->n_chunks,
          pq_dists, nh_scores.data());
      if (stats != nullptr) stats->n_cmps += count;
    };
    auto nh_consider_region = [](auto &pool, size_t capacity,
                                 float distance, size_t index) {
      const auto item = std::make_pair(distance,index);
      // C++ std::pair comparison matches legacy std::sort
      // (distance first, region index second), including equal-score ties.
      if (pool.size() == capacity && !(item < pool.back())) return;
      auto pos=std::lower_bound(pool.begin(),pool.end(),item);
      pool.insert(pos,item);
      if (pool.size() > capacity) pool.pop_back();
    };

'''
    part=once(part,anchor,pre+anchor,"fast query-local PQ scorer")

    a=part.index("    if (navhints_active && navhints_recent != nullptr")
    b=part.index("    // Two-stage Hint-IVF search:",a)
    recent=part[a:b]
    recent=once(recent,
      "      constexpr size_t batch_cap = 16;\n"
      "      for (size_t off = 0; off < navhints_recent->size(); off += batch_cap) {",
      r'''      // Replacing 32 tiny PQ batches by a single gather plus LUT pass
      // retains identical scoring and the original ID order.
      if (nh_fast) {
        nh_fast_score(navhints_recent->data(),navhints_recent->size());
        for (size_t i=0; i<navhints_recent->size(); ++i) {
          const unsigned id=(*navhints_recent)[i];
          if (visited.find(id) != visited.end()) continue;
          const float d=nh_scores[i];
          if (navhints_diverse) navhints_top(recent_pool,d,id);
          if (!found || d < best_hint_distance) {
            found=true; best_hint_id=id; best_hint_distance=d;
          }
        }
      }
      constexpr size_t batch_cap = 16;
      if (!nh_fast)
      for (size_t off = 0; off < navhints_recent->size(); off += batch_cap) {''',
      "fuse 512 recent scores into one PQ pass")
    part=part[:a]+recent+part[b:]

    a=part.index("    // Two-stage Hint-IVF search:")
    b=part.index("    std::sort(retset.begin(), retset.begin() + cur_list_size);",a)
    junction=part[a:b]
    junction=once(junction,
      "      constexpr size_t batch_cap = 16;\n"
      "      for (size_t off = 0; off < dir.medoids.size(); off += batch_cap) {",
      r'''      constexpr size_t batch_cap = 16;
      if (nh_fast) {
        nh_fast_score(dir.medoids.data(),dir.medoids.size());
        for (size_t cell=0; cell<dir.medoids.size(); ++cell)
          nh_consider_region(
              best_regions,dir.nprobe,nh_scores[cell],cell);
      }
      if (!nh_fast)
      for (size_t off = 0; off < dir.medoids.size(); off += batch_cap) {''',
      "score 512 native medoids once with exact bounded top8")
    junction=once(junction,
      "        for (size_t off = begin; off < end; off += batch_cap) {",
      r'''        if (nh_fast && begin < end) {
          // Preserve directory order and the original 'strictly less'
          // winner tie rule, while avoiding repeated 16-way PQ gathers.
          nh_fast_score(dir.children.data()+begin,end-begin);
          for (size_t i=0; i<end-begin; ++i) {
            const unsigned id=dir.children[begin+i];
            if (visited.find(id) != visited.end()) continue;
            const float d=nh_scores[i];
            if (navhints_diverse) navhints_top(junction_pool,d,id);
            if (!found_junction || d < best_junction_distance) {
              best_junction_distance=d;
              best_junction_id=id;
              found_junction=true;
            }
          }
        }
        if (!nh_fast)
        for (size_t off = begin; off < end; off += batch_cap) {''',
      "fuse each selected bucket in one PQ pass")
    part=part[:a]+junction+part[b:]
    p.write_text(begin+part+after)

def benchmark(root):
    p=root/"tests/search_disk_index.cpp"
    s=p.read_text()
    s=once(s,
       '        nav_mode != "gate50_cover") {',
       '        nav_mode != "gate50_cover" &&\n'
       '        nav_mode != "fast_core" && nav_mode != "fast_recent512") {',
       "register fast modes as identical candidate policies")
    s=once(s,
       '             nav_mode == "gate50_cover") ? &recent :',
       '             nav_mode == "gate50_cover" || nav_mode == "fast_core" ||\n'
       '             nav_mode == "fast_recent512") ? &recent :',
       "pass identical online Recent512")
    s=once(s,
       '             nav_mode == "gate50_cover")\n'
       '            ? native_junctions.get() : nullptr;',
       '             nav_mode == "gate50_cover" || nav_mode == "fast_core")\n'
       '            ? native_junctions.get() : nullptr;',
       "pass same native ID-only junction directory")
    s=once(s,
       '        nav_diag[i].coverage_policy =',
       '        nav_diag[i].fast_selector =\n'
       '            (nav_mode == "fast_core" || nav_mode == "fast_recent512");\n'
       '        nav_diag[i].coverage_policy =',
       "toggle optimized scorer not routing policy")
    s=once(s,
       '            nav_mode == "gate50_cover") {',
       '            nav_mode == "gate50_cover" ||\n'
       '            nav_mode == "fast_core" || nav_mode == "fast_recent512") {',
       "causal FIFO consistency")
    p.write_text(s)

def self_test():
    # Python reference check for top8 sorted distance+region tie behavior;
    # source-level exact query paths are separately audited by the workflow.
    rnd=random.Random(20261011)
    for n in (8,9,32,512):
        for k in (1,2,8,16):
            for _ in range(50):
                vals=[(rnd.randrange(16)*0.125,i) for i in range(n)]
                full=[]
                bounded=[]
                for pair in vals:
                    full.append(pair)
                    full.sort()
                    if len(full)>k:full.pop()
                    if len(bounded)<k or pair<bounded[-1]:
                        import bisect
                        bisect.insort(bounded,pair)
                        if len(bounded)>k:bounded.pop()
                if full!=bounded:
                    raise ValueError(f"top{ k} tie-order mismatch: {full} != {bounded}")
    print("STARLING_FAST_SELECTOR_UNIT_TIES_VERIFIED",flush=True)

def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("source",nargs="?",type=Path)
    arg.add_argument("--self-test",action="store_true")
    args=arg.parse_args()
    if args.self_test:
        self_test()
        if args.source is None:return
    if args.source is None:raise SystemExit("require temporary Starling checkout path")
    root=args.source.resolve()
    header(root)
    page(root)
    benchmark(root)
    print("STARLING_FAST_CORE_ADAPTER_READY "
          "same_hints=same_pq=same_policy; packed_batch_and_top8_no_extra_RAM",
          flush=True)

if __name__=="__main__":main()
