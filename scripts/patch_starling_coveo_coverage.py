#!/usr/bin/env python3
"""Stronger complementary-entry objective and chronological Coveo native replay.

Apply to a TEMPORARY checkout of pinned original Starling after the
Recent512, native-junction, diagnostic and selective-diverse patches.
The author's graph layout, navigator and search policy are not replaced.

Coveo 5K static training / 20K online warmup / 5K measured preserves the
frozen production stream. All norms are nonzero and both sides L2-normalized,
so native squared L2 and the original cosine/IP GT have identical ranking.
"""
from pathlib import Path
import argparse

def once(text,old,new,name):
    n=text.count(old)
    if n!=1:raise RuntimeError(f"{name}: expected one exact anchor, got {n}")
    return text.replace(old,new,1)

def header(root):
    p=root/"include/pq_flash_index.h"
    s=p.read_text()
    s=once(s,"    bool want_diverse = false;",
        "    bool want_diverse = false;\n"
        "    unsigned coverage_policy = 0; // 0: weak filter, 1: maxmin, 2: coverage/cost",
        "new candidate objective selector")
    p.write_text(s)

def page(root):
    p=root/"src/page_search.cpp"
    s=p.read_text()
    first=s.index("  void PQFlashIndex<T>::page_search(\n")
    last=s.index("  void PQFlashIndex<T>::page_search_interim(\n",first)
    before,section,after=s[:first],s[first:last],s[last:]
    old=r"""    auto navhints_select = [&](
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
    };"""
    new=r"""    auto navhints_select = [&](
        const std::vector<std::pair<float,unsigned>>& pool,
        unsigned &chosen, float &query_d, float &novelty) -> bool {
      if (pool.empty()) return false;
      // Original weak filtering preserved as a scientifically controlled arm.
      if (navhints_debug->coverage_policy == 0) {
        const float minimum =
            0.35f * std::max(1.f,navhints_debug->native_best);
        for (const auto &option : pool) {
          const float sep = navhints_novelty(option.second);
          if (sep >= minimum) {
            query_d=option.first; chosen=option.second; novelty=sep;
            return true;
          }
        }
        return false;
      }
      // Enforce a query-relevance budget BEFORE maximizing complementary
      // coverage: no more than 2x the best PQ(query,hint) squared distance.
      // All same-index resident PQ codes, no extra SSD access or training.
      const float max_qdist = std::max(0.001f,2.f*pool.front().first);
      float best_score=-std::numeric_limits<float>::infinity();
      bool found=false;
      for (const auto &option : pool) {
        if (option.first > max_qdist) break; // pool sorted by query distance
        const float sep=navhints_novelty(option.second);
        if (sep < 0.f) continue;  // page collision with native starts
        // 1: maximize distance to the nearest Starling start.
        // 2: favor novel coverage per unit of extra query distance.
        const float score = navhints_debug->coverage_policy == 1
            ? sep : sep / std::max(0.001f,option.first);
        if (score > best_score) {
          found=true; best_score=score; chosen=option.second;
          query_d=option.first; novelty=sep;
        }
      }
      return found;
    };"""
    section=once(section,old,new,"maximum marginal coverage on top-24")
    p.write_text(before+section+after)

def benchmark(root):
    p=root/"tests/search_disk_index.cpp"
    s=p.read_text()
    s=once(s,"  const size_t nav_warmup = 4000;",
           "  const size_t nav_warmup = 20000;",
           "chronological twenty-thousand online warmup")
    s=once(s,"query_num != 5000","query_num != 25000",
           "chronological 25K queries after 5K static training")
    s=once(s,"5K heldout uint8/L2 queries","25K chronological Coveo float/L2 queries",
           "correct native protocol assertion")
    s=once(s,
        '        nav_mode != "gate25_diverse") {',
        '        nav_mode != "gate25_diverse" &&\n'
        '        nav_mode != "cover_core" && nav_mode != "maxmin_core" &&\n'
        '        nav_mode != "gate50_cover") {',
        "new objective arms")
    s=once(s,
        '             nav_mode == "gate25_diverse") ? &recent :',
        '             nav_mode == "gate25_diverse" ||\n'
        '             nav_mode == "cover_core" || nav_mode == "maxmin_core" ||\n'
        '             nav_mode == "gate50_cover") ? &recent :',
        "online causal destinations in coverage modes")
    s=once(s,
        '             nav_mode == "gate25_diverse")\n'
        '            ? native_junctions.get() : nullptr;',
        '             nav_mode == "gate25_diverse" ||\n'
        '             nav_mode == "cover_core" || nav_mode == "maxmin_core" ||\n'
        '             nav_mode == "gate50_cover")\n'
        '            ? native_junctions.get() : nullptr;',
        "static learned junctions in coverage modes")
    s=once(s,
        '            (nav_mode == "gate25_diverse");',
        '            (nav_mode == "gate25_diverse");',
        "strict gate source preserved") if False else s
    s=once(s,
        '(nav_mode == "gate50_core" || nav_mode == "gate50_diverse");',
        '(nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
        '               nav_mode == "gate50_cover");',
        "calibrate coverage gate before heldout suffix")
    s=once(s,
        '        nav_diag[i].inject = (nav_mode != "score_only");\n'
        '        nav_diag[i].activation_gate =\n'
        '            (nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
        '             nav_mode == "gate25_diverse");\n'
        '        nav_diag[i].want_diverse =\n'
        '            (nav_mode == "diverse_core" || nav_mode == "gate50_diverse" ||\n'
        '             nav_mode == "gate25_diverse");\n'
        '        nav_diag[i].gate_threshold = gate_threshold;',
        '        nav_diag[i].inject = (nav_mode != "score_only");\n'
        '        nav_diag[i].activation_gate =\n'
        '            (nav_mode == "gate50_core" || nav_mode == "gate50_diverse" ||\n'
        '             nav_mode == "gate25_diverse" || nav_mode == "gate50_cover");\n'
        '        nav_diag[i].want_diverse =\n'
        '            (nav_mode == "diverse_core" || nav_mode == "gate50_diverse" ||\n'
        '             nav_mode == "gate25_diverse" || nav_mode == "cover_core" ||\n'
        '             nav_mode == "maxmin_core" || nav_mode == "gate50_cover");\n'
        '        nav_diag[i].coverage_policy =\n'
        '            (nav_mode == "cover_core" || nav_mode == "gate50_cover") ? 2 :\n'
        '            (nav_mode == "maxmin_core" ? 1 : 0);\n'
        '        nav_diag[i].gate_threshold = gate_threshold;',
        "enable novel entry selection and no-leak causal activation")
    s=once(s,
        '            nav_mode == "gate25_diverse") {',
        '            nav_mode == "gate25_diverse" ||\n'
        '            nav_mode == "cover_core" || nav_mode == "maxmin_core" ||\n'
        '            nav_mode == "gate50_cover") {',
        "maintain causal FIFO for coverage arms")
    s=once(s,
        '          << "gate_open,gate_threshold,diverse,recent_novelty,junction_novelty,";',
        '          << "gate_open,gate_threshold,diverse,recent_novelty,junction_novelty,";',
        "preserve diagnostic CSV format") if False else s
    p.write_text(s)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    root=ap.parse_args().source.resolve()
    header(root)
    page(root)
    benchmark(root)
    print("STARLING_COVEO_COMPLEMENTARY_ADAPTER_READY "
          "train5000 warm20000 eval5000; maxmin and cover/rank on 24 PQ candidates",
          flush=True)

if __name__=="__main__":main()
