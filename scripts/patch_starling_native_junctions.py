#!/usr/bin/env python3
"""Native Starling full-core extension: learn its own 16K junctions offline.

Apply *after* scripts/patch_starling_recent_hints.py to pinned original Starling.
Starling exports its 16K most visited graph vertices by replaying 5K
disjoint training queries with its author-native page search. The existing
build_hint_ivf_generic.py partitions those *Starling-generated IDs* into
512 geometrical regions, storing only IDs/CSR offsets.

For query q, this patch PQ-scores all 512 region representatives using the
PQ codes Starling already keeps resident, PQ-scores the members of the
eight closest regions, and supplies the best unused junction ID as one
additional native frontier start. In 'core16k_recent512' mode the previous
512 successful destinations supply one additional PQ-scored candidate.

No graph edges, disk page layouts, author indexing procedure, in-memory
navigator, PQ codes, page scheduler, or search termination are altered.
The extra CPU scoring is part of Starling's per-query time and n_cmps.
"""
from __future__ import annotations
from pathlib import Path
import argparse

def once(s:str,old:str,new:str,name:str)->str:
    n=s.count(old)
    if n!=1:raise ValueError(f"{name}: expected one match, saw {n}")
    return s.replace(old,new,1)

def header(root:Path)->None:
    path=root/"include/pq_flash_index.h"
    s=path.read_text()
    s=once(s,
        "namespace diskann {\n",
        """namespace diskann {
  // ID-only compact directory; database vector IDs remain original
  // Starling IDs, not new graph vertices or independent landing pages.
  struct StarlingHintIvf {
    std::vector<unsigned> medoids;
    std::vector<unsigned> offsets;
    std::vector<unsigned> children;
    unsigned nprobe = 8;
  };
""","Starling 16K directory state")
    s=once(s,
        """        const std::vector<unsigned> *navhints_recent = nullptr);""",
        """        const std::vector<unsigned> *navhints_recent = nullptr,
        const StarlingHintIvf *navhints_junctions = nullptr);""",
        "static directory optional native page-search interface")
    path.write_text(s)

def page_search(root:Path)->None:
    path=root/"src/page_search.cpp"
    s=path.read_text()
    marker="  void PQFlashIndex<T>::page_search(\n"
    a=s.index(marker)
    b=s.index("  void PQFlashIndex<T>::page_search_interim(\n",a)
    head,part,tail=s[:a],s[a:b],s[b:]
    part=once(part,
        """      const std::vector<unsigned> *navhints_recent) {""",
        """      const std::vector<unsigned> *navhints_recent,
      const StarlingHintIvf *navhints_junctions) {""",
        "native junction signature")
    start=r'''    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;'''
    extension=r'''    // Two-stage Hint-IVF search: 512 PQ-scored medoids, then
    // the children of the eight most promising regions. This selects
    // exactly ONE additional junction start, without replacing Starling's
    // original in-memory graph navigator or reading an extra graph page.
    if (navhints_junctions != nullptr) {
      const auto &dir = *navhints_junctions;
      if (dir.medoids.size() != 512 || dir.offsets.size() != 513 ||
          dir.nprobe == 0 || dir.nprobe > 64) {
        throw ANNException("Invalid Starling native Hint-IVF directory",
                           -1, __FUNCSIG__, __FILE__, __LINE__);
      }
      std::vector<std::pair<float, size_t>> best_regions;
      best_regions.reserve(dir.nprobe + 1);
      constexpr size_t batch_cap = 16;
      for (size_t off = 0; off < dir.medoids.size(); off += batch_cap) {
        const size_t n = std::min(batch_cap, dir.medoids.size() - off);
        compute_pq_dists(dir.medoids.data() + off, n, dist_scratch);
        if (stats != nullptr) stats->n_cmps += n;
        for (size_t i = 0; i < n; ++i) {
          best_regions.emplace_back(dist_scratch[i], off + i);
          std::sort(best_regions.begin(), best_regions.end());
          if (best_regions.size() > dir.nprobe) best_regions.pop_back();
        }
      }
      float best_junction_distance = std::numeric_limits<float>::infinity();
      unsigned best_junction_id = 0;
      bool found_junction = false;
      for (const auto &selected : best_regions) {
        const size_t cell = selected.second;
        const unsigned medoid_id = dir.medoids[cell];
        if (visited.find(medoid_id) == visited.end() &&
            (!found_junction || selected.first < best_junction_distance)) {
          best_junction_distance = selected.first;
          best_junction_id = medoid_id;
          found_junction = true;
        }
        const size_t begin = dir.offsets[cell];
        const size_t end = dir.offsets[cell+1];
        for (size_t off = begin; off < end; off += batch_cap) {
          const size_t n = std::min(batch_cap, end - off);
          compute_pq_dists(dir.children.data() + off, n, dist_scratch);
          if (stats != nullptr) stats->n_cmps += n;
          for (size_t i = 0; i < n; ++i) {
            const unsigned id = dir.children[off+i];
            if (visited.find(id) != visited.end()) continue;
            const float d = dist_scratch[i];
            if (!found_junction || d < best_junction_distance) {
              best_junction_distance = d;
              best_junction_id = id;
              found_junction = true;
            }
          }
        }
      }
      if (found_junction && cur_list_size < l_search) {
        retset[cur_list_size++] =
            Neighbor(best_junction_id, best_junction_distance, true);
        visited.insert(best_junction_id);
      }
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;'''
    part=once(part,start,extension,"native ID-only junction scorer")
    path.write_text(head+part+tail)

def benchmark(root:Path)->None:
    p=root/"tests/search_disk_index.cpp"
    s=p.read_text()
    s=once(s,"#include <cstdlib>\n",
           "#include <cstdlib>\n#include <fstream>\n",
           "Starling packed file support")
    marker="namespace po = boost::program_options;\n"
    helper=r'''
// Strict reader for the same compact GHIVF001 ID+CSR format as NavHints.
// Verified once at startup; PQ lookup and graph search remain native Starling.
static diskann::StarlingHintIvf load_starling_hint_ivf(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("missing native Starling junction directory");
  char magic[8] = {};
  in.read(magic,8);
  if (!in || std::memcmp(magic,"GHIVF001",8)!=0)
    throw std::runtime_error("wrong native Starling junction magic");
  uint32_t fields[4] = {};
  in.read(reinterpret_cast<char*>(fields),sizeof(fields));
  const uint32_t nlist=fields[0], children=fields[1], total=fields[2], reserved=fields[3];
  if (!in || nlist!=512 || total!=16000 || children+nlist!=total || reserved!=0)
    throw std::runtime_error("wrong native Starling junction shape");
  diskann::StarlingHintIvf dir;
  dir.medoids.resize(nlist);
  dir.offsets.resize(nlist+1);
  dir.children.resize(children);
  in.read(reinterpret_cast<char*>(dir.medoids.data()),nlist*4);
  in.read(reinterpret_cast<char*>(dir.offsets.data()),(nlist+1)*4);
  in.read(reinterpret_cast<char*>(dir.children.data()),children*4);
  if (!in || in.peek()!=std::char_traits<char>::eof())
    throw std::runtime_error("truncated or trailing Starling hint bytes");
  if (dir.offsets[0]!=0 || dir.offsets.back()!=children)
    throw std::runtime_error("invalid native Hint-IVF bucket offsets");
  for (size_t i=1; i<dir.offsets.size(); ++i) {
    if (dir.offsets[i]<dir.offsets[i-1])
      throw std::runtime_error("unordered Hint-IVF offsets");
  }
  std::unordered_set<unsigned> unique;
  for (auto id:dir.medoids) {
    if (id>=10000000 || !unique.insert(id).second)
      throw std::runtime_error("invalid duplicate or out-of-range junction medoid");
  }
  for (auto id:dir.children) {
    if (id>=10000000 || !unique.insert(id).second)
      throw std::runtime_error("invalid duplicate or out-of-range junction child");
  }
  if (unique.size()!=16000)
    throw std::runtime_error("incomplete 16K native junction directory");
  dir.nprobe=8;
  return dir;
}

'''
    s=once(s,marker,marker+helper,"Starling packed directory loader")
    before=r'''  // cache bfs levels
  std::vector<uint32_t> node_list;'''
    after=r'''  // Export exactly the top-16K Starling *native traversal* junction IDs
  // on a disjoint 5K-query training prefix, before any heldout replay.
  if (const char* target = std::getenv("STARLING_HINT_EXPORT_TOP16K")) {
    const char* train = std::getenv("STARLING_NAVHINTS_TRAIN_FILE");
    if (!train || !*train || !use_page_search || mem_L == 0)
      throw std::runtime_error("native Starling junction export needs train file and navigator");
    std::vector<uint32_t> native_hubs;
    _pFlashIndex->generate_cache_list_from_sample_queries(
        train, 40, 8, 16000, num_threads, native_hubs, true, mem_L);
    if (native_hubs.size()!=16000)
      throw std::runtime_error("Starling native top16K export incomplete");
    std::ofstream fout(target,std::ios::binary|std::ios::trunc);
    const char magic[8]={'G','I','D','S','T','0','0','1'};
    const uint32_t count=static_cast<uint32_t>(native_hubs.size()), reserved=0;
    fout.write(magic,sizeof(magic));
    fout.write(reinterpret_cast<const char*>(&count),4);
    fout.write(reinterpret_cast<const char*>(&reserved),4);
    fout.write(reinterpret_cast<const char*>(native_hubs.data()),count*4);
    if (!fout) throw std::runtime_error("failed Starling junction export");
    std::cout << "STARLING_NATIVE_JUNCTION_EXPORT count=" << count
              << " bytes=" << 16+count*4 << std::endl;
    return 0;
  }

  std::unique_ptr<diskann::StarlingHintIvf> native_junctions;
  if (const char* path = std::getenv("STARLING_NAVHINTS_IVF_FILE")) {
    native_junctions = std::make_unique<diskann::StarlingHintIvf>(
        load_starling_hint_ivf(path));
    std::cout << "STARLING_NATIVE_JUNCTIONS_LOADED nlist="
              << native_junctions->medoids.size()
              << " children=" << native_junctions->children.size()
              << " nprobe=" << native_junctions->nprobe << std::endl;
  }

  // cache bfs levels
  std::vector<uint32_t> node_list;'''
    s=once(s,before,after,"export and load native Starling junctions")
    s=once(s,
        '''    if (nav_mode != "baseline" && nav_mode != "recent512" &&
        nav_mode != "random512") {''',
        '''    if (nav_mode != "baseline" && nav_mode != "recent512" &&
        nav_mode != "random512" && nav_mode != "learned16k" &&
        nav_mode != "core16k_recent512") {''',
        "additional learned modes")
    s=once(s,
        '''            nav_mode == "recent512" ? &recent :
            (nav_mode == "random512" ? &random_ids : nullptr);''',
        '''            (nav_mode == "recent512" || nav_mode == "core16k_recent512")
                ? &recent :
            (nav_mode == "random512" ? &random_ids : nullptr);
        const diskann::StarlingHintIvf *dir =
            (nav_mode == "learned16k" || nav_mode == "core16k_recent512")
            ? native_junctions.get() : nullptr;
        if ((nav_mode == "learned16k" || nav_mode == "core16k_recent512") &&
            dir == nullptr) {
          throw std::runtime_error("learned NavHints modes require Starling-trained IVF");
        }''',
        "learned and online hints")
    s=once(s,
        '''            stats + i, candidates);''',
        '''            stats + i, candidates, dir);''',
        "pass learn-to-route IDs into native search")
    s=once(s,
        '''        if (nav_mode == "recent512") {''',
        '''        if (nav_mode == "recent512" || nav_mode == "core16k_recent512") {''',
        "online FIFO updates in Core")
    p.write_text(s)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("source",type=Path)
    x=a.parse_args()
    root=x.source.resolve()
    for name in ("include/pq_flash_index.h","src/page_search.cpp",
                 "tests/search_disk_index.cpp"):
        if not (root/name).is_file():raise SystemExit(f"no original Starling file {name}")
    header(root)
    page_search(root)
    benchmark(root)
    print("STARLING_NATIVE_FULL_CORE_ADAPTER_READY",flush=True)

if __name__=="__main__":main()
