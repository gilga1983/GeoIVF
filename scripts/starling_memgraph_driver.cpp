// Minimal driver for Starling's official in-memory Vamana Index.
//
// This file intentionally replaces only Starling's Boost.Program_options test
// CLI. The Index implementation, build/search algorithms, distance functions,
// tag semantics, graph serialization, and parameters come from the pinned
// upstream Starling sources.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "index.h"
#include "parameters.h"

namespace {

template <typename T>
struct Bin {
  uint32_t rows = 0;
  uint32_t dim = 0;
  std::vector<T> data;
};

template <typename T>
Bin<T> load_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Bin<T> b;
  f.read(reinterpret_cast<char*>(&b.rows), 4);
  f.read(reinterpret_cast<char*>(&b.dim), 4);
  if (!f || b.rows == 0 || b.dim == 0) throw std::runtime_error("bad bin header");
  const size_t n = static_cast<size_t>(b.rows) * b.dim;
  b.data.resize(n);
  f.read(reinterpret_cast<char*>(b.data.data()), n * sizeof(T));
  if (!f) throw std::runtime_error("truncated bin " + path);
  char extra;
  if (f.read(&extra, 1)) throw std::runtime_error("extra bytes in " + path);
  return b;
}

void save_ids(const std::string& path, const std::vector<uint32_t>& ids,
              uint32_t rows, uint32_t cols) {
  if (ids.size() != static_cast<size_t>(rows) * cols)
    throw std::runtime_error("result shape mismatch");
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot create " + path);
  f.write(reinterpret_cast<const char*>(&rows), 4);
  f.write(reinterpret_cast<const char*>(&cols), 4);
  f.write(reinterpret_cast<const char*>(ids.data()), ids.size() * sizeof(uint32_t));
  if (!f) throw std::runtime_error("write failed " + path);
}

unsigned parse_u(const char* s, const char* what) {
  unsigned long v = std::stoul(s);
  if (v > std::numeric_limits<unsigned>::max())
    throw std::runtime_error(std::string("overflow ") + what);
  return static_cast<unsigned>(v);
}

int build_cmd(int argc, char** argv) {
  // build <data_prefix> <index_prefix> <R> <L> <alpha> <threads>
  if (argc != 8) {
    std::cerr << "usage: starling_memgraph_driver build DATA_PREFIX INDEX_PREFIX R L ALPHA THREADS\n";
    return 2;
  }
  const std::string data_prefix = argv[2];
  const std::string index_prefix = argv[3];
  const unsigned R = parse_u(argv[4], "R");
  const unsigned L = parse_u(argv[5], "L");
  const float alpha = std::stof(argv[6]);
  const unsigned threads = parse_u(argv[7], "threads");

  const auto tags_bin = load_bin<uint32_t>(data_prefix + "_ids.bin");
  if (tags_bin.dim != 1) throw std::runtime_error("tag file must be Nx1");
  const auto data_meta = load_bin<float>(data_prefix + "_data.bin");
  if (data_meta.rows != tags_bin.rows) throw std::runtime_error("data/tag row mismatch");

  std::vector<uint32_t> tags(tags_bin.data.begin(), tags_bin.data.end());
  diskann::Parameters paras;
  paras.Set<unsigned>("R", R);
  paras.Set<unsigned>("L", L);
  paras.Set<unsigned>("C", 750);
  paras.Set<float>("alpha", alpha);
  paras.Set<bool>("saturate_graph", false);
  paras.Set<unsigned>("num_threads", threads);

  diskann::Index<float, uint32_t> index(
      diskann::Metric::INNER_PRODUCT, data_meta.dim, data_meta.rows, false, true);

  const auto begin = std::chrono::steady_clock::now();
  index.build((data_prefix + "_data.bin").c_str(), data_meta.rows, paras, tags);
  const auto end = std::chrono::steady_clock::now();
  index.save(index_prefix.c_str());

  std::cout << "BUILD_SECONDS "
            << std::chrono::duration<double>(end - begin).count()
            << " ROWS " << data_meta.rows << " DIM " << data_meta.dim << "\n";
  return 0;
}

int search_cmd(int argc, char** argv) {
  // search <index_prefix> <query.fbin> <result_prefix> <threads> <L> [L...]
  if (argc < 8) {
    std::cerr << "usage: starling_memgraph_driver search INDEX QUERY RESULT_PREFIX THREADS L [L...]\n";
    return 2;
  }
  const std::string index_prefix = argv[2];
  const std::string query_path = argv[3];
  const std::string result_prefix = argv[4];
  const unsigned threads = parse_u(argv[5], "threads");

  std::vector<unsigned> ls;
  for (int i = 6; i < argc; ++i) ls.push_back(parse_u(argv[i], "L"));
  if (ls.empty()) throw std::runtime_error("no search L");
  const unsigned max_l = *std::max_element(ls.begin(), ls.end());

  auto q = load_bin<float>(query_path);
  diskann::Index<float, uint32_t> index(
      diskann::Metric::INNER_PRODUCT, q.dim, 0, false, true);
  index.load(index_prefix.c_str(), threads, max_l);

  omp_set_num_threads(threads);
  for (const unsigned L : ls) {
    std::vector<uint32_t> out(q.rows);
    std::vector<double> latency(q.rows);

    const auto batch_begin = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < static_cast<int64_t>(q.rows); ++i) {
      auto start = std::chrono::steady_clock::now();
      uint32_t tag = 0;
      std::vector<float*> res;
      index.search_with_tags(
          q.data.data() + static_cast<size_t>(i) * q.dim,
          1, L, &tag, nullptr, nullptr, res);
      out[static_cast<size_t>(i)] = tag;
      auto stop = std::chrono::steady_clock::now();
      latency[static_cast<size_t>(i)] =
          std::chrono::duration<double, std::micro>(stop - start).count();
    }
    const auto batch_end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(batch_end - batch_begin).count();
    const double qps = q.rows / seconds;
    const double mean_us =
        std::accumulate(latency.begin(), latency.end(), 0.0) / q.rows;
    std::sort(latency.begin(), latency.end());
    const size_t pidx = std::min<size_t>(
        latency.size() - 1, static_cast<size_t>(0.999 * latency.size()));
    const double p999_us = latency[pidx];

    save_ids(result_prefix + "_" + std::to_string(L) + "_idx_uint32.bin",
             out, q.rows, 1);
    std::cout << std::fixed << std::setprecision(3)
              << L << " " << qps << " " << mean_us << " " << p999_us << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) return 2;
    const std::string cmd = argv[1];
    if (cmd == "build") return build_cmd(argc, argv);
    if (cmd == "search") return search_cmd(argc, argv);
    std::cerr << "unknown command: " << cmd << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "starling_memgraph_driver: " << e.what() << "\n";
    return 1;
  }
}
