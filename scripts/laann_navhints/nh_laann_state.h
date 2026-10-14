// Native LAANN + NavHints experiment, outside the authors' PageANN/LAANN tree.
// Licensed under the MIT license, matching PageANN. No changes to SSD graph.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>

namespace navhints_laann {

struct Region {
  uint32_t representative = 0;
  std::vector<uint32_t> vertices;
};

inline unsigned env_uint(const char *name, unsigned fallback, unsigned maximum) {
  const char *s = std::getenv(name);
  if (!s || !*s) return fallback;
  const std::string v(s);
  if (v.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error(std::string(name) + ": expected unsigned integer");
  unsigned long n = std::stoul(v);
  if (n > maximum) throw std::runtime_error(std::string(name) + ": above limit");
  return static_cast<unsigned>(n);
}

class State {
 public:
  State() {
    const char *raw = std::getenv("NH_LAANN_MODE");
    mode_ = raw ? raw : "off";
    if (mode_ != "off" && mode_ != "trace" && mode_ != "junction" &&
        mode_ != "history" && mode_ != "core")
      throw std::runtime_error("NH_LAANN_MODE: expected off, trace, junction, history or core");
    trace_ = mode_ == "trace";
    active_ = mode_ == "junction" || mode_ == "history" || mode_ == "core";
    use_junction_ = mode_ == "junction" || mode_ == "core";
    use_history_ = mode_ == "history" || mode_ == "core";
    capacity_ = env_uint("NH_LAANN_HISTORY", 2048, 100000);
    starts_ = env_uint("NH_LAANN_STARTS", 8, 64);
    nprobe_ = env_uint("NH_LAANN_NPROBE", 8, 512);
    if (use_junction_) {
      const char *path = std::getenv("NH_LAANN_DIRECTORY");
      if (!path || !*path) throw std::runtime_error("NH_LAANN_DIRECTORY required for junction routing");
      std::ifstream input(path);
      std::string magic;
      size_t num_regions = 0;
      if (!(input >> magic >> num_regions) || magic != "NHLAANN1" ||
          num_regions == 0 || num_regions > 512)
        throw std::runtime_error("Invalid NHLAANN1 region directory header");
      size_t ids = 0;
      for (size_t r = 0; r < num_regions; ++r) {
        Region region;
        size_t n = 0;
        if (!(input >> region.representative >> n) || n == 0 || n > 14464)
          throw std::runtime_error("Invalid or empty region in LAANN directory");
        region.vertices.resize(n);
        for (size_t j = 0; j < n; ++j)
          if (!(input >> region.vertices[j]))
            throw std::runtime_error("Truncated LAANN junction directory");
        ids += n;
        regions_.push_back(std::move(region));
      }
      if (ids == 0 || ids > 14464)
        throw std::runtime_error("LAANN directory outside B=14464 budget");
      std::string extra;
      if (input >> extra) throw std::runtime_error("Unexpected data after LAANN directory");
      junction_count_ = ids;
    }
    if (trace_) {
      const char *p = std::getenv("NH_LAANN_TRACE_FILE");
      if (!p || !*p) throw std::runtime_error("Trace mode requires NH_LAANN_TRACE_FILE");
      trace_file_ = p;
    }
    if (active_ || trace_) {
      std::cout << "NH_LAANN " << mode_
                << " B=" << junction_count_
                << " C=" << (use_history_ ? capacity_ : 0)
                << " k=" << (use_history_ ? starts_ : 0)
                << " regions=" << regions_.size()
                << " approximate_id_and_region_payload_bytes="
                << (4 * junction_count_ + 12 * regions_.size() +
                    (use_history_ ? 4 * capacity_ : 0))
                << std::endl;
    }
  }

  static State &get() {
    static State state;
    return state;
  }

  bool active() const { return active_; }
  bool tracing() const { return trace_; }

  std::vector<uint32_t> history_snapshot() {
    if (!use_history_) return {};
    std::lock_guard<std::mutex> g(mu_);
    return std::vector<uint32_t>(history_.begin(), history_.end());
  }

  // score_batch(ids, dists) must evaluate native LAANN-resident PQ codes
  // against the already constructed per-query PQ lookup table.
  template <typename Queue, typename BatchScorer>
  bool inject(Queue &queue, uint32_t total_points, uint32_t vectors_per_page,
              BatchScorer score_batch) {
    if (!active_) return false;
    if (vectors_per_page == 0) throw std::runtime_error("LAANN page capacity is zero");
    auto valid = [total_points](uint32_t id) { return id < total_points; };
    auto comparator = [](const std::pair<float,uint32_t> &a,
                         const std::pair<float,uint32_t> &b) {
      return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    bool inserted = false;
    uint32_t selected_junction = UINT32_MAX;
    if (use_junction_) {
      std::vector<uint32_t> representatives;
      representatives.reserve(regions_.size());
      for (const auto &r : regions_) {
        if (!valid(r.representative))
          throw std::runtime_error("Junction representative outside native LAANN ID range");
        representatives.push_back(r.representative);
      }
      std::vector<float> scores;
      score_batch(representatives, scores);
      if (scores.size() != representatives.size())
        throw std::runtime_error("LAANN batch distance count mismatch");
      std::vector<std::pair<float,size_t>> best_regions;
      for (size_t i = 0; i < scores.size(); ++i)
        best_regions.emplace_back(scores[i], i);
      std::sort(best_regions.begin(), best_regions.end(),
        [](const auto &a,const auto &b) {
          return a.first < b.first || (a.first == b.first && a.second < b.second);
        });
      std::vector<uint32_t> eligible;
      for (size_t i = 0; i < std::min<size_t>(nprobe_,best_regions.size()); ++i)
        for (uint32_t id : regions_[best_regions[i].second].vertices) {
          if (!valid(id))
            throw std::runtime_error("Junction vertex outside native LAANN ID range");
          eligible.push_back(id);
        }
      if (!eligible.empty()) {
        score_batch(eligible, scores);
        std::vector<std::pair<float,uint32_t>> ranked;
        for (size_t i = 0; i < eligible.size(); ++i)
          ranked.emplace_back(scores[i], eligible[i]);
        auto it = std::min_element(ranked.begin(), ranked.end(), comparator);
        selected_junction = it->second;
        queue.insert(diskann::Neighbor(it->second, it->first));
        inserted = true;
      }
    }
    if (use_history_) {
      const auto history = history_snapshot();
      if (!history.empty() && starts_ > 0) {
        std::vector<float> scores;
        score_batch(history, scores);
        if (scores.size() != history.size())
          throw std::runtime_error("History PQ distance count mismatch");
        std::vector<std::pair<float,uint32_t>> ranked;
        ranked.reserve(history.size());
        for (size_t i = 0; i < history.size(); ++i) {
          if (!valid(history[i]))
            throw std::runtime_error("Historical vertex outside native LAANN ID range");
          ranked.emplace_back(scores[i], history[i]);
        }
        std::sort(ranked.begin(), ranked.end(), comparator);
        std::unordered_set<uint32_t> selected_pages;
        if (selected_junction != UINT32_MAX)
          selected_pages.insert(selected_junction / vectors_per_page);
        uint32_t emitted = 0;
        for (const auto &item : ranked) {
          if (emitted >= starts_) break;
          if (!selected_pages.insert(item.second / vectors_per_page).second)
            continue;  // One starting point per SSD page.
          queue.insert(diskann::Neighbor(item.second, item.first));
          ++emitted;
          inserted = true;
        }
      }
    }
    return inserted;
  }

  // Called after each completed request, before LAANN captures total_us.
  void completed(uint32_t result, const std::vector<uint32_t> &page_visits) {
    if (!active_ && !trace_) return;
    std::lock_guard<std::mutex> g(mu_);
    if (trace_) {
      std::ofstream out(trace_file_, std::ios::app);
      if (!out) throw std::runtime_error("Cannot append LAANN training trace");
      // One line per query, IDs in first-expansion order. Values are
      // PAGE REPRESENTATIVE VECTOR IDs (not original-dataset IDs).
      for (size_t i = 0; i < page_visits.size(); ++i) {
        if (i) out << ' ';
        out << page_visits[i];
      }
      out << '\n';
    }
    if (use_history_ && capacity_ > 0) {
      if (history_set_.insert(result).second) {
        if (history_.size() == capacity_) {
          history_set_.erase(history_.front());
          history_.pop_front();
        }
        history_.push_back(result);
      }
    }
  }

 private:
  std::string mode_;
  bool active_ = false, trace_ = false;
  bool use_junction_ = false, use_history_ = false;
  unsigned capacity_ = 2048, starts_ = 8, nprobe_ = 8;
  std::vector<Region> regions_;
  size_t junction_count_ = 0;
  std::string trace_file_;
  std::deque<uint32_t> history_;
  std::unordered_set<uint32_t> history_set_;
  std::mutex mu_;
};

} // namespace navhints_laann
