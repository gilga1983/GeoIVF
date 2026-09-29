// Shared FP64 full-vector scan and bounded top-k heap over unchanged FP32 pages.
// No approximate ranking, fast-math, or reduced-precision payload is used.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Neighbor {
    double distance;
    int64_t id;
    bool operator<(const Neighbor& b) const {
        return distance < b.distance || (distance == b.distance && id < b.id);
    }
};
struct TopK {
    unsigned dims, capacity, page_bytes, k;
    bool failed = false;
    std::vector<double> query;
    std::vector<Neighbor> heap;
};
static void message(char* err, size_t size, const char* text) {
    if (err && size) std::snprintf(err, size, "%s", text);
}
extern "C" void* gk_create(const double* query, unsigned dims, unsigned capacity,
                          unsigned page_bytes, unsigned k, char* err, size_t size) {
    TopK* s = nullptr;
    try {
        if (!query || !dims || !capacity || !k || k > 1000000 ||
            uint64_t(capacity) * dims * sizeof(float) > page_bytes)
            throw std::invalid_argument("invalid top-k parameters");
        s = new TopK{dims, capacity, page_bytes, k, false, {}, {}};
        s->query.assign(query, query + dims);
        for (double v : s->query)
            if (!std::isfinite(v) || std::abs(v) > 1e10)
                throw std::invalid_argument("invalid query coordinate");
        s->heap.reserve(k);
        return s;
    } catch (const std::exception& e) { message(err, size, e.what()); delete s; return nullptr; }
}
extern "C" int gk_consume(void* handle, uint32_t n, const void* const* buffers,
        const uint64_t* lengths, const int64_t* starts, const uint32_t* counts,
        const int64_t* ids, const uint16_t* valid, uint64_t total_pages,
        uint64_t* evaluated, uint32_t* filled, double* threshold2, char* err, size_t size) {
    auto* s = static_cast<TopK*>(handle);
    try {
        if (!s || s->failed || !evaluated || !filled || !threshold2 || !ids || !valid ||
            (n && (!buffers || !lengths || !starts || !counts)))
            throw std::invalid_argument("invalid scan arguments or failed accumulator");
        uint64_t done = 0;
        for (uint32_t b = 0; b < n; ++b) {
            if (!buffers[b] || starts[b] < 0 || !counts[b] ||
                uint64_t(starts[b]) >= total_pages || counts[b] > total_pages-uint64_t(starts[b]) ||
                lengths[b] != uint64_t(counts[b]) * s->page_bytes)
                throw std::invalid_argument("invalid extent or buffer length");
            auto* raw = static_cast<const unsigned char*>(buffers[b]);
            for (uint32_t j = 0; j < counts[b]; ++j) {
                const uint64_t p = uint64_t(starts[b])+j;
                if (valid[p] > s->capacity) throw std::invalid_argument("invalid page occupancy");
                for (unsigned i = 0; i < valid[p]; ++i) {
                    const auto* point = raw + uint64_t(j)*s->page_bytes + uint64_t(i)*s->dims*4;
                    double dd = 0.0;
                    for (unsigned d = 0; d < s->dims; ++d) {
                        float v;
                        // memcpy also supports unaligned byte buffers and avoids aliasing UB.
                        std::memcpy(&v, point + d*4, sizeof(v));
                        const double delta = double(v)-s->query[d];
                        dd += delta*delta;
                    }
                    const int64_t id = ids[p*s->capacity+i];
                    if (!std::isfinite(dd) || id < 0)
                        throw std::invalid_argument("nonfinite payload distance or invalid ID");
                    Neighbor candidate{dd, id};
                    if (s->heap.size() < s->k) {
                        s->heap.push_back(candidate);
                        std::push_heap(s->heap.begin(), s->heap.end());
                    } else if (candidate < s->heap.front()) {
                        std::pop_heap(s->heap.begin(), s->heap.end());
                        s->heap.back() = candidate;
                        std::push_heap(s->heap.begin(), s->heap.end());
                    }
                    ++done;
                }
            }
        }
        *evaluated = done;
        *filled = static_cast<uint32_t>(s->heap.size());
        *threshold2 = s->heap.size() == s->k ? s->heap.front().distance : std::numeric_limits<double>::infinity();
        return 0;
    } catch (const std::exception& e) {
        if (s) s->failed = true;
        message(err, size, e.what()); return -1;
    }
}
extern "C" int gk_finish(void* handle, int64_t* ids, double* distances, unsigned capacity,
                         char* err, size_t size) {
    auto* s = static_cast<TopK*>(handle);
    try {
        if (!s || s->failed || !ids || !distances || capacity < s->heap.size())
            throw std::invalid_argument("invalid output or failed accumulator");
        auto sorted = s->heap;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i=0; i<sorted.size(); ++i) { ids[i]=sorted[i].id; distances[i]=sorted[i].distance; }
        return static_cast<int>(sorted.size());
    } catch (const std::exception& e) { message(err,size,e.what()); return -1; }
}
extern "C" void gk_close(void* handle) { delete static_cast<TopK*>(handle); }
