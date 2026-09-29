// Prepared query, contiguous-page selection and coalescing in one native call.
// No reassociation, fast-math, oracle threshold, or approximate ranking.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// Compile-time widths remove generic decoding from the common 8-bit path.
template<unsigned Bits>
static unsigned code_at(const uint8_t* row, unsigned coordinate) {
    if constexpr (Bits == 8) return row[coordinate];
    const unsigned pos = coordinate * Bits, byte = pos / 8, shift = pos % 8;
    unsigned value = row[byte];
    if (shift + Bits > 8) value |= unsigned(row[byte + 1]) << 8;
    return (value >> shift) & ((1u << Bits) - 1);
}

template<unsigned Bits>
static bool survives(const uint8_t* row, const float* radii, unsigned valid,
        unsigned dims, const double* origin, const double* scale,
        const double* query, double safe_tau) {
    for (unsigned i = 0; i < valid; ++i) {
        const double r = radii[i];
        // Invalid inputs must not silently cause rejection.
        if (!(r >= 0) || !std::isfinite(r)) return true;
        const double limit = (safe_tau + r) * (safe_tau + r);
        double sum = 0;
        bool reject = false;
        for (unsigned first = 0; first < dims; first += 16) {
            const unsigned end = std::min(dims, first + 16);
            for (unsigned j = first; j < end; ++j) {
                const double c = origin[j] + scale[j] * code_at<Bits>(row, i*dims+j);
                const double delta = query[j] - c;
                sum += delta * delta;
            }
            if (sum > limit) { reject = true; break; }
        }
        if (!reject) return true;
    }
    return false;
}

extern "C" int gp_plan(const uint8_t* packed, const float* radii,
        const uint16_t* valid, const float* radial, uint64_t total_pages,
        unsigned stride, unsigned cap, unsigned dims, unsigned bits,
        const double* origin, const double* scale, const double* query,
        double guard, double query_radial, uint64_t first, unsigned count,
        unsigned mode, double tau, unsigned gap, unsigned max_extent,
        uint8_t* keep, int64_t* extents, unsigned output_capacity,
        uint64_t* selected) {
    // mode: 0=none, 1=balls, 2=radial, 3=combined. Array shapes are
    // checked once by the prepared Python context and retained for its lifetime.
    if (!valid || !keep || !extents || !selected || !cap || !dims || mode > 3 ||
        !max_extent || count > output_capacity || first > total_pages ||
        count > total_pages-first || count > uint64_t(std::numeric_limits<int>::max()) ||
        std::isnan(tau) || tau < 0 || !std::isfinite(guard) || guard < 0 ||
        ((mode & 1) && (!packed || !radii || !origin || !scale || !query ||
          (bits != 4 && bits != 6 && bits != 8) || stride != (cap*dims*bits+7)/8)) ||
        ((mode & 2) && (!radial || !std::isfinite(query_radial) || query_radial < 0)))
        return -1;
    unsigned runs = 0;
    *selected = 0;
    const double safe_tau = tau + guard;
    for (unsigned offset = 0; offset < count; ++offset) {
        const uint64_t p = first + offset;
        if (valid[p] > cap) return -2;
        bool retain = true;
        if (std::isfinite(tau)) {
            if (mode & 2) {
                const double lb = std::max(double(radial[2*p])-query_radial,
                                          query_radial-double(radial[2*p+1]));
                retain = !(std::max(0.0, lb-guard) > tau);
            }
            if (retain && (mode & 1)) {
                const uint8_t* row = packed + p*stride;
                const float* rr = radii + p*cap;
                if (bits == 8) retain = survives<8>(row,rr,valid[p],dims,origin,scale,query,safe_tau);
                else if (bits == 6) retain = survives<6>(row,rr,valid[p],dims,origin,scale,query,safe_tau);
                else retain = survives<4>(row,rr,valid[p],dims,origin,scale,query,safe_tau);
            }
        }
        keep[offset] = retain;
        if (!retain) continue;
        ++*selected;
        if (runs) {
            const uint64_t start = uint64_t(extents[2*(runs-1)]);
            const uint64_t size = uint64_t(extents[2*(runs-1)+1]);
            if (p-start < max_extent && p-(start+size) <= gap) {
                extents[2*(runs-1)+1] = int64_t(p-start+1);
                continue;
            }
        }
        extents[2*runs] = int64_t(p);
        extents[2*runs+1] = 1;
        ++runs;
    }
    return int(runs);
}
