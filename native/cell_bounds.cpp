// Independent scalar codes as balls, quantization cells, or both.
// Build without fast-math; callers apply conservative roundoff allowances.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

static unsigned code_at(const uint8_t* row, unsigned coord, unsigned bits) {
    if (bits == 8) return row[coord];
    const unsigned pos = coord * bits, byte = pos / 8, shift = pos % 8;
    unsigned value = row[byte];
    if (shift + bits > 8) value |= unsigned(row[byte + 1]) << 8;
    return (value >> shift) & ((1u << bits) - 1);
}

template<int Shape, bool Select, bool Early>
static void kernel(const uint8_t* packed, const float* radii, const uint16_t* valid,
                   const int64_t* pages, uint64_t n, unsigned stride, unsigned cap,
                   unsigned dims, unsigned bits, const double* origin,
                   const double* scale, const double* query, double padding,
                   double guard, double tau, double* bounds, uint8_t* keep) {
    const double safe_tau = tau + guard, box_limit = safe_tau * safe_tau;
    for (uint64_t row = 0; row < n; ++row) {
        const uint64_t p = static_cast<uint64_t>(pages[row]);
        const uint8_t* codes = packed + p * stride;
        double page_lb = std::numeric_limits<double>::infinity();
        bool any = false;
        for (unsigned i = 0; i < valid[p]; ++i) {
            double sb = 0.0, sx = 0.0;
            double r = 0.0;
            if constexpr (Shape != 1) r = radii[p * cap + i];
            const double ball_limit = (safe_tau + r) * (safe_tau + r);
            bool rejected = false;
            for (unsigned j = 0; j < dims; ++j) {
                const double c = origin[j] + scale[j] * code_at(codes, i * dims + j, bits);
                const double delta = query[j] - c;
                if constexpr (Shape != 1) sb += delta * delta;
                if constexpr (Shape != 0) {
                    // Covers closed nearest-rounding bins, including endpoint ties.
                    const double gap = std::max(0.0, std::abs(delta) - (scale[j] * .5 + padding));
                    sx += gap * gap;
                }
                if constexpr (Select && Early) {
                    if ((j + 1) % 16 == 0 || j + 1 == dims) {
                        if constexpr (Shape != 1) rejected = sb > ball_limit;
                        if constexpr (Shape != 0) rejected = rejected || sx > box_limit;
                        if (rejected) break;
                    }
                }
            }
            if constexpr (Select) {
                if constexpr (Shape != 1) rejected = rejected || sb > ball_limit;
                if constexpr (Shape != 0) rejected = rejected || sx > box_limit;
                if (!rejected) any = true;
                // Same page-level exit rule for full and partial-coordinate kernels.
                if (any) break;
            } else {
                double lb = 0.0;
                if constexpr (Shape != 1) lb = std::max(0.0, std::sqrt(sb) - r);
                if constexpr (Shape != 0) lb = std::max(lb, std::sqrt(sx));
                page_lb = std::min(page_lb, lb);
            }
        }
        if constexpr (Select) keep[row] = any;
        else bounds[row] = std::max(0.0, page_lb - guard);
    }
}

extern "C" int gc_evaluate(const uint8_t* packed, const float* radii,
        const uint16_t* valid, const int64_t* pages, uint64_t n,
        unsigned stride, unsigned cap, unsigned dims, unsigned bits,
        const double* origin, const double* scale, const double* query,
        double padding, double guard, int shape, int strategy, double tau,
        double* bounds, uint8_t* keep) {
    if (!packed || !valid || !pages || !origin || !scale || !query || !dims ||
        bits < 1 || bits > 8 || !cap || stride != (cap * dims * bits + 7) / 8 ||
        shape < 0 || shape > 2 || strategy < 0 || strategy > 2 ||
        padding < 0 || guard < 0 || std::isnan(tau) || tau < 0 ||
        (shape != 1 && !radii) || (strategy == 0 && !bounds) || (strategy && !keep)) return -1;
#define RUN(S) \
    if (strategy == 0) kernel<S,false,false>(packed,radii,valid,pages,n,stride,cap,dims,bits,origin,scale,query,padding,guard,tau,bounds,keep); \
    else if (strategy == 1) kernel<S,true,false>(packed,radii,valid,pages,n,stride,cap,dims,bits,origin,scale,query,padding,guard,tau,bounds,keep); \
    else kernel<S,true,true>(packed,radii,valid,pages,n,stride,cap,dims,bits,origin,scale,query,padding,guard,tau,bounds,keep)
    if (shape == 0) { RUN(0); }
    else if (shape == 1) { RUN(1); }
    else { RUN(2); }
#undef RUN
    return 0;
}
