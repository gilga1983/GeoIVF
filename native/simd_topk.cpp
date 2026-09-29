// Same scalar per-vector addition order, SIMD across four different vectors.
// Existing heap, ID tie-break, error messages and ABI remain available.
#define gk_consume gk_consume_scalar
#include "topk.cpp"
#undef gk_consume
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define GEO_X86 1
#else
#define GEO_X86 0
#endif
extern "C" int gk_has_simd() {
#if GEO_X86
    return __builtin_cpu_supports("avx2") != 0;
#else
    return 0;
#endif
}
#if GEO_X86
__attribute__((target("avx2")))
static void distance4(const unsigned char* point, unsigned dims,
                       const double* query, double* output) {
    __m256d sum = _mm256_setzero_pd();
    unsigned d = 0;
    for (; d + 4 <= dims; d += 4) {
        __m128 v0, v1, v2, v3;
        // memcpy supports unaligned byte buffers without type-punning UB.
        std::memcpy(&v0, point + 4*d, 16);
        std::memcpy(&v1, point + 4*(dims+d), 16);
        std::memcpy(&v2, point + 4*(2*dims+d), 16);
        std::memcpy(&v3, point + 4*(3*dims+d), 16);
        _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
        const __m128 values[4] = {v0, v1, v2, v3};
        for (unsigned j = 0; j < 4; ++j) {
            const __m256d delta = _mm256_sub_pd(_mm256_cvtps_pd(values[j]),
                                               _mm256_set1_pd(query[d+j]));
            sum = _mm256_add_pd(sum, _mm256_mul_pd(delta, delta));
        }
    }
    for (; d < dims; ++d) {
        float v[4];
        for (unsigned i=0; i<4; ++i) std::memcpy(v+i,point+4*(i*dims+d),4);
        const __m256d delta = _mm256_sub_pd(_mm256_set_pd(v[3],v[2],v[1],v[0]),
                                           _mm256_set1_pd(query[d]));
        sum = _mm256_add_pd(sum, _mm256_mul_pd(delta, delta));
    }
    _mm256_storeu_pd(output, sum);
}
#endif
static void offer(TopK* s, double dd, int64_t id) {
    if (!std::isfinite(dd) || id < 0)
        throw std::invalid_argument("nonfinite payload distance or invalid ID");
    Neighbor candidate{dd,id};
    if (s->heap.size() < s->k) {
        s->heap.push_back(candidate); std::push_heap(s->heap.begin(),s->heap.end());
    } else if (candidate < s->heap.front()) {
        std::pop_heap(s->heap.begin(),s->heap.end()); s->heap.back()=candidate;
        std::push_heap(s->heap.begin(),s->heap.end());
    }
}
extern "C" int gk_consume(void* handle, uint32_t n, const void* const* buffers,
        const uint64_t* lengths, const int64_t* starts, const uint32_t* counts,
        const int64_t* ids, const uint16_t* valid, uint64_t total_pages,
        uint64_t* evaluated, uint32_t* filled, double* threshold2, char* err, size_t size) {
    auto* s = static_cast<TopK*>(handle);
    try {
        if (!gk_has_simd()) throw std::runtime_error("AVX2 required for explicit SIMD scanner");
        if (!s || s->failed || !evaluated || !filled || !threshold2 || !ids || !valid ||
            (n && (!buffers || !lengths || !starts || !counts)))
            throw std::invalid_argument("invalid scan arguments or failed accumulator");
        uint64_t done=0;
        for (uint32_t b=0; b<n; ++b) {
            if (!buffers[b] || starts[b]<0 || !counts[b] || uint64_t(starts[b])>=total_pages ||
                counts[b]>total_pages-uint64_t(starts[b]) || lengths[b]!=uint64_t(counts[b])*s->page_bytes)
                throw std::invalid_argument("invalid extent or buffer length");
            const auto* raw=static_cast<const unsigned char*>(buffers[b]);
            for (uint32_t j=0; j<counts[b]; ++j) {
                const uint64_t p=uint64_t(starts[b])+j;
                if (valid[p]>s->capacity) throw std::invalid_argument("invalid page occupancy");
                const auto* page=raw+uint64_t(j)*s->page_bytes;
                unsigned i=0;
#if GEO_X86
                for (; i+4<=valid[p]; i+=4) {
                    double dd[4]; distance4(page+uint64_t(i)*s->dims*4,s->dims,s->query.data(),dd);
                    // Same heap insertion order as the scalar scanner.
                    for (unsigned lane=0; lane<4; ++lane) offer(s,dd[lane],ids[p*s->capacity+i+lane]);
                }
#endif
                for (; i<valid[p]; ++i) {
                    const auto* point=page+uint64_t(i)*s->dims*4;
                    double dd=0;
                    for (unsigned d=0; d<s->dims; ++d) {
                        float v; std::memcpy(&v,point+4*d,4);
                        const double delta=double(v)-s->query[d]; dd+=delta*delta;
                    }
                    offer(s,dd,ids[p*s->capacity+i]);
                }
                done+=valid[p];
            }
        }
        *evaluated=done; *filled=static_cast<uint32_t>(s->heap.size());
        *threshold2=s->heap.size()==s->k?s->heap.front().distance:std::numeric_limits<double>::infinity();
        return 0;
    } catch (const std::exception& e) {
        if (s) s->failed=true;
        message(err,size,e.what()); return -1;
    }
}
