// Rank compressed singleton proxies before any disk read. Optional page bounds
// certify completion in the same candidate set, never an approximate threshold.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

template<unsigned Bits> static unsigned code(const uint8_t* row, unsigned j) {
    if constexpr (Bits == 8) return row[j];
    unsigned bit=j*Bits, at=bit/8, shift=bit%8, v=row[at];
    if (shift+Bits>8) v|=unsigned(row[at+1])<<8;
    return (v>>shift)&((1u<<Bits)-1);
}
struct Candidate {
    double d2; int64_t id, slot;
    bool operator<(const Candidate& b) const {
        return d2 < b.d2 || (d2 == b.d2 && id < b.id);
    }
};
template<unsigned Bits> static int rank_impl(const uint8_t* packed,
    const float* radii, const uint16_t* valid, const int64_t* ids,
    const int64_t* pages, uint64_t n, uint64_t total_pages,
    unsigned stride, unsigned cap, unsigned dims, unsigned wanted,
    const double* origin, const double* scale, const double* query,
    double guard, bool certified, int64_t* slots, double* distances,
    double* lower_bounds, uint64_t* comparisons) {
    std::vector<Candidate> heap; heap.reserve(wanted); *comparisons=0;
    for (uint64_t r=0;r<n;++r) {
        if (pages[r]<0 || uint64_t(pages[r])>=total_pages) return -2;
        const uint64_t p=uint64_t(pages[r]);
        if (!valid[p] || valid[p]>cap) return -3;
        const auto* row=packed+p*stride;
        double lb=std::numeric_limits<double>::infinity();
        for (unsigned i=0;i<valid[p];++i) {
            double sum=0.;
            for (unsigned j=0;j<dims;++j) {
                const double c=origin[j]+scale[j]*code<Bits>(row,i*dims+j);
                const double delta=query[j]-c; sum+=delta*delta;
            }
            const int64_t slot=int64_t(p*cap+i), id=ids[slot];
            if (id<0 || !std::isfinite(sum)) return -4;
            Candidate c{sum,id,slot};
            if (heap.size()<wanted) {
                heap.push_back(c); std::push_heap(heap.begin(),heap.end());
            } else if (c<heap.front()) {
                std::pop_heap(heap.begin(),heap.end()); heap.back()=c;
                std::push_heap(heap.begin(),heap.end());
            }
            if (certified) {
                const double radius=radii[slot];
                if (!(radius>=0.) || !std::isfinite(radius)) return -5;
                lb=std::min(lb,std::max(0.,std::sqrt(sum)-radius));
            }
            ++*comparisons;
        }
        if (certified) lower_bounds[r]=std::max(0.,lb-guard);
    }
    std::sort(heap.begin(),heap.end());
    for (unsigned i=0;i<heap.size();++i) {slots[i]=heap[i].slot; distances[i]=heap[i].d2;}
    return int(heap.size());
}
extern "C" int gr_rank(const uint8_t* packed, const float* radii,
    const uint16_t* valid,const int64_t* ids,const int64_t* pages,
    uint64_t n,uint64_t total_pages,unsigned stride,unsigned cap,unsigned dims,
    unsigned bits,unsigned wanted,const double* origin,const double* scale,
    const double* query,double guard,int certified,int64_t* slots,
    double* distances,double* lower_bounds,uint64_t* comparisons) {
    if (!packed||!valid||!ids||!pages||!origin||!scale||!query||!slots||
        !distances||!comparisons||!cap||!dims||!wanted||wanted>1000000||
        (bits!=4&&bits!=6&&bits!=8)||stride!=(uint64_t(cap)*dims*bits+7)/8||
        (certified!=0&&certified!=1)||!std::isfinite(guard)||guard<0||
        (certified&&(!radii||!lower_bounds))) return -1;
    try {
#define RUN(B) rank_impl<B>(packed,radii,valid,ids,pages,n,total_pages,stride,cap,dims,wanted,origin,scale,query,guard,certified,slots,distances,lower_bounds,comparisons)
        if (bits==8) return RUN(8);
        if (bits==6) return RUN(6);
        return RUN(4);
#undef RUN
    } catch (...) { return -9; }
}
