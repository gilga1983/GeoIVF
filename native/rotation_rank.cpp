// Common FP64 lookup-table scanner for scalar and product-quantized codes.
// Certification uses outward radii and separately bounded transform norms.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

extern "C" int grot_rank(const uint8_t* packed, const float* radii,
    const uint16_t* valid, const int64_t* ids, const int64_t* pages,
    uint64_t count, uint64_t total_pages, unsigned cap, unsigned groups,
    unsigned bits, unsigned stride, unsigned wanted, unsigned k,
    const double* lut, double guard, double norm_upper, double norm_lower,
    int certified, int64_t* slots, double* scores, double* lower_bounds,
    double* threshold_upper, uint64_t* compared) {
    if (!packed || !valid || !ids || !pages || !lut || !slots || !scores ||
        !threshold_upper || !compared || !cap || !groups || !wanted ||
        !k || k>wanted || (bits!=4 && bits!=8) ||
        stride!=(cap*groups*bits+7)/8 || !std::isfinite(guard) || guard<0 ||
        !std::isfinite(norm_upper) || norm_upper<=0 || norm_lower<0 ||
        !std::isfinite(norm_lower) || (certified && (!radii || !lower_bounds))) return -1;
    using Entry=std::tuple<double,int64_t,int64_t>;
    std::priority_queue<Entry> heap;
    std::priority_queue<double> upper;
    const double inf=std::numeric_limits<double>::infinity();
    const unsigned levels=1u<<bits;
    *compared=0; *threshold_upper=inf;
    for (uint64_t row=0;row<count;++row) {
        if (pages[row]<0 || uint64_t(pages[row])>=total_pages) return -2;
        const uint64_t p=uint64_t(pages[row]);
        if (valid[p]>cap) return -2;
        double page_lb=inf;
        for (unsigned i=0;i<valid[p];++i) {
            double ds=0.;
            for (unsigned j=0;j<groups;++j) {
                const unsigned pos=i*groups+j;
                const uint8_t byte=packed[p*stride+(bits==8?pos:pos/2)];
                const unsigned code=bits==8?byte:((byte>>(4*(pos%2)))&15);
                ds+=lut[j*levels+code];
            }
            if (!std::isfinite(ds) || ds<0 || ids[p*cap+i]<0) return -3;
            Entry e{ds,ids[p*cap+i],int64_t(p*cap+i)};
            if (heap.size()<wanted) heap.push(e);
            else if (e<heap.top()) {heap.pop();heap.push(e);}
            ++*compared;
            if (certified) {
                const double r=double(radii[p*cap+i]);
                if (!(r>=0) || !std::isfinite(r)) return -4;
                const double d=std::sqrt(ds);
                page_lb=std::min(page_lb,std::max(0.,(d-r-guard)/norm_upper));
                if (norm_lower>0) {
                    const double ub=std::nextafter((d+r+guard)/norm_lower,inf);
                    if (upper.size()<k) upper.push(ub);
                    else if (ub<upper.top()) {upper.pop();upper.push(ub);}
                }
            }
        }
        if (certified) lower_bounds[row]=std::nextafter(page_lb,0.);
    }
    const int n=int(heap.size());
    for (int j=n-1;j>=0;--j) {
        scores[j]=std::get<0>(heap.top());slots[j]=std::get<2>(heap.top());heap.pop();
    }
    if (upper.size()==k) *threshold_upper=upper.top();
    return n;
}
