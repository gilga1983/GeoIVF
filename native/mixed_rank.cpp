// Grouped fixed-length scalar decoding. Uniform and mixed controls share this
// scanner, LUT construction, FP64 accumulation order and optional certificate.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

template<unsigned Bits>
static void accumulate(const uint8_t* row, unsigned count, unsigned bit,
        const double* table, double& sum) {
    for (unsigned j=0;j<count;++j) {
        const unsigned pos=bit+j*Bits, byte=pos/8, shift=pos%8;
        unsigned value=row[byte];
        if constexpr (Bits != 8) {
            if (shift+Bits>8) value |= unsigned(row[byte+1])<<8;
        } else if (shift) value |= unsigned(row[byte+1])<<8;
        const unsigned code=(value>>shift)&((1u<<Bits)-1);
        sum+=table[j*(1u<<Bits)+code];
    }
}

extern "C" int gmix_rank(const uint8_t* packed,const float* radii,
    const uint16_t* valid,const int64_t* ids,const int64_t* pages,
    uint64_t count,uint64_t total_pages,unsigned cap,unsigned dims,unsigned code_bytes,
    unsigned stride,unsigned wanted,unsigned k,const uint32_t* segments,unsigned nsegments,
    const double* origin,const double* scale,const double* query,
    double guard,double norm_upper,double norm_lower,int certified,
    int64_t* slots,double* scores,double* lower_bounds,double* threshold_upper,uint64_t* compared) {
    if (!packed || !valid || !ids || !pages || !segments || !nsegments || !origin || !scale ||
        !query || !slots || !scores || !threshold_upper || !compared || !cap || !dims || dims>4096 ||
        !code_bytes || !wanted || wanted>1000000 || !k || k>wanted || nsegments>dims ||
        uint64_t(stride)!=uint64_t(cap)*code_bytes || !std::isfinite(guard) || guard<0 ||
        !std::isfinite(norm_upper) || norm_upper<=0 || !std::isfinite(norm_lower) || norm_lower<0 ||
        (certified && (!radii || !lower_bounds))) return -1;
    unsigned coordinates=0,total_bits=0,table_size=0;
    for (unsigned g=0;g<nsegments;++g) {
        const auto* s=segments+5*g; const unsigned b=s[2];
        if ((b!=2&&b!=4&&b!=6&&b!=8) || !s[1] || s[0]!=coordinates ||
            s[1]>dims-coordinates || s[3]!=total_bits || s[4]!=table_size) return -2;
        coordinates+=s[1];total_bits+=s[1]*b;table_size+=s[1]*(1u<<b);
    }
    if (coordinates!=dims || total_bits!=code_bytes*8) return -2;
    try {
        std::vector<double> table(table_size);
        for (unsigned g=0;g<nsegments;++g) {
            const auto* s=segments+5*g;const unsigned levels=1u<<s[2];
            for (unsigned j=0;j<s[1];++j) {
                const unsigned d=s[0]+j;
                if (!std::isfinite(query[d]) || !std::isfinite(origin[d]) ||
                    !std::isfinite(scale[d]) || scale[d]<=0) return -3;
                for (unsigned code=0;code<levels;++code) {
                    const double delta=query[d]-(origin[d]+scale[d]*code);
                    const double v=delta*delta;if (!std::isfinite(v)) return -3;
                    table[s[4]+j*levels+code]=v;
                }
            }
        }
        using Entry=std::tuple<double,int64_t,int64_t>;
        std::priority_queue<Entry> heap;std::priority_queue<double> upper;
        const double inf=std::numeric_limits<double>::infinity();*compared=0;*threshold_upper=inf;
        for (uint64_t pidx=0;pidx<count;++pidx) {
            if (pages[pidx]<0 || uint64_t(pages[pidx])>=total_pages) return -4;
            const uint64_t p=uint64_t(pages[pidx]);if (valid[p]>cap) return -4;
            double page_lb=inf;
            for (unsigned i=0;i<valid[p];++i) {
                const uint8_t* row=packed+p*stride+uint64_t(i)*code_bytes;double ds=0.;
                for (unsigned g=0;g<nsegments;++g) {
                    const auto* s=segments+5*g;const double* t=table.data()+s[4];
                    switch (s[2]) {
                        case 2:accumulate<2>(row,s[1],s[3],t,ds);break;
                        case 4:accumulate<4>(row,s[1],s[3],t,ds);break;
                        case 6:accumulate<6>(row,s[1],s[3],t,ds);break;
                        case 8:accumulate<8>(row,s[1],s[3],t,ds);break;
                    }
                }
                if (!std::isfinite(ds)||ds<0||ids[p*cap+i]<0) return -5;
                Entry e{ds,ids[p*cap+i],int64_t(p*cap+i)};
                if (heap.size()<wanted) heap.push(e);else if (e<heap.top()){heap.pop();heap.push(e);}
                ++*compared;
                if (certified) {
                    const double r=double(radii[p*cap+i]);if (!(r>=0)||!std::isfinite(r)) return -6;
                    const double d=std::sqrt(ds);page_lb=std::min(page_lb,std::max(0.,(d-r-guard)/norm_upper));
                    // Kept identical to the preceding uniform/PQ comparator.
                    if (norm_lower>0) {
                        const double ub=std::nextafter((d+r+guard)/norm_lower,inf);
                        if (upper.size()<k)upper.push(ub);else if (ub<upper.top()){upper.pop();upper.push(ub);}
                    }
                }
            }
            if (certified)lower_bounds[pidx]=std::nextafter(page_lb,0.);
        }
        const int n=int(heap.size());
        for (int j=n-1;j>=0;--j){scores[j]=std::get<0>(heap.top());slots[j]=std::get<2>(heap.top());heap.pop();}
        if (upper.size()==k)*threshold_upper=upper.top();
        return n;
    } catch (...) {return -7;}
}
