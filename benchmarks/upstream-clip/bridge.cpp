// Calls released Faiss/CLIP/HIVF-CLIP search code. No search logic copied or changed.
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <omp.h>
#include <memory>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include "index_ivf_clip.h"
#include "index_hivf_clip.h"
struct External {
    int kind;
    std::unique_ptr<faiss::IndexIVFFlat> flat;
    std::unique_ptr<vlsm::IndexIVFCLIP> clip;
    std::unique_ptr<vlsm::IndexHIVFCLIP> hivf;
};
static void error(char* b,size_t n,const char* s){if(b&&n)std::snprintf(b,n,"%s",s);}
extern "C" void* gu_build(int kind,const float* x,long long n,unsigned d,
        const float* centers,unsigned nlist,char* err,size_t errn) {
    try {
        if(kind<0||kind>2||!x||n<1||!d||!centers||!nlist)throw std::invalid_argument("bad external build input");
        omp_set_num_threads(1);
        auto h=std::make_unique<External>();h->kind=kind;
        auto q=std::make_unique<faiss::IndexFlatL2>(d);q->add(nlist,centers);
        if(kind==0){
            h->flat=std::make_unique<faiss::IndexIVFFlat>(q.get(),d,nlist);
            h->flat->own_fields=true;q.release();h->flat->is_trained=true;
            h->flat->add(n,x);
        }else{
            auto clip=std::make_unique<vlsm::IndexIVFCLIP>(std::move(q),d,nlist);
            clip->is_trained=true;clip->add(n,x);
            // Match released example settings; no query-file vectors train lambda.
            clip->compute_lambda_by_par(int(n),x,kind==1?1000:3000,kind==1?1000:3000,
                                         kind==1?100:4,kind==1?.9999f:.999f);
            if(kind==1)h->clip=std::move(clip);
            else{
                h->hivf=std::make_unique<vlsm::IndexHIVFCLIP>(d,faiss::METRIC_L2);
                h->hivf->src_ivf=std::move(clip);
                h->hivf->buildIndex(nlist,2,int(n),x,30000,500,5,.99999f);
            }
        }
        return h.release();
    }catch(const std::exception& e){error(err,errn,e.what());return nullptr;}
}
extern "C" int gu_search(void* ptr,const float* q,unsigned nprobe,unsigned k,
         long long* ids,float* distances,char* err,size_t errn){
    try{
        auto* h=static_cast<External*>(ptr);if(!h||!q||!ids||!distances||!k||!nprobe)throw std::invalid_argument("bad query");
        omp_set_num_threads(1);
        // MyIVFStats embeds Faiss stats as its first field; upstream expects this layout.
        vlsm::MyIVFStats stats{};
        auto out=reinterpret_cast<faiss::idx_t*>(ids);
        if(h->kind==0){h->flat->nprobe=nprobe;h->flat->search(1,q,k,distances,out);}
        else if(h->kind==1){h->clip->nprobe=nprobe;h->clip->search_alabation(1,q,k,distances,out,nullptr,&stats.base,0);}
        else{h->hivf->nprobe=nprobe;h->hivf->search(1,q,k,distances,out,nullptr,&stats.base);}
        return 0;
    }catch(const std::exception& e){error(err,errn,e.what());return -1;}
}
extern "C" int gu_save(void* ptr,const char* path,char* err,size_t errn){
    try{
        auto* h=static_cast<External*>(ptr);if(!h||!path)throw std::invalid_argument("bad save");
        if(h->kind==1)h->clip->save_index(path);
        else if(h->kind==2)h->hivf->save_index(std::string(path)+".hier",path);
        return 0;
    }catch(const std::exception& e){error(err,errn,e.what());return -1;}
}
extern "C" void gu_free(void* ptr){delete static_cast<External*>(ptr);}
