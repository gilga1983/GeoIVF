// Same completed-stage interface, rolling replenishment instead of QD barriers.
#define gio_read gio_read_batch
#include "reader.cpp"
#undef gio_read
#include <vector>
extern "C" int gio_read(void* handle, uint32_t n, const uint64_t* offsets,
        const uint32_t* lengths, void** buffers, char* err, size_t cap) {
    auto* r=static_cast<Reader*>(handle);
    try {
        if (!r || r->poisoned || !r->ring_ready)
            throw std::runtime_error("rolling reader requires a healthy io_uring handle");
        if (n && (!offsets || !lengths || !buffers)) throw std::runtime_error("null request arrays");
        for (uint32_t i=0; i<n; ++i) {
            if (!buffers[i] || !lengths[i] || lengths[i]>(1u<<30))
                throw std::runtime_error("invalid read length or buffer");
            if (r->direct && ((offsets[i]%4096)||(lengths[i]%4096)||
                              (reinterpret_cast<uintptr_t>(buffers[i])%4096)))
                throw std::runtime_error("unaligned direct I/O request");
        }
#ifdef GEOIVF_URING
        std::vector<uint8_t> seen(n,0);
        uint32_t next=0, inflight=0, completed=0;
        std::string failure;
        while (completed<n) {
            const uint32_t count=failure.empty()?std::min<uint32_t>(r->depth-inflight,n-next):0;
            for (uint32_t j=0; j<count; ++j) {
                const uint32_t i=next+j;
                auto* sqe=io_uring_get_sqe(&r->ring);
                if (!sqe) throw std::runtime_error("rolling submission queue full");
                io_uring_prep_read(sqe,r->fd,buffers[i],lengths[i],offsets[i]);
                io_uring_sqe_set_data64(sqe,i);
            }
            uint32_t sent=0;
            while (sent<count) {
                const int rc=io_uring_submit(&r->ring);
                if (rc==-EINTR) continue;
                if (rc<=0 || uint32_t(rc)>count-sent) throw std::runtime_error("rolling submit failed");
                sent+=uint32_t(rc);
            }
            next+=count; inflight+=count;
            if (!inflight) {
                if (!failure.empty()) throw std::runtime_error(failure);
                break;
            }
            io_uring_cqe* cqe=nullptr;
            int rc;
            do {rc=io_uring_wait_cqe(&r->ring,&cqe);} while (rc==-EINTR);
            if (rc<0) throw std::runtime_error("rolling completion wait failed");
            // Drain ready completions, then refill freed slots in one submission.
            do {
                const uint64_t i=io_uring_cqe_get_data64(cqe);
                const int result=cqe->res;
                io_uring_cqe_seen(&r->ring,cqe);
                if (i>=next || i>=n || seen[i]) throw std::runtime_error("invalid or duplicate completion ID");
                seen[i]=1; --inflight; ++completed;
                if (result<0) failure=std::strerror(-result);
                else if (uint32_t(result)!=lengths[i]) failure="short rolling read";
                if (!inflight) break;
                rc=io_uring_peek_cqe(&r->ring,&cqe);
                if (rc && rc!=-EAGAIN && rc!=-EINTR) throw std::runtime_error("rolling peek failed");
            } while (rc==0);
            // After an I/O error, stop submitting and drain outstanding requests.
            if (!failure.empty() && !inflight) throw std::runtime_error(failure);
        }
        if (completed!=n) throw std::runtime_error("incomplete rolling read stage");
        return 0;
#else
        throw std::runtime_error("rolling io_uring support not compiled");
#endif
    } catch (const std::exception& e) {
        // Closing the ring prevents in-flight I/O from outliving borrowed buffers.
        if (r) {r->poisoned=true; r->stop_ring();}
        error(err,cap,e.what()); return -1;
    }
}
