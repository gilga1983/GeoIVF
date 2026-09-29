// Small adapter to POSIX pread and upstream liburing, not an SSD simulator.
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#ifdef GEOIVF_URING
#include <liburing.h>
#endif
struct Reader {
    int fd = -1;
    bool direct = false, ring_ready = false, poisoned = false;
    unsigned depth = 1;
#ifdef GEOIVF_URING
    io_uring ring{};
#endif
    void stop_ring() {
#ifdef GEOIVF_URING
        if (ring_ready) io_uring_queue_exit(&ring);
#endif
        ring_ready = false;
    }
    ~Reader() { stop_ring(); if (fd >= 0) ::close(fd); }
};
static void error(char* out, size_t cap, const std::string& message) {
    if (out && cap) std::snprintf(out, cap, "%s", message.c_str());
}
extern "C" int gio_has_uring() {
#ifdef GEOIVF_URING
    return 1;
#else
    return 0;
#endif
}
extern "C" void* gio_open(const char* path, int direct, int uring,
                          unsigned depth, char* err, size_t cap) {
    Reader* r = new Reader();
    try {
        if (!depth || depth > 1024) throw std::runtime_error("invalid queue depth");
        r->depth = depth; r->direct = direct != 0;
        r->fd = ::open(path, O_RDONLY | O_CLOEXEC | (direct ? O_DIRECT : 0));
        if (r->fd < 0) throw std::runtime_error(std::string("open: ") + std::strerror(errno));
        if (uring) {
#ifdef GEOIVF_URING
            int rc = io_uring_queue_init(depth, &r->ring, 0);
            if (rc < 0) throw std::runtime_error(std::string("io_uring init: ") + std::strerror(-rc));
            r->ring_ready = true;
#else
            throw std::runtime_error("liburing support not compiled; run make URING=1");
#endif
        }
        return r;
    } catch (const std::exception& e) { error(err, cap, e.what()); delete r; return nullptr; }
}
extern "C" int gio_read(void* handle, uint32_t n, const uint64_t* offsets,
                         const uint32_t* lengths, void** buffers, char* err, size_t cap) {
    auto* r = static_cast<Reader*>(handle);
    try {
        if (!r || r->poisoned) throw std::runtime_error("invalid or failed reader");
        for (uint32_t i = 0; i < n; ++i) {
            if (!lengths[i] || lengths[i] > (1u<<30)) throw std::runtime_error("invalid read length");
            if (r->direct && ((offsets[i]%4096) || (lengths[i]%4096)
                             || (reinterpret_cast<uintptr_t>(buffers[i])%4096)))
                throw std::runtime_error("unaligned direct I/O request");
        }
#ifdef GEOIVF_URING
        if (r->ring_ready) {
            for (uint32_t first = 0; first < n; first += r->depth) {
                uint32_t count = std::min<uint32_t>(r->depth, n-first);
                for (uint32_t j = 0; j < count; ++j) {
                    uint32_t i = first+j;
                    auto* sqe = io_uring_get_sqe(&r->ring);
                    if (!sqe) throw std::runtime_error("submission queue unexpectedly full");
                    io_uring_prep_read(sqe, r->fd, buffers[i], lengths[i], offsets[i]);
                    io_uring_sqe_set_data64(sqe, i);
                }
                uint32_t sent = 0;
                while (sent < count) {
                    int rc = io_uring_submit(&r->ring);
                    if (rc == -EINTR) continue;
                    if (rc <= 0) throw std::runtime_error("io_uring submit failed");
                    sent += static_cast<uint32_t>(rc);
                }
                std::string failure;
                for (uint32_t j = 0; j < count; ++j) {
                    io_uring_cqe* cqe = nullptr;
                    int rc;
                    do { rc = io_uring_wait_cqe(&r->ring, &cqe); } while (rc == -EINTR);
                    if (rc < 0) throw std::runtime_error("io_uring wait failed");
                    auto i = io_uring_cqe_get_data64(cqe);
                    if (cqe->res < 0) failure = std::strerror(-cqe->res);
                    else if (i >= n || static_cast<uint32_t>(cqe->res) != lengths[i])
                        failure = "short io_uring read";
                    io_uring_cqe_seen(&r->ring, cqe);
                }
                if (!failure.empty()) throw std::runtime_error(failure);
            }
            return 0;
        }
#endif
        for (uint32_t i = 0; i < n; ++i) {
            size_t done = 0;
            while (done < lengths[i]) {
                ssize_t got = ::pread(r->fd, static_cast<char*>(buffers[i])+done,
                                      lengths[i]-done, offsets[i]+done);
                if (got < 0 && errno == EINTR) continue;
                if (got < 0) throw std::runtime_error(std::strerror(errno));
                if (!got) throw std::runtime_error("unexpected EOF");
                done += static_cast<size_t>(got);
                if (r->direct && done < lengths[i] && done%4096)
                    throw std::runtime_error("unaligned short direct read");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        if (r) { r->poisoned = true; r->stop_ring(); }
        error(err, cap, e.what()); return -1;
    }
}
extern "C" void gio_close(void* handle) { delete static_cast<Reader*>(handle); }
