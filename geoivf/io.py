"""Interchangeable byte readers. Trace capture is an observer, not a fake reader."""
from __future__ import annotations
import ctypes as C
import mmap
import os
from pathlib import Path


class MemoryReplay:
    """Explicit test-only payload oracle. Its timings are NOT storage timings."""
    def __init__(self, path: Path):
        self.data = path.read_bytes()
        self.mode = 'memory-replay-not-a-storage-benchmark'

    def read(self, requests: list[tuple[int, int]]) -> list[bytes]:
        out = [self.data[o:o+n] for o, n in requests]
        if any(len(b) != n for b, (_, n) in zip(out, requests)):
            raise EOFError('short replay read')
        return out

    def close(self):
        self.data = b''


class Pread:
    def __init__(self, path: Path):
        self.fd = os.open(path, os.O_RDONLY)
        self.mode = 'buffered-pread'

    def read(self, requests: list[tuple[int, int]]) -> list[bytes]:
        out = []
        for offset, size in requests:
            parts, left = [], size
            while left:
                b = os.pread(self.fd, left, offset+size-left)
                if not b:
                    raise EOFError('short payload read')
                parts.append(b)
                left -= len(b)
            out.append(b''.join(parts))
        return out

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class Native:
    def __init__(self, path: Path, *, direct: bool = False, uring: bool = False,
                 depth: int = 16, library: str | None = None):
        libpath = library or os.environ.get('GEOIVF_IO_LIBRARY') or str(
            Path(__file__).resolve().parent.parent/'build/libgeoivf_io.so')
        self.lib = C.CDLL(libpath)
        self.lib.gio_open.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_uint,
                                      C.c_char_p, C.c_size_t]
        self.lib.gio_open.restype = C.c_void_p
        self.lib.gio_read.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(C.c_uint64),
                                      C.POINTER(C.c_uint32), C.POINTER(C.c_void_p),
                                      C.c_char_p, C.c_size_t]
        self.lib.gio_read.restype = C.c_int
        self.lib.gio_close.argtypes = [C.c_void_p]
        self.lib.gio_close.restype = None
        self.err = C.create_string_buffer(1024)
        self.handle = self.lib.gio_open(os.fsencode(path), direct, uring, depth,
                                        self.err, len(self.err))
        if not self.handle:
            raise OSError(self.err.value.decode())
        self.mode = ('uring' if uring else 'native-pread')+('-direct' if direct else '-buffered')

    def read(self, requests: list[tuple[int, int]]) -> list[bytes]:
        if not requests:
            return []
        if any(o < 0 or n < 1 or n > 1<<30 for o, n in requests):
            raise ValueError('invalid native request')
        buffers = []
        try:
            for _, n in requests:
                buffers.append(mmap.mmap(-1, n))
            addresses = [C.addressof(C.c_char.from_buffer(b)) for b in buffers]
            n = len(requests)
            offsets = (C.c_uint64*n)(*(r[0] for r in requests))
            lengths = (C.c_uint32*n)(*(r[1] for r in requests))
            pointers = (C.c_void_p*n)(*addresses)
            rc = self.lib.gio_read(self.handle, n, offsets, lengths, pointers,
                                  self.err, len(self.err))
            if rc:
                raise OSError(self.err.value.decode())
            return [b[:] for b in buffers]
        finally:
            for b in buffers:
                b.close()

    def close(self):
        if self.handle:
            self.lib.gio_close(self.handle)
            self.handle = None


class PooledNative(Native):
    """Reusable aligned buffers with explicit one-stage borrowing.

    Call release(buffers) after consuming a stage. A new read is forbidden while
    views are outstanding. No output byte copy is made, and allocated buffer
    capacity is bounded and reported. Underlying I/O completion rules are unchanged.
    """
    def __init__(self, path, *, max_pool_bytes=64 << 20, **kwargs):
        if max_pool_bytes < 4096: raise ValueError('pool budget is too small')
        super().__init__(path, **kwargs)
        self.pool = []
        self.active = None
        self.max_pool_bytes=max_pool_bytes
        self.reserved_bytes=0
        self.allocations=0
        self.mode += '-pooled-borrowed'

    def read(self, requests):
        if not self.handle: raise ValueError('closed reader')
        if self.active is not None: raise RuntimeError('release previous stage before reading')
        if not requests: return []
        if any(o<0 or n<1 or n>1<<30 for o,n in requests): raise ValueError('invalid native request')
        sizes=[1 << max(12,(n-1).bit_length()) for _,n in requests]
        if sum(sizes)>self.max_pool_bytes: raise MemoryError('stage exceeds bounded I/O pool')
        # Drop oversized idle reservations before growing the pool when needed.
        projected=sum(max(sizes[i],len(self.pool[i]) if i<len(self.pool) else 0)
                      for i in range(len(sizes)))
        projected+=sum(len(b) for b in self.pool[len(sizes):])
        if projected>self.max_pool_bytes:
            for b in self.pool: b.close()
            self.pool=[]
        for i,size in enumerate(sizes):
            if i==len(self.pool):
                self.pool.append(mmap.mmap(-1,size));self.allocations+=1
            elif len(self.pool[i])<size:
                self.pool[i].close();self.pool[i]=mmap.mmap(-1,size);self.allocations+=1
        self.reserved_bytes=sum(len(b) for b in self.pool)
        n=len(requests)
        offsets=(C.c_uint64*n)(*(o for o,_ in requests))
        lengths=(C.c_uint32*n)(*(sz for _,sz in requests))
        pointers=(C.c_void_p*n)(*(C.addressof(C.c_char.from_buffer(b)) for b in self.pool[:n]))
        rc=self.lib.gio_read(self.handle,n,offsets,lengths,pointers,self.err,len(self.err))
        if rc: raise OSError(self.err.value.decode())
        self.active=[memoryview(b)[:size] for b,(_,size) in zip(self.pool,requests)]
        return self.active

    def release(self, buffers):
        if self.active is None:
            if buffers: raise ValueError('no outstanding read stage')
            return
        if buffers is not self.active: raise ValueError('wrong borrowed buffer batch')
        for view in self.active: view.release()
        self.active=None

    def close(self):
        if self.active is not None: self.release(self.active)
        for b in self.pool: b.close()
        self.pool=[];self.reserved_bytes=0
        super().close()
