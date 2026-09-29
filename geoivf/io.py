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
