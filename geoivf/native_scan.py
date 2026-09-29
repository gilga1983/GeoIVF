"""Native exact scan over page buffers. Ownership stays with the byte reader."""
from __future__ import annotations
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np


@lru_cache(maxsize=4)
def library(path):
    lib = ct.CDLL(str(path))
    lib.gk_create.argtypes = [ct.c_void_p] + [ct.c_uint]*4 + [ct.c_void_p,ct.c_size_t]
    lib.gk_create.restype = ct.c_void_p
    lib.gk_consume.argtypes = [ct.c_void_p,ct.c_uint32] + [ct.c_void_p]*6 + [ct.c_uint64] + [ct.c_void_p]*4 + [ct.c_size_t]
    lib.gk_consume.restype = ct.c_int
    lib.gk_finish.argtypes = [ct.c_void_p]*3 + [ct.c_uint,ct.c_void_p,ct.c_size_t]
    lib.gk_finish.restype = ct.c_int
    lib.gk_close.argtypes = [ct.c_void_p]
    lib.gk_close.restype = None
    return lib


class NativeTopK:
    def __init__(self, index, query, k=10, libpath=None):
        if sys.byteorder != 'little':
            raise ValueError('native payload reader currently requires little endian')
        self.lib = library(libpath or Path(__file__).resolve().parents[1]/'build/libgeoivf_topk.so')
        self.err = ct.create_string_buffer(1024)
        self.ids = np.asarray(index.page_ids)
        self.valid = np.asarray(index.valid)
        m=index.meta
        if (self.ids.dtype != np.int64 or self.valid.dtype != np.uint16 or
            not self.ids.flags.c_contiguous or not self.valid.flags.c_contiguous or
            self.ids.shape != (m['n_pages'],m['capacity']) or self.valid.shape != (m['n_pages'],)):
            raise ValueError('invalid native directory arrays')
        q=np.ascontiguousarray(query,dtype=np.float64)
        if q.shape != (m['d'],) or not 1 <= k <= m['n']:
            raise ValueError('invalid native query/k')
        self.page_size=m['page_size'];self.n_pages=m['n_pages'];self.k=k
        self.handle=self.lib.gk_create(q.ctypes.data,m['d'],m['capacity'],m['page_size'],k,self.err,len(self.err))
        if not self.handle: raise ValueError(self.err.value.decode())
        self.count=0;self.tau2=float('inf')

    def consume(self, buffers, extents):
        if not self.handle: raise ValueError('closed native accumulator')
        if len(buffers) != len(extents): raise ValueError('wrong number of buffers')
        n=len(buffers)
        if n>2**32-1: raise ValueError('too many extents')
        if any(p<0 or c<1 or p+c>self.n_pages or c>2**32-1 for p,c in extents):
            raise ValueError('invalid native extents')
        # These are zero-copy views; keep them alive until the C call returns.
        arrays=[np.frombuffer(b,dtype=np.uint8) for b in buffers]
        starts=np.ascontiguousarray([p for p,_ in extents],dtype=np.int64)
        counts=np.ascontiguousarray([c for _,c in extents],dtype=np.uint32)
        lengths=np.ascontiguousarray([a.nbytes for a in arrays],dtype=np.uint64)
        if any(len(a)!=c*self.page_size for a,(_,c) in zip(arrays,extents)):
            raise EOFError('short native scan buffer')
        pointers=(ct.c_void_p*n)(*(a.ctypes.data for a in arrays))
        evaluated=ct.c_uint64();filled=ct.c_uint32();tau2=ct.c_double()
        rc=self.lib.gk_consume(self.handle,n,pointers,lengths.ctypes.data,starts.ctypes.data,
                counts.ctypes.data,self.ids.ctypes.data,self.valid.ctypes.data,self.n_pages,
                ct.byref(evaluated),ct.byref(filled),ct.byref(tau2),self.err,len(self.err))
        if rc: raise ValueError(self.err.value.decode())
        self.count=filled.value;self.tau2=tau2.value
        return evaluated.value

    def finish(self):
        if not self.handle: raise ValueError('closed native accumulator')
        ids=np.empty(self.k,dtype=np.int64);dd=np.empty(self.k,dtype=np.float64)
        n=self.lib.gk_finish(self.handle,ids.ctypes.data,dd.ctypes.data,self.k,self.err,len(self.err))
        if n<0: raise ValueError(self.err.value.decode())
        return ids[:n],dd[:n]

    def close(self):
        if self.handle:
            self.lib.gk_close(self.handle);self.handle=None
