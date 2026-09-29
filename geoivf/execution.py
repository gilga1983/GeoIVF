"""Opt-in, same-answer SIMD scanner and completed-stage rolling I/O reader."""
import ctypes as ct
from pathlib import Path
from .native_scan import NativeTopK
from .io import PooledNative

ROOT = Path(__file__).resolve().parents[1]

class SIMDTopK(NativeTopK):
    def __init__(self,index,query,k=10,libpath=None):
        path=libpath or ROOT/'build/libgeoivf_topk_simd.so'
        lib=ct.CDLL(str(path)); lib.gk_has_simd.restype=ct.c_int
        if not lib.gk_has_simd():
            raise RuntimeError('explicit SIMD scan requires AVX2; use scan=native otherwise')
        super().__init__(index,query,k,libpath=path)

class RollingPooled(PooledNative):
    def __init__(self,path,**kwargs):
        if kwargs.get('uring') is not True:
            raise ValueError('rolling schedule requires explicit uring=True')
        kwargs.setdefault('library',str(ROOT/'build/libgeoivf_rolling.so'))
        super().__init__(path,**kwargs)
        self.mode+='-rolling'
