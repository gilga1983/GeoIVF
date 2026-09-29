"""Query-local prepared ball planner; immutable packed index arrays are borrowed.

Projection/guard computation happens once per query, radial distance once per
visited IVF list. Contiguous page selection and extent coalescing share one C
call and bounded reusable scratch. Neither codes nor read schedules are changed.
"""
from __future__ import annotations
import ctypes as ct
from functools import lru_cache
from pathlib import Path
import numpy as np

MODES = {'none': 0, 'balls': 1, 'radial': 2, 'combined': 3}


@lru_cache(maxsize=4)
def library(path):
    lib = ct.CDLL(str(path))
    fn = lib.gp_plan
    fn.argtypes = ([ct.c_void_p]*4 + [ct.c_uint64] + [ct.c_uint]*4 +
                   [ct.c_void_p]*3 + [ct.c_double]*2 + [ct.c_uint64] +
                   [ct.c_uint]*2 + [ct.c_double] + [ct.c_uint]*2 +
                   [ct.c_void_p]*2 + [ct.c_uint, ct.c_void_p])
    fn.restype = ct.c_int
    return lib


def array(value, dtype, shape):
    a = np.asarray(value)
    if a.dtype != np.dtype(dtype) or a.shape != shape or not a.flags.c_contiguous:
        raise ValueError('invalid prepared index array dtype, shape, or contiguity')
    return a


class PreparedPlanner:
    """Independent per-query context, not an index-global mutable query cache."""
    def __init__(self, index, query, *, mode='combined', window_pages=64,
                 gap_pages=0, max_extent_pages=256, libpath=None):
        if mode not in MODES or window_pages < 0 or not 0 <= gap_pages < 2**32 or not 1 <= max_extent_pages < 2**32:
            raise ValueError('invalid prepared planning options')
        self.index = index  # retain all borrowed arrays
        self.q = np.array(query, dtype=np.float64, order='C', copy=True)
        m = index.meta
        if self.q.shape != (m['d'],) or not np.isfinite(self.q).all() or np.max(np.abs(self.q)) > 1e10:
            raise ValueError('invalid prepared query')
        pages, cap = m['n_pages'], m['capacity']
        valid = array(index.valid, 'uint16', (pages,))
        radial = array(index.radial, 'float32', (pages, 2))
        self.centers = array(index.centers, 'float32', (m['nlist'], m['d']))
        self.ranges = array(index.ranges, 'int64', (m['nlist'], 2))
        self.mode = MODES[mode]
        self.gap, self.max_extent = int(gap_pages), int(max_extent_pages)
        self.radial_cache = {}
        self.guard = 0.0
        if self.mode:
            if getattr(index, 'shape', None) != 'ball' or index.summary['bits'] not in (4, 6, 8):
                raise ValueError('prepared filtering requires independent 4/6/8-bit CellIndex balls')
            a = index.arrays
            dims, bits = m['dims'], index.summary['bits']
            stride = (cap*dims*bits+7)//8
            packed = array(a['packed'], 'uint8', (pages, stride))
            radii = array(a['radii'], 'float32', (pages, cap))
            origin = array(a['origin'], 'float64', (dims,))
            scale = array(a['scale'], 'float64', (dims,))
            matrix = array(a['matrix'], 'float64', (m['d'], m['d']))
            mean = array(a['mean'], 'float64', (m['d'],))
            # Same full transform/reduction as the existing comparator, once only.
            self.zq = np.ascontiguousarray((self.q-mean) @ matrix)
            self.guard = float(index.summary['build_guard'] +
                2048*np.finfo(float).eps*m['d']**2*(1+np.max(np.abs(self.q))))
            self.prefix = (packed.ctypes.data, radii.ctypes.data, valid.ctypes.data,
                radial.ctypes.data, pages, stride, cap, dims, bits,
                origin.ctypes.data, scale.ctypes.data, self.zq.ctypes.data)
            self._refs = (packed, radii, valid, radial, origin, scale, matrix, mean)
        else:
            self.prefix = (None, None, valid.ctypes.data, radial.ctypes.data,
                           pages, 0, cap, 1, 8, None, None, None)
            self._refs = (valid, radial)
        # Query scratch is grown as needed, never an expanded-center index.
        self.capacity = min(pages, max(1, window_pages, 16))
        self.keep = np.empty(self.capacity, dtype=np.uint8)
        self.extents = np.empty((self.capacity, 2), dtype=np.int64)
        self.selected = ct.c_uint64()
        self.lib = library(libpath or Path(__file__).resolve().parents[1]/'build/libgeoivf_prepared.so')
        self.fn = self.lib.gp_plan
        self.calls = 0

    def plan(self, first, stop, li, tau):
        if (not 0 <= li < len(self.ranges) or not self.ranges[li,0] <= first <= stop <= self.ranges[li,1]
                or np.isnan(tau) or tau < 0):
            raise ValueError('invalid page window, list, or threshold')
        count = int(stop-first)
        if count > self.capacity:
            self.capacity = count
            self.keep = np.empty(count, dtype=np.uint8)
            self.extents = np.empty((count, 2), dtype=np.int64)
        qr = 0.0
        if self.mode & 2:
            if li not in self.radial_cache:
                self.radial_cache[li] = float(np.linalg.norm(self.q-self.centers[li]))
            qr = self.radial_cache[li]
        rc = self.fn(*self.prefix, self.guard, qr, first, count, self.mode, tau,
            self.gap, self.max_extent, self.keep.ctypes.data, self.extents.ctypes.data,
            self.capacity, ct.byref(self.selected))
        if rc < 0:
            raise ValueError(f'native prepared planner rejected parameters ({rc})')
        self.calls += 1
        return [tuple(x) for x in self.extents[:rc].tolist()], self.selected.value

    @property
    def scratch_bytes(self):
        return self.keep.nbytes + self.extents.nbytes + self.q.nbytes + (self.zq.nbytes if self.mode else 0)
