"""Implicit nearest-rounding cells on unchanged independent PCA scalar codes.

Static range contract: every indexed projected coordinate is in its finite cell.
No arbitrary overflow clipping is allowed. Native full bounds drive online
search; square-root-free selection kernels are measured separately on a fixed
workload. Neither requires stored per-coordinate box endpoints.
"""
from __future__ import annotations
import ctypes as ct
import json
from pathlib import Path
import numpy as np
from .index import Index
from .dependent import unpack_codes, file_hash

EPS = np.finfo(np.float64).eps
SHAPES = {'ball': 0, 'box': 1, 'hybrid': 2}


def certify_cells(layout_dir, summary_dir, projected, out):
    """Audit each point's ORIGINAL projected coordinates, not just its code."""
    base = Index(layout_dir)
    info = json.loads((Path(summary_dir)/'dependent.json').read_text())
    if info['scheme'] != 'independent' or info['anchors'] != base.meta['capacity']:
        raise ValueError('cells require independently encoded singleton proxies')
    if info['layout_payload_sha256'] != file_hash(Path(layout_dir)/'vectors.pages'):
        raise ValueError('layout hash mismatch')
    with np.load(Path(summary_dir)/'dependent.npz', allow_pickle=False) as f:
        packed, origin, scale = f['packed'], f['origin'], f['scale']
    dims, cap = info['dims'], base.meta['capacity']
    if projected.shape != (base.meta['n'], base.meta['d']):
        raise ValueError('wrong build-time projected scratch')
    padding = info['build_guard']
    worst = -float('inf'); audited = 0
    for start in range(0, base.meta['n_pages'], 1024):
        stop = min(base.meta['n_pages'], start + 1024)
        ids = base.page_ids[start:stop]
        active = np.arange(cap)[None] < base.valid[start:stop, None]
        code = unpack_codes(packed[start:stop], info['bits'], (cap, dims))
        center = origin + code.astype(float) * scale
        points = projected[np.maximum(ids, 0), :dims]
        excess = np.abs(points-center) - (scale*.5 + padding)
        worst = max(worst, float(excess[active].max()))
        if np.any(excess[active] > 0):
            raise AssertionError('quantization overflow or cell containment failure')
        audited += int(active.sum())
    cert = dict(layout_payload_sha256=info['layout_payload_sha256'],
                code_file_sha256=file_hash(Path(summary_dir)/'dependent.npz'),
                basis_sha256=info['basis_sha256'], audited_points=audited,
                maximum_coordinate_excess=worst, cell_padding=padding,
                interval_rule='closed decoded_center +/- (step/2 + padding)',
                overflow_policy='reject construction, never silently clip outliers')
    Path(out).write_text(json.dumps(cert, indent=2)+'\n')
    return cert


class CellIndex(Index):
    def __init__(self, layout_dir, summary_dir, *, shape='box', library=None):
        super().__init__(layout_dir)
        if shape not in SHAPES: raise ValueError('unknown shape')
        directory = Path(summary_dir)
        info = json.loads((directory/'dependent.json').read_text())
        cert = json.loads((directory/'cells.json').read_text())
        if (info['scheme'] != 'independent' or info['anchors'] != self.meta['capacity']
                or self.meta['layout_payload_sha256'] != info['layout_payload_sha256']
                or cert['layout_payload_sha256'] != info['layout_payload_sha256']
                or cert['code_file_sha256'] != file_hash(directory/'dependent.npz')):
            raise ValueError('wrong layout, unaudited codes, or unsupported representation')
        for key in ('codes', 'radii', 'coordinates', 'origin', 'scale'): delattr(self, key)
        keys = ['packed', 'origin', 'scale', 'mean', 'matrix']
        if shape != 'box': keys.append('radii')
        with np.load(directory/'dependent.npz', allow_pickle=False) as f:
            self.arrays = {k: np.ascontiguousarray(f[k]) for k in keys}
        a = self.arrays
        self.shape, self.summary = shape, info
        self.padding = cert['cell_padding']
        self.geometry_bytes = a['packed'].nbytes + (a['radii'].nbytes if 'radii' in a else 0)
        structural = sum(getattr(self, k).nbytes for k in ('centers','page_ids','valid','radial','ranges'))
        shared = sum(a[k].nbytes for k in ('origin','scale','mean','matrix'))
        self.meta = dict(self.meta, dims=info['dims'], balls=self.meta['capacity'],
                         summary_bytes=self.geometry_bytes,
                         directory_array_bytes=structural+shared+self.geometry_bytes)
        path = library or Path(__file__).resolve().parents[1]/'build/libgeoivf_cells.so'
        self.lib = ct.CDLL(str(path))
        self.fn = self.lib.gc_evaluate
        self.fn.argtypes = ([ct.c_void_p]*4 + [ct.c_uint64] + [ct.c_uint]*4 +
                            [ct.c_void_p]*3 + [ct.c_double]*2 + [ct.c_int]*2 +
                            [ct.c_double, ct.c_void_p, ct.c_void_p])
        self.fn.restype = ct.c_int
        self._query = None; self._zq = None

    def prepare(self, q):
        if self._query is None or not np.array_equal(q, self._query):
            self._query = np.array(q, dtype=np.float64, copy=True)
            self._zq = np.ascontiguousarray((self._query-self.arrays['mean']) @ self.arrays['matrix'])
        return self.summary['build_guard'] + 2048*EPS*self.meta['d']**2*(1+np.max(np.abs(q)))

    def geometry(self, q, pages, *, strategy=0, tau=float('inf')):
        pages = np.ascontiguousarray(pages, dtype=np.int64)
        if pages.ndim != 1 or np.any(pages < 0) or np.any(pages >= self.meta['n_pages']):
            raise ValueError('invalid page IDs')
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.meta['d'],) or not np.isfinite(q).all():
            raise ValueError('invalid query')
        if strategy not in (0,1,2) or np.isnan(tau) or tau < 0:
            raise ValueError('invalid native strategy/threshold')
        guard = self.prepare(q); a = self.arrays
        out = np.empty(len(pages), dtype=np.float64 if strategy == 0 else np.uint8)
        if not len(pages): return out
        rc = self.fn(a['packed'].ctypes.data,
                     a['radii'].ctypes.data if 'radii' in a else None,
                     self.valid.ctypes.data, pages.ctypes.data, len(pages),
                     a['packed'].shape[1], self.meta['capacity'], self.meta['dims'], self.summary['bits'],
                     a['origin'].ctypes.data, a['scale'].ctypes.data, self._zq.ctypes.data,
                     self.padding, guard, SHAPES[self.shape], strategy, tau,
                     out.ctypes.data if strategy == 0 else None,
                     out.ctypes.data if strategy else None)
        if rc: raise ValueError('native cell kernel rejected parameters')
        return out

    def bounds(self, q, pages, li, mode):
        if mode not in ('none','balls','radial','combined'): raise ValueError('invalid mode')
        pages = np.asarray(pages, dtype=np.int64)
        if mode == 'none' or not len(pages): return np.zeros(len(pages))
        guard = self.prepare(q)
        result = self.geometry(q, pages) if mode in ('balls','combined') else np.zeros(len(pages))
        if mode in ('radial','combined'):
            r = np.linalg.norm(np.asarray(q, dtype=float)-self.centers[li])
            radial = np.maximum(self.radial[pages, 0]-r, r-self.radial[pages, 1])
            result = np.maximum(result, np.maximum(0, radial-guard))
        return result

    def select(self, q, pages, li, mode, tau, *, strategy=2):
        """Threshold-aware page mask. Preserves radial/geometry conjunction.

        Does not return numerical lower bounds. Audit code may separately request
        them; timed runs leave that diagnostic work disabled.
        """
        if mode not in ('none','balls','radial','combined') or strategy not in (1,2):
            raise ValueError('invalid selection mode/strategy')
        pp=np.ascontiguousarray(pages,dtype=np.int64)
        if pp.ndim != 1 or np.any(pp<0) or np.any(pp>=self.meta['n_pages']):
            raise ValueError('invalid page IDs')
        if np.isnan(tau) or tau<0: raise ValueError('invalid threshold')
        keep=np.ones(len(pp),dtype=bool)
        if not len(pp) or mode=='none': return keep
        q=np.asarray(q,dtype=np.float64)
        if q.shape != (self.meta['d'],) or not np.isfinite(q).all():
            raise ValueError('invalid query')
        if np.isinf(tau): return keep  # initial seed reads cannot be disproved
        guard=self.prepare(q)
        if mode in ('radial','combined'):
            if not 0 <= li < self.meta['nlist']: raise ValueError('invalid IVF list')
            r=np.linalg.norm(q-self.centers[li])
            radial=np.maximum(self.radial[pp,0]-r,r-self.radial[pp,1])
            keep=np.maximum(0,radial-guard)<=tau
        if mode in ('balls','combined') and np.any(keep):
            slots=np.flatnonzero(keep)
            keep[slots]=self.geometry(q,pp[slots],strategy=strategy,tau=tau).astype(bool)
        return keep
