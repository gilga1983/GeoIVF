"""Conservative projected summaries of frozen pages; no payload changes.

PCA is not whitened. A complete, norm-bounded transform is partitioned into
head coordinates and an optional tail-norm interval relative to the IVF center.
Floating-point guards are engineering safeguards, not formal interval proofs.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .index import Index, checked

EPS = np.finfo(np.float64).eps


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def contraction(matrix):
    """Scale using a Gram-matrix row-sum bound plus roundoff allowance.

    ||T||_2^2 <= ||T.T T||_inf. This does not assume eigenvectors are exactly
    orthogonal, and unlike a power iteration cannot overlook a large singular
    value. The allowance covers FP64 dot products/row sums in this dimension.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError('invalid transform')
    gram = a.T @ a
    absolute = np.abs(a).T @ np.abs(a)
    upper = np.max(np.abs(gram).sum(axis=1))
    upper += 64 * EPS * max(a.shape) * np.max(absolute.sum(axis=1))
    if upper <= 0 or not np.isfinite(upper):
        raise ValueError('degenerate transform')
    divisor = np.nextafter(np.sqrt(upper) * (1 + 256*max(a.shape)*EPS), np.inf)
    return a / divisor, float(divisor)


@dataclass
class Basis:
    mean: np.ndarray
    matrix: np.ndarray
    kind: str
    normalization: float = 1.0
    fit_count: int = 0
    explained_fraction: np.ndarray | None = None

    @classmethod
    def fit(cls, training, kind='pca'):
        a = checked(training).astype(np.float64)
        mean = a.mean(axis=0)
        centered = a - mean
        covariance = centered.T @ centered / len(a)
        if kind == 'pca':
            values, matrix = np.linalg.eigh(covariance)
            order = np.argsort(-values, kind='stable')
            values, matrix = np.maximum(values[order], 0), matrix[:, order]
            # Reproducible eigenvector signs (degenerate subspaces can still vary).
            pivots = np.argmax(np.abs(matrix), axis=0)
            matrix *= np.where(matrix[pivots, np.arange(len(pivots))] < 0, -1., 1.)
            matrix, normalizer = contraction(matrix)
        elif kind == 'coordinates':
            order = np.argsort(-np.diag(covariance), kind='stable')
            values = np.maximum(np.diag(covariance)[order], 0)
            matrix, normalizer = np.eye(a.shape[1])[:, order], 1.0
        else:
            raise ValueError('unknown projection kind')
        fraction = np.cumsum(values)/max(float(values.sum()), np.finfo(float).tiny)
        return cls(mean, matrix, kind, normalizer, len(a), fraction)

    def project(self, points):
        return (np.asarray(points, dtype=np.float64)-self.mean) @ self.matrix

    def fingerprint(self):
        return hashlib.sha256(self.mean.tobytes()+self.matrix.tobytes()).hexdigest()

    def save(self, path):
        np.savez(path, mean=self.mean, matrix=self.matrix, kind=self.kind,
                 normalization=self.normalization, fit_count=self.fit_count,
                 explained_fraction=self.explained_fraction)


def summarize_projection(layout_dir, out, basis: Basis, *, dims=16, balls=8,
                         tail=False, center_dtype='uint8', projected_by_id=None):
    """Build a sidecar; full projected_by_id is optional build-time scratch.

    Tail intervals cost 8 additional bytes/ball. They bound norms of the omitted
    coordinates about the same list centroid used at query time. Head and tail
    bounds are combined per ball BEFORE taking the minimum over balls.
    """
    source, out = Path(layout_dir).resolve(), Path(out)
    base = Index(source); m = base.meta
    n, cap, d, pages = m['n'], m['capacity'], m['d'], m['n_pages']
    if not m.get('physical_layout_frozen'):
        raise ValueError('a frozen layout is required')
    if not 1 <= dims <= d or not 1 <= balls <= cap:
        raise ValueError('invalid dimension/ball count')
    if center_dtype not in ('uint8', 'float32'):
        raise ValueError('unsupported center precision')
    if basis.matrix.shape != (d, d) or basis.mean.shape != (d,):
        raise ValueError('basis shape mismatch')
    if projected_by_id is not None and projected_by_id.shape != (n, d):
        raise ValueError('projection scratch shape mismatch')
    expected = m['layout_payload_sha256']
    if digest_file(source/'vectors.pages') != expected:
        raise ValueError('frozen payload changed')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.memmap(source/'vectors.pages', dtype='<f4', mode='r',
                    shape=(pages, m['page_size']//4))

    def batch(start, stop):
        if projected_by_id is not None:
            return np.asarray(projected_by_id[np.maximum(base.page_ids[start:stop], 0)])
        a = np.asarray(raw[start:stop, :cap*d]).reshape(-1, cap, d)
        return basis.project(a)

    # Calibration uses database points, never evaluation queries.
    low = np.full(dims, np.inf); high = np.full(dims, -np.inf)
    max_magnitude = 0.0
    for start in range(0, pages, 2048):
        stop = min(pages, start+2048)
        z = batch(start, stop)
        valid = np.arange(cap)[None, :] < base.valid[start:stop, None]
        points = z[:, :, :dims][valid]
        low = np.minimum(low, points.min(axis=0)); high = np.maximum(high, points.max(axis=0))
        max_magnitude = max(max_magnitude, float(np.abs(z).max()))
    scale = (high-low)/255.; scale[scale == 0] = 1.
    codes = np.zeros((pages, balls, dims), dtype=center_dtype)
    radii = np.full((pages, balls), -1, dtype=np.float32)
    tail_bounds = np.zeros((pages, balls, 2), dtype=np.float32) if tail else np.empty((0,), np.float32)
    list_id = np.empty(pages, dtype=np.int64)
    for li, (first, last) in enumerate(base.ranges):
        list_id[first:last] = li
    projected_centers = basis.project(base.centers)
    slots_by_ball = np.array_split(np.arange(cap), balls)
    # Generous norm/error allowance at the declared finite-input magnitude range.
    build_guard = 2048*EPS*d*d*(1+max_magnitude+float(np.abs(basis.mean).max()))
    for start in range(0, pages, 2048):
        stop = min(pages, start+2048)
        z = batch(start, stop)
        valid = np.arange(cap)[None, :] < base.valid[start:stop, None]
        norms = (np.linalg.norm(z[:, :, dims:]-projected_centers[list_id[start:stop], None, dims:], axis=2)
                 if tail else None)
        for j, slots in enumerate(slots_by_ball):
            good = valid[:, slots]; count = good.sum(axis=1)
            a = z[:, slots, :dims]
            center = (a*good[:, :, None]).sum(axis=1)/np.maximum(count, 1)[:, None]
            if center_dtype == 'uint8':
                code = np.clip(np.rint((center-low)/scale), 0, 255).astype(np.uint8)
                decoded = low + code*scale
            else:
                code = center.astype(np.float32); decoded = code.astype(np.float64)
            radius = np.where(good, np.linalg.norm(a-decoded[:, None, :], axis=2), 0).max(axis=1)
            rr = np.nextafter((radius+build_guard).astype(np.float32), np.float32(np.inf))
            rr[count == 0] = -1
            codes[start:stop, j], radii[start:stop, j] = code, rr
            if tail:
                lo = np.where(good, norms[:, slots], np.inf).min(axis=1)
                hi = np.where(good, norms[:, slots], -np.inf).max(axis=1)
                lo[count == 0] = 0.; hi[count == 0] = 0.
                tail_bounds[start:stop, j, 0] = np.nextafter(np.maximum(0, lo-build_guard).astype(np.float32), np.float32(-np.inf))
                tail_bounds[start:stop, j, 1] = np.nextafter((hi+build_guard).astype(np.float32), np.float32(np.inf))
    del raw
    arrays = dict(mean=basis.mean, matrix=basis.matrix, origin=low, scale=scale,
                  codes=codes, radii=radii, tail_bounds=tail_bounds,
                  projected_centers=projected_centers)
    np.savez(out/'summary.npz', **arrays)
    summary_bytes = codes.nbytes+radii.nbytes+tail_bounds.nbytes
    shared_bytes = sum(arrays[k].nbytes for k in ('mean', 'matrix', 'origin', 'scale', 'projected_centers'))
    directory_bytes = sum(getattr(base, k).nbytes for k in ('centers', 'page_ids', 'valid', 'radial', 'ranges'))
    meta = dict(layout_payload_sha256=expected, basis_sha256=basis.fingerprint(),
                projection=basis.kind, normalization=basis.normalization,
                fit_count=basis.fit_count, dims=dims, balls=balls, tail=bool(tail),
                center_dtype=center_dtype, summary_bytes=summary_bytes,
                bytes_per_page=summary_bytes/pages, radial_bytes=base.radial.nbytes,
                shared_projection_bytes=shared_bytes,
                directory_array_bytes=directory_bytes+shared_bytes+summary_bytes,
                build_guard=build_guard, max_magnitude=max_magnitude,
                fit_explained_fraction=(basis.explained_fraction[:dims].tolist()
                    if basis.explained_fraction is not None else None))
    (out/'summary.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    if digest_file(source/'vectors.pages') != expected:
        raise ValueError('payload changed during summary build')
    return meta


class ProjectedIndex(Index):
    """Use the existing staged executor/readers with a different RAM directory."""
    def __init__(self, layout_dir, summary_dir):
        super().__init__(layout_dir)
        info = json.loads((Path(summary_dir)/'summary.json').read_text())
        if self.meta.get('layout_payload_sha256') != info['layout_payload_sha256']:
            raise ValueError('summary belongs to another physical layout')
        for key in ('codes', 'radii', 'coordinates', 'origin', 'scale'):
            delattr(self, key)
        with np.load(Path(summary_dir)/'summary.npz', allow_pickle=False) as f:
            for key in f.files:
                setattr(self, key, f[key])
        self.summary = info
        self.meta = dict(self.meta, **{k: info[k] for k in ('dims', 'balls', 'projection',
                         'summary_bytes', 'directory_array_bytes')})
        self._query = None
        self._zq = None

    def bounds(self, q, pages, li, mode):
        if mode not in ('none', 'radial', 'balls', 'combined'):
            raise ValueError('unknown filtering mode')
        pages = np.asarray(pages, dtype=np.int64)
        result = np.zeros(len(pages), dtype=np.float64)
        if not len(pages) or mode == 'none':
            return result
        # One full rotation/query; charged inside the existing filter timer.
        if self._query is None or not np.array_equal(q, self._query):
            self._query = np.array(q, dtype=np.float64, copy=True)
            self._zq = (self._query-self.mean) @ self.matrix
        if mode in ('balls', 'combined'):
            dims = self.meta['dims']
            c = self.codes[pages].astype(np.float64)
            if self.summary['center_dtype'] == 'uint8':
                c = self.origin+c*self.scale
            lb = np.maximum(0, np.linalg.norm(c-self._zq[:dims], axis=2)-self.radii[pages])
            if self.summary['tail']:
                nq = np.linalg.norm(self._zq[dims:]-self.projected_centers[li, dims:])
                interval = self.tail_bounds[pages]
                tb = np.maximum(0, np.maximum(interval[:, :, 0]-nq, nq-interval[:, :, 1]))
                lb = np.hypot(lb, tb)
            lb[self.radii[pages] < 0] = np.inf
            result = lb.min(axis=1)
        if mode in ('radial', 'combined'):
            r = np.linalg.norm(np.asarray(q, dtype=np.float64)-self.centers[li])
            result = np.maximum(result, np.maximum(self.radial[pages, 0]-r,
                                                   r-self.radial[pages, 1]))
        guard = self.summary['build_guard']+2048*EPS*self.meta['d']**2*(1+float(np.max(np.abs(q))))
        return np.maximum(0, result-guard)
