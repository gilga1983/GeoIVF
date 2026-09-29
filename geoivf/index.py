"""Faiss-backed IVF construction; FP32 page payload and a compact RAM directory.

Coordinate selection is deliberately used for the first adapter: it is a
contraction without a numerically estimated PCA operator norm. This is not the
PCA variant discussed in the exploratory experiments.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np


def vectors(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == '.npy':
        a = np.load(path, mmap_mode='r', allow_pickle=False)
    elif path.suffix == '.fvecs':
        raw = np.fromfile(path, dtype='<i4')
        if not len(raw) or raw[0] < 1 or len(raw) % (int(raw[0]) + 1):
            raise ValueError('invalid fvecs length/dimension')
        raw = raw.reshape(-1, int(raw[0]) + 1)
        if not np.all(raw[:, 0] == raw[0, 0]):
            raise ValueError('inconsistent fvecs dimensions')
        a = raw[:, 1:].copy().view('<f4')
    else:
        raise ValueError('expected .npy or .fvecs')
    return checked(a)


def checked(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32, order='C')
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError('vectors must be a nonempty, finite, 2D array')
    if np.max(np.abs(a)) > 1e10:
        raise ValueError('vector magnitude exceeds the supported numeric range')
    return a


def train_faiss(x: np.ndarray, nlist: int, seed: int, train_size: int,
                threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    import faiss  # Required for real builds; never silently replace Faiss.
    if not 1 <= nlist <= len(x) or threads < 1:
        raise ValueError('invalid nlist or thread count')
    faiss.omp_set_num_threads(threads)
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(x), min(train_size, len(x)), replace=False)
    if len(ids) < nlist:
        raise ValueError('training sample is smaller than nlist')
    km = faiss.Kmeans(x.shape[1], nlist, niter=20, seed=seed, verbose=False)
    km.train(np.ascontiguousarray(x[ids]))
    centers = np.asarray(km.centroids, dtype=np.float32)
    quantizer = faiss.IndexFlatL2(x.shape[1])
    quantizer.add(centers)
    labels = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), 65536):
        labels[s:s+65536] = quantizer.search(x[s:s+65536], 1)[1][:, 0]
    return centers, labels


def pack(ids: np.ndarray, z: np.ndarray, capacity: int) -> np.ndarray:
    """Balanced recursive coordinate splits. Pages stay within one IVF list."""
    if len(ids) <= capacity:
        return ids
    a = z[ids]
    axis = int(np.argmax(np.var(a, axis=0)))
    order = np.lexsort((ids, a[:, axis]))
    ids = ids[order]
    cut = capacity * (math.ceil(len(ids) / capacity) // 2)
    return np.concatenate((pack(ids[:cut], z, capacity),
                           pack(ids[cut:], z, capacity)))


def build_assigned(x: np.ndarray, centers: np.ndarray, labels: np.ndarray,
                   out: str | Path, *, dims: int = 16, balls: int = 2,
                   layout: str = 'geopack', page_size: int = 4096,
                   provenance: dict | None = None) -> dict:
    """Build using externally fixed assignments (also used by unit fixtures)."""
    x, centers = checked(x), checked(centers)
    labels = np.asarray(labels, dtype=np.int64)
    n, d = x.shape
    if centers.shape[1] != d or labels.shape != (n,):
        raise ValueError('assignment/centroid dimension mismatch')
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise ValueError('invalid list assignment')
    if page_size < 4096 or page_size % 4096 or page_size & (page_size - 1):
        raise ValueError('page size must be a power of two, >= 4096')
    capacity = page_size // (4*d)
    if not 1 <= balls <= capacity or not 1 <= dims <= d or capacity > 65535:
        raise ValueError('invalid dimensions, balls, or vectors/page')
    if layout not in ('input', 'radial', 'geopack'):
        raise ValueError('unknown layout')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty index: {out}')
    out.mkdir(parents=True, exist_ok=True)
    coordinates = np.argsort(-np.var(x.astype(np.float64), axis=0),
                             kind='stable')[:dims].astype(np.int32)
    z = x[:, coordinates].astype(np.float64)
    origin = z.min(axis=0)
    scale = (z.max(axis=0) - origin) / 255.0
    scale[scale == 0] = 1.0
    # Avoid repeated O(n*nlist) scans when constructing buckets.
    by_list = np.argsort(labels, kind='stable')
    counts = np.bincount(labels, minlength=len(centers))
    splits = np.r_[0, np.cumsum(counts)]
    n_pages = int(np.sum((counts + capacity - 1) // capacity))
    page_ids = np.full((n_pages, capacity), -1, dtype=np.int64)
    valid = np.zeros(n_pages, dtype=np.uint16)
    codes = np.zeros((n_pages, balls, dims), dtype=np.uint8)
    radii = np.full((n_pages, balls), -1, dtype=np.float32)
    radial = np.zeros((n_pages, 2), dtype=np.float32)
    ranges = np.zeros((len(centers), 2), dtype=np.int64)
    p = 0
    with (out/'vectors.pages').open('wb') as f:
        for li in range(len(centers)):
            ids = by_list[splits[li]:splits[li+1]]
            if layout == 'radial':
                r = np.linalg.norm(x[ids].astype(np.float64)-centers[li], axis=1)
                ids = ids[np.lexsort((ids, r))]
            elif layout == 'geopack':
                ids = pack(ids, z, capacity)
            ranges[li, 0] = p
            for s in range(0, len(ids), capacity):
                group = ids[s:s+capacity]
                valid[p] = len(group)
                page_ids[p, :len(group)] = group
                payload = x[group].astype('<f4', copy=False).tobytes()
                f.write(payload + bytes(page_size-len(payload)))
                r = np.linalg.norm(x[group].astype(np.float64)-centers[li], axis=1)
                radial[p] = [np.nextafter(np.float32(r.min()), np.float32(-np.inf)),
                             np.nextafter(np.float32(r.max()), np.float32(np.inf))]
                for j, g in enumerate(np.array_split(group, min(balls, len(group)))):
                    c = z[g].mean(axis=0)
                    code = np.clip(np.rint((c-origin)/scale), 0, 255).astype(np.uint8)
                    decoded = origin + scale * code
                    # Re-cover the ORIGINAL projected points after quantization.
                    radius = np.linalg.norm(z[g]-decoded, axis=1).max()
                    guard = 128*np.finfo(float).eps*d*(1+np.max(np.abs(z[g]))
                                                       +np.max(np.abs(decoded)))
                    codes[p, j] = code
                    radii[p, j] = np.nextafter(np.float32(radius+guard), np.float32(np.inf))
                p += 1
            ranges[li, 1] = p
    arrays = dict(centers=centers, coordinates=coordinates, origin=origin,
                  scale=scale, page_ids=page_ids, valid=valid, codes=codes,
                  radii=radii, radial=radial, ranges=ranges)
    np.savez(out/'directory.npz', **arrays)
    meta = dict(format_version=2, n=n, d=d, nlist=len(centers), page_size=page_size,
                capacity=capacity, n_pages=n_pages, dims=dims, balls=balls,
                layout=layout, projection='variance-ranked-coordinate-subset',
                payload_dtype='float32', center_dtype='uint8', radius_dtype='float32',
                summary_bytes=codes.nbytes+radii.nbytes,
                radial_bytes=radial.nbytes,
                directory_array_bytes=sum(a.nbytes for a in arrays.values()),
                payload_bytes=n_pages*page_size, provenance=provenance or {})
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta


class Index:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.meta = json.loads((self.path/'manifest.json').read_text())
        if self.meta['format_version'] != 2:
            raise ValueError('unsupported index format')
        with np.load(self.path/'directory.npz', allow_pickle=False) as f:
            for key in f.files:
                setattr(self, key, f[key])
        m = self.meta
        if (self.path/'vectors.pages').stat().st_size != m['payload_bytes']:
            raise ValueError('truncated or oversized vector payload')
        if self.page_ids.shape != (m['n_pages'], m['capacity']):
            raise ValueError('invalid page directory shape')
        self.quantizer = None
        self.numeric_scale = (np.max(np.abs(self.centers)) + np.max(np.abs(self.origin))
                              + 255*np.max(self.scale))

    def route(self, q: np.ndarray, nprobe: int, *, numpy_test: bool = False) -> np.ndarray:
        if not 1 <= nprobe <= self.meta['nlist']:
            raise ValueError('nprobe outside [1,nlist]')
        if numpy_test:
            dist = np.sum((self.centers.astype(np.float64)-q)**2, axis=1)
            return np.argsort(dist, kind='stable')[:nprobe]
        import faiss
        if self.quantizer is None:
            self.quantizer = faiss.IndexFlatL2(self.meta['d'])
            self.quantizer.add(self.centers)
        return self.quantizer.search(q[None, :], nprobe)[1][0]

    def bounds(self, q: np.ndarray, pages: np.ndarray, li: int, mode: str) -> np.ndarray:
        if mode not in ('none', 'balls', 'radial', 'combined'):
            raise ValueError('unknown filter')
        result = np.zeros(len(pages), dtype=np.float64)
        if mode == 'none':
            return result
        if mode in ('balls', 'combined'):
            c = self.origin + self.codes[pages].astype(np.float64)*self.scale
            delta = c-q[self.coordinates].astype(np.float64)
            lb = np.linalg.norm(delta, axis=2)-self.radii[pages]
            lb[self.radii[pages] < 0] = np.inf
            result = np.maximum(0, lb.min(axis=1))
        if mode in ('radial', 'combined'):
            r = np.linalg.norm(q.astype(np.float64)-self.centers[li])
            # Deterministic reverse-triangle bound, NOT learned CLIP lambda.
            result = np.maximum(result, np.maximum(self.radial[pages, 0]-r,
                                                   r-self.radial[pages, 1]))
        guard = 128*np.finfo(float).eps*self.meta['d']*(1+np.max(np.abs(q))
                            +self.numeric_scale)
        return np.maximum(0, result-guard)
