"""Packed dependent-center page summaries, with explicit quantization controls.

Static singleton-proxy experiment: keep pages and payloads fixed. Every active
ball covers its assigned original projected point after *all* decoding steps.
Only packed codes are retained; decoded centers are query-window scratch.
"""
from __future__ import annotations
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from .index import Index

EPS = np.finfo(np.float64).eps


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pack_codes(values, bits):
    """Pack each first-axis record separately, including actual final-byte padding."""
    if not 1 <= bits <= 8:
        raise ValueError('bits must be in [1,8]')
    a = np.asarray(values)
    if a.ndim < 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('expected batched integer codes')
    if np.any(a < 0) or np.any(a >= 2**bits):
        raise ValueError('code out of range')
    flat = a.reshape(len(a), -1).astype(np.uint8)
    b = ((flat[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(len(a), -1)
    return np.packbits(b, axis=1, bitorder='little')


def unpack_codes(packed, bits, shape):
    if not 1 <= bits <= 8 or not shape or min(shape) < 0:
        raise ValueError('invalid decoding shape or precision')
    a = np.asarray(packed)
    count = int(np.prod(shape))
    if a.ndim != 2 or a.dtype != np.uint8 or a.shape[1] != (count*bits+7)//8:
        raise ValueError('packed record size mismatch')
    b = np.unpackbits(a, axis=1, bitorder='little')[:, :count*bits]
    v = (b.reshape(len(a), count, bits) * (1 << np.arange(bits))).sum(axis=2).astype(np.uint8)
    return v.reshape((len(a), *shape))


def fit_anchors(points, valid, origin, scale, anchors, *, midpoint=False):
    """Enumerate anchor subsets (<=8 points/page), minimizing sum squared errors.

    The exhaustive subset search is exact for its center-error surrogate, given
    data-point anchors quantized with shared 8-bit scales and the specified pair
    grid. It is NOT an optimizer for query false positives. Training queries are
    never used. Returns target slot order for transient radius certification.
    """
    z = np.asarray(points, dtype=np.float64)
    if z.ndim != 3 or z.shape[1] > 8 or not 2 <= anchors < z.shape[1]:
        raise ValueError('requires 2 <= anchors < capacity <= 8')
    batch, cap, dims = z.shape
    valid = np.asarray(valid)
    if valid.shape != (batch,) or np.any(valid < 1) or np.any(valid > cap):
        raise ValueError('invalid occupancy')
    codes = np.clip(np.rint((z-origin)/scale), 0, 255).astype(np.uint8)
    a = origin + codes*scale
    # Distances to all potential decoded anchors, directly evaluated for stability.
    dx = np.sum((z[:, :, None, :]-a[:, None, :, :])**2, axis=3)
    pairs = np.array(list(combinations(range(cap), 2)), dtype=np.int64)
    left, right = pairs[:, 0], pairs[:, 1]
    sep = np.sum((a[:, left]-a[:, right])**2, axis=2)
    dl, dr = dx[:, :, left], dx[:, :, right]
    if midpoint:
        alpha = np.full(dl.shape, 8, dtype=np.uint8)
    else:
        den = np.where(sep > 0, 2*sep, 1)[:, None, :]
        t = np.clip((dl-dr+sep[:, None, :])/den, 0, 1)
        alpha = np.rint(t*16).astype(np.uint8)
    t = alpha.astype(np.float64)/16
    error = np.maximum(0, (1-t)*dl + t*dr - t*(1-t)*sep[:, None, :])
    subsets = list(combinations(range(cap), anchors))
    best_score = np.full(batch, np.inf)
    best_subset = np.zeros(batch, dtype=np.int64)
    point_valid = np.arange(cap)[None, :] < valid[:, None]
    for number, subset in enumerate(subsets):
        permitted = np.isin(left, subset) & np.isin(right, subset)
        e = error[:, :, permitted].min(axis=2)
        for i in subset:
            e[:, i] = dx[:, i, i]  # explicit anchors still have quantization error
        score = np.where(point_valid, e, 0).sum(axis=1)
        score[valid < (max(subset)+1)] = np.inf
        change = score < best_score
        best_score[change] = score[change]
        best_subset[change] = number
    # Short pages use the first slots; invalid summaries are disabled by radius=-1.
    best_subset[valid < anchors] = 0
    selected = np.asarray(subsets, dtype=np.int64)[best_subset]
    complement = np.array([[j for j in range(cap) if j not in s] for s in subsets])[best_subset]
    order = np.concatenate((selected, complement), axis=1)
    explicit = np.take_along_axis(codes, selected[:, :, None], axis=1)
    pair_codes = np.zeros((batch, cap-anchors), dtype=np.uint8)
    weights = np.zeros_like(pair_codes)
    pred = np.empty((batch, cap, dims), dtype=np.float64)
    pred[:, :anchors] = origin+explicit*scale
    local_pairs = np.array(list(combinations(range(anchors), 2)))
    lookup = {(int(u), int(v)): k for k, (u, v) in enumerate(pairs)}
    global_pair = np.empty((batch, len(local_pairs)), dtype=np.int64)
    # Small fixed loop, not a Python loop over the millions of points.
    lookup_matrix = np.zeros((cap, cap), dtype=np.int64)
    for (u, v), k in lookup.items():
        lookup_matrix[u, v] = lookup_matrix[v, u] = k
    global_pair[:] = lookup_matrix[selected[:, local_pairs[:, 0]], selected[:, local_pairs[:, 1]]]
    rows = np.arange(batch)
    for j in range(cap-anchors):
        target = complement[:, j]
        errs = error[rows[:, None], target[:, None], global_pair]
        chosen = np.argmin(errs, axis=1)
        pair = local_pairs[chosen]
        global_p = global_pair[rows, chosen]
        w = alpha[rows, target, global_p]
        weights[:, j] = w
        pair_codes[:, j] = (pair[:, 0] << 4) | pair[:, 1]
        weight = w[:, None]/16
        pred[:, anchors+j] = ((1-weight)*pred[rows, pair[:, 0]]
                                  + weight*pred[rows, pair[:, 1]])
    return explicit, pair_codes, weights, pred, order


def decode_records(arrays, info, pages):
    """Decode only requested records. No full-index expanded center cache."""
    pages = np.asarray(pages, dtype=np.int64)
    cap, dims, anchors = info['balls'], info['dims'], info['anchors']
    bits = info['bits']
    c = unpack_codes(arrays['packed'][pages], bits, (anchors, dims)).astype(float)
    c = arrays['origin'] + c*arrays['scale']
    if info['scheme'] == 'independent':
        return c
    result = np.empty((len(pages), cap, dims), dtype=float)
    result[:, :anchors] = c
    pair = arrays['pairs'][pages]
    lo, hi = pair >> 4, pair & 15
    if np.any(lo >= anchors) or np.any(hi >= anchors):
        raise ValueError('invalid anchor reference')
    if info['scheme'] == 'midpoint':
        t = np.full((*pair.shape, 1), .5)
    else:
        t = arrays['alpha'][pages, :, None].astype(float)/16
        if np.any(t > 1):
            raise ValueError('invalid interpolation coefficient')
    rows = np.arange(len(pages))[:, None]
    result[:, anchors:] = (1-t)*c[rows, lo] + t*c[rows, hi]
    rb = info['residual_bits']
    if rb:
        mid = 2**(rb-1)-1
        r = unpack_codes(arrays['residual'][pages], rb, (cap-anchors, dims)).astype(float)-mid
        result[:, anchors:] += r*arrays['residual_scale'][pages, :, None].astype(float)
    return result


def summarize_dependencies(layout_dir, out, basis, *, dims=64, scheme='independent',
                           anchors=4, bits=8, residual_bits=0, projected_by_id=None):
    """Create compact singleton-cover sidecar, keeping original page files intact."""
    source, out = Path(layout_dir).resolve(), Path(out)
    base = Index(source); m = base.meta
    cap, d, pages = m['capacity'], m['d'], m['n_pages']
    if not m.get('physical_layout_frozen'):
        raise ValueError('expected frozen pages')
    if scheme not in ('independent', 'midpoint', 'interpolate') or not 1 <= dims <= d:
        raise ValueError('invalid representation')
    if scheme == 'independent':
        anchors = cap
        if residual_bits:
            raise ValueError('independent centers do not use interpolation residuals')
    elif bits != 8 or not 2 <= anchors < cap <= 8:
        raise ValueError('dependent schemes require 8-bit anchors and capacity <= 8')
    if not 1 <= bits <= 8 or residual_bits not in (0, 2, 4):
        raise ValueError('unsupported precision')
    if residual_bits and scheme != 'interpolate':
        raise ValueError('residuals require interpolation')
    if projected_by_id is None or projected_by_id.shape != (m['n'], d):
        raise ValueError('full build-time projection scratch is required')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    expected = m['layout_payload_sha256']
    if file_hash(source/'vectors.pages') != expected:
        raise ValueError('payload hash changed')
    out.mkdir(parents=True, exist_ok=True)
    zall = projected_by_id[:, :dims]
    low = np.asarray(zall.min(axis=0), dtype=float)
    high = np.asarray(zall.max(axis=0), dtype=float)
    scale = (high-low)/(2**bits-1); scale[scale == 0] = 1.
    packed = np.zeros((pages, (anchors*dims*bits+7)//8), dtype=np.uint8)
    radii = np.full((pages, cap), -1, dtype=np.float32)
    dependent = cap-anchors
    pairs = np.zeros((pages, dependent), dtype=np.uint8)
    alpha = np.zeros((pages, dependent if scheme == 'interpolate' else 0), dtype=np.uint8)
    residual = np.zeros((pages, (dependent*dims*residual_bits+7)//8), dtype=np.uint8)
    residual_scale = np.zeros((pages, dependent if residual_bits else 0), dtype=np.float32)
    maxmag = float(max(np.max(np.abs(low)), np.max(np.abs(high))))
    guard = 2048*EPS*d*d*(1+maxmag+float(np.max(np.abs(basis.mean))))
    info = dict(scheme=scheme, anchors=anchors, balls=cap, dims=dims, bits=bits,
                residual_bits=residual_bits)
    errors = []
    for start in range(0, pages, 512):
        stop = min(pages, start+512); count = stop-start
        ids = base.page_ids[start:stop]
        z = np.asarray(projected_by_id[np.maximum(ids, 0), :dims], dtype=float)
        occupancy = base.valid[start:stop]
        if scheme == 'independent':
            code = np.clip(np.rint((z-low)/scale), 0, 2**bits-1).astype(np.uint8)
            order = np.broadcast_to(np.arange(cap), (count, cap))
            pred = low+code*scale
        else:
            code, pair, weights, pred, order = fit_anchors(z, occupancy, low, scale, anchors,
                                                          midpoint=scheme == 'midpoint')
            pairs[start:stop] = pair
            if scheme == 'interpolate': alpha[start:stop] = weights
        packed[start:stop] = pack_codes(code, bits)
        target = np.take_along_axis(z, order[:, :, None], axis=1)
        if residual_bits:
            delta = target[:, anchors:]-pred[:, anchors:]
            mid = 2**(residual_bits-1)-1
            sc = (np.max(np.abs(delta), axis=2)/mid).astype(np.float32)
            sc[sc == 0] = 1.
            rc = np.clip(np.rint(delta/sc[:, :, None]), -mid, mid).astype(np.int16)+mid
            residual[start:stop] = pack_codes(rc, residual_bits)
            residual_scale[start:stop] = sc
            pred[:, anchors:] += (rc.astype(float)-mid)*sc[:, :, None].astype(float)
        # The exact decoded representation, not an ideal unquantized prediction.
        error = np.linalg.norm(target-pred, axis=2)
        valid = order < occupancy[:, None]
        rr = np.nextafter((error+guard).astype(np.float32), np.float32(np.inf))
        rr[~valid] = -1
        radii[start:stop] = rr
        errors.append(error[valid])
    arrays = dict(packed=packed, radii=radii, pairs=pairs, alpha=alpha,
                  residual=residual, residual_scale=residual_scale,
                  origin=low, scale=scale, mean=basis.mean, matrix=basis.matrix)
    # Ensure builder prediction and deployed decoder agree (cover audit, all points).
    for start in range(0, pages, 512):
        pp = np.arange(start, min(pages, start+512))
        decoded = decode_records(arrays, info, pp)
        ids = base.page_ids[pp]
        z = np.asarray(projected_by_id[np.maximum(ids, 0), :dims], dtype=float)
        # Every original point must lie in at least one active decoded ball.
        ds = np.linalg.norm(z[:, :, None]-decoded[:, None], axis=3)
        ds -= np.where(radii[pp] < 0, -np.inf, radii[pp])[:, None]
        valid = np.arange(cap)[None, :] < base.valid[pp, None]
        if np.any(ds.min(axis=2)[valid] > 1e-8):
            raise AssertionError('decoded ball union does not cover original points')
    np.savez(out/'dependent.npz', **arrays)
    per_page = {k: v.nbytes for k, v in arrays.items()
                if k in ('packed', 'radii', 'pairs', 'alpha', 'residual', 'residual_scale')}
    shared = sum(arrays[k].nbytes for k in ('origin', 'scale', 'mean', 'matrix'))
    basebytes = sum(getattr(base, k).nbytes for k in ('centers', 'page_ids', 'valid', 'radial', 'ranges'))
    err = np.concatenate(errors)
    info.update(summary_bytes=sum(per_page.values()), bytes_per_page=sum(per_page.values())/pages,
                component_bytes=per_page, shared_projection_bytes=shared,
                directory_array_bytes=basebytes+shared+sum(per_page.values()),
                radial_bytes=base.radial.nbytes, layout_payload_sha256=expected,
                basis_sha256=basis.fingerprint(), projection=basis.kind,
                build_guard=guard, radius_error_mean=float(err.mean()),
                radius_error_median=float(np.median(err)),
                radius_error_p95=float(np.quantile(err, .95)),
                radius_error_max=float(err.max()), all_points_cover_audited=int(m['n']),
                runtime_expanded_centers_retained=False)
    (out/'dependent.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n')
    return info


class DependentIndex(Index):
    def __init__(self, layout_dir, summary_dir):
        super().__init__(layout_dir)
        info = json.loads((Path(summary_dir)/'dependent.json').read_text())
        if self.meta.get('layout_payload_sha256') != info['layout_payload_sha256']:
            raise ValueError('sidecar belongs to another layout')
        for key in ('codes', 'radii', 'coordinates', 'origin', 'scale'):
            delattr(self, key)
        with np.load(Path(summary_dir)/'dependent.npz', allow_pickle=False) as f:
            self.arrays = {k: f[k] for k in f.files}
        self.summary = info
        self.meta = dict(self.meta, **{k: info[k] for k in ('dims', 'balls', 'projection',
                         'summary_bytes', 'directory_array_bytes')})
        self._query = None; self._zq = None

    def bounds(self, q, pages, li, mode):
        if mode not in ('none', 'radial', 'balls', 'combined'):
            raise ValueError('invalid bound mode')
        pages = np.asarray(pages, dtype=np.int64)
        result = np.zeros(len(pages), dtype=float)
        if not len(pages) or mode == 'none': return result
        a = self.arrays
        if mode in ('balls', 'combined'):
            if self._query is None or not np.array_equal(q, self._query):
                self._query = np.array(q, dtype=float, copy=True)
                self._zq = (self._query-a['mean']) @ a['matrix']
            c = decode_records(a, self.summary, pages)
            lb = np.linalg.norm(c-self._zq[:self.meta['dims']], axis=2)-a['radii'][pages]
            lb[a['radii'][pages] < 0] = np.inf
            result = np.maximum(0, lb.min(axis=1))
        if mode in ('radial', 'combined'):
            r = np.linalg.norm(np.asarray(q, dtype=float)-self.centers[li])
            result = np.maximum(result, np.maximum(self.radial[pages, 0]-r, r-self.radial[pages, 1]))
        guard = self.summary['build_guard']+2048*EPS*self.meta['d']**2*(1+float(np.max(np.abs(q))))
        return np.maximum(0, result-guard)
