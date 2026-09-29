"""Quantized local-line descriptions and capacity-preserving line-aware packing.

Payloads stay FP32. A point is described by its coordinate along a decoded line
and its perpendicular norm, both conservatively enclosed in scalar intervals.
Numeric guards are engineering safeguards, not formally verified arithmetic.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .index import Index

EPS = np.finfo(np.float64).eps


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fit_lines(points, valid):
    """Batched free least-squares lines, including singleton/duplicate pages."""
    z = np.asarray(points, dtype=np.float64)
    good = np.asarray(valid, dtype=bool)
    if z.ndim != 3 or good.shape != z.shape[:2] or not np.isfinite(z).all():
        raise ValueError('invalid point batch')
    count = good.sum(axis=1)
    if np.any(count == 0):
        raise ValueError('empty page')
    a = (z*good[..., None]).sum(axis=1)/count[:, None]
    centered = (z-a[:, None])*good[..., None]
    gram = centered @ centered.transpose(0, 2, 1)
    _, vectors = np.linalg.eigh(gram)
    u = np.einsum('bnd,bn->bd', centered, vectors[:, :, -1])
    norm = np.linalg.norm(u, axis=1)
    bad = norm <= np.finfo(float).tiny
    u[bad] = 0.; u[bad, 0] = 1.; norm[bad] = 1.
    u /= norm[:, None]
    pivot = np.argmax(np.abs(u), axis=1)
    u *= np.where(u[np.arange(len(u)), pivot] < 0, -1., 1.)[:, None]
    return a, u


def coordinates(z, a, u):
    delta = z-a[:, None]
    t = np.einsum('bnd,bd->bn', delta, u)
    residual = delta-t[..., None]*u[:, None]
    return t, np.linalg.norm(residual, axis=2)


def line_score(points, valid, worst_weight=1., span_weight=.1):
    a, u = fit_lines(points, valid)
    t, rho = coordinates(points, a, u)
    rr = np.where(valid, rho*rho, 0.)
    span = np.where(valid, t, -np.inf).max(axis=1)-np.where(valid, t, np.inf).min(axis=1)
    score = rr.sum(axis=1)+worst_weight*rr.max(axis=1)+span_weight*span*span
    return score, a, u, t, rho


def refine_layout(source, out, x, projected, *, dims=64, passes=4):
    """Alternating adjacent-page balanced two-line reassignment, not global optimum.

    Candidate assignments use fixed fitted lines. Accept only if refitting lowers
    sum squared residuals + worst residual squared + 0.1 * axial span squared.
    Full pages exchange vectors only within the SAME IVF list; tails stay put.
    """
    source, out = Path(source), Path(out)
    base = Index(source); m = base.meta
    if not m.get('physical_layout_frozen') or not 1 <= dims <= projected.shape[1] or passes < 1:
        raise ValueError('invalid frozen layout/refinement')
    if digest(source/'vectors.pages') != m['layout_payload_sha256']:
        raise ValueError('source payload changed')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    original = base.page_ids.copy(); ids = original.copy(); cap = m['capacity']
    accepted = []; objectives = []
    for sweep in range(passes):
        left = []
        for first, last in base.ranges:
            full_end = int(last)-(int(last)>int(first) and base.valid[int(last)-1] < cap)
            left.extend(range(int(first)+(sweep % 2), full_end-1, 2))
        changes = 0; improvement = 0.
        for start in range(0, len(left), 512):
            pp = np.asarray(left[start:start+512], dtype=np.int64)
            current = np.stack((ids[pp], ids[pp+1]), axis=1)
            zz = np.asarray(projected[current, :dims], dtype=float)
            flat = zz.reshape(-1, cap, dims); good = np.ones(flat.shape[:2], bool)
            old, a, u, tt, _ = line_score(flat, good)
            a = a.reshape(-1, 2, dims); u = u.reshape(-1, 2, dims)
            tt = tt.reshape(-1, 2, cap)
            pool = zz.reshape(-1, 2*cap, dims)
            delta = pool[:, :, None]-a[:, None]
            t = np.einsum('bnkd,bkd->bnk', delta, u)
            res = delta-t[..., None]*u[:, None]
            cost = np.sum(res*res, axis=3)
            axial = np.maximum(0., np.maximum(tt.min(axis=2)[:, None]-t,
                                              t-tt.max(axis=2)[:, None]))
            cost += .1*axial*axial
            order = np.argsort(cost[:, :, 0]-cost[:, :, 1], axis=1, kind='stable')
            proposal = np.take_along_axis(current.reshape(-1, 2*cap), order, axis=1).reshape(-1, 2, cap)
            zp = np.asarray(projected[proposal, :dims], dtype=float).reshape(-1, cap, dims)
            new = line_score(zp, good)[0].reshape(-1, 2).sum(axis=1)
            old = old.reshape(-1, 2).sum(axis=1)
            use = new < old-1e-10*np.maximum(1., old)
            ids[pp[use]], ids[pp[use]+1] = proposal[use, 0], proposal[use, 1]
            changes += int(use.sum()); improvement += float((old[use]-new[use]).sum())
        accepted.append(changes); objectives.append(improvement)
    for first, last in base.ranges:
        np.testing.assert_array_equal(np.sort(ids[first:last].ravel()),
                                      np.sort(original[first:last].ravel()))
    with np.load(source/'directory.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays['page_ids'] = ids
    radial = np.empty_like(base.radial)
    with (out/'vectors.pages').open('wb') as f:
        for li, (first, last) in enumerate(base.ranges):
            for p in range(int(first), int(last)):
                n = int(base.valid[p]); group = ids[p, :n]
                payload = x[group].astype('<f4', copy=False).tobytes()
                f.write(payload+bytes(m['page_size']-len(payload)))
                r = np.linalg.norm(x[group].astype(float)-base.centers[li], axis=1)
                radial[p] = [np.nextafter(np.float32(r.min()), np.float32(-np.inf)),
                             np.nextafter(np.float32(r.max()), np.float32(np.inf))]
    arrays['radial'] = radial
    # The legacy filter is disabled on a repacked payload. A sidecar is REQUIRED.
    arrays['codes'][:] = 0
    arrays['radii'][:] = -1
    arrays['radii'][:, 0] = np.inf  # safe but unselective fallback
    np.savez(out/'directory.npz', **arrays)
    info = dict(m, layout='line-refined', layout_payload_sha256=digest(out/'vectors.pages'),
                layout_directory_sha256=digest(out/'directory.npz'),
                refinement=dict(dims=dims, passes=passes, accepted_pairs=accepted,
                                objective_decreases=objectives,
                                changed_membership_pages=int(np.sum(np.any(np.sort(ids, axis=1)
                                            != np.sort(original, axis=1), axis=1))),
                                objective='sum_rho2 + max_rho2 + 0.1*span2',
                                query_fitted=False, parent_payload=m['layout_payload_sha256']))
    (out/'manifest.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


def decode_axes(arrays, pages, precision):
    if precision == 'uint8':
        a = arrays['origin']+arrays['anchor_codes'][pages].astype(float)*arrays['scale']
        u = arrays['direction_codes'][pages].astype(float)/127.
    else:
        a = arrays['anchor_codes'][pages].astype(float)
        u = arrays['direction_codes'][pages].astype(float)
    norm = np.linalg.norm(u, axis=1)
    if np.any(norm <= 0):
        raise ValueError('zero decoded direction')
    return a, u/norm[:, None]


def scalar_intervals(arrays, pages, guard):
    sc = arrays['scalar_scale'][pages].astype(float)
    tl = sc[:, 0, None]+arrays['t_codes'][pages].astype(float)*sc[:, 1, None]
    rl = arrays['rho_codes'][pages].astype(float)*sc[:, 2, None]
    return tl-guard, tl+sc[:, 1, None]+guard, np.maximum(0., rl-guard), rl+sc[:, 2, None]+guard


def summarize_lines(layout_dir, out, basis, projected, *, dims=64, precision='uint8'):
    source, out = Path(layout_dir), Path(out)
    base = Index(source); m = base.meta; pages, cap = m['n_pages'], m['capacity']
    if not m.get('physical_layout_frozen') or not 1 <= dims <= m['d']:
        raise ValueError('invalid layout/dimensions')
    if precision not in ('uint8', 'float32') or projected.shape != (m['n'], m['d']):
        raise ValueError('invalid precision or projected scratch')
    if digest(source/'vectors.pages') != m['layout_payload_sha256']:
        raise ValueError('source payload changed')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    origin = np.asarray(projected[:, :dims].min(axis=0), dtype=float)
    scale = (np.asarray(projected[:, :dims].max(axis=0), dtype=float)-origin)/255.
    scale[scale == 0] = 1.
    maxmag = float(np.max(np.abs(projected[:, :dims])))
    guard = 8192*EPS*m['d']**2*(1+maxmag+float(np.max(np.abs(basis.mean))))
    arrays = dict(anchor_codes=np.zeros((pages, dims), dtype=precision),
                  direction_codes=np.zeros((pages, dims), dtype='int8' if precision=='uint8' else 'float32'),
                  t_codes=np.zeros((pages, cap), dtype=np.uint16),
                  rho_codes=np.zeros((pages, cap), dtype=np.uint16),
                  scalar_scale=np.zeros((pages, 3), dtype=np.float32),
                  origin=origin, scale=scale, mean=basis.mean, matrix=basis.matrix)
    rho_stats = []; fractions = []; max_rho = []
    for start in range(0, pages, 1024):
        pp = np.arange(start, min(pages, start+1024))
        z = np.asarray(projected[np.maximum(base.page_ids[pp], 0), :dims], dtype=float)
        good = np.arange(cap)[None] < base.valid[pp, None]
        a, u = fit_lines(z, good)
        if precision == 'uint8':
            arrays['anchor_codes'][pp] = np.clip(np.rint((a-origin)/scale), 0, 255).astype(np.uint8)
            arrays['direction_codes'][pp] = np.clip(np.rint(127*u), -127, 127).astype(np.int8)
        else:
            arrays['anchor_codes'][pp] = a.astype(np.float32)
            arrays['direction_codes'][pp] = u.astype(np.float32)
        aa, uu = decode_axes(arrays, pp, precision)
        t, rho = coordinates(z, aa, uu)
        lo = np.where(good, t, np.inf).min(axis=1)-guard
        hi = np.where(good, t, -np.inf).max(axis=1)+guard
        t0 = np.nextafter(lo.astype(np.float32), np.float32(-np.inf))
        ts = np.nextafter(((hi-t0.astype(float))/65534.).astype(np.float32), np.float32(np.inf))
        rs = np.nextafter(((np.where(good, rho, 0).max(axis=1)+guard)/65534.).astype(np.float32), np.float32(np.inf))
        arrays['scalar_scale'][pp] = np.stack((t0, ts, rs), axis=1)
        arrays['t_codes'][pp] = np.clip(np.floor((t-t0[:, None])/ts[:, None]), 0, 65534).astype(np.uint16)
        arrays['rho_codes'][pp] = np.clip(np.floor(rho/rs[:, None]), 0, 65534).astype(np.uint16)
        tl, th, rl, rh = scalar_intervals(arrays, pp, guard)
        if not np.all(((tl <= t)&(t <= th)&(rl <= rho)&(rho <= rh))[good]):
            raise AssertionError('scalar intervals fail to cover original point')
        rho_stats.append(rho[good]); max_rho.append(np.where(good, rho, 0).max(axis=1))
        _, ideal_u = fit_lines(z, good)
        ideal_t, ideal_rho = coordinates(z, a, ideal_u)
        total = np.where(good, ideal_t**2+ideal_rho**2, 0).sum(axis=1)
        explained = np.where(good, ideal_t**2, 0).sum(axis=1)
        fractions.append(np.divide(explained, total, out=np.ones_like(total), where=total>1e-20))
    np.savez(out/'line.npz', **arrays)
    components = {k: arrays[k].nbytes for k in ('anchor_codes','direction_codes','t_codes','rho_codes','scalar_scale')}
    shared = sum(arrays[k].nbytes for k in ('origin','scale','mean','matrix'))
    structural = sum(getattr(base,k).nbytes for k in ('centers','page_ids','valid','radial','ranges'))
    r = np.concatenate(rho_stats); ff = np.concatenate(fractions); mr = np.concatenate(max_rho)
    info = dict(dims=dims, precision=precision, summary_bytes=sum(components.values()),
                bytes_per_page=sum(components.values())/pages, component_bytes=components,
                directory_array_bytes=structural+shared+sum(components.values()),
                shared_projection_bytes=shared, radial_bytes=base.radial.nbytes,
                layout_payload_sha256=m['layout_payload_sha256'], basis_sha256=basis.fingerprint(),
                build_guard=guard, all_point_intervals_audited=int(m['n']),
                mean_perpendicular_distance=float(r.mean()), p95_perpendicular_distance=float(np.quantile(r,.95)),
                mean_page_max_perpendicular_distance=float(mr.mean()),
                mean_local_variance_explained=float(ff.mean()), median_local_variance_explained=float(np.median(ff)),
                expanded_axes_retained=False)
    (out/'line.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


class LineIndex(Index):
    def __init__(self, layout_dir, summary_dir, *, bound='shell'):
        super().__init__(layout_dir)
        self.summary = json.loads((Path(summary_dir)/'line.json').read_text())
        if bound not in ('shell','ball') or self.meta.get('layout_payload_sha256') != self.summary['layout_payload_sha256']:
            raise ValueError('invalid bound or mismatched layout')
        for key in ('codes','radii','coordinates','origin','scale'):
            delattr(self,key)
        with np.load(Path(summary_dir)/'line.npz', allow_pickle=False) as f:
            self.line_arrays = {k:f[k] for k in f.files}
        self.bound = bound
        self.meta = dict(self.meta, **{k:self.summary[k] for k in ('dims','summary_bytes','directory_array_bytes')})
        self._query = None; self._zq = None

    def bounds(self, q, pages, li, mode):
        if mode not in ('none','radial','balls','combined'):
            raise ValueError('invalid mode')
        pages = np.asarray(pages,dtype=np.int64); result = np.zeros(len(pages))
        if not len(pages) or mode == 'none': return result
        if self._query is None or not np.array_equal(q,self._query):
            self._query = np.asarray(q,dtype=float).copy()
            self._zq = (self._query-self.line_arrays['mean']) @ self.line_arrays['matrix']
        guard = self.summary['build_guard']+8192*EPS*self.meta['d']**2*(1+float(np.max(np.abs(q))))
        if mode in ('balls','combined'):
            a,u = decode_axes(self.line_arrays,pages,self.summary['precision'])
            tq,rq = coordinates(self._zq[None,None,:self.meta['dims']],a,u)
            tl,th,rl,rh = scalar_intervals(self.line_arrays,pages,self.summary['build_guard'])
            if self.bound == 'shell':
                dt = np.maximum(0.,np.maximum(tl-tq,tq-th))
                dr = np.maximum(0.,np.maximum(rl-rq,rq-rh))
                lb = np.hypot(dt,dr)
            else:
                # The enclosing ball loses axial/perpendicular structure.
                mid = (tl+th)/2
                radius = np.hypot((th-tl)/2,rh)
                lb = np.maximum(0.,np.hypot(tq-mid,rq)-radius)
            lb[np.arange(self.meta['capacity'])[None] >= self.valid[pages,None]] = np.inf
            result = lb.min(axis=1)
        if mode in ('radial','combined'):
            r = np.linalg.norm(np.asarray(q,dtype=float)-self.centers[li])
            result = np.maximum(result,np.maximum(self.radial[pages,0]-r,r-self.radial[pages,1]))
        return np.maximum(0.,result-guard)
