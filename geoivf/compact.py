"""Short-segment page refinement with a fixed full-space diameter budget.

Offline heuristic only. Queries never enter packing. Full pages exchange IDs
within 64-vector IVF pools, while every output slot obeys its original diameter
budget. Partial pages remain unchanged. Bounds are rebuilt by existing sidecars.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .index import Index


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def geometry(points, valid=None):
    """Refit a least-squares line; return span, worst residual and tube score."""
    z = np.asarray(points, dtype=np.float64)
    if z.ndim != 3 or not np.isfinite(z).all():
        raise ValueError('finite batched points required')
    good = np.ones(z.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    if good.shape != z.shape[:2] or np.any(good.sum(axis=1) == 0):
        raise ValueError('invalid occupancy')
    center = (z*good[..., None]).sum(axis=1)/good.sum(axis=1)[:, None]
    delta = (z-center[:, None])*good[..., None]
    _, vv = np.linalg.eigh(delta@delta.transpose(0, 2, 1))
    u = np.einsum('bnd,bn->bd', delta, vv[:, :, -1])
    length = np.linalg.norm(u, axis=1)
    bad = length <= np.finfo(float).tiny
    u[bad] = 0; u[bad, 0] = 1; length[bad] = 1
    u /= length[:, None]
    piv = np.argmax(np.abs(u), axis=1)
    u *= np.where(u[np.arange(len(u)), piv] < 0, -1., 1.)[:, None]
    t = np.einsum('bnd,bd->bn', delta, u)
    residual = delta-t[..., None]*u[:, None]
    r2 = np.where(good, np.sum(residual**2, axis=2), 0.)
    lo = np.where(good, t, np.inf).min(axis=1)
    hi = np.where(good, t, -np.inf).max(axis=1)
    span2 = (hi-lo)**2
    worst2 = r2.max(axis=1)
    return dict(score=span2+4*worst2, span=np.sqrt(span2),
                worst=np.sqrt(worst2), center=center, direction=u)


def diameter2(points, valid=None):
    """Direct differences, not a cancellation-prone norm/dot identity."""
    z = np.asarray(points, dtype=np.float64)
    good = np.ones(z.shape[:2], bool) if valid is None else np.asarray(valid, bool)
    out = np.zeros(len(z))
    for i in range(z.shape[1]):
        for j in range(i):
            dd = np.sum((z[:, i]-z[:, j])**2, axis=1)
            out = np.maximum(out, np.where(good[:, i]&good[:, j], dd, 0.))
    return out


def constrained_pair(points, full_points, budgets2, *, shortlist=4):
    """Propose balanced pages inside a 16-point pool.

    Candidate generation includes bounded axial neighborhoods, compact joint
    axial/residual assignments and full-space nearest-neighbor subsets. Reject
    diameter-infeasible proposals BEFORE ranking. Freely refit a small shortlist,
    accepting only a strict decrease in span^2+4*max_residual^2. Identity fallback.
    budgets2 belong to fixed original page slots, not the previous iteration.
    """
    z, x = np.asarray(points, float), np.asarray(full_points, float)
    b = len(z); cap = z.shape[1]//2
    if z.ndim != 3 or x.shape[:2] != z.shape[:2] or z.shape[1] != 16:
        raise ValueError('requires sixteen-point pools')
    budgets2 = np.asarray(budgets2, float)
    if budgets2.shape != (b, 2) or np.any(budgets2 < 0) or shortlist < 1:
        raise ValueError('invalid budget or shortlist')
    gg = geometry(z.reshape(b*2, cap, -1))
    a = gg['center'].reshape(b, 2, -1)
    u = gg['direction'].reshape(b, 2, -1)
    delta = z[:, None]-a[:, :, None]
    t = np.einsum('bknd,bkd->bkn', delta, u)
    rho2 = np.sum((delta-t[..., None]*u[:, :, None])**2, axis=3)
    # Short neighborhoods, instead of residual-only selection over the whole pool.
    candidates = [np.broadcast_to(np.arange(cap), (b, cap))]
    for axis in range(2):
        order = np.argsort(t[:, axis], axis=1, kind='stable')
        candidates.extend(order[:, j:j+cap] for j in range(cap+1))
    cost = t*t+4*rho2
    candidates.append(np.argsort(cost[:, 0]-cost[:, 1], axis=1, kind='stable')[:, :cap])
    centers = x.reshape(b, 2, cap, -1).mean(axis=2)
    dc = np.sum((x[:, :, None]-centers[:, None])**2, axis=3)
    candidates.append(np.argsort(dc[:, :, 0]-dc[:, :, 1], axis=1, kind='stable')[:, :cap])
    pairdist = np.sum((x[:, :, None]-x[:, None])**2, axis=3)
    for seed in range(16):
        candidates.append(np.argsort(pairdist[:, seed], axis=1, kind='stable')[:, :cap])
    left = np.stack(candidates, axis=1)
    count = left.shape[1]
    mask = np.zeros((b, count, 16), bool)
    np.put_along_axis(mask, left, True, axis=2)
    right = np.argsort(mask, axis=2, kind='stable')[:, :, :cap]
    both = np.stack((left, right), axis=2)
    # Full-space diameter feasibility applies to each page, not the pair sum.
    diam = np.zeros((b, count, 2))
    rr = np.arange(b)[:, None, None]
    for i in range(cap):
        for j in range(i):
            diam = np.maximum(diam, pairdist[rr, both[..., i], both[..., j]])
    feasible = np.all(diam <= budgets2[:, None]*(1+1e-12)+1e-10, axis=2)
    approx = np.zeros((b, count))
    for slot, subset in enumerate((left, right)):
        tt = np.take_along_axis(t[:, slot, None], subset, axis=2)
        r2 = np.take_along_axis(rho2[:, slot, None], subset, axis=2)
        approx += (tt.max(axis=2)-tt.min(axis=2))**2+4*r2.max(axis=2)
    approx[~feasible] = np.inf
    chosen = np.argsort(approx, axis=1, kind='stable')[:, :min(shortlist, count)]
    selected = both[np.arange(b)[:, None], chosen]
    gathered = z[np.arange(b)[:, None, None, None], selected]
    new = geometry(gathered.reshape(-1, cap, z.shape[2]))['score'].reshape(b, -1, 2).sum(axis=2)
    ok = feasible[np.arange(b)[:, None], chosen]
    new[~ok] = np.inf
    old = gg['score'].reshape(b, 2).sum(axis=1)
    best = np.argmin(new, axis=1)
    best_score = new[np.arange(b), best]
    use = best_score < old-1e-10*np.maximum(old, 1.)
    order = np.broadcast_to(np.arange(16), (b, 16)).copy()
    order[use] = selected[np.arange(b), best].reshape(b, 16)[use]
    return order, use, dict(candidates=int(b*count), feasible=int(feasible.sum()),
                           score_decrease=float(np.sum(old[use]-best_score[use])))


def rounds():
    """Seven disjoint rounds visit all 28 page pairs within an eight-page pool."""
    ring = list(range(8))
    result = []
    for _ in range(7):
        result.append([(ring[i], ring[7-i]) for i in range(4)])
        ring = [ring[0], ring[-1], *ring[1:-1]]
    return result


def layout_metrics(index, x, projected, dims=64):
    values = {k: [] for k in ('diameter', 'span', 'worst', 'score')}
    for s in range(0, index.meta['n_pages'], 1024):
        pp = np.arange(s, min(s+1024, index.meta['n_pages']))
        ids = index.page_ids[pp]
        good = np.arange(index.meta['capacity'])[None] < index.valid[pp, None]
        g = geometry(projected[np.maximum(ids, 0), :dims], good)
        values['diameter'].append(np.sqrt(diameter2(x[np.maximum(ids, 0)], good)))
        for k in ('span', 'worst', 'score'):
            values[k].append(g[k])
    return {k: np.concatenate(v) for k, v in values.items()}


def compact_layout(source, out, x, projected, *, dims=64, relaxation=0., passes=1):
    source, out = Path(source), Path(out)
    base = Index(source); m = base.meta; cap = m['capacity']
    if cap != 8 or not m.get('physical_layout_frozen') or not 0 <= relaxation <= .25 or passes < 1:
        raise ValueError('invalid frozen layout or compactness parameters')
    if x.shape != (m['n'], m['d']) or projected.shape != x.shape or not 1 <= dims <= m['d']:
        raise ValueError('invalid data shapes')
    if digest(source/'vectors.pages') != m['layout_payload_sha256']:
        raise ValueError('source payload changed')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    before = layout_metrics(base, x, projected, dims)
    # Fixed per-slot bounds prevent cumulative slack across successive rounds.
    budget2 = (before['diameter']*(1+relaxation))**2
    blocks = []
    for first, last in base.ranges:
        end = int(last)-(int(last)>int(first) and base.valid[int(last)-1] < cap)
        blocks.extend((p, min(8, end-p)) for p in range(int(first), end, 8))
    ids = base.page_ids.copy(); progress = []
    for sweep in range(passes):
        for pairs in rounds():
            addresses = [(p+i, p+j) for p, n in blocks for i, j in pairs if i<n and j<n]
            accepted = 0; feasible = 0; candidates = 0; gain = 0.
            for start in range(0, len(addresses), 128):
                pp = np.asarray(addresses[start:start+128], dtype=np.int64)
                if not len(pp): continue
                pool = ids[pp].reshape(-1, 16)
                order, use, stats = constrained_pair(projected[pool, :dims], x[pool], budget2[pp])
                proposal = np.take_along_axis(pool, order, axis=1).reshape(-1, 2, cap)
                ids[pp[use, 0]] = proposal[use, 0]
                ids[pp[use, 1]] = proposal[use, 1]
                accepted += int(use.sum()); feasible += stats['feasible']
                candidates += stats['candidates']; gain += stats['score_decrease']
            progress.append(dict(accepted_pairs=accepted, feasible_candidates=feasible,
                                 candidates=candidates, score_decrease=gain))
            print(json.dumps(dict(compact_relaxation=relaxation, round=len(progress), **progress[-1])), flush=True)
    for first, last in base.ranges:
        np.testing.assert_array_equal(np.sort(ids[first:last].ravel()), np.sort(base.page_ids[first:last].ravel()))
    with np.load(source/'directory.npz', allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    arrays['page_ids'] = ids
    with (out/'vectors.pages').open('wb') as f:
        for li, (first, last) in enumerate(base.ranges):
            for p in range(int(first), int(last)):
                group = ids[p, :base.valid[p]]
                payload = x[group].astype('<f4', copy=False).tobytes()
                f.write(payload+bytes(m['page_size']-len(payload)))
                r = np.linalg.norm(x[group].astype(float)-base.centers[li], axis=1)
                arrays['radial'][p] = [np.nextafter(np.float32(r.min()), np.float32(-np.inf)),
                                       np.nextafter(np.float32(r.max()), np.float32(np.inf))]
    arrays['codes'][:] = 0; arrays['radii'][:] = -1; arrays['radii'][:, 0] = np.inf
    np.savez(out/'directory.npz', **arrays)
    meta = dict(m, layout=f'compact-segment-{relaxation:g}',
                layout_payload_sha256=digest(out/'vectors.pages'),
                layout_directory_sha256=digest(out/'directory.npz'))
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    after = layout_metrics(Index(out), x, projected, dims)
    tolerance = 1e-8*np.maximum(1., before['diameter'])
    if np.any(after['diameter'] > before['diameter']*(1+relaxation)+tolerance):
        raise AssertionError('fixed full-space diameter budget violated')
    if after['score'].sum() > before['score'].sum()*(1+1e-10):
        raise AssertionError('refitted segment objective increased')
    np.savez(out/'compact-metrics.npz', **{'before_'+k:v for k,v in before.items()},
             **{'after_'+k:v for k,v in after.items()})
    meta['refinement'] = dict(dims=dims, passes=passes, diameter_relaxation=relaxation,
                             hard_diameter_space='original full-dimensional FP32 vectors',
                             objective='span_squared + 4*max_perpendicular_squared',
                             before={k:float(v.mean()) for k,v in before.items()},
                             after={k:float(v.mean()) for k,v in after.items()}, rounds=progress,
                             changed_membership_pages=int(np.any(np.sort(ids, axis=1)!=np.sort(base.page_ids, axis=1), axis=1).sum()),
                             full_space_diameter_violations=0, query_fitted=False, global_optimum=False)
    (out/'manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
    return meta
