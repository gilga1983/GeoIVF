#!/usr/bin/env python3
"""PCA/tail-norm ablation on canonical SIFT1M; counts, not storage timings."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import Index, vectors, train_faiss
from geoivf.layouts import freeze_layout, summarize
from geoivf.projections import Basis, ProjectedIndex, summarize_projection, digest_file
from geoivf.search import search
from geoivf.io import MemoryReplay
from scripts.qualify_sift1m import reference, audit_rejections


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class SeededNoFilter:
    """Zero bounds with the SAME seed/window scheduling as filtered search."""
    def __init__(self, index): self.index = index
    def __getattr__(self, key): return getattr(self.index, key)
    def bounds(self, q, pages, li, mode): return np.zeros(len(pages))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--queries', type=int, default=64)
    ap.add_argument('--nprobe', type=int, nargs='+', default=[16, 64])
    ap.add_argument('--seed', type=int, default=12345)
    args = ap.parse_args()
    if not 1 <= args.queries <= 1000 or not args.nprobe or not 1 <= min(args.nprobe) <= max(args.nprobe) <= 1024:
        ap.error('invalid development query count or nprobe')
    for p in (args.work, args.out):
        if p.exists() and any(p.iterdir()): raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1)
    start_time = time.monotonic()
    dataset = json.loads((args.data/'dataset.json').read_text())
    # Lock to the previously validated canonical corpus, not just its shape.
    expected = '21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
    if digest_file(args.data/'sift_base.fvecs') != expected:
        raise ValueError('base hash differs from canonical qualification')
    x = vectors(args.data/'sift_base.fvecs')
    allq = vectors(args.data/'sift_query.fvecs')
    learn = vectors(args.data/'sift_learn.fvecs')
    if x.shape != (1000000, 128) or allq.shape != (10000, 128) or learn.shape != (100000, 128):
        raise ValueError('wrong canonical dataset shape')
    gt = np.memmap(args.data/'sift_groundtruth.ivecs', dtype='<i4', mode='r', shape=(10000, 101))[:, 1:]
    qids = np.random.default_rng(20260929).permutation(10000)[:1000][:args.queries]
    queries = allq[qids]
    print('Fit shared IVF and independent-learning-file PCA', flush=True)
    centers, labels = train_faiss(x, 1024, args.seed, 100000)
    quantizer = faiss.IndexFlatL2(128); quantizer.add(centers)
    routes = quantizer.search(queries, max(args.nprobe))[1]
    np.savez(args.out/'shared-candidates.npz', query_ids=qids, lists=routes, centers=centers)
    refs = {(npb, qi): reference(x, labels, q, routes[qi, :npb])
            for npb in args.nprobe for qi, q in enumerate(queries)}
    basis = Basis.fit(learn, 'pca'); del learn
    basis.save(args.out/'pca-basis.npz')
    projected = np.lib.format.open_memmap(args.work/'projection-scratch.npy', mode='w+', dtype=np.float64, shape=x.shape)
    for s in range(0, len(x), 32768): projected[s:s+32768] = basis.project(x[s:s+32768])
    projected.flush()
    rows, builds, indices = [], [], {}
    # Freeze geometry exactly as in the preceding qualification, not in PCA space.
    physical = args.work/'geopack-physical'
    frozen = freeze_layout(x, centers, labels, physical, layout='geopack', pack_dims=16)
    reader = MemoryReplay(physical/'vectors.pages')
    legacy_dir = args.work/'legacy-c16'
    legacy_meta = summarize(physical, legacy_dir, dims=16, balls=8)
    legacy = Index(legacy_dir); indices['coordinates16-u8'] = legacy
    builds.append(dict(name='coordinates16-u8', **legacy_meta))
    variants = [('pca16-u8', 16, False, 'uint8'),
                ('pca16-f32', 16, False, 'float32'),
                ('pca32-u8', 32, False, 'uint8'),
                ('pca64-u8', 64, False, 'uint8'),
                ('pca8-tail-u8', 8, True, 'uint8'),
                ('pca16-tail-u8', 16, True, 'uint8')]
    for name, dims, tail, precision in variants:
        print('Build '+name, flush=True)
        path = args.work/name; t = time.monotonic()
        meta = summarize_projection(physical, path, basis, dims=dims, balls=8,
                                   tail=tail, center_dtype=precision, projected_by_id=projected)
        builds.append(dict(name=name, build_seconds=time.monotonic()-t, **meta))
        indices[name] = ProjectedIndex(physical, path)
    comparisons = 0; audited = 0

    def evaluate(name, idx, npb, *, mode='combined', window=64, gap=0, count=None,
                 category='representation', zero_bounds=False):
        nonlocal comparisons, audited
        count = min(count or len(queries), len(queries))
        used = SeededNoFilter(idx) if zero_bounds else idx
        records = []; checked_pages = 0
        for qi, q in enumerate(queries[:count]):
            audit = [] if qi < 2 else None
            ids, stat = search(used, q, reader, nprobe=npb, filtering=mode,
                               window_pages=window, gap_pages=gap,
                               preassigned_lists=routes[qi, :npb], audit_sink=audit)
            np.testing.assert_array_equal(ids, refs[npb, qi])
            if audit: checked_pages += audit_rejections(idx, x, q, audit)
            recall = len(set(map(int, ids)) & set(map(int, gt[qids[qi], :10])))/10
            saved = stat['candidate_pages']-stat['read_pages']
            records.append(dict(query_id=int(qids[qi]), recall_at_10=recall,
                                saved_fraction=saved/max(1, stat['candidate_pages']), **stat))
        tag = f'{category}-{name}-np{npb}-w{window}-g{gap}'
        with (args.out/(tag+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        savings = np.array([r['candidate_pages']-r['read_pages'] for r in records])
        row = dict(name=name, category=category, nprobe=npb, window=window, gap=gap,
                   independent_queries=count, exact_matches=count, audited_rejections=checked_pages,
                   summary_bytes=idx.meta['summary_bytes'],
                   directory_array_bytes=idx.meta['directory_array_bytes'],
                   means={k: float(np.mean([r[k] for r in records])) for k in records[0] if k != 'query_id'},
                   median_saved_fraction=float(np.median([r['saved_fraction'] for r in records])),
                   queries_saving_zero=int(np.sum(savings == 0)),
                   largest_query_share_of_savings=float(savings.max()/max(1, savings.sum())), csv=tag+'.csv')
        rows.append(row); comparisons += count; audited += checked_pages
        save(args.out/'partial-results.json', rows)
        print(json.dumps(dict(name=tag, pages=row['means']['read_pages'],
                              extents=row['means']['read_requests'], median_saving=row['median_saved_fraction'])), flush=True)
        return row

    for npb in args.nprobe:
        evaluate('none-whole-list', legacy, npb, mode='none', window=0)
        evaluate('none-seeded-w64', legacy, npb, zero_bounds=True)
        for name, idx in indices.items(): evaluate(name, idx, npb)

    # Offline headroom analysis only: no oracle threshold is fed to online search.
    diagnostics = []
    npb = max(args.nprobe)
    for qi, q in enumerate(queries):
        ids = refs[npb, qi]
        tau = float(np.linalg.norm(x[ids[-1]].astype(float)-q))
        qz = basis.project(q)
        counts = {m: 0 for m in (16, 32, 64, 128)}; total = 0
        for li in routes[qi, :npb]:
            first, last = map(int, legacy.ranges[li]); total += last-first
            for s in range(first, last, 1024):
                end = min(last, s+1024); pid = legacy.page_ids[s:end]
                z = np.asarray(projected[np.maximum(pid, 0)])
                cumulative = np.cumsum((z-qz)**2, axis=2)
                for m in counts:
                    distances = np.sqrt(cumulative[:, :, m-1]); distances[pid < 0] = np.inf
                    counts[m] += int(np.sum(distances.min(axis=1) > tau+1e-5))
        diagnostics.append(dict(query_id=int(qids[qi]), nprobe=npb, final_ivf_tau=tau,
                                candidate_pages=total, ideal_unquantized_head_rejections=counts))
    save(args.out/'offline-headroom.json', dict(scope='optimistic-final-threshold-not-an-online-algorithm',
                                               observations=diagnostics))

    # Matched seed/window controls. No storage-latency measurements here.
    # Parameter choices below are explicit development diagnostics, not held-out tuning.
    for window in (16, 256, 0):
        evaluate('none-seeded', legacy, npb, window=window, zero_bounds=True,
                 count=16, category='schedule')
        for name in ('pca16-u8', 'pca64-u8'):
            evaluate(name, indices[name], npb, window=window, count=16, category='schedule')
    for gap in (1, 2):
        for name in ('pca16-u8', 'pca64-u8'):
            evaluate(name, indices[name], npb, gap=gap, count=16, category='schedule')
    reader.close()
    if digest_file(physical/'vectors.pages') != frozen['layout_payload_sha256']:
        raise AssertionError('payload mutated during evaluation')
    report = dict(dataset=dataset, independent_queries=len(queries), query_ids=qids.tolist(),
                  query_split='development-only', seed=args.seed, nlist=1024,
                  candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                  layout_payload_sha256=frozen['layout_payload_sha256'],
                  projection_basis_sha256=basis.fingerprint(),
                  pca_fit_source='canonical 100000-vector sift_learn.fvecs; no evaluation queries',
                  pca_normalization=basis.normalization,
                  pca_explained_fraction=basis.explained_fraction.tolist(),
                  configuration_query_matches=comparisons, audited_rejection_decisions=audited,
                  builds=builds, results=rows, elapsed_seconds=time.monotonic()-start_time,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  numpy=np.__version__, faiss=faiss.__version__, python=platform.python_version(),
                  storage_latency_valid=False, production_speedup_valid=False,
                  limitations=['RAM payload replay, not SSD traffic', 'no CLIP performance comparison',
                               'PCA full rotation and Python loops are not optimized',
                               '64 development queries, not full held-out evaluation',
                               'engineering numerical guards, not formally verified arithmetic',
                               'larger summaries and tail bounds have explicitly different memory costs'])
    save(args.out/'projection-results.json', report)
    print(f'PASS: {comparisons} configuration-query matches; {audited} rejection decisions audited', flush=True)

if __name__ == '__main__': main()
