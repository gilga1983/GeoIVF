#!/usr/bin/env python3
"""Canonical-data qualification: real data, measured plans, NOT SSD timings."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import faiss
from geoivf.index import Index, vectors, train_faiss
from geoivf.layouts import freeze_layout, summarize
from geoivf.search import search
from geoivf.io import MemoryReplay


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def reference(x, labels, q, lists, k=10):
    ids = np.flatnonzero(np.isin(labels, lists))
    a = x[ids].astype(np.float64)-q
    dd = np.einsum('ij,ij->i', a, a)
    # Full ID tie-break, independent of search order and Faiss float32 ties.
    order = np.lexsort((ids, dd))[:k]
    return ids[order]


def audit_rejections(index, x, query, decisions):
    for start in range(0, len(decisions), 1024):
        part = decisions[start:start+1024]
        pp = np.array([p for p, _, _ in part], dtype=np.int64)
        ids = index.page_ids[pp]
        mask = np.arange(index.meta['capacity'])[None, :] < index.valid[pp, None]
        delta = x[np.maximum(ids, 0)].astype(np.float64)-query
        exact = np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))
        minimum = np.where(mask, exact, np.inf).min(axis=1)
        taus = np.array([t for _, t, _ in part]); lb = np.array([v for _, _, v in part])
        if not np.all(minimum > taus) or not np.all(lb <= minimum+1e-9):
            raise AssertionError('unsafe page rejection or invalid lower bound')
    return len(decisions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--nlist', type=int, default=1024)
    ap.add_argument('--nprobe', type=int, nargs='+', default=[4, 8, 16, 32, 64, 128, 256])
    ap.add_argument('--queries', type=int, default=32)
    ap.add_argument('--split', choices=['development', 'heldout'], default='development')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--audit-queries', type=int, default=2)
    args = ap.parse_args()
    if not 1 <= args.queries <= (1000 if args.split == 'development' else 9000):
        ap.error('query count exceeds the selected disjoint split')
    if not args.nprobe or min(args.nprobe) < 1 or max(args.nprobe) > args.nlist:
        ap.error('nprobe outside the index range')
    for path in [args.out, args.work]:
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f'refusing to overwrite nonempty run: {path}')
        path.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((args.data/'dataset.json').read_text())
    faiss.omp_set_num_threads(1)
    started = time.monotonic()
    print('Loading canonical SIFT1M', flush=True)
    x = vectors(args.data/'sift_base.fvecs')
    all_q = vectors(args.data/'sift_query.fvecs')
    if x.shape != (1000000, 128) or all_q.shape != (10000, 128):
        raise ValueError('not canonical SIFT1M shapes')
    gt = np.memmap(args.data/'sift_groundtruth.ivecs', dtype='<i4', mode='r', shape=(10000, 101))[:, 1:]
    perm = np.random.default_rng(20260929).permutation(len(all_q))
    pool = perm[:1000] if args.split == 'development' else perm[1000:]
    qids = pool[:args.queries]; queries = all_q[qids]
    print('Training shared IVF assignments', flush=True)
    centers, labels = train_faiss(x, args.nlist, args.seed, 100000)
    quantizer = faiss.IndexFlatL2(128); quantizer.add(centers)
    routes = quantizer.search(queries, max(args.nprobe))[1]
    np.savez(args.out/'shared-candidates.npz', query_ids=qids, lists=routes,
             centers=centers)
    refs = {(npb, qi): reference(x, labels, q, routes[qi, :npb])
            for npb in args.nprobe for qi, q in enumerate(queries)}
    # Check canonical ground truth independently on a small audit subset.
    global_audits = []
    for qi in range(min(2, len(queries))):
        ids = reference(x, np.zeros(len(x), dtype=np.int64), queries[qi], [0])
        truth = np.asarray(gt[qids[qi], :10], dtype=np.int64)
        mine = np.sort(np.sum((x[ids].astype(np.float64)-queries[qi])**2, axis=1))
        theirs = np.sort(np.sum((x[truth].astype(np.float64)-queries[qi])**2, axis=1))
        np.testing.assert_array_equal(mine, theirs)
        global_audits.append(dict(query_id=int(qids[qi]), top10_distances_agree=True,
                                  exact_ids_agree=bool(np.array_equal(ids, truth))))
    rows = []; builds = []; comparisons = 0; audited_pages = 0
    for layout in ['input', 'radial', 'geopack']:
        physical = args.work/(layout+'-physical')
        t = time.monotonic()
        meta = freeze_layout(x, centers, labels, physical, layout=layout, pack_dims=16,
                             provenance=dict(dataset=provenance, seed=args.seed))
        layout_seconds = time.monotonic()-t
        reader = MemoryReplay(physical/'vectors.pages')
        for dims in ([16, 4, 8] if layout == 'geopack' else [16]):
            path = args.work/f'{layout}-d{dims}-b8'
            t = time.monotonic(); sm = summarize(physical, path, dims=dims, balls=8)
            idx = Index(path)
            builds.append(dict(layout=layout, dims=dims, balls=8,
                               layout_seconds=layout_seconds,
                               summary_seconds=time.monotonic()-t,
                               payload_sha256=sm['layout_payload_sha256'],
                               summary_bytes=sm['summary_bytes'],
                               radial_bytes=sm['radial_bytes'],
                               allocated_directory_array_bytes=sm['directory_array_bytes']))
            modes = ['none', 'radial', 'balls', 'combined'] if dims == 16 else ['balls', 'combined']
            for mode in modes:
                for npb in args.nprobe:
                    items = []; match = 0; recall = []; checked_pages = 0
                    for qi, q in enumerate(queries):
                        audit = [] if qi < args.audit_queries else None
                        ids, stats = search(idx, q, reader, nprobe=npb, filtering=mode,
                                            window_pages=0 if mode == 'none' else 64,
                                            preassigned_lists=routes[qi, :npb], audit_sink=audit)
                        np.testing.assert_array_equal(ids, refs[npb, qi])
                        match += 1
                        recall.append(len(set(map(int, ids)) & set(map(int, gt[qids[qi], :10])))/10)
                        if audit:
                            checked_pages += audit_rejections(idx, x, q, audit)
                        items.append(dict(query_id=int(qids[qi]), recall_at_10=recall[-1], **stats))
                    name = f'{layout}-d{dims}-{mode}-np{npb}'
                    with (args.out/(name+'.csv')).open('w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=list(items[0])); w.writeheader(); w.writerows(items)
                    row = dict(layout=layout, dims=dims, balls=8, filtering=mode,
                               nprobe=npb, queries=len(queries), exact_matches=match,
                               recall_at_10=float(np.mean(recall)), audited_rejections=checked_pages,
                               summary_bytes=sm['summary_bytes'], radial_bytes=sm['radial_bytes'],
                               allocated_directory_array_bytes=sm['directory_array_bytes'],
                               means={k: float(np.mean([s[k] for s in items]))
                                      for k in items[0] if k not in ('query_id', 'recall_at_10')})
                    rows.append(row); comparisons += match; audited_pages += checked_pages
                    save_json(args.out/'partial-results.json', rows)
                    print(json.dumps(dict(name=name, recall=row['recall_at_10'],
                                          pages=row['means']['read_pages'],
                                          requests=row['means']['read_requests'], matches=match)), flush=True)
        reader.close()
    report = dict(dataset=provenance, seed=args.seed, nlist=args.nlist,
                  query_split=args.split, query_ids=qids.tolist(),
                  independent_queries=len(queries), configuration_query_matches=comparisons,
                  audited_page_decisions=audited_pages, global_groundtruth_audits=global_audits,
                  layouts=builds, results=rows, elapsed_seconds=time.monotonic()-started,
                  numpy=np.__version__, faiss=faiss.__version__, python=platform.python_version(),
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
                  metric_scope='algorithmic qualification with actual FP32 payload replay',
                  storage_latency_valid=False, production_speedup_valid=False,
                  limitations=['Python/NumPy execution', 'RAM replay, not NVMe measurement',
                               'development queries are not held-out performance evidence',
                               'radial is deterministic reverse triangle, not CLIP',
                               'total RSS includes dataset/oracle and is not deployed directory RAM'])
    save_json(args.out/'qualification.json', report)
    print(f'PASS: {comparisons} comparisons; {audited_pages} rejected pages audited', flush=True)

if __name__ == '__main__':
    main()
