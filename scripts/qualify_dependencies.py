#!/usr/bin/env python3
"""Dependent centers on canonical frozen SIFT1M pages. No SSD timing claims."""
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
import numpy as np
import faiss
from geoivf.index import Index, vectors, train_faiss
from geoivf.layouts import freeze_layout
from geoivf.projections import Basis
from geoivf.dependent import DependentIndex, summarize_dependencies, file_hash
from geoivf.search import search
from geoivf.io import MemoryReplay
from scripts.qualify_sift1m import reference, audit_rejections


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path.home()/'.cache/geoivf/datasets/sift1m')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--queries', type=int, default=64)
    ap.add_argument('--nprobe', type=int, default=64)
    args = ap.parse_args()
    if not 1 <= args.queries <= 1000 or not 1 <= args.nprobe <= 1024:
        ap.error('invalid development sample/nprobe')
    for p in (args.work, args.out):
        if p.exists() and any(p.iterdir()): raise FileExistsError(p)
        p.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1)
    started = time.monotonic()
    expected = '21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816'
    if file_hash(args.data/'sift_base.fvecs') != expected:
        raise ValueError('not the qualified canonical SIFT1M base')
    x = vectors(args.data/'sift_base.fvecs')
    allq = vectors(args.data/'sift_query.fvecs')
    learn = vectors(args.data/'sift_learn.fvecs')
    gt = np.memmap(args.data/'sift_groundtruth.ivecs', dtype='<i4', mode='r', shape=(10000, 101))[:, 1:]
    qids = np.random.default_rng(20260929).permutation(10000)[:args.queries]
    queries = allq[qids]
    print('Train common IVF, learn PCA, freeze previous GeoPack layout', flush=True)
    centers, labels = train_faiss(x, 1024, 12345, 100000)
    quantizer = faiss.IndexFlatL2(128); quantizer.add(centers)
    routes = quantizer.search(queries, args.nprobe)[1]
    refs = [reference(x, labels, q, routes[qi]) for qi, q in enumerate(queries)]
    np.savez(args.out/'shared-candidates.npz', query_ids=qids, lists=routes, centers=centers)
    basis = Basis.fit(learn, 'pca'); del learn
    basis.save(args.out/'pca-basis.npz')
    projected = np.lib.format.open_memmap(args.work/'projection.npy', mode='w+',
                                         dtype=np.float64, shape=x.shape)
    for s in range(0, len(x), 32768): projected[s:s+32768] = basis.project(x[s:s+32768])
    projected.flush()
    physical = args.work/'geopack'
    frozen = freeze_layout(x, centers, labels, physical, layout='geopack', pack_dims=16)
    # This test must isolate center encoding, not layout or basis changes.
    if frozen['layout_payload_sha256'] != '8f1da77ff41f7e37c89fc3449ef6ba18a1901a50a394245a10d6b882fdee97e7':
        raise AssertionError('frozen layout differs from preceding experiment')
    base = Index(physical)
    reader = MemoryReplay(physical/'vectors.pages')
    # Controlled byte ladder; identifier and residual-scale overhead is included.
    variants = [
        ('pca16-independent8', 16, 'independent', 8, 8, 0),
        ('pca32-independent8', 32, 'independent', 8, 8, 0),
        ('pca64-independent8', 64, 'independent', 8, 8, 0),
        ('pca64-independent4', 64, 'independent', 8, 4, 0),
        ('pca64-independent6', 64, 'independent', 8, 6, 0),
        ('pca64-midpoint2', 64, 'midpoint', 2, 8, 0),
        ('pca64-midpoint4', 64, 'midpoint', 4, 8, 0),
        ('pca64-midpoint6', 64, 'midpoint', 6, 8, 0),
        ('pca64-interpolate4', 64, 'interpolate', 4, 8, 0),
        ('pca64-interpolate6', 64, 'interpolate', 6, 8, 0),
        ('pca64-interpolate4-res2', 64, 'interpolate', 4, 8, 2),
        ('pca64-interpolate4-res4', 64, 'interpolate', 4, 8, 4)]
    rows, builds = [], []
    audit_total = 0
    for name, dims, scheme, anchors, bits, rb in [('none', 0, '', 0, 0, 0)]+variants:
        info = None
        if name == 'none':
            index = base
        else:
            print('Building '+name, flush=True)
            t = time.monotonic()
            info = summarize_dependencies(physical, args.work/name, basis, dims=dims,
                scheme=scheme, anchors=anchors, bits=bits, residual_bits=rb, projected_by_id=projected)
            info['name'] = name; info['build_seconds'] = time.monotonic()-t
            builds.append(info)
            save(args.out/(name+'-encoding.json'), info)
            index = DependentIndex(physical, args.work/name)
        records = []; audits = 0
        for qi, q in enumerate(queries):
            decisions = [] if qi < 2 and name != 'none' else None
            ids, stat = search(index, q, reader, nprobe=args.nprobe,
                              filtering='none' if name == 'none' else 'combined',
                              window_pages=0 if name == 'none' else 64,
                              preassigned_lists=routes[qi], audit_sink=decisions)
            np.testing.assert_array_equal(ids, refs[qi])
            if decisions: audits += audit_rejections(index, x, q, decisions)
            recall = len(set(map(int, ids)) & set(map(int, gt[qids[qi], :10])))/10
            records.append(dict(query_id=int(qids[qi]), recall_at_10=recall,
                                saved_fraction=1-stat['read_pages']/stat['candidate_pages'], **stat))
        audit_total += audits
        with (args.out/(name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
        row = dict(name=name, nprobe=args.nprobe, independent_queries=len(queries),
            exact_matches=len(queries), audited_rejections=audits,
            geometry_bytes_per_page=info['bytes_per_page'] if info else 0,
            directory_array_bytes=info['directory_array_bytes'] if info else None,
            radius_error_mean=info['radius_error_mean'] if info else None,
            radius_error_median=info['radius_error_median'] if info else None,
            radius_error_p95=info['radius_error_p95'] if info else None,
            means={k: float(np.mean([r[k] for r in records])) for k in records[0] if k != 'query_id'},
            median_saved_fraction=float(np.median([r['saved_fraction'] for r in records])),
            queries_saving_zero=sum(r['saved_fraction'] == 0 for r in records))
        rows.append(row); save(args.out/'partial-results.json', rows)
        print(json.dumps(dict(name=name, bytes_per_page=row['geometry_bytes_per_page'],
            pages=row['means']['read_pages'], extents=row['means']['read_requests'],
            radius=row['radius_error_mean'])), flush=True)
        if name != 'none': del index
    reader.close()
    if file_hash(physical/'vectors.pages') != frozen['layout_payload_sha256']:
        raise AssertionError('payload changed')
    save(args.out/'dependency-results.json', dict(
        dataset=json.loads((args.data/'dataset.json').read_text()), independent_queries=len(queries),
        query_ids=qids.tolist(), query_split='development-only', seed=12345,
        nlist=1024, nprobe=args.nprobe, k=10,
        layout_payload_sha256=frozen['layout_payload_sha256'],
        basis_sha256=basis.fingerprint(),
        candidate_routes_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
        configuration_query_matches=len(rows)*len(queries), audited_rejection_decisions=audit_total,
        builds=builds, results=rows, elapsed_seconds=time.monotonic()-started,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        numpy=np.__version__, faiss=faiss.__version__, python=platform.python_version(),
        storage_latency_valid=False, production_speedup_valid=False,
        limitations=['one dataset/seed and development sample only',
            'RAM replay counts, not SSD traffic or speedup',
            'midpoint/interpolation data-point anchor subsets optimize center SSE, not page false positives',
            'residual variants reuse interpolation-optimized anchors, not residual-optimal selection',
            'prototype decoding overhead, not optimized native throughput',
            'all radii use four bytes; all identifiers, residual scales and padding counted',
            'no CLIP result; radial bound is deterministic reverse triangle',
            'numerical guards not a formally verified library']))
    print(f'PASS: {len(rows)*len(queries)} matches, {audit_total} rejection decisions audited', flush=True)

if __name__ == '__main__': main()
