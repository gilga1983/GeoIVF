#!/usr/bin/env python3
"""Reproducible integration fixture, never a substitute for canonical ANN data."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np
from geoivf.index import build_assigned, train_faiss, Index
from geoivf.io import Pread
from geoivf.search import search, Trace

p = argparse.ArgumentParser()
p.add_argument('--out', default='artifacts/smoke')
p.add_argument('--seed', type=int, default=12345)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
faiss.omp_set_num_threads(1)
rng = np.random.default_rng(a.seed)
prototypes = rng.normal(0, 4, (32, 128))
x = (prototypes[rng.integers(32, size=4096)] + rng.normal(size=(4096, 128))).astype('float32')
queries = (prototypes[rng.integers(32, size=32)] + rng.normal(size=(32, 128))).astype('float32')
centers, labels = train_faiss(x, 16, a.seed, 4096)
coarse = faiss.IndexFlatL2(128); coarse.add(centers)
baseline = faiss.IndexIVFFlat(coarse, 128, 16, faiss.METRIC_L2)
baseline.is_trained = True; baseline.add(x); baseline.nprobe = 8
expected = baseline.search(queries, 10)[1]
results = []
for layout in ['input', 'radial', 'geopack']:
    root = out/layout
    manifest = build_assigned(x, centers, labels, root, layout=layout, dims=16, balls=8,
                             provenance={'dataset': 'synthetic-integration-fixture', 'seed': a.seed,
                                         'faiss_version': faiss.__version__})
    index = Index(root)
    for filtering in ['none', 'combined']:
        trace = Trace(out/f'{layout}-{filtering}.jsonl')
        reader = Pread(root/'vectors.pages')
        rows = []
        try:
            for qi, q in enumerate(queries):
                ids, stats = search(index, q, reader, nprobe=8, filtering=filtering,
                                    window_pages=32, trace=trace, qid=qi)
                np.testing.assert_array_equal(ids, expected[qi])
                rows.append(stats)
        finally:
            trace.close(); reader.close()
        results.append(dict(layout=layout, filtering=filtering, correct_queries=len(rows),
                            pages=float(np.mean([r['read_pages'] for r in rows])),
                            requests=float(np.mean([r['read_requests'] for r in rows])),
                            summary_bytes=manifest['summary_bytes']))
report = dict(dataset='synthetic-integration-fixture-not-performance-evidence',
              base_vectors=len(x), queries=len(queries), seed=a.seed,
              numpy_version=np.__version__, faiss_version=faiss.__version__, results=results)
(out/'smoke.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
