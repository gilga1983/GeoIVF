from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .index import Index, vectors, train_faiss, build_assigned
from .io import MemoryReplay, Pread, Native, PooledNative
from .search import search, Trace
from .mqsim import export


def main():
    ap = argparse.ArgumentParser(description='GeoIVF research harness')
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--base', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--nlist', type=int, default=64)
    b.add_argument('--train-size', type=int, default=100000)
    b.add_argument('--threads', type=int, default=1)
    b.add_argument('--seed', type=int, default=12345)
    b.add_argument('--layout', choices=['input', 'radial', 'geopack'], default='geopack')
    b.add_argument('--dims', type=int, default=16)
    b.add_argument('--balls', type=int, default=2)
    b.add_argument('--page-size', type=int, default=4096)
    s = sub.add_parser('search')
    s.add_argument('--index', required=True)
    s.add_argument('--queries', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--backend', choices=['replay', 'pread', 'native', 'uring'], default='pread')
    s.add_argument('--direct', action='store_true')
    s.add_argument('--summary', help='Certified independent PCA sidecar with cells.json')
    s.add_argument('--shape', choices=['ball','box','hybrid'], default='ball')
    s.add_argument('--selection', choices=['bounds','fixed','adaptive'], default='bounds')
    s.add_argument('--scan', choices=['python','native'], default='python')
    s.add_argument('--pooled', action='store_true', help='Borrow reusable native read buffers')
    s.add_argument('--queue-depth', type=int, default=16)
    s.add_argument('--filter', choices=['none', 'radial', 'balls', 'combined'], default='combined')
    s.add_argument('--nprobe', type=int, default=8)
    s.add_argument('--k', type=int, default=10)
    s.add_argument('--window-pages', type=int, default=64)
    s.add_argument('--gap-pages', type=int, default=0)
    s.add_argument('--max-extent-pages', type=int, default=256)
    s.add_argument('--max-queries', type=int, default=0)
    s.add_argument('--trace', action='store_true')
    e = sub.add_parser('export-mqsim')
    e.add_argument('--trace', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--ssd-config', required=True)
    e.add_argument('--request-gap-ns', required=True, type=int)
    args = ap.parse_args()
    if args.command == 'build':
        import faiss
        x = vectors(args.base)
        centers, labels = train_faiss(x, args.nlist, args.seed, args.train_size, args.threads)
        result = build_assigned(x, centers, labels, args.out, dims=args.dims, balls=args.balls,
                               layout=args.layout, page_size=args.page_size,
                               provenance=dict(builder='faiss.Kmeans+IndexFlatL2',
                                               faiss_version=faiss.__version__, seed=args.seed))
    elif args.command == 'export-mqsim':
        result = export(args.trace, args.out, request_gap_ns=args.request_gap_ns,
                        ssd_config=args.ssd_config)
    else:
        import faiss
        faiss.omp_set_num_threads(1)
        if args.summary:
            from .cells import CellIndex
            index = CellIndex(args.index, args.summary, shape=args.shape)
        else:
            index = Index(args.index)
        if args.selection != 'bounds' and args.filter != 'none' and not args.summary:
            ap.error('--selection requires a certified --summary sidecar')
        if args.pooled and args.backend not in ('native','uring'):
            ap.error('--pooled requires native or uring')
        q = vectors(args.queries)
        if args.max_queries < 0:
            ap.error('--max-queries must be nonnegative')
        if args.max_queries:
            q = q[:args.max_queries]
        if args.direct and args.backend not in ('native', 'uring'):
            ap.error('--direct requires native or uring (no silent fallback)')
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'refusing to overwrite results: {out}')
        out.mkdir(parents=True, exist_ok=True)
        if args.backend == 'replay':
            reader = MemoryReplay(index.path/'vectors.pages')
        elif args.backend == 'pread':
            reader = Pread(index.path/'vectors.pages')
        else:
            reader_type = PooledNative if args.pooled else Native
            reader = reader_type(index.path/'vectors.pages', direct=args.direct,
                            uring=args.backend == 'uring', depth=args.queue_depth)
        trace = Trace(out/'requests.jsonl') if args.trace else None
        rows, answers = [], []
        try:
            for qi, query in enumerate(q):
                ids, stats = search(index, query, reader, k=args.k, nprobe=args.nprobe,
                                    filtering=args.filter, window_pages=args.window_pages,
                                    gap_pages=args.gap_pages, max_extent_pages=args.max_extent_pages,
                                    trace=trace, qid=qi, selection=args.selection, scan=args.scan)
                padded = np.full(args.k, -1, dtype=np.int64)
                padded[:len(ids)] = ids
                answers.append(padded)
                rows.append(dict(query_id=qi, **stats))
        finally:
            reader.close()
            if trace:
                trace.close()
        np.save(out/'neighbors.npy', np.asarray(answers))
        with (out/'queries.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = dict(backend=reader.mode, queries=len(rows), options=vars(args),
                      index=index.meta, numpy_version=np.__version__, faiss_version=faiss.__version__,
                      mean={key: float(np.mean([r[key] for r in rows])) for key in rows[0]
                            if key != 'query_id'},
                      timing_scope='Online staged search with selected scan/selection backend; Python coordinator',
                      io_scope='requested extents/bytes, not measured NVMe commands',
                      payload_backend_real_reads=args.backend != 'replay',
                      device_latency_measured=False)
        (out/'run.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
