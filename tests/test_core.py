import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from geoivf.index import build_assigned, Index, checked, vectors
from geoivf.io import MemoryReplay, Pread, Native
from geoivf.search import search, coalesce, Trace
from geoivf.mqsim import export


def fixture_data():
    rng = np.random.default_rng(20260929)
    centers = rng.normal(0, 8, (4, 128)).astype('float32')
    labels = np.arange(267) % 4
    x = (centers[labels]+rng.normal(size=(267, 128))).astype('float32')
    # An exact duplicate tests deterministic tie handling.
    x[4] = x[0]
    q = np.r_[x[:2], centers+rng.normal(size=(4, 128)).astype('float32')]
    return x, centers, labels, q


def oracle(index, x, q, k=10, nprobe=3):
    selected = index.route(q, nprobe, numpy_test=True)
    ids = []
    for li in selected:
        start, end = index.ranges[li]
        for p in range(start, end):
            ids.extend(index.page_ids[p, :index.valid[p]])
    dist = np.sum((x[ids].astype('float64')-q)**2, axis=1)
    return np.asarray([i for _, i in sorted(zip(dist, ids))[:k]])


@pytest.mark.parametrize('layout', ['input', 'radial', 'geopack'])
@pytest.mark.parametrize('filtering', ['none', 'radial', 'balls', 'combined'])
def test_readers_and_results(tmp_path, layout, filtering):
    x, centers, labels, queries = fixture_data()
    build_assigned(x, centers, labels, tmp_path/'index', layout=layout, balls=4)
    idx = Index(tmp_path/'index')
    path = idx.path/'vectors.pages'
    expected = []
    plans = []
    for cls in (MemoryReplay, Pread, Native):
        reader = cls(path)
        trace = Trace(tmp_path/f'{cls.__name__}.jsonl')
        try:
            current = []
            for qi, q in enumerate(queries):
                ids, stats = search(idx, q, reader, filtering=filtering, nprobe=3,
                                    window_pages=5, trace=trace, qid=qi, numpy_test=True)
                np.testing.assert_array_equal(ids, oracle(idx, x, q))
                assert stats['read_bytes'] == 4096*stats['read_pages']
                assert stats['read_pages'] <= stats['candidate_pages']
                current.append(ids.tolist())
        finally:
            trace.close()
            reader.close()
        records = [json.loads(s) for s in (tmp_path/f'{cls.__name__}.jsonl').read_text().splitlines()]
        plans.append([(r['query_id'], r['stage_id'], r['offset_bytes'], r['length_bytes'])
                      for r in records])
        expected.append(current)
    assert expected[0] == expected[1] == expected[2]
    assert plans[0] == plans[1] == plans[2]


@pytest.mark.parametrize('balls', [1, 2, 4, 8])
@pytest.mark.parametrize('dims', [4, 16, 128])
def test_bounds_cover_original_points(tmp_path, balls, dims):
    x, centers, labels, queries = fixture_data()
    meta = build_assigned(x, centers, labels, tmp_path/'index', balls=balls, dims=dims)
    idx = Index(tmp_path/'index')
    assert meta['summary_bytes'] == meta['n_pages']*balls*(dims+4)
    for li, (start, end) in enumerate(idx.ranges):
        pages = np.arange(start, end)
        for q in np.r_[queries, np.zeros((1, 128), dtype=np.float32)]:
            bounds = idx.bounds(q, pages, li, 'combined')
            for lb, p in zip(bounds, pages):
                ids = idx.page_ids[p, :idx.valid[p]]
                true = np.linalg.norm(x[ids].astype('float64')-q, axis=1).min()
                assert lb <= true + 1e-10


def test_coalescing():
    assert coalesce([8, 1, 2, 2, 4, 9]) == [(1, 2), (4, 1), (8, 2)]
    assert coalesce([1, 2, 4], gap_pages=1) == [(1, 4)]
    assert coalesce([1, 2, 4], gap_pages=1, max_pages=2) == [(1, 2), (4, 1)]
    for kwargs in ({'gap_pages': -1}, {'max_pages': 0}):
        with pytest.raises(ValueError):
            coalesce([1], **kwargs)


def test_gap_accounting_and_seed(tmp_path):
    x, centers, labels, queries = fixture_data()
    build_assigned(x, centers, labels, tmp_path/'index', balls=8, dims=64)
    idx = Index(tmp_path/'index')
    reader = MemoryReplay(idx.path/'vectors.pages')
    trace = Trace(tmp_path/'trace.jsonl')
    try:
        ids, s = search(idx, queries[0], reader, nprobe=4, gap_pages=2, trace=trace,
                        window_pages=8, max_extent_pages=8, numpy_test=True)
        np.testing.assert_array_equal(ids, oracle(idx, x, queries[0], nprobe=4))
        assert s['gap_pages_read'] == s['read_pages']-s['selected_pages']
    finally:
        trace.close(); reader.close()
    records = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert records[0]['threshold_before'] is None
    for r in records:
        assert r['depends_on_stage'] == (r['stage_id']-1 if r['stage_id'] else None)


def test_payload_and_invalid_input(tmp_path):
    x, centers, labels, _ = fixture_data()
    root = tmp_path/'index'
    build_assigned(x, centers, labels, root)
    idx = Index(root)
    data = (root/'vectors.pages').read_bytes()
    seen = []
    for p, n in enumerate(idx.valid):
        a = np.frombuffer(data, dtype='<f4', count=int(n)*128, offset=p*4096).reshape(n, 128)
        ids = idx.page_ids[p, :n]
        np.testing.assert_array_equal(a, x[ids])
        seen.extend(ids)
    assert sorted(seen) == list(range(len(x)))
    with pytest.raises(FileExistsError):
        build_assigned(x, centers, labels, root)
    with pytest.raises(ValueError):
        checked(np.array([[np.nan]], dtype='float32'))
    for page_size in (2048, 6144):
        with pytest.raises(ValueError):
            build_assigned(x, centers, labels, tmp_path/str(page_size), page_size=page_size)
    with pytest.raises(ValueError):
        idx.route(x[0], 0, numpy_test=True)
    f = tmp_path/'bad.fvecs'; f.write_bytes(np.array([128, 0], dtype='<i4').tobytes())
    with pytest.raises(ValueError):
        vectors(f)


def test_native_short_read(tmp_path):
    path = tmp_path/'small'; path.write_bytes(b'x'*4096)
    r = Native(path)
    try:
        with pytest.raises(OSError, match='EOF'):
            r.read([(4096, 4096)])
    finally:
        r.close()


@pytest.mark.skipif(os.environ.get('GEOIVF_TEST_DIRECT') != '1', reason='direct I/O opt-in')
def test_direct_reader(tmp_path):
    path = tmp_path/'payload'; data = bytes(range(256))*48; path.write_bytes(data)
    r = Native(path, direct=True)
    try:
        assert r.read([(4096, 8192), (0, 4096)]) == [data[4096:], data[:4096]]
    finally:
        r.close()


@pytest.mark.skipif(os.environ.get('GEOIVF_TEST_URING') != '1', reason='io_uring opt-in')
def test_uring_reader(tmp_path):
    path = tmp_path/'payload'; data = bytes(range(256))*128; path.write_bytes(data)
    r = Native(path, direct=True, uring=True, depth=2)
    try:
        requests = [(p*4096, 4096) for p in [6, 0, 5, 1, 4, 3, 2]]
        assert r.read(requests) == [data[o:o+n] for o, n in requests]
    finally:
        r.close()


def test_mqsim_export(tmp_path):
    fields = dict(Flash_Channel_Count=1, Chip_No_Per_Channel=1, Die_No_Per_Chip=1,
                  Plane_No_Per_Die=1, Block_No_Per_Plane=64, Page_No_Per_Block=64,
                  Page_Capacity=4096, Overprovisioning_Ratio=0.1)
    root = ET.Element('SSD_Configuration')
    for key, value in fields.items():
        ET.SubElement(root, key).text = str(value)
    config = tmp_path/'ssd.xml'; ET.ElementTree(root).write(config)
    trace = tmp_path/'source.jsonl'
    records = [dict(offset_bytes=0, length_bytes=8192, operation='read'),
               dict(offset_bytes=16384, length_bytes=4096, operation='read')]
    trace.write_text(''.join(json.dumps(r)+'\n' for r in records))
    out = tmp_path/'mqsim'
    report = export(trace, out, request_gap_ns=1000, ssd_config=config)
    assert (out/'requests.trace').read_text() == '1000 0 0 16 1\n2000 0 32 8 1\n'
    assert report['read_bytes'] == 12288
    assert report['query_latency_valid'] is False
    assert ET.parse(out/'workload.xml').find('.//Time_Unit').text == 'NANOSECOND'
    records[0]['offset_bytes'] = 1
    trace.write_text(json.dumps(records[0])+'\n')
    with pytest.raises(ValueError):
        export(trace, out, request_gap_ns=1000, ssd_config=config)


def test_faiss_integration(tmp_path):
    faiss = pytest.importorskip('faiss')
    from geoivf.index import train_faiss
    rng = np.random.default_rng(731)
    x = rng.normal(size=(1024, 32)).astype('float32')
    q = rng.normal(size=(16, 32)).astype('float32')
    centers, labels = train_faiss(x, 8, 17, 1024)
    quantizer = faiss.IndexFlatL2(32); quantizer.add(centers)
    baseline = faiss.IndexIVFFlat(quantizer, 32, 8, faiss.METRIC_L2)
    baseline.is_trained = True; baseline.add(x)
    build_assigned(x, centers, labels, tmp_path/'index', dims=16, balls=8)
    idx = Index(tmp_path/'index')
    r = Pread(idx.path/'vectors.pages')
    try:
        for nprobe in [1, 3, 8]:
            baseline.nprobe = nprobe
            expected = baseline.search(q, 10)[1]
            for i, query in enumerate(q):
                ids, _ = search(idx, query, r, nprobe=nprobe, window_pages=5)
                np.testing.assert_array_equal(ids, expected[i])
    finally:
        r.close()
