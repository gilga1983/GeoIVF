import os
import numpy as np
import pytest
from geoivf.index import Index
from geoivf.layouts import freeze_layout, summarize, sha256
from geoivf.io import MemoryReplay, Native
from geoivf.search import search


def data():
    rng = np.random.default_rng(41)
    centers = rng.normal(0, 8, (4, 128)).astype('float32')
    labels = np.arange(259) % 4
    x = (centers[labels]+rng.normal(size=(259, 128))).astype('float32')
    return x, centers, labels


@pytest.mark.parametrize('layout', ['input', 'radial', 'geopack'])
def test_frozen_layout_and_audit(tmp_path, layout):
    x, centers, labels = data()
    frozen = tmp_path/'physical'
    freeze_layout(x, centers, labels, frozen, layout=layout, pack_dims=8)
    src = Index(frozen); before = sha256(frozen/'vectors.pages')
    for dims, balls in [(4, 1), (8, 4), (16, 8), (128, 2)]:
        path = tmp_path/f'summary-{dims}-{balls}'
        meta = summarize(frozen, path, dims=dims, balls=balls)
        assert meta['layout_payload_sha256'] == before == sha256(path/'vectors.pages')
        idx = Index(path)
        np.testing.assert_array_equal(idx.page_ids, src.page_ids)
        reader = MemoryReplay(path/'vectors.pages')
        for q in x[[0, 81]]:
            lists = np.array([0, 1, 2, 3])
            audit = []
            ids, _ = search(idx, q, reader, nprobe=4, preassigned_lists=lists,
                            audit_sink=audit, window_pages=4)
            dist = np.sum((x.astype('float64')-q)**2, axis=1)
            expected = np.lexsort((np.arange(len(x)), dist))[:10]
            np.testing.assert_array_equal(ids, expected)
            for p, tau, lb in audit:
                page_ids = idx.page_ids[p, :idx.valid[p]]
                assert np.sqrt(dist[page_ids].min()) > tau
                assert lb <= np.sqrt(dist[page_ids].min()) + 1e-10
        reader.close()
    assert sha256(frozen/'vectors.pages') == before


@pytest.mark.parametrize('depth', [1, 2, 8, 32])
@pytest.mark.skipif(os.environ.get('GEOIVF_TEST_URING') != '1', reason='io_uring opt-in')
def test_uring_unique_pages_and_eof(tmp_path, depth):
    # Different bytes on every page catch misplaced/out-of-order completions.
    path = tmp_path/'payload'
    data = b''.join(bytes([i])*4096 for i in range(64))
    path.write_bytes(data)
    r = Native(path, direct=True, uring=True, depth=depth)
    try:
        requests = [(p*4096, n*4096) for p, n in [(41, 3), (0, 2), (6, 1),
                     (31, 4), (11, 2), (60, 4), (5, 1), (0, 1), (17, 2)]]
        assert r.read(requests) == [data[o:o+n] for o, n in requests]
        # Completion failure/EOF must propagate; no cached or sync fallback.
        with pytest.raises(OSError, match='short|EOF'):
            r.read([(0, 4096), (len(data), 4096)])
        with pytest.raises(OSError, match='failed'):
            r.read([(0, 4096)])
    finally:
        r.close()


@pytest.mark.skipif(os.environ.get('GEOIVF_TEST_URING') != '1', reason='io_uring opt-in')
def test_uring_full_search_equivalence(tmp_path):
    x, centers, labels = data()
    freeze_layout(x, centers, labels, tmp_path/'layout')
    summarize(tmp_path/'layout', tmp_path/'index')
    idx = Index(tmp_path/'index')
    r = Native(idx.path/'vectors.pages', direct=True, uring=True, depth=8)
    try:
        for q in x[:8]:
            ids, _ = search(idx, q, r, nprobe=4, preassigned_lists=np.arange(4),
                            window_pages=5)
            dd = np.sum((x.astype('float64')-q)**2, axis=1)
            np.testing.assert_array_equal(ids, np.lexsort((np.arange(len(x)), dd))[:10])
    finally:
        r.close()
