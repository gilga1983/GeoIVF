import json
import numpy as np
import pytest
from geoivf.index import build_assigned
from geoivf.projections import Basis, contraction, digest_file, summarize_projection, ProjectedIndex
from geoivf.io import MemoryReplay
from geoivf.search import search


def fixture(tmp_path, d=16):
    rng = np.random.default_rng(731)
    x = rng.normal(size=(181, d)).astype(np.float32)
    # Include duplicates, a zero vector, and unequal/partly-filled lists.
    x[0] = 0; x[1] = x[2]
    centers = rng.normal(size=(3, d)).astype(np.float32)
    labels = np.arange(len(x)) % 3
    layout = tmp_path/'physical'
    meta = build_assigned(x, centers, labels, layout, dims=8, balls=1)
    meta.update(physical_layout_frozen=True,
                layout_payload_sha256=digest_file(layout/'vectors.pages'))
    (layout/'manifest.json').write_text(json.dumps(meta))
    return x, centers, labels, layout


@pytest.mark.parametrize('kind', ['pca', 'coordinates'])
@pytest.mark.parametrize('tail', [False, True])
@pytest.mark.parametrize('precision', ['uint8', 'float32'])
@pytest.mark.parametrize('balls', [1, 8])
def test_bounds_and_search(tmp_path, kind, tail, precision, balls):
    x, centers, labels, layout = fixture(tmp_path)
    basis = Basis.fit(x[10:], kind)
    meta = summarize_projection(layout, tmp_path/'summary', basis, dims=8,
                                 balls=balls, tail=tail, center_dtype=precision)
    idx = ProjectedIndex(layout, tmp_path/'summary')
    pages = np.arange(idx.meta['n_pages'])
    assert meta['summary_bytes'] == idx.codes.nbytes+idx.radii.nbytes+idx.tail_bounds.nbytes
    assert digest_file(layout/'vectors.pages') == meta['layout_payload_sha256']
    reader = MemoryReplay(layout/'vectors.pages')
    for q in np.concatenate((x[:4], x[8:10]*5)):
        for li, (first, last) in enumerate(idx.ranges):
            pp = pages[first:last]
            ids = idx.page_ids[pp]
            actual = np.linalg.norm(x[np.maximum(ids, 0)].astype(float)-q, axis=2)
            actual[ids < 0] = np.inf
            for mode in ('balls', 'combined'):
                lb = idx.bounds(q, pp, li, mode)
                assert np.all(lb <= actual.min(axis=1)+1e-11)
        got, _ = search(idx, q, reader, nprobe=3, window_pages=2, numpy_test=True)
        exact = np.sum((x.astype(float)-q)**2, axis=1)
        expected = np.lexsort((np.arange(len(x)), exact))[:10]
        np.testing.assert_array_equal(got, expected)
    reader.close()


def test_cache_matches_direct_and_full_dimension(tmp_path):
    x, _, _, layout = fixture(tmp_path)
    basis = Basis.fit(x)
    a = summarize_projection(layout, tmp_path/'a', basis, dims=16, balls=8,
                              tail=True, projected_by_id=basis.project(x))
    summarize_projection(layout, tmp_path/'b', basis, dims=16, balls=8, tail=True)
    ai = ProjectedIndex(layout, tmp_path/'a'); bi = ProjectedIndex(layout, tmp_path/'b')
    np.testing.assert_allclose(ai.codes, bi.codes, rtol=0, atol=0)
    np.testing.assert_allclose(ai.radii, bi.radii, rtol=1e-5, atol=1e-7)
    assert a['layout_payload_sha256'] == digest_file(layout/'vectors.pages')


def test_residual_can_reject_when_head_cannot(tmp_path):
    x, _, _, layout = fixture(tmp_path)
    basis = Basis(np.zeros(16), np.eye(16), 'coordinates')
    summarize_projection(layout, tmp_path/'head', basis, dims=1, balls=8)
    summarize_projection(layout, tmp_path/'tail', basis, dims=1, balls=8, tail=True)
    head = ProjectedIndex(layout, tmp_path/'head'); tail = ProjectedIndex(layout, tmp_path/'tail')
    q = np.zeros(16, np.float32); q[-1] = 100
    p = np.arange(head.ranges[0, 0], head.ranges[0, 1])
    assert np.all(tail.bounds(q, p, 0, 'balls') > head.bounds(q, p, 0, 'balls')+50)


def test_transform_norm_and_invalid_inputs():
    rng = np.random.default_rng(93)
    for a in (rng.normal(size=(11, 11)), np.eye(11)*100):
        t, _ = contraction(a)
        assert np.linalg.norm(t, 2) <= 1
    with pytest.raises(ValueError): contraction(np.zeros((4, 4)))
    with pytest.raises(ValueError): contraction(np.full((4, 4), np.nan))
    with pytest.raises(ValueError): Basis.fit(np.ones((5, 4)), 'whitened')


def test_constant_data_and_payload_validation(tmp_path):
    x, _, _, layout = fixture(tmp_path)
    basis = Basis.fit(np.ones((20, 16)))
    summarize_projection(layout, tmp_path/'ok', basis, dims=2, balls=1)
    with pytest.raises(ValueError):
        summarize_projection(layout, tmp_path/'bad-d', basis, dims=17)
    with pytest.raises(FileExistsError):
        summarize_projection(layout, tmp_path/'ok', basis)
    with open(layout/'vectors.pages', 'r+b') as f:
        f.write(b'xxxx')
    with pytest.raises(ValueError):
        summarize_projection(layout, tmp_path/'tampered', basis, dims=2)
