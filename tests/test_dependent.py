from __future__ import annotations
import hashlib
import json
import numpy as np
import pytest
from geoivf.dependent import (pack_codes, unpack_codes, fit_anchors,
                             summarize_dependencies, DependentIndex, file_hash)
from geoivf.index import build_assigned, Index
from geoivf.io import MemoryReplay
from geoivf.search import search


@pytest.mark.parametrize('bits', range(1, 9))
def test_packing(bits):
    a = np.random.default_rng(17).integers(0, 2**bits, size=(5, 3, 7), dtype=np.uint8)
    p = pack_codes(a, bits)
    assert p.shape == (5, (21*bits+7)//8)
    np.testing.assert_array_equal(unpack_codes(p, bits, (3, 7)), a)


def test_code_rejection():
    with pytest.raises(ValueError): pack_codes(np.array([[16]], np.uint8), 4)
    with pytest.raises(ValueError): unpack_codes(np.zeros((2, 2), np.uint8), 8, (3,))


@pytest.mark.parametrize('midpoint,middle', [(True, 2.), (False, 1.)])
def test_exact_line(midpoint, middle):
    z = np.array([[[0., 0.], [middle, 0.], [4., 0.]]])
    codes, pair, alpha, pred, order = fit_anchors(z, np.array([3]), np.zeros(2),
                                                 np.ones(2), 2, midpoint=midpoint)
    target = np.take_along_axis(z, order[:, :, None], axis=1)
    np.testing.assert_allclose(pred, target, atol=0, rtol=0)
    assert pair.shape == (1, 1)


class IdentityBasis:
    kind = 'identity-test'
    def __init__(self, d):
        self.mean = np.zeros(d); self.matrix = np.eye(d)
    def fingerprint(self):
        return hashlib.sha256(self.matrix.tobytes()).hexdigest()


@pytest.fixture
def layout(tmp_path):
    rng = np.random.default_rng(98)
    x = rng.normal(size=(35, 128)).astype(np.float32)
    c = x.mean(axis=0, keepdims=True)
    path = tmp_path/'physical'
    m = build_assigned(x, c, np.zeros(len(x), np.int64), path,
                       dims=16, balls=1, layout='geopack')
    m.update(physical_layout_frozen=True, layout_payload_sha256=file_hash(path/'vectors.pages'))
    (path/'manifest.json').write_text(json.dumps(m))
    return x, path, tmp_path


@pytest.mark.parametrize('scheme,anchors,bits,residual', [
    ('independent', 8, 2, 0), ('independent', 8, 4, 0),
    ('independent', 8, 6, 0), ('independent', 8, 8, 0),
    ('midpoint', 2, 8, 0), ('midpoint', 4, 8, 0), ('midpoint', 6, 8, 0),
    ('interpolate', 4, 8, 0), ('interpolate', 6, 8, 0),
    ('interpolate', 4, 8, 2), ('interpolate', 4, 8, 4)])
def test_cover_search_and_accounting(layout, scheme, anchors, bits, residual):
    x, path, root = layout
    before = file_hash(path/'vectors.pages')
    info = summarize_dependencies(path, root/'summary', IdentityBasis(128), dims=64,
                                   scheme=scheme, anchors=anchors, bits=bits,
                                   residual_bits=residual, projected_by_id=x.astype(float))
    index = DependentIndex(path, root/'summary')
    reader = MemoryReplay(path/'vectors.pages')
    for q in np.random.default_rng(31).normal(size=(3, 128)).astype(np.float32):
        p = np.arange(index.meta['n_pages'])
        lb = index.bounds(q, p, 0, 'combined')
        for page in p:
            ids = index.page_ids[page, :index.valid[page]]
            truth = np.linalg.norm(x[ids].astype(float)-q, axis=1).min()
            assert lb[page] <= truth + 1e-10
        expected, _ = search(index, q, reader, nprobe=1, filtering='none', numpy_test=True)
        actual, _ = search(index, q, reader, nprobe=1, filtering='combined', numpy_test=True)
        np.testing.assert_array_equal(actual, expected)
    reader.close()
    assert info['all_points_cover_audited'] == len(x)
    assert info['summary_bytes'] == sum(info['component_bytes'].values())
    assert before == file_hash(path/'vectors.pages')
    expected_bpp = ((anchors*64*bits+7)//8+8*4 if scheme != 'independent'
                    else (8*64*bits+7)//8+8*4)
    if scheme != 'independent': expected_bpp += 8-anchors
    if scheme == 'interpolate': expected_bpp += 8-anchors
    if residual: expected_bpp += ((8-anchors)*64*residual+7)//8 + 4*(8-anchors)
    assert info['bytes_per_page'] == expected_bpp
    assert not info['runtime_expanded_centers_retained']
