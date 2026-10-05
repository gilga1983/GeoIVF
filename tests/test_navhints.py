import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


learn = load_script("learn_global_navigation_landmarks")
build = load_script("build_hint_ivf")


def test_skip_ranking_uses_first_occurrence_only():
    records = [
        {"query": 0, "ids": [99, 10, 20, 10, 30]},
        {"query": 1, "ids": [99, 20, 30]},
        {"query": 2, "ids": [99, 10, 40]},
    ]
    ranked = learn.rank_hints(records)
    by_id = {
        vertex: (score, support)
        for score, support, vertex in ranked
    }

    # Repeated vertex 10 in q0 contributes only its first position (=1).
    assert by_id[10] == (2, 2)
    assert by_id[20] == (3, 2)
    assert by_id[30] == (6, 2)
    assert by_id[40] == (2, 1)
    # Teacher starts are excluded.
    assert 99 not in by_id
    assert [vertex for _, _, vertex in ranked] == [30, 20, 10, 40]


def test_start_id_binary_round_trip_and_duplicate_rejection(tmp_path):
    path = tmp_path / "ids.bin"
    expected = np.asarray([7, 3, 11, 19], dtype=np.uint32)
    learn.write_start_ids(path, expected)
    np.testing.assert_array_equal(learn.load_start_ids(path), expected)

    with pytest.raises(ValueError, match="unique"):
        learn.write_start_ids(tmp_path / "dup.bin", [1, 2, 1])


def test_budget_parser_rejects_ambiguous_inputs():
    assert learn.parse_budgets("8,16,32") == [8, 16, 32]
    with pytest.raises(ValueError):
        learn.parse_budgets("")
    with pytest.raises(ValueError):
        learn.parse_budgets("8,0")
    with pytest.raises(ValueError):
        learn.parse_budgets("8,8")


def normalized_fixture(rows=64, dim=12):
    rng = np.random.default_rng(20261005)
    x = rng.normal(size=(rows, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def test_spherical_partition_is_seed_deterministic():
    x = normalized_fixture()
    c1, a1 = build.lloyd_spherical(x, 8, 4, 17, 731)
    c2, a2 = build.lloyd_spherical(x, 8, 4, 17, 731)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_allclose(c1, c2, rtol=0, atol=0)


def test_id_only_ivf_covers_each_hint_exactly_once(tmp_path):
    x = normalized_fixture(rows=48, dim=10)
    hint_ids = np.arange(100, 148, dtype=np.uint32)

    centers, initial = build.lloyd_spherical(x, 6, 4, 16, 17)
    medoid_local = build.choose_medoids(x, centers, initial)
    medoid_vecs = x[medoid_local]
    final = build.assign_batched(x, medoid_vecs, 16)
    medoid_ids = hint_ids[medoid_local]
    buckets = [hint_ids[final == c] for c in range(6)]

    path = tmp_path / "ivf.bin"
    offsets, children = build.write_ivf(
        path, medoid_ids, buckets, len(hint_ids)
    )

    assert path.read_bytes()[:8] == build.IVF_MAGIC
    assert offsets[0] == 0
    assert offsets[-1] == len(children)
    deployed = list(map(int, medoid_ids)) + list(map(int, children))
    assert len(deployed) == len(hint_ids)
    assert len(set(deployed)) == len(hint_ids)
    assert sorted(deployed) == sorted(map(int, hint_ids))


def test_builder_rejects_invalid_parameters():
    x = normalized_fixture(rows=16, dim=4)
    with pytest.raises(ValueError):
        build.lloyd_spherical(x, 0, 2, 4, 1)
    with pytest.raises(ValueError):
        build.lloyd_spherical(x, 17, 2, 4, 1)
    with pytest.raises(ValueError):
        build.lloyd_spherical(x, 4, 0, 4, 1)
    with pytest.raises(ValueError):
        build.lloyd_spherical(x, 4, 2, 0, 1)
