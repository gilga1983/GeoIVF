#!/usr/bin/env python3
"""Demand-adaptive spatial resolution for regional navigation maps.

Use the 512-cell geometric partition as atomic spatial cells. Training demand is
the number of navigation checkpoints entering each atomic cell. Build a genuine
nested partition by repeatedly splitting the CURRENT highest-demand leaf into
two geometric children. Thus hot areas receive finer resolution while cold
areas remain coarse.

For each target leaf count (64/128/256/512), compare the demand-adaptive nested
partition against the ordinary uniform k-means partition with the same nominal
number of regions. All arms share:
  * the same held-out checkpoint events,
  * the same total stored (state, hint) entry budget,
  * the same per-state candidate cap,
  * the same routed proxy.

This isolates whether spending spatial resolution according to observed demand
is better than spending it uniformly across the graph.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"GIPIP001"


def load_fbin(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated fbin: {path}")
    rows, dim = struct.unpack("<II", raw)
    if path.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"bad fbin size: {path}")
    return rows, dim, np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_centers(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != MAGIC:
        raise ValueError(f"bad router: {path}")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nf = nlist * dim
    centers = np.frombuffer(raw, dtype="<f4", count=nf, offset=16).reshape(nlist, dim).copy()
    expected = 16 + nf * 4 + nlist * 4 + nf * 4
    if len(raw) != expected:
        raise ValueError(f"bad router size: {path}")
    return int(nlist), int(dim), centers


def stage_band(pos):
    if pos < 16:
        return "8-15"
    if pos < 32:
        return "16-31"
    if pos < 64:
        return "32-63"
    if pos < 128:
        return "64-127"
    return "128+"


def qstats(vals):
    if not vals:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, .95)),
        "max": float(a.max()),
    }


def assign_ids(unique_ids, base, centers_by_level, batch=2048):
    ids = np.asarray(sorted(unique_ids), dtype=np.int64)
    result = {level: np.empty(len(ids), dtype=np.int32) for level in centers_by_level}
    for lo in range(0, len(ids), batch):
        hi = min(len(ids), lo + batch)
        x = np.asarray(base[ids[lo:hi]], dtype=np.float32, order="C")
        for level, centers in centers_by_level.items():
            result[level][lo:hi] = np.argmax(x @ centers.T, axis=1).astype(np.int32)
    return {
        int(v): {level: int(result[level][i]) for level in centers_by_level}
        for i, v in enumerate(ids)
    }


def make_events(records, region_of, finest, min_pos):
    events = []
    for rec in records:
        qid = int(rec["query"])
        ids = [int(v) for v in rec["ids"]]
        seen = set()
        for pos, vid in enumerate(ids):
            atom = int(region_of[vid][finest])
            if atom in seen:
                continue
            seen.add(atom)
            if pos < min_pos or pos + 1 >= len(ids):
                continue
            future = {}
            for p in range(pos + 1, len(ids)):
                future.setdefault(ids[p], p)
            if future:
                events.append({
                    "query": qid,
                    "pos": pos,
                    "atom": atom,
                    "vid": vid,
                    "ids": ids,
                    "regions": region_of[vid],
                    "future": future,
                })
    return events


def normalize(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def split_atoms(atoms, atomic_centers, atomic_demand):
    atoms = np.asarray(atoms, dtype=np.int32)
    if len(atoms) < 2:
        raise ValueError("cannot split singleton")

    # Deterministic weighted spherical 2-means. Start from the most demanded
    # atom and its geometrically farthest mate.
    weights = np.asarray([atomic_demand[int(a)] for a in atoms], dtype=np.float64)
    seed1_i = int(np.argmax(weights))
    seed1 = atomic_centers[int(atoms[seed1_i])]
    sims = atomic_centers[atoms] @ seed1
    seed2_i = int(np.argmin(sims))
    if seed2_i == seed1_i:
        seed2_i = (seed1_i + 1) % len(atoms)
    centers = np.stack([
        normalize(seed1.astype(np.float64)),
        normalize(atomic_centers[int(atoms[seed2_i])].astype(np.float64)),
    ])

    assign = np.zeros(len(atoms), dtype=np.int8)
    for _ in range(12):
        scores = atomic_centers[atoms] @ centers.T
        new_assign = np.argmax(scores, axis=1).astype(np.int8)
        # Repair an empty side by moving the atom least compatible with the
        # populated side.
        if np.all(new_assign == 0) or np.all(new_assign == 1):
            side = int(new_assign[0])
            other = 1 - side
            move = int(np.argmin(scores[:, side]))
            new_assign[move] = other
        if np.array_equal(new_assign, assign):
            assign = new_assign
            break
        assign = new_assign
        for side in (0, 1):
            idx = np.flatnonzero(assign == side)
            w = weights[idx] + 1.0  # keep zero-demand atoms geometrically represented
            vec = np.sum(atomic_centers[atoms[idx]].astype(np.float64) * w[:, None], axis=0)
            centers[side] = normalize(vec)

    left = [int(a) for a in atoms[assign == 0]]
    right = [int(a) for a in atoms[assign == 1]]
    if not left or not right:
        raise RuntimeError("degenerate split")
    return left, right


def adaptive_snapshots(atomic_centers, atomic_demand, targets):
    targets = sorted(targets)
    natoms = len(atomic_centers)
    if targets[-1] > natoms:
        raise ValueError("target exceeds atomic cells")

    leaves = {0: list(range(natoms))}
    next_id = 1
    snapshots = {}

    def leaf_demand(atoms):
        return int(sum(atomic_demand[a] for a in atoms))

    while len(leaves) <= targets[-1]:
        if len(leaves) in targets:
            atom_to_leaf = {}
            meta = {}
            for leaf_id, atoms in leaves.items():
                for a in atoms:
                    atom_to_leaf[int(a)] = int(leaf_id)
                meta[str(leaf_id)] = {
                    "atoms": len(atoms),
                    "train_demand": leaf_demand(atoms),
                }
            snapshots[len(leaves)] = {"atom_to_leaf": atom_to_leaf, "meta": meta}
            if len(leaves) == targets[-1]:
                break

        candidates = [
            (leaf_demand(atoms), len(atoms), -leaf_id, leaf_id)
            for leaf_id, atoms in leaves.items() if len(atoms) > 1
        ]
        if not candidates:
            raise RuntimeError("ran out of splittable leaves")
        _, _, _, leaf_id = max(candidates)
        atoms = leaves.pop(leaf_id)
        left, right = split_atoms(atoms, atomic_centers, atomic_demand)
        leaves[next_id] = left
        next_id += 1
        leaves[next_id] = right
        next_id += 1

    return snapshots


def learn(train_events, key_fn):
    score = collections.defaultdict(lambda: collections.defaultdict(int))
    support = collections.defaultdict(lambda: collections.defaultdict(int))
    event_count = collections.Counter()
    total_mass = 0
    for ev in train_events:
        key = key_fn(ev)
        event_count[key] += 1
        e = int(ev["pos"])
        for vid, p in ev["future"].items():
            residual = int(p) - e
            if residual <= 0:
                continue
            score[key][int(vid)] += residual
            support[key][int(vid)] += 1
            total_mass += residual
    return score, support, event_count, total_mass


def allocate(score, support, budget, cap):
    rows = []
    for key, srow in score.items():
        for vid, s in srow.items():
            rows.append((int(s), int(support[key][vid]), key, int(vid)))
    rows.sort(key=lambda x: (-x[0], -x[1], repr(x[2]), x[3]))
    maps = collections.defaultdict(list)
    counts = collections.Counter()
    selected_mass = 0
    stored = 0
    for s, sup, key, vid in rows:
        if stored >= budget:
            break
        if counts[key] >= cap:
            continue
        maps[key].append(vid)
        counts[key] += 1
        selected_mass += s
        stored += 1
    return dict(maps), stored, selected_mass


def evaluate(events, key_fn, maps, queries, base):
    n = len(events)
    available = usable_events = hit = total_skip = 0
    hit_skips = []
    cache = {}
    for ev in events:
        candidates = maps.get(key_fn(ev), ())
        if not candidates:
            continue
        available += 1
        e = int(ev["pos"])
        visited = set(ev["ids"][:e+1])
        usable = [v for v in candidates if v not in visited]
        if not usable:
            continue
        usable_events += 1
        ck = tuple(usable)
        vecs = cache.get(ck)
        if vecs is None:
            vecs = np.asarray(base[np.asarray(usable, dtype=np.int64)], dtype=np.float32, order="C")
            cache[ck] = vecs
        q = np.asarray(queries[int(ev["query"])], dtype=np.float32)
        chosen = int(usable[int(np.argmax(vecs @ q))])
        p = ev["future"].get(chosen)
        if p is not None:
            skip = int(p) - e
            hit += 1
            total_skip += skip
            hit_skips.append(skip)
    return {
        "events": n,
        "map_available_fraction": available / n if n else 0.0,
        "usable_map_fraction": usable_events / n if n else 0.0,
        "routed_suffix_hit_fraction": hit / n if n else 0.0,
        "routed_mean_skip_all_events": total_skip / n if n else 0.0,
        "routed_skip_when_hit": qstats(hit_skips),
    }


def run_arm(train_events, held_events, key_fn, queries, base, budget, cap):
    score, support, event_count, total_mass = learn(train_events, key_fn)
    maps, stored, selected_mass = allocate(score, support, budget, cap)
    out = evaluate(held_events, key_fn, maps, queries, base)
    out.update({
        "stored_entries": stored,
        "states_with_entries": len(maps),
        "train_events_per_state": qstats(list(event_count.values())),
        "selected_train_score_mass_fraction": selected_mass / total_mass if total_mass else 0.0,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--router-root", type=Path, required=True)
    ap.add_argument("--levels", default="64,128,256,512")
    ap.add_argument("--train-queries", type=int, default=9000)
    ap.add_argument("--min-entry-pos", type=int, default=8)
    ap.add_argument("--budgets", default="2048,4096,8192")
    ap.add_argument("--per-state-cap", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    records.sort(key=lambda r: int(r["query"]))
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("dense query IDs required")

    nbase, bdim, base = load_fbin(args.base)
    nq, qdim, queries = load_fbin(args.queries)
    if qdim != bdim or nq < len(records):
        raise ValueError("shape mismatch")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    finest = levels[-1]

    centers = {}
    for level in levels:
        n, d, c = load_centers(args.router_root / f"nlist-{level}" / "router.bin")
        if n != level or d != bdim:
            raise ValueError(f"router mismatch {level}")
        centers[level] = c

    unique_ids = {int(v) for r in records for v in r["ids"]}
    region_of = assign_ids(unique_ids, base, centers)

    train_records = records[:args.train_queries]
    held_records = records[args.train_queries:]
    train_events = make_events(train_records, region_of, finest, args.min_entry_pos)
    held_events = make_events(held_records, region_of, finest, args.min_entry_pos)

    atomic_demand = np.zeros(finest, dtype=np.int64)
    for ev in train_events:
        atomic_demand[int(ev["atom"])] += 1

    snapshots = adaptive_snapshots(centers[finest], atomic_demand, levels)

    result = {
        "split": {"train_queries": len(train_records), "heldout_queries": len(held_records)},
        "event_population": {
            "train_events": len(train_events),
            "heldout_events": len(held_events),
            "checkpoint": f"first entry into unseen {finest}-cell atomic region",
        },
        "adaptive_policy": "repeatedly split highest-demand current leaf; weighted spherical 2-means over atomic 512-cell centroids",
        "fairness": {
            "total_entry_budgets": budgets,
            "per_state_cap": args.per_state_cap,
            "same_event_population": True,
        },
        "atomic_demand": qstats(atomic_demand.tolist()),
        "levels": {},
        "results": {},
    }

    for level in levels:
        meta = snapshots[level]["meta"]
        result["levels"][str(level)] = {
            "adaptive_leaf_atom_count": qstats([m["atoms"] for m in meta.values()]),
            "adaptive_leaf_train_demand": qstats([m["train_demand"] for m in meta.values()]),
        }

    for budget in budgets:
        arms = {}
        for level in levels:
            amap = snapshots[level]["atom_to_leaf"]
            arms[f"adaptive_region_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, amap=amap: ("ar", int(amap[int(ev["atom"])])),
                queries, base, budget, args.per_state_cap,
            )
            arms[f"adaptive_region_stage_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, amap=amap: ("ars", int(amap[int(ev["atom"])]), stage_band(int(ev["pos"]))),
                queries, base, budget, args.per_state_cap,
            )
            arms[f"uniform_region_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, level=level: ("ur", level, int(ev["regions"][level])),
                queries, base, budget, args.per_state_cap,
            )
            arms[f"uniform_region_stage_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, level=level: ("urs", level, int(ev["regions"][level]), stage_band(int(ev["pos"]))),
                queries, base, budget, args.per_state_cap,
            )

        best = sorted(
            ((m["routed_suffix_hit_fraction"], m["routed_mean_skip_all_events"], name)
             for name, m in arms.items()),
            reverse=True,
        )
        result["results"][str(budget)] = {
            "arms": arms,
            "top12": [
                {"arm": name, "hit": hit, "mean_skip": skip}
                for hit, skip, name in best[:12]
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
