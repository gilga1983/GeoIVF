#!/usr/bin/env python3
"""Fixed-memory and evidence-backed hierarchical navigation-map diagnostic.

This diagnostic uses one common event population: first entry into a previously
unseen finest-resolution region (default 512) at expansion position >= 8.

It compares:
  * exact-stage maps,
  * flat regional maps at 64/128/256/512 resolution,
  * adaptive regional maps that choose the finest resolution with enough
    training evidence, backing off to coarser resolutions otherwise,
  * the same adaptive policy with coarse stage-band conditioning.

Every arm receives the same TOTAL stored (state, vertex) entry budget and the
same per-state candidate cap. Thus finer state descriptions do not win merely
because they create more maps.

The routed proxy is conservative: from the learned candidates for the current
state, choose the highest exact query-inner-product unvisited vertex, and count
success only if that vertex occurs later in the natural traversal.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path

import numpy as np

ROUTER_MAGIC = b"GIPIP001"


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
    if len(raw) < 16 or raw[:8] != ROUTER_MAGIC:
        raise ValueError(f"bad router: {path}")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nf = nlist * dim
    centers = np.frombuffer(raw, dtype="<f4", count=nf, offset=16).reshape(nlist, dim).copy()
    expected = 16 + nf * 4 + nlist * 4 + nf * 4
    if len(raw) != expected:
        raise ValueError(f"bad router size: {path}")
    return int(nlist), int(dim), centers


def stage_band(pos: int) -> str:
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


def assign_all(unique_ids, base, level_centers, batch=2048):
    ids = np.asarray(sorted(unique_ids), dtype=np.int64)
    out = {level: np.empty(len(ids), dtype=np.int32) for level in level_centers}
    for lo in range(0, len(ids), batch):
        hi = min(len(ids), lo + batch)
        x = np.asarray(base[ids[lo:hi]], dtype=np.float32, order="C")
        for level, centers in level_centers.items():
            out[level][lo:hi] = np.argmax(x @ centers.T, axis=1).astype(np.int32)
    return {
        int(v): {level: int(out[level][i]) for level in level_centers}
        for i, v in enumerate(ids)
    }


def make_events(records, region_of, finest, min_entry_pos):
    events = []
    for rec in records:
        qid = int(rec["query"])
        ids = [int(v) for v in rec["ids"]]
        seen = set()
        for pos, vid in enumerate(ids):
            fine = region_of[vid][finest]
            if fine in seen:
                continue
            seen.add(fine)
            if pos < min_entry_pos or pos + 1 >= len(ids):
                continue
            future = {}
            for p in range(pos + 1, len(ids)):
                future.setdefault(ids[p], p)
            if future:
                events.append({
                    "query": qid,
                    "pos": pos,
                    "ids": ids,
                    "regions": region_of[vid],
                    "future": future,
                })
    return events


def state_key(ev, kind, level=None):
    if kind == "stage_exact":
        return ("s", int(ev["pos"]))
    if kind == "flat_region":
        assert level is not None
        return ("r", int(level), int(ev["regions"][level]))
    if kind == "flat_region_stage":
        assert level is not None
        return ("rs", int(level), int(ev["regions"][level]), stage_band(int(ev["pos"])))
    raise ValueError(kind)


def build_counts(events, levels, with_stage):
    counts = collections.Counter()
    for ev in events:
        band = stage_band(int(ev["pos"]))
        for level in levels:
            key = (level, int(ev["regions"][level]), band) if with_stage else (
                level, int(ev["regions"][level])
            )
            counts[key] += 1
    return counts


def adaptive_key(ev, levels, counts, threshold, with_stage):
    band = stage_band(int(ev["pos"]))
    # Prefer the finest state with enough training evidence.
    for level in reversed(levels):
        key = (level, int(ev["regions"][level]), band) if with_stage else (
            level, int(ev["regions"][level])
        )
        if counts.get(key, 0) >= threshold:
            return ("hrs",) + key if with_stage else ("hr",) + key
    # The coarsest state is the mandatory backoff even when sparse.
    level = levels[0]
    key = (level, int(ev["regions"][level]), band) if with_stage else (
        level, int(ev["regions"][level])
    )
    return ("hrs",) + key if with_stage else ("hr",) + key


def learn_scores(events, key_fn):
    score = collections.defaultdict(lambda: collections.defaultdict(int))
    support = collections.defaultdict(lambda: collections.defaultdict(int))
    event_count = collections.Counter()
    total_mass = 0
    for ev in events:
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


def allocate(score, support, total_budget, per_state_cap):
    ranked = []
    for key, row in score.items():
        for vid, s in row.items():
            ranked.append((int(s), int(support[key][vid]), key, int(vid)))
    ranked.sort(key=lambda x: (-x[0], -x[1], repr(x[2]), x[3]))

    selected = collections.defaultdict(list)
    selected_count = collections.Counter()
    selected_mass = 0
    for s, sup, key, vid in ranked:
        if sum(selected_count.values()) >= total_budget:
            break
        if selected_count[key] >= per_state_cap:
            continue
        selected[key].append(vid)
        selected_count[key] += 1
        selected_mass += s
    return dict(selected), int(sum(selected_count.values())), int(selected_mass)


def evaluate(events, key_fn, maps, queries, base):
    available = 0
    usable_events = 0
    hit = 0
    total_skip = 0
    hit_skips = []
    candidate_cache = {}

    for ev in events:
        key = key_fn(ev)
        candidates = maps.get(key, ())
        if not candidates:
            continue
        available += 1
        e = int(ev["pos"])
        visited = set(ev["ids"][: e + 1])
        usable = [v for v in candidates if v not in visited]
        if not usable:
            continue
        usable_events += 1
        ck = tuple(usable)
        vecs = candidate_cache.get(ck)
        if vecs is None:
            vecs = np.asarray(base[np.asarray(usable, dtype=np.int64)], dtype=np.float32, order="C")
            candidate_cache[ck] = vecs
        q = np.asarray(queries[int(ev["query"])], dtype=np.float32)
        chosen = int(usable[int(np.argmax(vecs @ q))])
        p = ev["future"].get(chosen)
        if p is not None:
            skip = int(p) - e
            hit += 1
            total_skip += skip
            hit_skips.append(skip)

    n = len(events)
    return {
        "events": n,
        "map_available_fraction": available / n if n else 0.0,
        "usable_map_fraction": usable_events / n if n else 0.0,
        "routed_suffix_hit_fraction": hit / n if n else 0.0,
        "routed_mean_skip_all_events": total_skip / n if n else 0.0,
        "routed_skip_when_hit": qstats(hit_skips),
    }


def run_arm(train_events, held_events, key_fn, queries, base, budget, cap):
    score, support, event_count, total_mass = learn_scores(train_events, key_fn)
    maps, stored, selected_mass = allocate(score, support, budget, cap)
    metrics = evaluate(held_events, key_fn, maps, queries, base)
    metrics.update({
        "stored_entries": stored,
        "active_states_with_entries": len(maps),
        "train_events_per_state": qstats(list(event_count.values())),
        "selected_train_score_mass_fraction": selected_mass / total_mass if total_mass else 0.0,
    })
    return metrics


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
    ap.add_argument("--thresholds", default="32,64,128,256")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    records.sort(key=lambda r: int(r["query"]))
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query IDs must be dense")
    if not 1 <= args.train_queries < len(records):
        raise ValueError("invalid train split")

    nbase, bdim, base = load_fbin(args.base)
    nq, qdim, queries = load_fbin(args.queries)
    if qdim != bdim or nq < len(records):
        raise ValueError("query/base shape mismatch")

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    if levels != sorted(levels) or len(levels) < 2:
        raise ValueError("levels must be increasing")
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]
    if min(budgets) <= 0 or min(thresholds) <= 0 or args.per_state_cap <= 0:
        raise ValueError("invalid budget/threshold/cap")

    centers = {}
    for level in levels:
        path = args.router_root / f"nlist-{level}" / "router.bin"
        nlist, dim, c = load_centers(path)
        if nlist != level or dim != bdim:
            raise ValueError(f"router mismatch at {level}")
        centers[level] = c

    unique_ids = {int(v) for r in records for v in r["ids"]}
    if not unique_ids or max(unique_ids) >= nbase:
        raise ValueError("trace vertex outside base")
    region_of = assign_all(unique_ids, base, centers)

    train_records = records[:args.train_queries]
    held_records = records[args.train_queries:]
    finest = levels[-1]
    train_events = make_events(train_records, region_of, finest, args.min_entry_pos)
    held_events = make_events(held_records, region_of, finest, args.min_entry_pos)

    counts_region = build_counts(train_events, levels, with_stage=False)
    counts_region_stage = build_counts(train_events, levels, with_stage=True)

    result = {
        "split": {"train_queries": len(train_records), "heldout_queries": len(held_records)},
        "event_population": {
            "finest_region_count": finest,
            "definition": "first entry into each previously unseen finest-resolution region",
            "min_entry_position": args.min_entry_pos,
            "train_events": len(train_events),
            "heldout_events": len(held_events),
        },
        "fairness": {
            "same_total_stored_entry_budget_per_arm": True,
            "same_per_state_candidate_cap": args.per_state_cap,
            "budgets": budgets,
        },
        "levels": levels,
        "thresholds": thresholds,
        "results": {},
    }

    for budget in budgets:
        arms = {}
        arms["stage_exact"] = run_arm(
            train_events, held_events,
            lambda ev: state_key(ev, "stage_exact"),
            queries, base, budget, args.per_state_cap,
        )
        for level in levels:
            arms[f"flat_region_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, level=level: state_key(ev, "flat_region", level),
                queries, base, budget, args.per_state_cap,
            )
            arms[f"flat_region_stage_{level}"] = run_arm(
                train_events, held_events,
                lambda ev, level=level: state_key(ev, "flat_region_stage", level),
                queries, base, budget, args.per_state_cap,
            )

        for threshold in thresholds:
            key_r = lambda ev, threshold=threshold: adaptive_key(
                ev, levels, counts_region, threshold, False
            )
            key_rs = lambda ev, threshold=threshold: adaptive_key(
                ev, levels, counts_region_stage, threshold, True
            )
            arms[f"hier_region_t{threshold}"] = run_arm(
                train_events, held_events, key_r,
                queries, base, budget, args.per_state_cap,
            )
            arms[f"hier_region_stage_t{threshold}"] = run_arm(
                train_events, held_events, key_rs,
                queries, base, budget, args.per_state_cap,
            )

        best = sorted(
            (
                (v["routed_suffix_hit_fraction"], v["routed_mean_skip_all_events"], k)
                for k, v in arms.items()
            ),
            reverse=True,
        )[:10]
        result["results"][str(budget)] = {
            "arms": arms,
            "top10_by_routed_hit": [
                {"arm": k, "hit": h, "mean_skip": s} for h, s, k in best
            ],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
