#!/usr/bin/env python3
"""Held-out diagnostic for region-conditioned navigation hints.

The experiment asks whether "where the traversal has arrived" is more
informative than traversal age alone. Vertices are assigned to a geometric
partition using the same spherical-kmeans centers as the portal router.

For each query, we create an event when the natural search first enters a new
region at or after --min-entry-pos. Training events score every later, unique
expanded vertex by its residual expansion distance from that entry. We learn
top-k onward-junction maps under four conditioning schemes:

  global        no state
  stage_band    coarse traversal-age bands
  stage_exact   exact expansion position
  region        entered graph region
  region_stage  entered region plus coarse age band

Held-out evaluation reports both candidate-set coverage and a conservative
routing proxy: among unvisited map candidates, choose the one with maximum
exact query inner product and ask whether it occurs later in the natural trace.
No runtime search behavior is modified by this diagnostic.
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
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise ValueError(f"bad fbin size: {path}")
    return rows, dim, np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, dim))


def load_centers(path: Path):
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:8] != ROUTER_MAGIC:
        raise ValueError(f"bad portal router: {path}")
    nlist, dim = struct.unpack("<II", raw[8:16])
    nfloat = nlist * dim
    centers = np.frombuffer(raw, dtype="<f4", count=nfloat, offset=16).reshape(nlist, dim).copy()
    expected = 16 + nfloat * 4 + nlist * 4 + nfloat * 4
    if len(raw) != expected:
        raise ValueError(f"bad portal router size: {path}")
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


def assign_regions(unique_ids, base, centers, batch=2048):
    ids = np.asarray(sorted(unique_ids), dtype=np.int64)
    out = np.empty(len(ids), dtype=np.int32)
    for lo in range(0, len(ids), batch):
        hi = min(len(ids), lo + batch)
        x = np.asarray(base[ids[lo:hi]], dtype=np.float32, order="C")
        out[lo:hi] = np.argmax(x @ centers.T, axis=1).astype(np.int32)
    return {int(v): int(r) for v, r in zip(ids, out)}


def make_events(records, region_of, min_entry_pos):
    events = []
    per_query = []
    regions_seen = collections.Counter()
    for rec in records:
        qid = int(rec["query"])
        ids = [int(v) for v in rec["ids"]]
        seen_regions = set()
        qevents = 0
        for pos, vid in enumerate(ids):
            region = region_of[vid]
            if region in seen_regions:
                continue
            seen_regions.add(region)
            if pos < min_entry_pos or pos + 1 >= len(ids):
                continue
            future = {}
            for p in range(pos + 1, len(ids)):
                future.setdefault(ids[p], p)
            if not future:
                continue
            events.append({
                "query": qid,
                "pos": pos,
                "region": region,
                "ids": ids,
                "future": future,
            })
            regions_seen[region] += 1
            qevents += 1
        per_query.append(qevents)
    return events, per_query, regions_seen


def key_for(event, scheme):
    if scheme == "global":
        return 0
    if scheme == "stage_band":
        return stage_band(event["pos"])
    if scheme == "stage_exact":
        return int(event["pos"])
    if scheme == "region":
        return int(event["region"])
    if scheme == "region_stage":
        return (int(event["region"]), stage_band(event["pos"]))
    raise ValueError(scheme)


def learn(events, scheme):
    score = collections.defaultdict(lambda: collections.defaultdict(int))
    support = collections.defaultdict(lambda: collections.defaultdict(int))
    event_count = collections.Counter()
    total_mass = 0
    for ev in events:
        key = key_for(ev, scheme)
        event_count[key] += 1
        e = int(ev["pos"])
        for vid, p in ev["future"].items():
            residual = int(p) - e
            if residual <= 0:
                continue
            score[key][int(vid)] += residual
            support[key][int(vid)] += 1
            total_mass += residual

    ranked = {}
    for key, scores in score.items():
        ranked[key] = [
            v for _, _, v in sorted(
                ((int(s), int(support[key][v]), int(v)) for v, s in scores.items()),
                key=lambda x: (-x[0], -x[1], x[2]),
            )
        ]
    return ranked, score, event_count, total_mass


def qstats(vals):
    if not vals:
        return {"min": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "min": float(a.min()),
        "median": float(np.median(a)),
        "mean": float(a.mean()),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(a.max()),
    }


def eval_scheme(events, scheme, ranked, score, event_count, total_mass, k, queries, base):
    candidate_cache = {}
    routed_hit = 0
    routed_skip = 0
    routed_hit_skips = []
    oracle_hit = 0
    oracle_max_skip = 0
    oracle_hit_skips = []
    available = 0
    unvisited_available = 0

    for ev in events:
        key = key_for(ev, scheme)
        candidates = ranked.get(key, ())[:k]
        if not candidates:
            continue
        available += 1

        future = ev["future"]
        e = int(ev["pos"])
        future_hits = [int(future[v]) - e for v in candidates if v in future]
        if future_hits:
            oracle_hit += 1
            best = max(future_hits)
            oracle_max_skip += best
            oracle_hit_skips.append(best)

        visited = set(ev["ids"][: e + 1])
        usable = [v for v in candidates if v not in visited]
        if not usable:
            continue
        unvisited_available += 1
        ck = tuple(usable)
        vecs = candidate_cache.get(ck)
        if vecs is None:
            vecs = np.asarray(base[np.asarray(usable, dtype=np.int64)], dtype=np.float32, order="C")
            candidate_cache[ck] = vecs
        q = np.asarray(queries[int(ev["query"])], dtype=np.float32)
        chosen = int(usable[int(np.argmax(vecs @ q))])
        p = future.get(chosen)
        if p is not None:
            skip = int(p) - e
            routed_hit += 1
            routed_skip += skip
            routed_hit_skips.append(skip)

    n = len(events)
    selected_mass = 0
    stored = 0
    for key, ids in ranked.items():
        chosen = ids[:k]
        stored += len(chosen)
        selected_mass += sum(int(score[key][v]) for v in chosen)

    counts = list(event_count.values())
    return {
        "events": n,
        "map_keys": len(ranked),
        "stored_ids_at_k": stored,
        "train_events_per_key": qstats(counts),
        "train_topk_score_mass_fraction": selected_mass / total_mass if total_mass else 0.0,
        "heldout_map_available_fraction": available / n if n else 0.0,
        "heldout_unvisited_map_available_fraction": unvisited_available / n if n else 0.0,
        "heldout_oracle_suffix_hit_fraction": oracle_hit / n if n else 0.0,
        "heldout_oracle_mean_max_skip_all_events": oracle_max_skip / n if n else 0.0,
        "heldout_oracle_skip_when_hit": qstats(oracle_hit_skips),
        "heldout_routed_suffix_hit_fraction": routed_hit / n if n else 0.0,
        "heldout_routed_mean_skip_all_events": routed_skip / n if n else 0.0,
        "heldout_routed_skip_when_hit": qstats(routed_hit_skips),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--router-root", type=Path, required=True)
    ap.add_argument("--regions", default="64,128")
    ap.add_argument("--train-queries", type=int, default=4000)
    ap.add_argument("--min-entry-pos", type=int, default=8)
    ap.add_argument("--topks", default="8,16,32")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = [json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if not records:
        raise ValueError("empty trace")
    records.sort(key=lambda r: int(r["query"]))
    if [int(r["query"]) for r in records] != list(range(len(records))):
        raise ValueError("trace query IDs must be dense from zero")
    if not 1 <= args.train_queries < len(records):
        raise ValueError("invalid train/heldout split")

    nbase, bdim, base = load_fbin(args.base)
    nq, qdim, queries = load_fbin(args.queries)
    if qdim != bdim or nq < len(records):
        raise ValueError("query/base shape mismatch")

    unique_ids = {int(v) for r in records for v in r["ids"]}
    if not unique_ids or max(unique_ids) >= nbase:
        raise ValueError("trace vertex outside base dataset")

    region_counts = [int(x) for x in args.regions.split(",") if x.strip()]
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    if not region_counts or not topks or min(topks) <= 0:
        raise ValueError("empty/invalid regions or topks")

    train_records = records[:args.train_queries]
    held_records = records[args.train_queries:]
    schemes = ("global", "stage_band", "stage_exact", "region", "region_stage")

    result = {
        "trace": str(args.trace),
        "queries": len(records),
        "split": {"train": len(train_records), "heldout": len(held_records)},
        "min_entry_position": args.min_entry_pos,
        "event_definition": "first entry into each previously unseen graph region per query, with entry position >= min_entry_position",
        "score": "sum of residual natural expansion distance from region-entry event",
        "routing_proxy": "highest exact query inner product among unvisited learned candidates; success iff chosen vertex occurs later in natural trace",
        "topks": topks,
        "regions": {},
    }

    for nlist in region_counts:
        router = args.router_root / f"nlist-{nlist}" / "router.bin"
        rn, rdim, centers = load_centers(router)
        if rn != nlist or rdim != bdim:
            raise ValueError(f"router mismatch for {nlist}")

        region_of = assign_regions(unique_ids, base, centers)
        train_events, train_per_q, train_region_events = make_events(
            train_records, region_of, args.min_entry_pos
        )
        held_events, held_per_q, held_region_events = make_events(
            held_records, region_of, args.min_entry_pos
        )
        if not train_events or not held_events:
            raise ValueError(f"nlist={nlist}: no late region-entry events")

        learned = {}
        for scheme in schemes:
            learned[scheme] = learn(train_events, scheme)

        by_k = {}
        for k in topks:
            metrics = {}
            for scheme in schemes:
                ranked, score, event_count, total_mass = learned[scheme]
                metrics[scheme] = eval_scheme(
                    held_events, scheme, ranked, score, event_count, total_mass,
                    k, queries, base
                )

            st = metrics["stage_exact"]
            rg = metrics["region"]
            metrics["region_vs_stage_exact"] = {
                "routed_hit_delta_points": 100.0 * (
                    rg["heldout_routed_suffix_hit_fraction"] -
                    st["heldout_routed_suffix_hit_fraction"]
                ),
                "routed_mean_skip_delta": (
                    rg["heldout_routed_mean_skip_all_events"] -
                    st["heldout_routed_mean_skip_all_events"]
                ),
                "oracle_hit_delta_points": 100.0 * (
                    rg["heldout_oracle_suffix_hit_fraction"] -
                    st["heldout_oracle_suffix_hit_fraction"]
                ),
                "oracle_mean_skip_delta": (
                    rg["heldout_oracle_mean_max_skip_all_events"] -
                    st["heldout_oracle_mean_max_skip_all_events"]
                ),
            }
            by_k[str(k)] = metrics

        result["regions"][str(nlist)] = {
            "unique_trace_vertices": len(unique_ids),
            "train_events": len(train_events),
            "heldout_events": len(held_events),
            "train_events_per_query": qstats(train_per_q),
            "heldout_events_per_query": qstats(held_per_q),
            "train_regions_with_events": len(train_region_events),
            "heldout_regions_with_events": len(held_region_events),
            "train_region_event_count": qstats(list(train_region_events.values())),
            "heldout_region_event_count": qstats(list(held_region_events.values())),
            "results_by_k": by_k,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
