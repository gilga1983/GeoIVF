#!/usr/bin/env python3
"""Validate and summarize paired Starling baseline, score-only, Core and oracle.

All values are independent native Starling query records. Oracle uses exact
ground truth only as a diagnostic upper bound. It is NOT an ANN competitor.
"""
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


MODES = ("baseline", "score_only", "core16k_recent512", "oracle")
WIDTHS = (20, 40, 80)
REPS = (0, 1, 2)
COUNT = 1000


def avg(rows, field):
    return sum(float(r[field]) for r in rows) / len(rows)


def metric(rows, field, default=None):
    return sum(int(r[field] != default) for r in rows) / len(rows)


def read_files(path):
    raw = {}
    for file in sorted(path.glob("rep*-*/L*.csv")):
        m = re.fullmatch(r"rep([0-9]+)-(.+)", file.parent.name)
        w = re.fullmatch(r"L([0-9]+).csv", file.name)
        if not m or not w:
            raise RuntimeError(f"bad artifact path {file}")
        rep, mode, L = int(m.group(1)), m.group(2), int(w.group(1))
        if rep not in REPS or mode not in MODES or L not in WIDTHS:
            raise RuntimeError(f"unexpected mode {file}")
        key = (rep, mode, L)
        if key in raw:
            raise RuntimeError(f"duplicate arm {file}")
        with file.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != COUNT:
            raise RuntimeError(f"{file} has {len(rows)} measured rows, expected {COUNT}")
        if [int(r["query"]) for r in rows] != list(range(COUNT)):
            raise RuntimeError(f"noncausal or reordered queries in {file}")
        raw[key] = rows
    expected = {(rep, m, L) for rep in REPS for m in MODES for L in WIDTHS}
    if set(raw) != expected:
        raise RuntimeError(f"missing modes={sorted(expected-set(raw))}, extras={sorted(set(raw)-expected)}")
    return raw


def summarize(raw):
    result = {
        "protocol": {
            "dataset": "BigANN-10M",
            "system": "original pinned Starling with temporary native entry diagnostics",
            "warmup": 4000, "measured_per_rep": COUNT, "repetitions": 3,
            "widths": list(WIDTHS), "modes": list(MODES),
            "oracle": "exact heldout GT nearest neighbor; only diagnostic headroom",
            "timer": "native per-query total_us and logical page-search n_ios",
            "caveat": "GT oracle cannot be used as an ANN competitor; latency is subject to NVMe/CPU variation.",
        },
        "arms": {}, "paired": {}, "score_only_routes_preserved": True
    }
    for L in WIDTHS:
        for mode in MODES:
            rows = [r for rep in REPS for r in raw[(rep, mode, L)]]
            key = f"L{L}/{mode}"
            result["arms"][key] = {
                "queries": len(rows),
                "recall_note": "See native Starling per-rep summary; do not infer recall from CSV",
                "mean_ios": round(avg(rows, "ios"), 5),
                "mean_total_us": round(avg(rows, "total_us"), 3),
                "mean_cpu_us": round(avg(rows, "cpu_us"), 3),
                "mean_cache_hits": round(avg(rows, "cache_hits"), 4),
                "mean_recent_scoring_us": round(avg(rows, "recent_ns")/1000, 3),
                "mean_junction_scoring_us": round(avg(rows, "junction_ns")/1000, 3),
                "mean_oracle_scoring_us": round(avg(rows, "oracle_ns")/1000, 3),
                "recent_beats_best_native_fraction": round(
                    sum(float(r["recent_dist"])>=0 and
                        float(r["recent_dist"]) < float(r["native_best"])
                        for r in rows) / len(rows), 5),
                "junction_beats_best_native_fraction": round(
                    sum(float(r["junction_dist"])>=0 and
                        float(r["junction_dist"]) < float(r["native_best"])
                        for r in rows) / len(rows), 5),
                "recent_candidate_popped_fraction": round(
                    sum(int(r["recent_popped"])>0 for r in rows)/len(rows), 5),
                "junction_candidate_popped_fraction": round(
                    sum(int(r["junction_popped"])>0 for r in rows)/len(rows), 5),
                "oracle_candidate_popped_fraction": round(
                    sum(int(r["oracle_popped"])>0 for r in rows)/len(rows), 5),
                "recent_page_read_fraction": round(
                    sum(int(r["recent_page_read"])>0 for r in rows)/len(rows), 5),
                "junction_page_read_fraction": round(
                    sum(int(r["junction_page_read"])>0 for r in rows)/len(rows), 5),
                "oracle_page_read_fraction": round(
                    sum(int(r["oracle_page_read"])>0 for r in rows)/len(rows), 5),
            }
        changes = defaultdict(int)
        paired_deltas = defaultdict(list)
        for rep in REPS:
            base = raw[(rep, "baseline", L)]
            score = raw[(rep, "score_only", L)]
            core = raw[(rep, "core16k_recent512", L)]
            oracle = raw[(rep, "oracle", L)]
            for idx in range(COUNT):
                b, s, c, o = base[idx], score[idx], core[idx], oracle[idx]
                page_fields = [f"p{j}" for j in range(8)]
                if int(b["ios"]) != int(s["ios"]) or any(
                    b[p] != s[p] for p in page_fields
                ):
                    changes["score_only_route_changed"] += 1
                if int(c["ios"]) != int(b["ios"]):
                    changes["core_io_changed"] += 1
                if any(c[p] != b[p] for p in page_fields):
                    changes["core_first_pages_changed"] += 1
                if any(o[p] != b[p] for p in page_fields):
                    changes["oracle_first_pages_changed"] += 1
                paired_deltas["score_only_minus_baseline_us"].append(
                    float(s["total_us"])-float(b["total_us"]))
                paired_deltas["core_minus_baseline_us"].append(
                    float(c["total_us"])-float(b["total_us"]))
                paired_deltas["oracle_minus_baseline_us"].append(
                    float(o["total_us"])-float(b["total_us"]))
                paired_deltas["core_reads_saved"].append(
                    float(b["ios"])-float(c["ios"]))
                paired_deltas["oracle_reads_saved"].append(
                    float(b["ios"])-float(o["ios"]))
        result["paired"][f"L{L}"] = {
            **{f"{key}_fraction": round(val/(COUNT*len(REPS)), 5)
               for key, val in changes.items()},
            **{key+"_mean": round(statistics.mean(vals), 4)
               for key, vals in paired_deltas.items()},
        }
        if changes["score_only_route_changed"]:
            result["score_only_routes_preserved"] = False
    return result


def markdown(result):
    lines = [
        "# Native Starling entry-point diagnostic",
        "",
        "Full BigANN-10M, disjoint 5K training, 4K heldout warmup and 1K "
        "measured per repetition, three rotated repetitions.",
        "",
        "The GT oracle is **diagnostic headroom only**, not a competing ANN algorithm. "
        "Comparison is within Starling's pinned original index and search.",
        "",
        "| L | mode | I/Os/query | per-query latency (us) | CPU (us) | "
        "Recent scores (us) | Junction scores (us) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for L in WIDTHS:
        for mode in MODES:
            r = result["arms"][f"L{L}/{mode}"]
            lines.append(f"| {L} | {mode} | {r['mean_ios']:.3f} | "
                         f"{r['mean_total_us']:.1f} | {r['mean_cpu_us']:.1f} | "
                         f"{r['mean_recent_scoring_us']:.2f} | "
                         f"{r['mean_junction_scoring_us']:.2f} |")
    lines += ["", f"Score-only preserves native route: **{result['score_only_routes_preserved']}**.", ""]
    for L in WIDTHS:
        p = result["paired"][f"L{L}"]
        core = result["arms"][f"L{L}/core16k_recent512"]
        ora = result["arms"][f"L{L}/oracle"]
        lines += [
            f"## L={L}",
            f"- Core reads saved per query: {p['core_reads_saved_mean']:+.3f}. "
            f"Oracle reads saved per query: {p['oracle_reads_saved_mean']:+.3f}.",
            f"- Score-only extra latency: {p['score_only_minus_baseline_us_mean']:+.2f} us/query. "
            f"Core total latency difference: {p['core_minus_baseline_us_mean']:+.2f} us/query.",
            f"- Core changes first eight physical pages on "
            f"{100*p.get('core_first_pages_changed_fraction',0):.1f}% of queries; "
            f"oracle on {100*p.get('oracle_first_pages_changed_fraction',0):.1f}%.",
            f"- Full Core: recent beats native best "
            f"{100*core['recent_beats_best_native_fraction']:.1f}%, "
            f"junction beats native best "
            f"{100*core['junction_beats_best_native_fraction']:.1f}%; "
            f"recent candidate popped {100*core['recent_candidate_popped_fraction']:.1f}%, "
            f"junction popped {100*core['junction_candidate_popped_fraction']:.1f}%.",
            f"- Oracle candidate popped on {100*ora['oracle_candidate_popped_fraction']:.1f}% of queries.",
            "",
        ]
    lines += [
        "## Interpretation boundaries",
        "The oracle injects GT into the normal Starling frontier. Its I/O difference measures only "
        "the headroom of this interface, not what a deployable router can attain.",
        "Logical I/Os follow Starling's native page-search counters; latency includes hint CPU work. "
        "External physical NVMe accounting and cross-system comparisons remain separate.",
        "The score-only path must preserve all original page choices before its latency overhead is interpretable.",
        "",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: summarizer DIAG_DIR OUT_JSON OUT_MD")
    raw = read_files(Path(sys.argv[1]))
    result = summarize(raw)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2)+"\n")
    Path(sys.argv[3]).write_text(markdown(result))
    if not result["score_only_routes_preserved"]:
        raise RuntimeError("INVALID CONTROL: score-only changed Starling page choices")
    print("STARLING_ENTRY_DIAGNOSTIC_SUMMARY_VERIFIED", flush=True)
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
