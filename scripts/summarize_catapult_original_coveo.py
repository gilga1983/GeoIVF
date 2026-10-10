#!/usr/bin/env python3
"""Report the ACTUAL CatapultDB authors' native CPU/graph comparison.

Both Catapult and vanilla are measured within the SAME original-author Rust
library, same original-author C++ Vamana graph and frozen chronological Coveo
queries. This cannot be interpreted as a disk-I/O or same-RAM NavHints result.
"""
import json,statistics,sys
from pathlib import Path
LS=(12,20,40,80,160)
SEEDS=(42,123,2026)
MODES=("vanilla","catapult")
def avg(xs):return statistics.mean(xs)

def main():
    if len(sys.argv)!=2:raise SystemExit("usage: summarize_catapult_original_coveo.py ART")
    art=Path(sys.argv[1])
    results={}
    for rep,seed in enumerate(SEEDS):
        for L in LS:
            for mode in MODES:
                path=art/f"rep{rep}-L{L}-{mode}.json"
                if not path.is_file():raise ValueError(f"Missing authors' native result {path}")
                row=json.loads(path.read_text())
                assert row["source"]=="UNMODIFIED original MRandl/catapult-db library"
                assert row["mode"]==mode and row["seed"]==seed
                assert row["beam_width"]==L and row["k"]==10
                assert row["graph_nodes"]==31950
                assert row["measured_queries"]==5000 and row["warmup_queries"]==20000
                assert row["payload_dims"]==64 and row["original_unpadded_dims"]==50
                assert row["hash_bits"]==8 and row["bucket_capacity"]==40
                assert row["auxiliary_catapult_ID_payload_bytes"]==40960
                assert 0<=row["recall_at10_percent"]<=100
                assert row["wall_clock_mean_latency_us"]>0
                assert 0<=row["original_author_catapult_usage_fraction"]<=1
                results[rep,L,mode]=row
    records={}
    paired={}
    for L in LS:
        for mode in MODES:
            group=[results[rep,L,mode] for rep in range(3)]
            records[f"L{L}/{mode}"]={
                "mean_latency_us":round(avg([r["wall_clock_mean_latency_us"] for r in group]),3),
                "recall10_pct":round(avg([r["recall_at10_percent"] for r in group]),5),
                "mean_computed_dists":round(avg([r["original_author_distance_computations_per_query"] for r in group]),4),
                "mean_nodes_expanded":round(avg([r["original_author_nodes_visited_per_query"] for r in group]),4),
                "mean_catapult_used_percent":round(100*avg([r["original_author_catapult_usage_fraction"] for r in group]),3),
                "mean_single_thread_qps":round(avg([r["measured_query_only_qps"] for r in group]),2),
            }
        a=records[f"L{L}/vanilla"]
        b=records[f"L{L}/catapult"]
        paired[f"L{L}"]={
            "catapult_latency_reduction_percent":round(100*(1-b["mean_latency_us"]/a["mean_latency_us"]),4),
            "catapult_distance_comp_reduction_percent":round(100*(1-b["mean_computed_dists"]/a["mean_computed_dists"]),4),
            "recall_delta_pp":round(b["recall10_pct"]-a["recall10_pct"],5),
        }
    out={
        "experiment":"Original-author CatapultDB, not an independent port",
        "native_rust_author_sha":"a16eddd34b4339db5ec86e292470ce7929179bc3",
        "native_cpp_author_sha":"c1dbaecce5e7e02d02ad1493660c8048d965005d",
        "source":"https://github.com/MRandl/catapult-db",
        "source_experiments":"https://github.com/sacs-epfl/catapult-db-experiments",
        "dataset":"Coveo production chronological query stream, 31950 × 50, zero-padded to 64",
        "split":"5K static train disjoint; author Catapult learns only from 20K causal warmup; 5K measured",
        "graph":"Original authors' C++ Vamana memory graph, identical for both modes",
        "modes":"vanilla vs original Catapult; one chronological thread; three LSH seeds",
        "memory":"8-bit hash, 40 IDs per bucket, 40,960 auxiliary ID bytes + projection vectors/container overhead",
        "metric":"native original-author in-memory CPU search, NOT SSD I/O",
        "not_comparable_to":"DiskANN+NavHints disk-resident latency and SSD I/O without explicit architectural caveats",
        "records":records,"paired":paired,"success":True,
    }
    (art/"author-original-coveo-summary.json").write_text(json.dumps(out,indent=2)+"\n")
    lines=["# Original-author CatapultDB on chronological Coveo","",
           "This uses the **original authors' unmodified Rust search library** and their"
           " C++ Vamana graph builder, with original production query chronology.",
           "It is NOT a same-SSD/memory comparison with NavHints.",
           "",
           "| Beam L | Mode | Recall@10 | Latency (µs) | Native dist cmps | Nodes visited | Catapult used |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for L in LS:
        for mode in MODES:
            r=records[f"L{L}/{mode}"]
            lines.append(f"| {L} | {mode} | {r['recall10_pct']:.2f}% | "
                         f"{r['mean_latency_us']:.1f} | {r['mean_computed_dists']:.1f} | "
                         f"{r['mean_nodes_expanded']:.1f} | "
                         f"{r['mean_catapult_used_percent']:.1f}% |")
    lines+=["","## Original authors' Catapult versus own vanilla baseline"]
    for L in LS:
        r=paired[f"L{L}"]
        lines.append(f"- L{L}: latency reduction {r['catapult_latency_reduction_percent']:+.2f}%, "
                     f"distance-comparison reduction {r['catapult_distance_comp_reduction_percent']:+.2f}%, "
                     f"recall delta {r['recall_delta_pp']:+.3f} percentage points.")
    lines+=["",
      "## Scientific limitations",
      "The authors' original Rust benchmark uses an entirely memory-resident graph. "
      "We use the same source and original authors' DiskANN C++ Vamana builder, "
      "but do not claim physical SSD reads, byte-read ratios, or cross-backend speedups.",
      "Graph vectors are normalized 50D Coveo embeddings zero-padded to 64D only "
      "to meet the native SIMD alignment. This preserves L2 and cosine ranking.",
      "Every run processes 20K warmup requests before timing the 5K measured suffix; "
      "the source benchmark's parallel execution is replaced by a tiny caller of its "
      "unmodified library because parallel batch ordering would compromise causality.",
      "The original Catapult auxiliary ID budget is 40,960 bytes, plus projections "
      "and container overhead; it should not be described as memory-matched with "
      "NavHints' 102,940-byte RAM-only Core.",
      ""]
    report="\n".join(lines)
    (art/"author-original-coveo-summary.md").write_text(report)
    print("CATAPULT_ORIGINAL_AUTHOR_30_ARMS_VALIDATED",flush=True)
    print(report,flush=True)

if __name__=="__main__":main()
