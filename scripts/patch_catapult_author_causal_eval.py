#!/usr/bin/env python3
"""Add strictly causal, measured-suffix replay to the author-aligned CatapultDB
DiskANN adapter. Apply AFTER paper patch, snapshot patch, and author alignment.

The author's hash/LRU/insertion decisions remain untouched. This only changes
benchmark execution order (one completed query at a time) and restricts the
reported recall/cost statistics to the post-warm-up suffix. It does NOT claim
that the original author's in-memory engine is the SSD engine under test.
"""
from pathlib import Path
import argparse


def once(s: str, old: str, new: str, label: str) -> str:
    count = s.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected unique anchor, got {count}")
    return s.replace(old, new, 1)


GT_HELPER = r'''fn catapult_slice_ground_truth_context(
    ctx: &GroundTruthContext,
    start: usize,
) -> anyhow::Result<GroundTruthContext> {
    if ctx.gt_ids_variable_length.is_some() {
        anyhow::bail!("causal Catapult ground-truth slicing cannot handle filters");
    }
    let ids = ctx.gt_ids.as_ref().ok_or_else(|| anyhow::anyhow!("GT IDs absent"))?;
    let offset = start.checked_mul(ctx.gt_dim).ok_or_else(|| anyhow::anyhow!("GT slice overflow"))?;
    if offset > ids.len() {
        anyhow::bail!("Catapult GT start exceeds length");
    }
    Ok(GroundTruthContext {
        gt_ids: Some(ids[offset..].to_vec()),
        gt_ids_variable_length: None,
        gt_dists: ctx.gt_dists.as_ref().map(|x| x[offset..].to_vec()),
        gt_dim: ctx.gt_dim,
        recall_at: ctx.recall_at,
    })
}

'''


def patch(path: Path) -> None:
    s = path.read_text()
    anchor = '    let catapult_snapshot_load = std::env::var_os("DISKANN_CATAPULT_SNAPSHOT_LOAD");\n'
    conf = r'''    let catapult_causal_replay = std::env::var("DISKANN_CATAPULT_CAUSAL_REPLAY")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let catapult_causal_warmup = std::env::var("DISKANN_CATAPULT_CAUSAL_WARMUP")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4000);
    if catapult_causal_replay
        && (catapult_config.is_none()
            || catapult_causal_warmup >= num_queries
            || search_params.vector_filters_file.is_some())
    {
        anyhow::bail!("causal Catapult requires policy, no filters, and warmup < query count");
    }
'''
    s = once(s, anchor, conf + anchor, "causal environment config")
    s = once(s, "// Simplified internal structures to reduce parameter count\n",
             GT_HELPER + "// Simplified internal structures to reduce parameter count\n",
             "ground truth slicing")

    begin = '        let zipped = queries\n'
    end = '        let total_time = start.elapsed();\n'
    a = s.find(begin)
    b = s.find(end, a)
    if a < 0 or b < 0 or s.find(begin, a + 1) != -1:
        raise RuntimeError("Catapult parallel search block not uniquely found")
    parallel = s[a:b]
    closure = '|(((((q, vf), id_chunk), dist_chunk), stats), rc)| {'
    close = '\n            },\n        );\n'
    n = parallel.count(closure)
    if n != 1 or parallel.count(close) != 1:
        raise RuntimeError(f"Unexpected Catapult query-closure format: {n} opens")
    body = parallel.split(closure, 1)[1].rsplit(close, 1)[0]
    sequential = r'''        let mut catapult_measured_start = None::<Instant>;
        if catapult_causal_replay {
            // Explicit indexing guarantees a strict completed-query history.
            // Rayon (even with a one-thread pool) makes no ordering promise.
            for qi in 0..num_queries {
                if qi == catapult_causal_warmup {
                    catapult_measured_start = Some(Instant::now());
                }
                let q = queries.row(qi);
                let vf = &vector_filters[qi];
                let first = qi * search_params.recall_at as usize;
                let last = first + search_params.recall_at as usize;
                let id_chunk = &mut result_ids[first..last];
                let dist_chunk = &mut result_dists[first..last];
                let stats = &mut statistics_vec[qi];
                let rc = &mut result_counts[qi];
                (|| {
''' + body + r'''
                })();
                if has_any_search_failed.load(Ordering::Acquire) {
                    anyhow::bail!("Catapult query {} failed in causal replay", qi);
                }
            }
        } else {
''' + parallel + r'''        }
'''
    s = s[:a] + sequential + s[b:]
    s = once(s, end,
             '''        let total_time = if catapult_causal_replay {
            catapult_measured_start
                .ok_or_else(|| anyhow::anyhow!("Catapult measured suffix not started"))?
                .elapsed()
        } else {
            start.elapsed()
        };
''', "suffix wall time")
    old = '''        let mut search_result = DiskSearchResult::new(
            &statistics_vec,
            &result_ids,
            &result_counts,
            l,
            total_time.as_secs_f32(),
            num_queries,
            &gt_context,
        )?;
'''
    new = '''        let mut search_result = if catapult_causal_replay {
            let from = catapult_causal_warmup;
            let result_offset = from * search_params.recall_at as usize;
            let gt_suffix = catapult_slice_ground_truth_context(&gt_context, from)?;
            let measured_count = num_queries - from;
            eprintln!(
                "CATAPULT_CAUSAL_STATS L={} warmup={} measured={} residents={}",
                l, from, measured_count,
                catapult.as_ref().map(|c| c.resident_entries()).transpose()?.unwrap_or(0)
            );
            DiskSearchResult::new(
                &statistics_vec[from..],
                &result_ids[result_offset..],
                &result_counts[from..],
                l,
                total_time.as_secs_f32(),
                measured_count,
                &gt_suffix,
            )?
        } else {
            DiskSearchResult::new(
                &statistics_vec,
                &result_ids,
                &result_counts,
                l,
                total_time.as_secs_f32(),
                num_queries,
                &gt_context,
            )?
        };
'''
    s = once(s, old, new, "Catapult suffix statistics")
    path.write_text(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diskann", type=Path)
    a = ap.parse_args()
    root = a.diskann.resolve()
    target = root / "diskann-benchmark/src/disk_index/search.rs"
    if not target.is_file():
        raise SystemExit("Missing pinned DiskANN Rust benchmark")
    patch(target)
    print("CATAPULT_CAUSAL_REPLAY_PATCH_APPLIED", flush=True)


if __name__ == "__main__":
    main()
