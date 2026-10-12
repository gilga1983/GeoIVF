#!/usr/bin/env python3
"""Instrument *complete* causal requests without changing NavHints or Catapult policy.

Apply to pinned Microsoft DiskANN AFTER the existing NavHints regional-history
patch (mode=nav) or source-aligned Catapult causal patch (mode=catapult).

CSV covers the measured suffix only. service_us is a monotonic per-request
wall-clock interval encompassing routing, SSD search, result materialization,
and *post-query* feedback. engine_us is the engine/adapter's existing search
timer, not an independent measurement. remainder_us includes marshalling and
feedback, and must NOT be described as exclusively feedback.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: wanted one source anchor, found {count}")
    return source.replace(old, new, 1)


GUARD = r"""
// Captures the whole query scope even when the causal loop uses `continue`.
// The storage write occurs on Drop, before the next query starts.
struct CompleteRequestTimer<'a> {
    start: std::time::Instant,
    destination: &'a mut u128,
}
impl<'a> CompleteRequestTimer<'a> {
    fn new(destination: &'a mut u128) -> Self {
        Self { start: std::time::Instant::now(), destination }
    }
}
impl Drop for CompleteRequestTimer<'_> {
    fn drop(&mut self) {
        *self.destination = self.start.elapsed().as_micros();
    }
}

"""


def profile_code(measured_start: str, engine: str, indent: str) -> str:
    # Profiles are emitted *after* the timed loop, so CSV formatting, disk
    # output and JSON aggregation cannot contaminate individual timings.
    body = r"""if let Some(profile_dir) = std::env::var_os("DISKANN_SERVICE_PROFILE_DIR") {
    let profile_dir = std::path::PathBuf::from(profile_dir);
    std::fs::create_dir_all(&profile_dir)?;
    let mut csv = String::from("query_index,service_us,engine_us,remainder_us,io_us,pq_preprocess_us,reported_cpu_us,reads,comparisons,hops\n");
    let mut count = 0usize;
    for qi in MEASURED_START..num_queries {
        let st = &statistics_vec[qi];
        let elapsed = complete_service_us[qi];
        let native = st.total_execution_time_us;
        if elapsed == 0 || native == 0 || elapsed < native {
            anyhow::bail!("complete-service accounting invalid at L={} query={} service={} native={}", l, qi, elapsed, native);
        }
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            qi, elapsed, native, elapsed - native,
            st.io_time_us, st.query_pq_preprocess_time_us, st.cpu_time_us,
            st.total_io_operations, st.total_comparisons, st.search_hops,
        ));
        count += 1;
    }
    if count == 0 {
        anyhow::bail!("complete-service profile has empty measured suffix");
    }
    let dest = profile_dir.join(format!("L{}.csv", l));
    std::fs::write(&dest, csv)?;
    eprintln!("COMPLETE_SERVICE_PROFILE engine=ENGINE L={} measured={} path={}",
              l, count, dest.display());
}
"""
    body = body.replace("MEASURED_START", measured_start).replace("ENGINE", engine)
    return "".join(indent + line if line.strip() else line for line in body.splitlines(keepends=True))


def patch_nav(source: str) -> str:
    if "struct CompleteRequestTimer" in source:
        raise RuntimeError("complete-service timer already installed in NavHints source")
    s = once(source, "// Simplified internal structures to reduce parameter count\n",
             GUARD + "// Simplified internal structures to reduce parameter count\n",
             "NavHints service-scope guard")
    s = once(s,
        '    let experience_replay = std::env::var("DISKANN_EXPERIENCE_REPLAY")\n',
        '''    // Service experiment only: same engine and frozen state, three arms.
    // Baseline/Junctions reuse the causal request loop but learn no history.
    let service_mode = std::env::var("DISKANN_SERVICE_MODE")
        .unwrap_or_else(|_| "navhints".to_string());
    if !matches!(service_mode.as_str(), "baseline" | "junctions" | "navhints") {
        anyhow::bail!("invalid DISKANN_SERVICE_MODE");
    }
    let experience_replay = std::env::var("DISKANN_EXPERIENCE_REPLAY")
''', "NavHints service arm")
    s = once(s,
        '            for qi in 0..num_queries {\n                let q = queries.row(qi);\n',
        '''            let mut complete_service_us = vec![0u128; num_queries];
            for qi in 0..num_queries {
                let _service_timer = CompleteRequestTimer::new(&mut complete_service_us[qi]);
                let q = queries.row(qi);
''', "NavHints per-query scope")
    start = "                let (search_result, selected_hub, selected_region, cache_top) =\n                    searcher.search_with_hint_ivf_experience(\n"
    a = s.find(start)
    if a < 0 or s.find(start, a + 1) != -1:
        raise RuntimeError("NavHints causal dispatch not unique")
    end = "                    )?;\n"
    b = s.find(end, a)
    if b < 0:
        raise RuntimeError("NavHints causal call terminator missing")
    original = s[a:b + len(end)]
    if "local_region_nprobe," not in original or "&page_hubs," not in original:
        raise RuntimeError("not the selected regional-history Core implementation")
    indented = "                } else {\n" + original.replace(
        "                let (search_result, selected_hub, selected_region, cache_top) =\n", ""
    ).replace("                    )?;\n", "                    )?\n") + "                };\n"
    replacement = '''                let (search_result, selected_hub, selected_region, cache_top) =
                    if service_mode == "baseline" {
                        (searcher.search(
                            q, search_params.recall_at, l,
                            Some(search_params.beam_width), mode,
                        )?, 0u32, 0usize, Vec::<u32>::new())
                    } else if service_mode == "junctions" {
                        (searcher.search_with_hint_ivf(
                            q, search_params.recall_at, l,
                            Some(search_params.beam_width),
                            &index.medoid_ids, &index.coarse_local_ids,
                            &index.coarse_pq_codes, &index.offsets,
                            &index.hint_ids, hint_ivf_nprobe,
                        )?, 0u32, 0usize, Vec::<u32>::new())
''' + indented
    s = s[:a] + replacement + s[b + len(end):]
    s = once(s, '            let persisted_hubs = page_hubs.len();\n',
             profile_code("experience_warmup", "navhints", "            ")
             + '            let persisted_hubs = page_hubs.len();\n',
             "NavHints measured-suffix service profile")
    return s


def patch_catapult(source: str) -> str:
    if "struct CompleteRequestTimer" in source:
        raise RuntimeError("complete-service timer already installed in Catapult source")
    s = once(source, "// Simplified internal structures to reduce parameter count\n",
             GUARD + "// Simplified internal structures to reduce parameter count\n",
             "Catapult service-scope guard")
    s = once(s,
        '        let mut catapult_measured_start = None::<Instant>;\n',
        '''        let mut complete_service_us = vec![0u128; num_queries];
        let mut catapult_measured_start = None::<Instant>;
''', "Catapult profile storage")
    s = once(s,
        '                let q = queries.row(qi);\n',
        '''                let _service_timer = CompleteRequestTimer::new(&mut complete_service_us[qi]);
                let q = queries.row(qi);
''', "Catapult per-query scope")
    marker = '        let total_time = if catapult_causal_replay {\n'
    s = once(s, marker,
             '''        if catapult_causal_replay {
''' + profile_code("catapult_causal_warmup", "catapult", "            ")
             + '''        }
''' + marker, "Catapult measured-suffix service profile")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diskann", type=Path)
    ap.add_argument("--engine", required=True, choices=("nav", "catapult"))
    args = ap.parse_args()
    file = args.diskann / "diskann-benchmark/src/disk_index/search.rs"
    if not file.is_file():
        raise FileNotFoundError(file)
    source = file.read_text()
    result = patch_nav(source) if args.engine == "nav" else patch_catapult(source)
    if result == source:
        raise RuntimeError("service patch did not change search source")
    file.write_text(result)
    print(f"COMPLETE_SERVICE_PATCH_PASS engine={args.engine}", flush=True)


if __name__ == "__main__":
    main()
