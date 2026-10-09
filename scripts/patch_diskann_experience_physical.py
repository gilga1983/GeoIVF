#!/usr/bin/env python3
"""Instrument the *existing* frozen Sample1/Sample2 controller with real sector rewrites.

Run AFTER patch_diskann_experience_core.py against pinned Microsoft DiskANN.
Only a separately copied PubMed DiskANN graph is writable. The existing
in-memory per-hub hint map remains the routing lookup, so this quantifies the
physical read/modify/write/sync COST of page updates, not a crash-recovery
implementation or validated on-disk hint reader. No changes to graph edges.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from patch_diskann_start_points import once

def patch(path:Path)->None:
    s=path.read_text()
    marker="// Simplified internal structures to reduce parameter count\n"
    helper=r'''/// Issue one full, actual 4KiB page rewrite after a completed query.
/// The caller's graph copy is verified separately: 1 node/sector and >64
/// unused terminal bytes in each sector. All graph/PQ bytes are preserved.
fn navhints_physical_page_rewrite(
    file: &std::fs::File,
    hub: u32,
    ids: &[u32],
    nodes_per_sector: u64,
    node_len: u64,
    force_sync: bool,
) -> anyhow::Result<u64> {
    use std::os::unix::fs::FileExt;
    const SECTOR: usize = 4096;
    const TRAILER: usize = 64;
    if nodes_per_sector != 1 || ids.is_empty() || ids.len() > 10 {
        anyhow::bail!("unsupported physical-persistence copy: nodes_per_sector={}, ids={}", nodes_per_sector, ids.len());
    }
    if node_len as usize > SECTOR - TRAILER {
        anyhow::bail!("graph record overlaps reserved physical hint trailer");
    }
    let offset=(hub as u64 + 1)
        .checked_mul(SECTOR as u64)
        .ok_or_else(|| anyhow::anyhow!("physical page offset overflow"))?;
    let mut buf=[0u8;SECTOR];
    file.read_exact_at(&mut buf, offset)?;
    let trailer=&mut buf[SECTOR-TRAILER..];
    if trailer[..8] != *b"NHWT0001" && trailer.iter().any(|b| *b != 0) {
        anyhow::bail!("nonzero graph-index padding at hub {}; refusing overwrite",hub);
    }
    trailer.fill(0);
    trailer[..8].copy_from_slice(b"NHWT0001");
    trailer[8..12].copy_from_slice(&hub.to_le_bytes());
    trailer[12..16].copy_from_slice(&(ids.len() as u32).to_le_bytes());
    for (i,&id) in ids.iter().enumerate() {
        trailer[16+i*4..20+i*4].copy_from_slice(&id.to_le_bytes());
    }
    file.write_all_at(&buf, offset)?;
    if force_sync {
        file.sync_data()?;
    }
    Ok(SECTOR as u64)
}

'''
    s=once(s,marker,helper+marker,"physical page write helper")
    old="""            let mut eval_write_filler_slots = 0usize;

            for qi in 0..num_queries {
"""
    new="""            let mut eval_write_filler_slots = 0usize;
            let physical_graph_copy = std::env::var_os("DISKANN_EXPERIENCE_PHYSICAL_COPY");
            let mut physical_graph_file = physical_graph_copy.as_ref()
                .map(|path| std::fs::OpenOptions::new().read(true).write(true).open(path))
                .transpose()?;
            let physical_nps = std::env::var("DISKANN_EXPERIENCE_PHYSICAL_NODES_PER_SECTOR")
                .ok().map(|s| s.parse::<u64>()).transpose()?.unwrap_or(1);
            let physical_node_len = std::env::var("DISKANN_EXPERIENCE_PHYSICAL_NODE_LEN")
                .ok().map(|s| s.parse::<u64>()).transpose()?.unwrap_or(0);
            let physical_force_sync = std::env::var("DISKANN_EXPERIENCE_PHYSICAL_SYNC")
                .ok().is_some_and(|x| x == "1");
            let mut physical_pages = 0u64;
            let mut physical_bytes = 0u64;
            let mut physical_write_us = 0u128;
            let mut physical_eval_pages = 0u64;
            let mut physical_eval_write_us = 0u128;
            if physical_graph_file.is_some() {
                if experience_hub_capacity > 10 || physical_node_len == 0 {
                    anyhow::bail!("invalid graph-copy guard for physical persistence");
                }
            }

            for qi in 0..num_queries {
"""
    s=once(s,old,new,"physical setup")
    marker2="""                page_hubs.insert(selected_hub, page.clone());

                writes += 1;"""
    new2="""                page_hubs.insert(selected_hub, page.clone());
                if let Some(ref mut graph_copy) = physical_graph_file {
                    let t=Instant::now();
                    let bytes=navhints_physical_page_rewrite(
                        graph_copy, selected_hub, &page, physical_nps,
                        physical_node_len, physical_force_sync)?;
                    let elapsed=t.elapsed().as_micros();
                    physical_pages+=1;
                    physical_bytes+=bytes;
                    physical_write_us+=elapsed;
                    if qi >= experience_warmup {
                        physical_eval_pages+=1;
                        physical_eval_write_us+=elapsed;
                        // Query statistics from the upstream graph exclude post-search
                        // mutations; explicitly include the synchronous write cost.
                        statistics_vec[qi].total_execution_time_us += elapsed;
                        statistics_vec[qi].io_time_us += elapsed;
                    }
                }

                writes += 1;"""
    s=once(s,marker2,new2,"real writes")
    marker3="""            let final_page_slots: usize = page_hubs.values().map(Vec::len).sum();
            eprintln!(
                "EXPERIENCE_STATS"""
    repl3="""            let final_page_slots: usize = page_hubs.values().map(Vec::len).sum();
            eprintln!(
                "PHYSICAL_PERSISTENCE_STATS L={} enabled={} sync={} pages={} bytes={} write_us={} eval_pages={} eval_write_us={} copied_index_only=1",
                l, if physical_graph_file.is_some(){1}else{0},
                if physical_force_sync{1}else{0}, physical_pages,
                physical_bytes, physical_write_us, physical_eval_pages,
                physical_eval_write_us,
            );
            eprintln!(
                "EXPERIENCE_STATS"""
    s=once(s,marker3,repl3,"physical write summary")
    path.write_text(s)
def main():
    a=argparse.ArgumentParser()
    a.add_argument("diskann",type=Path)
    args=a.parse_args()
    p=args.diskann.resolve()/"diskann-benchmark/src/disk_index/search.rs"
    if not p.is_file():raise SystemExit(f"missing patched benchmark source: {p}")
    patch(p)
    print("Added validated 4KiB graph-page read-modify-write cost instrumentation")
if __name__=="__main__":main()
