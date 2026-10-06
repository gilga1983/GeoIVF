#!/usr/bin/env python3
"""Patch DiskANN benchmark to dump per-query returned result IDs for each L."""
from __future__ import annotations
import argparse
from pathlib import Path
from patch_diskann_start_points import once

def patch(path:Path)->None:
    s=path.read_text()
    anchor="""        let total_time = start.elapsed();
"""
    code=r'''        if let Some(base_path) = std::env::var_os("DISKANN_RESULT_IDS_FILE") {
            let base = std::path::PathBuf::from(base_path);
            let path = if search_params.search_list.len() == 1 {
                base
            } else {
                let stem = base
                    .file_stem()
                    .and_then(|x| x.to_str())
                    .unwrap_or("results")
                    .to_string();
                let ext = base
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("bin")
                    .to_string();
                base.with_file_name(format!("{stem}.L{l}.{ext}"))
            };
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let cols = search_params.recall_at as usize;
            let mut raw = Vec::with_capacity(8 + result_ids.len() * 4);
            raw.extend_from_slice(&(num_queries as u32).to_le_bytes());
            raw.extend_from_slice(&(cols as u32).to_le_bytes());
            for id in &result_ids {
                raw.extend_from_slice(&id.to_le_bytes());
            }
            std::fs::write(&path, raw)?;
            eprintln!("dumped returned result IDs to {:?}", path);
        }

'''
    s=once(s,anchor,code+anchor,"result-id dump")
    path.write_text(s)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("diskann",type=Path); args=ap.parse_args()
    p=args.diskann.resolve()/"diskann-benchmark/src/disk_index/search.rs"
    patch(p); print("patched benchmark for returned-ID dumps")
if __name__=="__main__": main()
