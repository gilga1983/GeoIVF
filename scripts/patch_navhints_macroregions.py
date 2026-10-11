#!/usr/bin/env python3
"""Add 16-256 macro bucket views after the *verified* regional-history patch.

Original PQ routing, 16K chosen junctions, and search frontier are unchanged.
All macro variants share an identical train-only 512->G mapping; no extra
per-query medoid scoring. This is a correctness-first LOCALITY/CPU diagnostic.

Apply in this exact order on pinned DiskANN3:
 patch_diskann_hint_ivf_packed_direct.py
 patch_diskann_experience_core.py
 patch_navhints_regional_history.py
 patch_navhints_macroregions.py
"""
from pathlib import Path
import argparse

def once(s,a,b,label):
    n=s.count(a)
    if n!=1:
        raise RuntimeError(f"{label} needs one exact anchor but found {n}")
    return s.replace(a,b,1)

def patch_provider(path):
    s=path.read_text()
    s=once(s,
        "    region_cache_ids: Option<&'a [Vec<u32>]>,\n    region_cache_nprobe: usize,\n}\n",
        "    region_cache_ids: Option<&'a [Vec<u32>]>,\n    region_cache_nprobe: usize,\n"
        "    macro_regions: Option<&'a [u16]>,\n}\n",
        "macro coarse mapping in HintIvfSearch")
    old="            region_cache_nprobe: 1,\n"
    count=s.count(old)
    if count<1: raise RuntimeError("No default Hint-IVF constructors")
    s=s.replace(old,old+"            macro_regions: None,\n")
    s=once(s,
        "            region_cache_ids,\n            region_cache_nprobe,\n",
        "            region_cache_ids,\n            region_cache_nprobe,\n"
        "            macro_regions,\n",
        "macro regions in selected experience search")
    s=once(s,
        "        region_cache_ids: Option<&[Vec<u32>]>,\n        region_cache_nprobe: usize,\n",
        "        region_cache_ids: Option<&[Vec<u32>]>,\n        region_cache_nprobe: usize,\n"
        "        macro_regions: Option<&[u16]>,\n",
        "extend experience argument")
    s=once(s,
        '''                    if let Some(candidates) = groups.get(cell) {
                        regional_candidates.extend_from_slice(candidates);
                    }
''',
        '''                    let bucket = ivf.macro_regions
                        .map(|mapping| mapping[cell] as usize)
                        .unwrap_or(cell);
                    if let Some(candidates) = groups.get(bucket) {
                        regional_candidates.extend_from_slice(candidates);
                    }
''',
        "map PQ-ranked coarse cells to macro-history buckets")
    path.write_text(s)
    print("NAVHINTS_MACRO_PROVIDER_PATCH_APPLIED",flush=True)

def patch_benchmark(path):
    s=path.read_text()
    anchor='''    let experience_replay = std::env::var("DISKANN_EXPERIENCE_REPLAY")
'''
    config=r'''    // Load a training-only mapping of 512 original PQ routing cells into
    // coarser query-local history buckets. Original search routing is unchanged.
    let macro_region_map: Option<(Vec<u16>, usize)> =
        match std::env::var_os("DISKANN_LOCAL_MACRO_MAP_FILE") {
            Some(path) => {
                let index = hint_ivf.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("macro map requires trained Hint-IVF"))?;
                let raw = std::fs::read(std::path::Path::new(&path))?;
                if raw.len() < 16 || &raw[..8] != b"GHMAC001" {
                    anyhow::bail!("invalid macro-region map magic");
                }
                let ncoarse = u32::from_le_bytes(raw[8..12].try_into()?) as usize;
                let nmacro = u32::from_le_bytes(raw[12..16].try_into()?) as usize;
                if ncoarse != index.medoid_ids.len()
                    || nmacro < 1 || nmacro > ncoarse
                    || raw.len() != 16 + 2*ncoarse
                {
                    anyhow::bail!("macro region mapping shape mismatch");
                }
                let mut ids=Vec::<u16>::with_capacity(ncoarse);
                for b in raw[16..].chunks_exact(2) {
                    let id=u16::from_le_bytes([b[0],b[1]]);
                    if id as usize >= nmacro {
                        anyhow::bail!("macro region id outside requested partition");
                    }
                    ids.push(id);
                }
                eprintln!(
                    "NAVHINTS_MACRO_MAP ncoarse={} nmacro={} map_bytes={}",
                    ncoarse,nmacro,raw.len()
                );
                Some((ids,nmacro))
            }
            None => None,
        };
'''
    s=once(s,anchor,config+anchor,"load and validate macro-region map")
    s=once(s,
        '''            let mut history = if region_mode {
                vec![Vec::<(u32, usize)>::new(); index.medoid_ids.len()]
            } else { Vec::new() };
            let mut history_view = if region_mode {
                vec![Vec::<u32>::new(); index.medoid_ids.len()]
            } else { Vec::new() };
''',
        '''            let history_groups = macro_region_map.as_ref()
                .map(|(_, n)| *n)
                .unwrap_or(index.medoid_ids.len());
            let mut history = if region_mode {
                vec![Vec::<(u32, usize)>::new(); history_groups]
            } else { Vec::new() };
            let mut history_view = if region_mode {
                vec![Vec::<u32>::new(); history_groups]
            } else { Vec::new() };
''',
        "allocate macro-bucket rather than 512 original regional histories")

    s=once(s,
        '''                        local_region_nprobe,
                    )?;
''',
        '''                        local_region_nprobe,
                        macro_region_map.as_ref().map(|(m, _)| m.as_slice()),
                    )?;
''',
        "pass macro-region view")
    s=once(s,
        '''                        local_region_nprobe,
                        macro_region_map.as_ref().map(|(m, _)| m.as_slice()),
                    )?;

                let base_count =''',
        '''                        local_region_nprobe,
                        macro_region_map.as_ref().map(|(m, _)| m.as_slice()),
                    )?;
                // The selected region was taken from the same PQ-ranked coarse
                // starts, but is remapped for the history-only update.
                let selected_region = macro_region_map.as_ref()
                    .map(|(m, _)| m[selected_region] as usize)
                    .unwrap_or(selected_region);

                let base_count =''',
        "associate completed winner with its macro query region")
    path.write_text(s)
    print("NAVHINTS_MACRO_BENCHMARK_PATCH_APPLIED",flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("diskann",type=Path)
    a=p.parse_args()
    root=a.diskann.resolve()
    provider=root/"diskann-disk/src/search/provider/disk_provider.rs"
    bench=root/"diskann-benchmark/src/disk_index/search.rs"
    if not (provider.is_file() and bench.is_file()):
        raise RuntimeError("expected pinned DiskANN checkout")
    patch_provider(provider)
    patch_benchmark(bench)

if __name__=="__main__":
    main()
