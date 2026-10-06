#!/usr/bin/env python3
"""Extend progressive NavHints with three independently routed retained-hint banks.

Primary Hint-IVF chooses the sole start. Extra banks are routed once per query,
retain fixed per-bank quotas, and become eligible only after configured natural
beam boundaries. The graph, fixed-L gate, and one-admission-per-native-beam
policy are unchanged.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

from patch_diskann_start_points import once


def replace_n(s, old, new, n, label):
    c=s.count(old)
    if c!=n: raise RuntimeError(f"{label}: expected {n} matches, found {c}")
    return s.replace(old,new)


def patch_glue(root: Path):
    p=root/"diskann/src/graph/glue.rs"; s=p.read_text()
    s=once(s,
"""    fn progressive_hint_distances<F>(
        &mut self,
        _f: F,
""",
"""    fn progressive_hint_distances<F>(
        &mut self,
        _hops: u32,
        _f: F,
""","portfolio progressive hook signature")
    p.write_text(s)

    p=root/"diskann/src/graph/index.rs"; s=p.read_text()
    s=once(s,
"""                    .progressive_hint_distances(|id, distance| {
""",
"""                    .progressive_hint_distances(scratch.hops, |id, distance| {
""","portfolio progressive hook call")
    p.write_text(s)


def patch_provider(path: Path):
    s=path.read_text()

    # Strategy and accessor both carry the three optional banks.
    s=replace_n(s,
"""    progressive_hint_limit: usize,
""",
"""    progressive_hint_limit: usize,
    temporal_hint_ivf1: Option<HintIvfSearch<'a>>,
    temporal_hint_limit1: usize,
    temporal_hint_hop1: u32,
    temporal_hint_ivf2: Option<HintIvfSearch<'a>>,
    temporal_hint_limit2: usize,
    temporal_hint_hop2: u32,
    temporal_hint_ivf3: Option<HintIvfSearch<'a>>,
    temporal_hint_limit3: usize,
    temporal_hint_hop3: u32,
""",2,"portfolio strategy/accessor fields")

    s=once(s,
"""    retained_hint_ranked: Vec<(u32, f32)>,
""",
"""    retained_hint_ranked: Vec<(u32, f32, u32)>,
""","activation-aware retained state")

    s=once(s,
"""            progressive_hint_limit: strategy.progressive_hint_limit,
            retained_hint_ranked: Vec::new(),
""",
"""            progressive_hint_limit: strategy.progressive_hint_limit,
            temporal_hint_ivf1: strategy.temporal_hint_ivf1,
            temporal_hint_limit1: strategy.temporal_hint_limit1,
            temporal_hint_hop1: strategy.temporal_hint_hop1,
            temporal_hint_ivf2: strategy.temporal_hint_ivf2,
            temporal_hint_limit2: strategy.temporal_hint_limit2,
            temporal_hint_hop2: strategy.temporal_hint_hop2,
            temporal_hint_ivf3: strategy.temporal_hint_ivf3,
            temporal_hint_limit3: strategy.temporal_hint_limit3,
            temporal_hint_hop3: strategy.temporal_hint_hop3,
            retained_hint_ranked: Vec::new(),
""","portfolio accessor constructor")

    # All non-progressive strategy constructors disable the portfolio.
    old="""            progressive_hint_limit: 0,
        }
"""
    new="""            progressive_hint_limit: 0,
            temporal_hint_ivf1: None,
            temporal_hint_limit1: 0,
            temporal_hint_hop1: 0,
            temporal_hint_ivf2: None,
            temporal_hint_limit2: 0,
            temporal_hint_hop2: 0,
            temporal_hint_ivf3: None,
            temporal_hint_limit3: 0,
            temporal_hint_hop3: 0,
        }
"""
    count=s.count(old)
    if count<2: raise RuntimeError(f"portfolio disabled constructors: expected >=2, found {count}")
    s=s.replace(old,new)

    # Extend the progressive strategy constructor.
    s=once(s,
"""        hint_ivf: HintIvfSearch<'a>,
        progressive_hint_limit: usize,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
""",
"""        hint_ivf: HintIvfSearch<'a>,
        progressive_hint_limit: usize,
        temporal_hint_ivf1: Option<HintIvfSearch<'a>>,
        temporal_hint_limit1: usize,
        temporal_hint_hop1: u32,
        temporal_hint_ivf2: Option<HintIvfSearch<'a>>,
        temporal_hint_limit2: usize,
        temporal_hint_hop2: u32,
        temporal_hint_ivf3: Option<HintIvfSearch<'a>>,
        temporal_hint_limit3: usize,
        temporal_hint_hop3: u32,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
""","portfolio strategy signature")
    s=once(s,
"""            progressive_hints: true,
            progressive_hint_limit,
        }
""",
"""            progressive_hints: true,
            progressive_hint_limit,
            temporal_hint_ivf1,
            temporal_hint_limit1,
            temporal_hint_hop1,
            temporal_hint_ivf2,
            temporal_hint_limit2,
            temporal_hint_hop2,
            temporal_hint_ivf3,
            temporal_hint_limit3,
            temporal_hint_hop3,
        }
""","portfolio strategy fields")

    # Route one extra bank into a fixed quota. It never chooses the graph start.
    anchor="""    async fn start_point_distances<F>(&mut self, mut f: F) -> ANNResult<()>
"""
    helper=r'''    fn retain_temporal_bank(
        &mut self,
        ivf: HintIvfSearch<'_>,
        limit: usize,
        activation_hop: u32,
    ) -> ANNResult<()> {
        const MAX_NPROBE: usize = 64;
        const MAX_TOPK: usize = 16;
        if limit == 0 || limit > MAX_TOPK
            || ivf.nprobe == 0
            || ivf.nprobe > MAX_NPROBE
            || ivf.nprobe > ivf.medoid_ids.len()
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid temporal Hint-IVF parameters",
            ));
        }

        let mut coarse_storage =
            [(f32::INFINITY, usize::MAX, u32::MAX); MAX_NPROBE];
        let coarse = &mut coarse_storage[..ivf.nprobe];
        self.pq_distances_packed(
            ivf.coarse_local_ids,
            ivf.coarse_pq_codes,
            |distance, local_id| {
                let cell = local_id as usize;
                let id = ivf.medoid_ids[cell];
                let worst = coarse[ivf.nprobe - 1];
                if distance
                    .total_cmp(&worst.0)
                    .then_with(|| cell.cmp(&worst.1))
                    .is_ge()
                {
                    return;
                }
                let mut pos = ivf.nprobe - 1;
                while pos > 0
                    && distance
                        .total_cmp(&coarse[pos - 1].0)
                        .then_with(|| cell.cmp(&coarse[pos - 1].1))
                        .is_lt()
                {
                    coarse[pos] = coarse[pos - 1];
                    pos -= 1;
                }
                coarse[pos] = (distance, cell, id);
            },
        )?;

        let mut top = [(f32::INFINITY, u32::MAX); MAX_TOPK];
        let mut consider = |distance: f32, id: u32| {
            let worst = top[MAX_TOPK - 1];
            if distance
                .total_cmp(&worst.0)
                .then_with(|| id.cmp(&worst.1))
                .is_ge()
            {
                return;
            }
            let mut pos = MAX_TOPK - 1;
            while pos > 0
                && distance
                    .total_cmp(&top[pos - 1].0)
                    .then_with(|| id.cmp(&top[pos - 1].1))
                    .is_lt()
            {
                top[pos] = top[pos - 1];
                pos -= 1;
            }
            top[pos] = (distance, id);
        };

        let mut routing_cmps = ivf.medoid_ids.len();
        let mut children = std::mem::take(&mut self.scratch.hint_ids_scratch);
        children.clear();
        for &(distance, cell, medoid_id) in coarse.iter() {
            consider(distance, medoid_id);
            let lo = ivf.offsets[cell] as usize;
            let hi = ivf.offsets[cell + 1] as usize;
            children.extend_from_slice(&ivf.hint_ids[lo..hi]);
        }
        routing_cmps = routing_cmps.checked_add(children.len()).ok_or_else(|| {
            diskann_error!(ErrorKind::IndexError, "temporal Hint-IVF comparison overflow")
        })?;
        let fine_result = self.pq_distances(&children, |distance, id| {
            consider(distance, id);
        });
        children.clear();
        self.scratch.hint_ids_scratch = children;
        fine_result?;

        self.io_tracker
            .routing_comparisons
            .fetch_add(routing_cmps, std::sync::atomic::Ordering::Relaxed);
        self.retained_hint_ranked.extend(
            top.into_iter()
                .take(limit)
                .filter(|(distance, id)| distance.is_finite() && *id != u32::MAX)
                .map(|(distance, id)| (id, distance, activation_hop)),
        );
        Ok(())
    }

'''
    s=once(s,anchor,helper+anchor,"temporal bank router")

    # Primary bank gets its own quota; route each extra bank once and merge by PQ rank.
    s=once(s,
"""                self.retained_hint_ranked.extend(
                    retained
                        .into_iter()
                        .filter(|(distance, id)| distance.is_finite() && *id != u32::MAX)
                        .map(|(distance, id)| (id, distance)),
                );
""",
"""                self.retained_hint_ranked.extend(
                    retained
                        .into_iter()
                        .take(self.progressive_hint_limit)
                        .filter(|(distance, id)| distance.is_finite() && *id != u32::MAX)
                        .map(|(distance, id)| (id, distance, 0)),
                );
                let temporal = [
                    (self.temporal_hint_ivf1, self.temporal_hint_limit1, self.temporal_hint_hop1),
                    (self.temporal_hint_ivf2, self.temporal_hint_limit2, self.temporal_hint_hop2),
                    (self.temporal_hint_ivf3, self.temporal_hint_limit3, self.temporal_hint_hop3),
                ];
                for (bank, limit, activation_hop) in temporal {
                    if let Some(bank) = bank {
                        self.retain_temporal_bank(bank, limit, activation_hop)?;
                    }
                }
                self.retained_hint_ranked.sort_by(|a, b| {
                    a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0))
                });
""","route temporal banks")

    # Only active banks participate at each natural beam boundary.
    s=once(s,
"""    async fn progressive_hint_distances<F>(&mut self, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if self.progressive_hints {
            for &(id, distance) in self
                .retained_hint_ranked
                .iter()
                .take(self.progressive_hint_limit)
            {
                f(id, distance);
            }
        }
        Ok(())
    }
""",
"""    async fn progressive_hint_distances<F>(&mut self, hops: u32, mut f: F) -> ANNResult<()>
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        if self.progressive_hints {
            for &(id, distance, activation_hop) in &self.retained_hint_ranked {
                if activation_hop <= hops {
                    f(id, distance);
                }
            }
        }
        Ok(())
    }
""","activation-aware progressive candidates")

    # Extend public search signature with three optional banks represented by empty slices/zero limits.
    s=once(s,
"""        nprobe: usize,
        progressive_hint_limit: usize,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
""",
"""        nprobe: usize,
        progressive_hint_limit: usize,
        t1_medoid_ids: &[u32],
        t1_coarse_local_ids: &[u32],
        t1_coarse_pq_codes: &[u8],
        t1_offsets: &[u32],
        t1_hint_ids: &[u32],
        t1_nprobe: usize,
        t1_limit: usize,
        t1_hop: u32,
        t2_medoid_ids: &[u32],
        t2_coarse_local_ids: &[u32],
        t2_coarse_pq_codes: &[u8],
        t2_offsets: &[u32],
        t2_hint_ids: &[u32],
        t2_nprobe: usize,
        t2_limit: usize,
        t2_hop: u32,
        t3_medoid_ids: &[u32],
        t3_coarse_local_ids: &[u32],
        t3_coarse_pq_codes: &[u8],
        t3_offsets: &[u32],
        t3_hint_ids: &[u32],
        t3_nprobe: usize,
        t3_limit: usize,
        t3_hop: u32,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
""","portfolio public signature")

    # Validate optional bank shapes and total retained state.
    marker="""        if search_list_size < return_list_size {
"""
    validate=r'''        if progressive_hint_limit + t1_limit + t2_limit + t3_limit > 32 {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "total retained Hint-IVF portfolio exceeds 32",
            ));
        }
        for (medoids, locals, codes, offsets, hints, nprobe, limit) in [
            (t1_medoid_ids, t1_coarse_local_ids, t1_coarse_pq_codes, t1_offsets, t1_hint_ids, t1_nprobe, t1_limit),
            (t2_medoid_ids, t2_coarse_local_ids, t2_coarse_pq_codes, t2_offsets, t2_hint_ids, t2_nprobe, t2_limit),
            (t3_medoid_ids, t3_coarse_local_ids, t3_coarse_pq_codes, t3_offsets, t3_hint_ids, t3_nprobe, t3_limit),
        ] {
            if limit == 0 {
                if !medoids.is_empty() || !locals.is_empty() || !codes.is_empty()
                    || !offsets.is_empty() || !hints.is_empty() || nprobe != 0
                {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "disabled temporal Hint-IVF bank must be empty",
                    ));
                }
                continue;
            }
            if limit > 16
                || medoids.is_empty()
                || locals.len() != medoids.len()
                || locals.iter().enumerate().any(|(i, id)| *id as usize != i)
                || offsets.len() != medoids.len() + 1
                || offsets.first().copied() != Some(0)
                || offsets.last().copied() != Some(hints.len() as u32)
                || nprobe == 0
                || nprobe > medoids.len()
                || nprobe > 64
            {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "invalid temporal Hint-IVF bank shape",
                ));
            }
            let num_chunks = self.index.provider().pq_data.get_num_chunks();
            if codes.len() != medoids.len() * num_chunks {
                return Err(diskann_error!(
                    ErrorKind::IndexError,
                    "temporal Hint-IVF coarse PQ size mismatch",
                ));
            }
        }

'''
    s=once(s,marker,validate+marker,"portfolio validation")

    # Construct the optional borrowed bank views.
    s=once(s,
"""        let strategy = self.search_strategy_with_progressive_hint_ivf(
            &io_tracker,
            hint_ivf,
            progressive_hint_limit,
        );
""",
"""        let temporal_hint_ivf1 = (t1_limit > 0).then_some(HintIvfSearch {
            medoid_ids: t1_medoid_ids,
            coarse_local_ids: t1_coarse_local_ids,
            coarse_pq_codes: t1_coarse_pq_codes,
            offsets: t1_offsets,
            hint_ids: t1_hint_ids,
            nprobe: t1_nprobe,
        });
        let temporal_hint_ivf2 = (t2_limit > 0).then_some(HintIvfSearch {
            medoid_ids: t2_medoid_ids,
            coarse_local_ids: t2_coarse_local_ids,
            coarse_pq_codes: t2_coarse_pq_codes,
            offsets: t2_offsets,
            hint_ids: t2_hint_ids,
            nprobe: t2_nprobe,
        });
        let temporal_hint_ivf3 = (t3_limit > 0).then_some(HintIvfSearch {
            medoid_ids: t3_medoid_ids,
            coarse_local_ids: t3_coarse_local_ids,
            coarse_pq_codes: t3_coarse_pq_codes,
            offsets: t3_offsets,
            hint_ids: t3_hint_ids,
            nprobe: t3_nprobe,
        });
        let strategy = self.search_strategy_with_progressive_hint_ivf(
            &io_tracker,
            hint_ivf,
            progressive_hint_limit,
            temporal_hint_ivf1,
            t1_limit,
            t1_hop,
            temporal_hint_ivf2,
            t2_limit,
            t2_hop,
            temporal_hint_ivf3,
            t3_limit,
            t3_hop,
        );
""","portfolio strategy construction")

    path.write_text(s)


def patch_benchmark(path: Path):
    s=path.read_text()

    # Load up to three extra banks and their quota/activation settings.
    marker="""    // Load the vector filters
"""
    load=r'''    let mut temporal_hint_ivf1 = match std::env::var_os("DISKANN_TEMPORAL_HINT_IVF1_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let mut temporal_hint_ivf2 = match std::env::var_os("DISKANN_TEMPORAL_HINT_IVF2_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let mut temporal_hint_ivf3 = match std::env::var_os("DISKANN_TEMPORAL_HINT_IVF3_FILE") {
        Some(path) => Some(HintIvfIndex::load(std::path::Path::new(&path))?),
        None => None,
    };
    let t1_nprobe = std::env::var("DISKANN_TEMPORAL_HINT_IVF1_NPROBE").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t2_nprobe = std::env::var("DISKANN_TEMPORAL_HINT_IVF2_NPROBE").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t3_nprobe = std::env::var("DISKANN_TEMPORAL_HINT_IVF3_NPROBE").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t1_limit = std::env::var("DISKANN_TEMPORAL_HINT_IVF1_LIMIT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t2_limit = std::env::var("DISKANN_TEMPORAL_HINT_IVF2_LIMIT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t3_limit = std::env::var("DISKANN_TEMPORAL_HINT_IVF3_LIMIT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(0);
    let t1_hop = std::env::var("DISKANN_TEMPORAL_HINT_IVF1_HOP").ok().map(|v| v.parse::<u32>()).transpose()?.unwrap_or(0);
    let t2_hop = std::env::var("DISKANN_TEMPORAL_HINT_IVF2_HOP").ok().map(|v| v.parse::<u32>()).transpose()?.unwrap_or(0);
    let t3_hop = std::env::var("DISKANN_TEMPORAL_HINT_IVF3_HOP").ok().map(|v| v.parse::<u32>()).transpose()?.unwrap_or(0);
    for (bank, nprobe, limit) in [
        (temporal_hint_ivf1.as_ref(), t1_nprobe, t1_limit),
        (temporal_hint_ivf2.as_ref(), t2_nprobe, t2_limit),
        (temporal_hint_ivf3.as_ref(), t3_nprobe, t3_limit),
    ] {
        if bank.is_some() != (limit > 0) || (limit > 0 && (nprobe == 0 || nprobe > 64)) {
            anyhow::bail!("invalid temporal Hint-IVF environment");
        }
    }

'''
    s=once(s,marker,load+marker,"load temporal banks")

    # Validate and pack coarse PQ codes once.
    marker="""    logger.log_checkpoint("index_loaded");
"""
    pack=r'''    for index in [
        temporal_hint_ivf1.as_mut(),
        temporal_hint_ivf2.as_mut(),
        temporal_hint_ivf3.as_mut(),
    ].into_iter().flatten() {
        searcher.validate_hint_ivf_ids(&index.medoid_ids, &index.hint_ids)?;
        index.coarse_local_ids = (0..index.medoid_ids.len() as u32).collect();
        index.coarse_pq_codes = searcher.pack_hint_ivf_coarse_pq(&index.medoid_ids)?;
    }

'''
    s=once(s,marker,pack+marker,"pack temporal banks")

    # Expand the progressive call. Empty banks are represented by empty slices.
    old="""                        searcher.search_with_progressive_hint_ivf(
                            q,
                            search_params.recall_at,
                            l,
                            Some(search_params.beam_width),
                            &index.medoid_ids,
                            &index.coarse_local_ids,
                            &index.coarse_pq_codes,
                            &index.offsets,
                            &index.hint_ids,
                            hint_ivf_nprobe,
                            progressive_hint_topk,
                        )
"""
    new=r'''                        let empty_u32: &[u32] = &[];
                        let empty_u8: &[u8] = &[];
                        let (t1m,t1l,t1c,t1o,t1h) = if let Some(t) = temporal_hint_ivf1.as_ref() {
                            (&t.medoid_ids[..], &t.coarse_local_ids[..], &t.coarse_pq_codes[..], &t.offsets[..], &t.hint_ids[..])
                        } else { (empty_u32, empty_u32, empty_u8, empty_u32, empty_u32) };
                        let (t2m,t2l,t2c,t2o,t2h) = if let Some(t) = temporal_hint_ivf2.as_ref() {
                            (&t.medoid_ids[..], &t.coarse_local_ids[..], &t.coarse_pq_codes[..], &t.offsets[..], &t.hint_ids[..])
                        } else { (empty_u32, empty_u32, empty_u8, empty_u32, empty_u32) };
                        let (t3m,t3l,t3c,t3o,t3h) = if let Some(t) = temporal_hint_ivf3.as_ref() {
                            (&t.medoid_ids[..], &t.coarse_local_ids[..], &t.coarse_pq_codes[..], &t.offsets[..], &t.hint_ids[..])
                        } else { (empty_u32, empty_u32, empty_u8, empty_u32, empty_u32) };
                        searcher.search_with_progressive_hint_ivf(
                            q,
                            search_params.recall_at,
                            l,
                            Some(search_params.beam_width),
                            &index.medoid_ids,
                            &index.coarse_local_ids,
                            &index.coarse_pq_codes,
                            &index.offsets,
                            &index.hint_ids,
                            hint_ivf_nprobe,
                            progressive_hint_topk,
                            t1m,t1l,t1c,t1o,t1h,t1_nprobe,t1_limit,t1_hop,
                            t2m,t2l,t2c,t2o,t2h,t2_nprobe,t2_limit,t2_hop,
                            t3m,t3l,t3c,t3o,t3h,t3_nprobe,t3_limit,t3_hop,
                        )
'''
    s=once(s,old,new,"portfolio benchmark dispatch")
    path.write_text(s)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("diskann",type=Path)
    a=ap.parse_args()
    root=a.diskann.resolve()
    subprocess.run([
        sys.executable,
        str(Path(__file__).resolve().parent/"patch_diskann_progressive_navhints.py"),
        str(root),
    ],check=True)
    patch_glue(root)
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("extended progressive NavHints with three-bank temporal portfolio")

if __name__=="__main__":
    main()
