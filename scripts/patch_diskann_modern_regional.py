#!/usr/bin/env python3
"""Compose legacy regional NavHints with the modern hub/cache/vertex stack.

Apply after, in order:
  patch_diskann_vertex_navhints.py --variants 8
  patch_diskann_hub_winners.py
  patch_diskann_online_value_cache.py

This revives the original regional mechanism unchanged in spirit:
after a natural beam enters a previously unseen geometric region after the
configured warmup position, PQ-score that region's learned hint list and offer
only its best unseen candidate through the ordinary fixed-L frontier gate.

The regional map is orthogonal to:
  * the 16K start,
  * support-2 exact-vertex continuation,
  * frozen hub winners, and
  * the fully-online seed-only value cache.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from patch_diskann_start_points import once


def patch_graph(root: Path) -> None:
    glue=root/"diskann/src/graph/glue.rs"
    s=glue.read_text()
    marker="""    /// Score the selected continuation overlay carried by the closest expanded node.
"""
    hook="""    /// Score hints attached to at most one newly entered geometric region.
    fn regional_hint_distances<F>(
        &mut self,
        _expanded: &[Self::Id],
        _hops: u32,
        _f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        std::future::ready(Ok(()))
    }

"""
    s=once(s,marker,hook+marker,"modern regional SearchAccessor hook")
    glue.write_text(s)

    idx=root/"diskann/src/graph/index.rs"
    s=idx.read_text()
    # Put the regional proposal before the other learned overlays. All proposals
    # still compete through the same fixed-L queue.
    anchor="""                let mut hub_candidates: Vec<(A::Id, f32)> = Vec::new();
"""
    insertion=r'''                let mut regional_best: Option<(A::Id, f32)> = None;
                accessor
                    .regional_hint_distances(
                        &scratch.beam_nodes,
                        scratch.hops,
                        |id, distance| {
                            if scratch.visited.contains(&id) {
                                return;
                            }
                            let better = regional_best
                                .is_none_or(|current| distance.total_cmp(&current.1).is_lt());
                            if better {
                                regional_best = Some((id, distance));
                            }
                        },
                    )
                    .await?;
                if let Some((id, distance)) = regional_best {
                    let queue_accepts = scratch.best.size() < scratch.best.capacity()
                        || *scratch.best.get(scratch.best.size() - 1).distance() >= distance;
                    if queue_accepts && scratch.visited.insert(id) {
                        scratch.best.insert(Neighbor::new(id, distance));
                    }
                }

'''
    s=once(s,anchor,insertion+anchor,"modern regional frontier injection")
    idx.write_text(s)


def patch_provider(path: Path) -> None:
    s=path.read_text()

    marker="""pub struct DiskSearchStrategy<'a, Data, ProviderFactory>
"""
    helper=r'''#[derive(Clone, Copy)]
struct RegionalHintSearch<'a> {
    vertex_regions: &'a [u16],
    offsets: &'a [u32],
    hint_ids: &'a [u32],
    min_entry_pos: u32,
}

'''
    s=once(s,marker,helper+marker,"modern regional search view")

    s=once(
        s,
        """    /// Selected associated-data continuation variant. None disables the overlay.
    vertex_hint_variant: Option<usize>,
}
""",
        """    /// Selected associated-data continuation variant. None disables the overlay.
    vertex_hint_variant: Option<usize>,

    /// Optional state-conditioned regional navigation table.
    regional_hints: Option<RegionalHintSearch<'a>>,
}
""",
        "modern regional strategy field",
    )

    # Hub patch has extended the accessor after the vertex fields.
    s=once(
        s,
        """    hub_winner_last: [u32; 10],
    hub_winner_last_valid: bool,
}
""",
        """    hub_winner_last: [u32; 10],
    hub_winner_last_valid: bool,
    regional_hints: Option<RegionalHintSearch<'a>>,
    regional_seen: Vec<bool>,
}
""",
        "modern regional accessor fields",
    )

    s=once(
        s,
        """            hub_winner_last: [u32::MAX; 10],
            hub_winner_last_valid: false,
        })
""",
        """            hub_winner_last: [u32::MAX; 10],
            hub_winner_last_valid: false,
            regional_hints: strategy.regional_hints,
            regional_seen: vec![
                false;
                strategy
                    .regional_hints
                    .map_or(0, |r| r.offsets.len().saturating_sub(1))
            ],
        })
""",
        "modern regional accessor init",
    )

    # Every existing strategy constructor disables regional routing.
    replacements=[
      ("""            vertex_hint_variant: None,
        }
""","""            vertex_hint_variant: None,
            regional_hints: None,
        }
""","generic/canonical regional off"),
      ("""            vertex_hint_variant: Some(vertex_hint_variant),
        }
""","""            vertex_hint_variant: Some(vertex_hint_variant),
            regional_hints: None,
        }
""","vertex regional off"),
    ]
    for old,new,label in replacements:
        n=s.count(old)
        if n<1:
            raise RuntimeError(f"{label}: pattern missing")
        s=s.replace(old,new)

    marker="""    /// Perform the ordinary graph search from caller-supplied starting vertices.
"""
    strategy=r'''    fn search_strategy_with_vertex_hint_ivf_regional<'a>(
        &'a self,
        io_tracker: &'a IOTracker,
        hint_ivf: HintIvfSearch<'a>,
        vertex_hint_variant: usize,
        regional_hints: RegionalHintSearch<'a>,
    ) -> DiskSearchStrategy<'a, Data, ProviderFactory> {
        DiskSearchStrategy {
            io_tracker,
            cache_indexed_vectors: false,
            postprocess_filter: PostprocessStrategy::AcceptAll,
            vertex_provider_factory: &self.vertex_provider_factory,
            scratch_pool: &self.scratch_pool,
            start_points: None,
            hint_ivf: Some(hint_ivf),
            vertex_hint_variant: Some(vertex_hint_variant),
            regional_hints: Some(regional_hints),
        }
    }

'''
    s=once(s,marker,strategy+marker,"modern regional strategy constructor")

    marker="""    fn hub_winner_distances<F>(
"""
    scorer=r'''    fn regional_hint_distances<F>(
        &mut self,
        expanded: &[Self::Id],
        hops: u32,
        mut f: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        F: FnMut(Self::Id, f32) + Send,
    {
        let result = (|| {
            let Some(regional) = self.regional_hints else {
                return Ok(());
            };
            let nregions = regional.offsets.len().saturating_sub(1);
            let mut chosen: Option<usize> = None;

            for &id in expanded {
                let pos=id as usize;
                if pos>=regional.vertex_regions.len() {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "regional vertex outside label table",
                    ));
                }
                let region=regional.vertex_regions[pos] as usize;
                if region>=nregions {
                    return Err(diskann_error!(
                        ErrorKind::IndexError,
                        "regional label outside offset table",
                    ));
                }
                if self.regional_seen[region] {
                    continue;
                }
                self.regional_seen[region]=true;
                if chosen.is_none() && hops>regional.min_entry_pos {
                    chosen=Some(region);
                }
            }

            let Some(region)=chosen else {
                return Ok(());
            };
            let lo=regional.offsets[region] as usize;
            let hi=regional.offsets[region+1] as usize;
            let ids=&regional.hint_ids[lo..hi];
            if ids.is_empty() {
                return Ok(());
            }
            self.io_tracker.routing_comparisons.fetch_add(
                ids.len(),
                std::sync::atomic::Ordering::Relaxed,
            );
            self.pq_distances(ids, |distance,id| f(id,distance))
        })();
        std::future::ready(result)
    }

'''
    s=once(s,marker,scorer+marker,"modern regional accessor scorer")

    marker="""    /// Search with the ordinary Hint-IVF + vertex overlay plus one online
"""
    method=r'''    /// Search with Hint-IVF + support-2 + online value-cache seed +
    /// regional state-conditioned routing. Hub-page winners remain enabled by
    /// the common accessor whenever DISKANN_HUB_WINNER_COUNT is set.
    pub fn search_with_vertex_hint_ivf_value_cache_regional(
        &self,
        query: &[Data::VectorDataType],
        return_list_size: u32,
        search_list_size: u32,
        beam_width: Option<usize>,
        medoid_ids: &[u32],
        coarse_local_ids: &[u32],
        coarse_pq_codes: &[u8],
        offsets: &[u32],
        hint_ids: &[u32],
        nprobe: usize,
        vertex_hint_variant: usize,
        value_cache_ids: &[u32],
        vertex_regions: &[u16],
        regional_offsets: &[u32],
        regional_hint_ids: &[u32],
        min_entry_pos: u32,
    ) -> ANNResult<SearchResult<Data::AssociatedDataType>> {
        if value_cache_ids.is_empty() {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "modern regional test expects a warmed value cache",
            ));
        }
        if medoid_ids.is_empty()
            || coarse_local_ids.len()!=medoid_ids.len()
            || offsets.len()!=medoid_ids.len()+1
            || offsets.first().copied()!=Some(0)
            || offsets.last().copied()!=Some(hint_ids.len() as u32)
            || regional_offsets.len()<2
            || nprobe==0
            || nprobe>medoid_ids.len()
            || nprobe>64
        {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "invalid modern regional search shape",
            ));
        }
        if search_list_size<return_list_size {
            return Err(diskann_error!(
                ErrorKind::IndexError,
                "search list size below return size",
            ));
        }

        let mut query_stats=QueryStatistics::default();
        let mut indices=vec![0u32;return_list_size as usize];
        let mut distances=vec![0f32;return_list_size as usize];
        let mut associated_data=
            vec![Data::AssociatedDataType::default();return_list_size as usize];
        let mut result_output_buffer=SearchOutput::new(
            &mut indices,&mut distances,&mut associated_data,None,
        )?;

        let timer=Instant::now();
        let io_tracker=IOTracker::default();
        let hint_ivf=HintIvfSearch {
            medoid_ids,
            coarse_local_ids,
            coarse_pq_codes,
            offsets,
            hint_ids,
            nprobe,
            value_cache_ids: Some(value_cache_ids),
        };
        let regional_hints=RegionalHintSearch {
            vertex_regions,
            offsets: regional_offsets,
            hint_ids: regional_hint_ids,
            min_entry_pos,
        };
        let strategy=self.search_strategy_with_vertex_hint_ivf_regional(
            &io_tracker,hint_ivf,vertex_hint_variant,regional_hints,
        );
        let knn_search=Knn::new(search_list_size as usize,beam_width)
            .map_err(|e| diskann_error!(ErrorKind::IndexError,e))?;
        let stats=self.runtime.block_on(self.index.search(
            knn_search,&strategy,&DefaultContext,query,&mut result_output_buffer,
        ))?;

        let routing_comparisons=io_tracker.routing_comparisons
            .load(std::sync::atomic::Ordering::Relaxed) as u32;
        query_stats.total_comparisons=stats.cmps.saturating_add(routing_comparisons);
        query_stats.search_hops=stats.hops;
        query_stats.total_execution_time_us=timer.elapsed().as_micros();
        query_stats.io_time_us=IOTracker::time(&io_tracker.io_time_us) as u128;
        query_stats.total_io_operations=io_tracker.io_count() as u32;
        query_stats.total_vertices_loaded=io_tracker.io_count() as u32;
        query_stats.query_pq_preprocess_time_us=
            IOTracker::time(&io_tracker.preprocess_time_us) as u128;
        query_stats.cpu_time_us=query_stats.total_execution_time_us
            .saturating_sub(query_stats.io_time_us)
            .saturating_sub(query_stats.query_pq_preprocess_time_us);

        let mut search_result=SearchResult {
            results:Vec::with_capacity(return_list_size as usize),
            stats:SearchResultStats {
                cmps:query_stats.total_comparisons,
                result_count:stats.result_count,
                query_statistics:query_stats,
            },
        };
        for ((vertex_id,distance),data) in
            indices.into_iter().zip(distances).zip(associated_data)
        {
            search_result.results.push(SearchResultItem {
                vertex_id,distance,data,
            });
        }
        Ok(search_result)
    }

'''
    s=once(s,marker,method+marker,"modern regional public search")
    path.write_text(s)


def patch_benchmark(path: Path) -> None:
    s=path.read_text()

    marker="""struct HintIvfIndex {
"""
    parser=r'''struct RegionalHintIndex {
    vertex_regions: Vec<u16>,
    offsets: Vec<u32>,
    hint_ids: Vec<u32>,
    min_entry_pos: u32,
}

impl RegionalHintIndex {
    fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let raw=std::fs::read(path)?;
        if raw.len()<24 || &raw[0..8]!=b"GIRGN001" {
            anyhow::bail!("invalid regional-hint header");
        }
        let nvertices=u32::from_le_bytes(raw[8..12].try_into()?) as usize;
        let nregions=u32::from_le_bytes(raw[12..16].try_into()?) as usize;
        let total=u32::from_le_bytes(raw[16..20].try_into()?) as usize;
        let min_entry_pos=u32::from_le_bytes(raw[20..24].try_into()?);
        let expected=24usize
            .checked_add(nvertices.checked_mul(2).ok_or_else(|| anyhow::anyhow!("regional size overflow"))?)
            .and_then(|x|x.checked_add((nregions+1).checked_mul(4)?))
            .and_then(|x|x.checked_add(total.checked_mul(4)?))
            .ok_or_else(|| anyhow::anyhow!("regional size overflow"))?;
        if raw.len()!=expected {
            anyhow::bail!("regional-hint size mismatch");
        }
        let mut off=24usize;
        let mut vertex_regions=Vec::with_capacity(nvertices);
        for _ in 0..nvertices {
            vertex_regions.push(u16::from_le_bytes(raw[off..off+2].try_into()?));
            off+=2;
        }
        let mut offsets=Vec::with_capacity(nregions+1);
        for _ in 0..=nregions {
            offsets.push(u32::from_le_bytes(raw[off..off+4].try_into()?));
            off+=4;
        }
        let mut hint_ids=Vec::with_capacity(total);
        for _ in 0..total {
            hint_ids.push(u32::from_le_bytes(raw[off..off+4].try_into()?));
            off+=4;
        }
        if offsets.first().copied()!=Some(0)
            || offsets.last().copied()!=Some(total as u32)
            || offsets.windows(2).any(|w|w[0]>w[1])
        {
            anyhow::bail!("invalid regional offsets");
        }
        Ok(Self {vertex_regions,offsets,hint_ids,min_entry_pos})
    }
}

'''
    s=once(s,marker,parser+marker,"modern regional parser")

    anchor="""    let value_cache_replay = std::env::var("DISKANN_VALUE_CACHE_REPLAY")
"""
    load=r'''    let regional_hints = match std::env::var_os("DISKANN_REGIONAL_HINT_FILE") {
        Some(path) => Some(RegionalHintIndex::load(std::path::Path::new(&path))?),
        None => None,
    };

'''
    s=once(s,anchor,load+anchor,"load modern regional map")

    # In the chronological replay, route through the regional method whenever
    # a map is supplied and the 512 cache is warm.
    old=r'''                let search_result = if value_cache_capacity > 0 && !cache_ids.is_empty() {
                    searcher.search_with_vertex_hint_ivf_value_cache(
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
                        variant,
                        &cache_ids,
                    )?
                } else {
'''
    new=r'''                let search_result = if value_cache_capacity > 0 && !cache_ids.is_empty() {
                    if let Some(regional) = regional_hints.as_ref() {
                        searcher.search_with_vertex_hint_ivf_value_cache_regional(
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
                            variant,
                            &cache_ids,
                            &regional.vertex_regions,
                            &regional.offsets,
                            &regional.hint_ids,
                            regional.min_entry_pos,
                        )?
                    } else {
                        searcher.search_with_vertex_hint_ivf_value_cache(
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
                            variant,
                            &cache_ids,
                        )?
                    }
                } else {
'''
    s=once(s,old,new,"modern regional chronological dispatch")
    path.write_text(s)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("diskann",type=Path);args=ap.parse_args()
    root=args.diskann.resolve()
    patch_graph(root)
    patch_provider(root/"diskann-disk/src/search/provider/disk_provider.rs")
    patch_benchmark(root/"diskann-benchmark/src/disk_index/search.rs")
    print("patched modern stack with revived regional NavHints")


if __name__=="__main__":main()
