//! Measurement-only CLI for the UNMODIFIED original authors' CatapultDB library.
//! This calls the original native AdjacencyGraph::beam_search for each query.
//! Sequential chronology, original 20K online warm-up, final 5K measured.

use catapult::{
    fs::Queries,
    numerics::AlignedBlock,
    search::{AdjacencyGraph, SearchStrategy},
    sets::catapults::LruSet,
    statistics::Stats,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const ONLINE_WARM: usize = 20000;
const MEASURE: usize = 5000;
const CATALOG: usize = 31950;
const K: usize = 10;
const HASH_BITS: usize = 8;
const BUCKET_CAPACITY: usize = 40;

fn getu32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().expect("truncated u32"))
}
fn load_gt(path: &str, rows: usize) -> (usize, Vec<u32>) {
    let buf=fs::read(path).expect("missing exact original Coveo ground truth");
    assert!(buf.len() >= 8, "truncated GT");
    let n=getu32(&buf,0) as usize;
    let d=getu32(&buf,4) as usize;
    assert_eq!(n,rows);
    assert!(d >= K);
    let off=8+rows*d*4;
    assert_eq!(buf.len(),off+rows*d*4, "GT must contain IDs and distances");
    let mut ids=Vec::with_capacity(rows*d);
    for j in 0..rows*d { ids.push(getu32(&buf,8+j*4)); }
    for &id in &ids { assert!((id as usize) < CATALOG); }
    (d,ids)
}
fn recall_at_10(found: &[usize], truth: &[u32]) -> f64 {
    assert!(found.len() >= K && truth.len() >= K);
    let mut count=0;
    for &id in &found[..K] {
        if truth[..K].iter().any(|&x| x as usize == id) {
            count+=1;
        }
    }
    count as f64 / K as f64
}

fn main() {
    let args:Vec<String>=std::env::args().collect();
    assert_eq!(args.len(),9,
        "usage: author_coveo_eval GRAPH PAYLOAD QUERIES_NPY REPLAY_GT MODE WIDTH SEED OUT_JSON");
    let graph=&args[1];
    let payload=&args[2];
    let queries_file=&args[3];
    let gt_file=&args[4];
    let mode=&args[5];
    assert!(mode=="vanilla"||mode=="catapult");
    let beam:usize=args[6].parse().expect("invalid beam width");
    let seed:u64=args[7].parse().expect("invalid seed");
    let out=&args[8];
    assert!(beam>=K);

    let queries=Vec::<Vec<AlignedBlock>>::load_from_npy(queries_file,None);
    assert_eq!(queries.len(),ONLINE_WARM+MEASURE);
    let (gt_dim,gt)=load_gt(gt_file,queries.len());

    let native=AdjacencyGraph::<LruSet>::load_flat_from_path(
        PathBuf::from(graph),
        PathBuf::from(payload),
        HASH_BITS,BUCKET_CAPACITY,seed,
        SearchStrategy::from_string(mode,None),
    );
    assert_eq!(native.len(),CATALOG);
    let mut stats=Stats::new();
    let mut durations=Vec::with_capacity(MEASURE);
    let mut recall_sum=0f64;
    let mut exact10=0usize;
    let mut checksum=0u64;
    let mut correct_ids=0usize;
    for (i,query) in queries.iter().enumerate() {
        if i==ONLINE_WARM {
            stats=Stats::new();
        }
        let start=Instant::now();
        let res=native.beam_search(query,K,beam,&mut stats);
        let duration=start.elapsed();
        assert!(res.len()>=K);
        if i>=ONLINE_WARM {
            durations.push(duration.as_secs_f64()*1_000_000.0);
            let idx:Vec<usize>=res.iter().map(|entry| entry.index.internal).collect();
            let truth=&gt[i*gt_dim..i*gt_dim+K];
            let recall=recall_at_10(&idx,truth);
            recall_sum+=recall;
            correct_ids+=(recall*K as f64).round() as usize;
            if recall==1.0 {exact10+=1}
            checksum=checksum.wrapping_add(idx[0] as u64);
        }
    }
    assert_eq!(durations.len(),MEASURE);
    let elapsed=durations.iter().sum::<f64>();
    durations.sort_by(|a,b| a.total_cmp(b));
    let p95=durations[((MEASURE as f64*0.95).ceil() as usize).min(MEASURE-1)];
    let p99=durations[((MEASURE as f64*0.99).ceil() as usize).min(MEASURE-1)];
    let result=json!({
      "source":"UNMODIFIED original MRandl/catapult-db library",
      "source_paper":"https://arxiv.org/abs/2603.02164",
      "mode":mode,"seed":seed,
      "graph_nodes":CATALOG,"graph_type":"author-linked C++ DiskANN Vamana in RAM",
      "payload_dims":64,"original_unpadded_dims":50,
      "query_order":"original production chronological, one native Rust thread",
      "static_train_disjoint_rows":5000,
      "warmup_queries":ONLINE_WARM,"measured_queries":MEASURE,
      "beam_width":beam,"k":K,"hash_bits":HASH_BITS,"bucket_capacity":BUCKET_CAPACITY,
      "auxiliary_catapult_ID_payload_bytes":(1usize<<HASH_BITS)*BUCKET_CAPACITY*4,
      "wall_clock_mean_latency_us":elapsed/MEASURE as f64,
      "wall_clock_p95_us":p95,"wall_clock_p99_us":p99,
      "measured_query_only_qps":1_000_000.0/(elapsed/MEASURE as f64),
      "recall_at10_percent":100.0*recall_sum/MEASURE as f64,
      "exact10_fraction":exact10 as f64/MEASURE as f64,
      "matched_neighbors_top10":correct_ids,
      "first_id_checksum":checksum,
      "original_author_distance_computations_per_query":stats.get_computed_dists() as f64/MEASURE as f64,
      "original_author_nodes_visited_per_query":stats.get_nodes_visited() as f64/MEASURE as f64,
      "original_author_catapult_usage_fraction":stats.get_searches_with_catapults() as f64/MEASURE as f64,
      "note":"CPU/memory-resident graph; does NOT measure SSD I/Os or establish whole-system parity with disk-resident NavHints."
    });
    let output=serde_json::to_string_pretty(&result).unwrap()+"\n";
    fs::write(out,output).expect("cannot write output");
    println!("AUTHOR_CATAPULT_COVEO_DONE mode={} seed={} beam={} recall={} latency_us={}",
        mode,seed,beam,result["recall_at10_percent"],result["wall_clock_mean_latency_us"]);
}
