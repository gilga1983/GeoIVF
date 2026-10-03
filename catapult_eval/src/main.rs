use catapult::{
    fs::Queries,
    numerics::{AlignedBlock, SIMD_LANECOUNT},
    search::{AdjacencyGraph, FlatCatapultChoice, FlatSearch, hash_start::EngineStarter},
    sets::catapults::FifoSet,
};
use std::{
    env,
    fs::File,
    hint::black_box,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const LOAD_LITTLE_ENDIAN: bool = cfg!(target_endian = "little");

fn usage() -> ! {
    eprintln!(
        "usage: catapult-official-eval <graph> <payload> <queries.npy> <out_prefix> <k> <beam> <off|on>"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 8 {
        usage();
    }

    let graph_path = PathBuf::from(&args[1]);
    let payload_path = PathBuf::from(&args[2]);
    let query_path = &args[3];
    let out_prefix = PathBuf::from(&args[4]);
    let k: usize = args[5].parse().unwrap();
    let beam: usize = args[6].parse().unwrap();
    let enabled = match args[7].as_str() {
        "off" => false,
        "on" => true,
        _ => usage(),
    };
    assert!(beam >= k);

    let adjacency =
        AdjacencyGraph::<FifoSet<30>, FlatSearch>::load_from_path::<LOAD_LITTLE_ENDIAN>(
            graph_path,
            payload_path,
        );
    let queries: Vec<Vec<AlignedBlock>> =
        Vec::<Vec<AlignedBlock>>::load_from_npy(query_path);

    let graph_size = adjacency.len();
    let num_hash = 10;
    let plane_dim = queries[0].len() * SIMD_LANECOUNT;
    let engine = EngineStarter::new(num_hash, plane_dim, graph_size, Some(42));
    let graph = AdjacencyGraph::new(
        adjacency,
        engine,
        if enabled {
            FlatCatapultChoice::CatapultsEnabled
        } else {
            FlatCatapultChoice::CatapultsDisabled
        },
    );

    let start = Instant::now();
    let chunk_size: usize = 1000;
    let mut chunk_start = Instant::now();
    let mut chunks: Vec<(usize, usize, f64, f64)> = Vec::new();
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(queries.len());
    for (i, query) in queries.iter().enumerate() {
        rows.push(black_box(graph.beam_search(query, k, beam)));
        let end = i + 1;
        if end % chunk_size == 0 || end == queries.len() {
            let chunk_elapsed = chunk_start.elapsed().as_secs_f64();
            let begin = end - (end % chunk_size).max(if end == queries.len() { end % chunk_size } else { 0 });
            let actual_begin = if end % chunk_size == 0 { end - chunk_size } else { end - (end % chunk_size) };
            let count = end - actual_begin;
            let chunk_qps = if chunk_elapsed > 0.0 {
                count as f64 / chunk_elapsed
            } else {
                0.0
            };
            chunks.push((actual_begin, end, chunk_elapsed, chunk_qps));
            chunk_start = Instant::now();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = if elapsed > 0.0 {
        queries.len() as f64 / elapsed
    } else {
        0.0
    };

    let ubin_path = out_prefix.with_extension("ubin");
    let mut ubin = BufWriter::new(File::create(&ubin_path).unwrap());
    ubin.write_all(&(rows.len() as u32).to_le_bytes()).unwrap();
    ubin.write_all(&(k as u32).to_le_bytes()).unwrap();
    for row in &rows {
        assert_eq!(row.len(), k);
        for &id in row {
            let id = u32::try_from(id).unwrap();
            ubin.write_all(&id.to_le_bytes()).unwrap();
        }
    }
    ubin.flush().unwrap();

    let chunks_path = out_prefix.with_extension("chunks.tsv");
    let mut chunks_out = BufWriter::new(File::create(&chunks_path).unwrap());
    writeln!(chunks_out, "begin\tend\telapsed_seconds\tqps").unwrap();
    for (begin, end, chunk_elapsed, chunk_qps) in &chunks {
        writeln!(chunks_out, "{begin}\t{end}\t{chunk_elapsed:.9}\t{chunk_qps:.6}").unwrap();
    }
    chunks_out.flush().unwrap();

    let summary_path = out_prefix.with_extension("txt");
    let mut summary = BufWriter::new(File::create(&summary_path).unwrap());
    writeln!(summary, "queries={}", rows.len()).unwrap();
    writeln!(summary, "k={k}").unwrap();
    writeln!(summary, "beam={beam}").unwrap();
    writeln!(summary, "catapults={}", if enabled { "on" } else { "off" }).unwrap();
    writeln!(summary, "num_hash=10").unwrap();
    writeln!(summary, "catapult_capacity=30").unwrap();
    writeln!(summary, "engine_seed=42").unwrap();
    writeln!(summary, "chunk_size={chunk_size}").unwrap();
    writeln!(summary, "elapsed_seconds={elapsed:.9}").unwrap();
    writeln!(summary, "qps={qps:.6}").unwrap();
    summary.flush().unwrap();

    println!(
        "queries={} k={} beam={} catapults={} elapsed_seconds={:.9} qps={:.6}",
        rows.len(),
        k,
        beam,
        if enabled { "on" } else { "off" },
        elapsed,
        qps
    );
}
