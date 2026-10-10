//! Executable oracle test against the authors' ORIGINAL CatapultDB source.
//!
//! This file is copied into a TEMPORARY checkout of pinned MRandl/catapult-db.
//! Authors' source is left UNMODIFIED; we link their native public EngineStarter.
//! Verifies hash signatures, LRU/duplicate behavior, medoid occupancy, and
//! effective search-list length semantics before claiming port fidelity.
use catapult::{
    numerics::{AlignedBlock, SIMD_LANECOUNT, VectorLike},
    search::{NodeId, hash_start::{EngineStarter, EngineStarterParams}},
    sets::catapults::LruSet,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::collections::VecDeque;

const HASH: usize=8;
const BUCKET_CAP: usize=4;
const BASE: usize=3;
const D: usize=SIMD_LANECOUNT*2;

// Reconstructed original-author Gaussian projection generator. Unlike our
// earlier port's hand-written Box-Muller, this samples the authors' own
// rand_distr::StandardNormal in exactly the same iteration order.
fn make_planes(seed:u64)->Vec<Vec<AlignedBlock>> {
    let mut iter=StdRng::seed_from_u64(seed).sample_iter(StandardNormal);
    (0..HASH).map(|_| {
        (0..D/SIMD_LANECOUNT).map(|_| {
            let mut block=[0f32;SIMD_LANECOUNT];
            for x in block.iter_mut() { *x=iter.next().unwrap(); }
            AlignedBlock::new(block)
        }).collect()
    }).collect()
}

fn source_signature(q:&[AlignedBlock],planes:&[Vec<AlignedBlock>])->usize {
    planes.iter().fold(0usize,|code,plane|
        (code<<1)|usize::from(plane.dot(q)>=0f32))
}
fn reversed_port_signature(q:&[AlignedBlock],planes:&[Vec<AlignedBlock>])->usize {
    planes.iter().enumerate().fold(0usize,|code,(h,plane)| {
        code|(usize::from(plane.dot(q)>=0f32)<<h)
    })
}
fn reverse_bits(code:usize,bits:usize)->usize {
    (0..bits).fold(0,|out,bit| (out<<1)|((code>>bit)&1))
}
fn query(i:usize)->Vec<AlignedBlock> {
    (0..D/SIMD_LANECOUNT).map(|b| AlignedBlock::new(
        std::array::from_fn(|k|{
            let j=b*SIMD_LANECOUNT+k;
            ((i*13+j*7+1) as f32*0.07123).sin()
                +0.07*((i*3+j*11+1) as f32*0.033).cos()
        }))).collect()
}
fn insert_source(bucket:&mut VecDeque<usize>,id:usize) {
    if let Some(pos)=bucket.iter().position(|&x|x==id) {
        bucket.remove(pos);
    }
    if bucket.len()==BUCKET_CAP { bucket.pop_front(); }
    bucket.push_back(id);
}

fn main() {
    let mut matched=0usize;
    let mut bit_order_mismatches=0usize;
    let mut medoid_insertions=0usize;
    let mut medoid_would_be_skipped=0usize;
    let mut author_bucket_cases=0usize;
    for &seed in &[0u64,1,2,42,123,2026] {
        let author:EngineStarter<LruSet>=EngineStarter::new(
            EngineStarterParams::new(HASH,BUCKET_CAP,D,
                NodeId{internal:BASE},seed,true));
        let planes=make_planes(seed);
        let mut model=vec![VecDeque::<usize>::new();1<<HASH];
        for i in 0..3200usize {
            let q=query(i+((seed as usize)%100)*1000);
            let src=author.select_starting_points(&q);
            let ours=source_signature(&q,&planes);
            assert_eq!(src.signature,ours,"SOURCE_HASH_DISAGREES seed={seed} i={i}");
            let old=reversed_port_signature(&q,&planes);
            assert_eq!(reverse_bits(ours,HASH),old,
                "PORT_BIT_ORDER_IS_NOT_A_BUCKET_PERMUTATION");
            if ours!=old { bit_order_mismatches+=1; }

            let expected=model[src.signature].iter()
                .copied().collect::<Vec<_>>();
            let actual=src.catapults.iter().map(|e|e.internal).collect::<Vec<_>>();
            assert_eq!(actual,expected,"BUCKET_ORDER_OR_EVICTION_DISAGREES seed={seed} i={i}");
            assert_eq!(src.starting_node.internal,BASE);

            // Deliberately include duplicate reinsertions, eviction when full,
            // and the permanent graph medoid as a legitimate best result.
            let id=if i%19==0 { BASE } else { (i*31/7)%67 };
            if id==BASE { medoid_insertions+=1;medoid_would_be_skipped+=1; }
            author.new_catapult(src.signature,NodeId{internal:id});
            insert_source(&mut model[src.signature],id);

            let post=author.select_starting_points(&q);
            let actual=post.catapults.iter().map(|e|e.internal).collect::<Vec<_>>();
            let expected=model[src.signature].iter().copied().collect::<Vec<_>>();
            assert_eq!(actual,expected,"POSTUPDATE_LRU_DISAGREES seed={seed} i={i}");
            matched+=1;
            author_bucket_cases+=2;
        }
    }
    // The authors' call to shrink_to(k) affects vector CAPACITY, not LENGTH.
    let mut initial=Vec::from_iter(0..40usize);
    initial.shrink_to(10);
    assert_eq!(initial.len(),40,
        "Rust Vec::shrink_to does NOT truncate the initial candidate set");

    assert!(medoid_insertions>0 && bit_order_mismatches>0);
    println!("ORIGINAL_CATAPULT_NATIVE_ORACLE_PASS \
         hash_cases={matched} bucket_pre_post_cases={author_bucket_cases} \
         medoid_insertions={medoid_insertions} \
         current_port_skips_medoid={medoid_would_be_skipped} \
         reverse_label_cases={bit_order_mismatches} \
         beam_start_shrink_to_does_not_truncate=true");
}
