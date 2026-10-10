#!/usr/bin/env python3
"""Match author CatapultDB semantics in a TEMPORARY patched DiskANN checkout.

Apply AFTER patch_diskann_paper_catapult.py. The experimental SSD backend,
PQ scorer and native search algorithm are untouched.

Adjust RNG to original StdRng + rand_distr StandardNormal, bit packing to MSB,
admit medoid winners to the LRU, and account for post-query feedback in latency.
"""
from pathlib import Path
import argparse

AUTHOR_RUST_SHA="a16eddd34b4339db5ec86e292470ce7929179bc3"
PINNED_DISKANN_SHA="fcf90534174cf29c78c9f13b4cccf1fcabff85f5"

def once(s,old,new,label):
    n=s.count(old)
    if n!=1:
        raise RuntimeError(f"{label}: expected 1 exact anchor; found {n}")
    return s.replace(old,new,1)

def align(root):
    cargo=root/"diskann-benchmark/Cargo.toml"
    s=cargo.read_text()
    s=once(s,
       "rand.workspace = true\nrayon.workspace = true",
       "rand.workspace = true\nrand_distr.workspace = true\nrayon.workspace = true",
       "pinned rand_distr workspace import")
    cargo.write_text(s)

    p=root/"diskann-benchmark/src/disk_index/search.rs"
    s=p.read_text()
    s=once(s,
       "use rand::{rngs::StdRng, Rng, SeedableRng};",
       "use rand::{rngs::StdRng, Rng, SeedableRng};\nuse rand_distr::StandardNormal;",
       "authors RNG distribution import")
    old=r'''        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut hyperplanes = Vec::with_capacity(cfg.hashes * dim);
        while hyperplanes.len() < cfg.hashes * dim {
            // Box-Muller transform from the already-pinned rand crate.
            // Random-hyperplane LSH needs isotropic normal directions.
            let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
            let u2 = rng.random::<f32>();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            hyperplanes.push(radius * theta.cos());
            if hyperplanes.len() < cfg.hashes * dim {
                hyperplanes.push(radius * theta.sin());
            }
        }
'''
    new=r'''        // Authors released StdRng + rand_distr::StandardNormal,
        // with f32 values emitted in plane-major order.
        let mut gaussian = StdRng::seed_from_u64(cfg.seed).sample_iter(StandardNormal);
        let mut hyperplanes = Vec::with_capacity(cfg.hashes * dim);
        while hyperplanes.len() < cfg.hashes * dim {
            let sample: f32 = gaussian.next().expect("infinite Gaussian iterator");
            hyperplanes.push(sample);
        }
'''
    s=once(s,old,new,"author-identical random hyperplanes")
    s=once(s,
       "            if dot >= 0.0 {\n                code |= 1usize << h;\n            }",
       "            code = (code << 1) | usize::from(dot >= 0.0);",
       "author-identical MSB-first LSH bit packing")
    s=once(s,
       "        if destination == self.medoid {\n            return Ok(());\n        }\n\n        let mut guard = self.buckets[bucket]",
       "        // Original author's bucket stores medoid winners too.\n"
       "        let mut guard = self.buckets[bucket]",
       "original author permits medoid residency")
    s=once(s,
       """                        // Report complete request time, including Catapult hashing/cache
                        // lookup and optional static portal routing.
                        stats.total_execution_time_us = query_timer.elapsed().as_micros();

""",
       "",
       "remove early timing boundary without assuming frozen vs online replay")
    # Original snapshot experiments wrap updates in if !catapult_freeze.
    # Both frozen and online variants must use the same correct end timing.
    if "                        if !catapult_freeze {" in s:
        anchor='''                                        has_any_search_failed.store(true, Ordering::Release);
                                    }
                                }
                            }
                        }
'''
        s=once(s,anchor,anchor+
            "                        // Include bucket lookup and conditional feedback in per-query timing.\n"
            "                        stats.total_execution_time_us = query_timer.elapsed().as_micros();\n",
            "time author-semantics snapshot online or frozen search")
    else:
        anchor='''                                    has_any_search_failed.store(true, Ordering::Release);
                                }
                            }
                        }
'''
        s=once(s,anchor,anchor+
            "                        // Include author bucket feedback in query time.\n"
            "                        stats.total_execution_time_us = query_timer.elapsed().as_micros();\n",
            "time author-semantics online search")
    p.write_text(s)
    print("CATAPULT_AUTHOR_SEMANTICS_ALIGNED "
          "StdNormal_f32=1 msb_bits=1 medoid_stored=1 "
          "feedback_in_timing=1 original_author="+AUTHOR_RUST_SHA,flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("diskann",type=Path)
    args=ap.parse_args()
    align(args.diskann.resolve())
if __name__=="__main__":main()
