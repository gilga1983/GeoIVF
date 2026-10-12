#!/usr/bin/env python3
"""Unit tests for complete-service instrumentation and aggregation."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patch_diskann_full_service import patch_catapult, patch_nav, once
from qualify_navhints_service_latency import (
    EXPECTED_FIELDS, frontier, interpolate, matched_recall, percentile,
    profile_summary,
)


class ServiceProfileTests(unittest.TestCase):
    def test_timer_on_early_continue_and_suffix_csv_hooks(self):
        stub = """// Simplified internal structures to reduce parameter count
        let mut catapult_measured_start = None::<Instant>;
        if catapult_causal_replay {
            for qi in 0..num_queries {
                if qi == catapult_causal_warmup {
                    catapult_measured_start = Some(Instant::now());
                }
                let q = queries.row(qi);
            }
        }
        let total_time = if catapult_causal_replay {
"""
        out = patch_catapult(stub)
        self.assertIn("impl Drop for CompleteRequestTimer", out)
        self.assertIn("CompleteRequestTimer::new(&mut complete_service_us[qi])", out)
        self.assertIn("for qi in catapult_causal_warmup..num_queries", out)
        self.assertLess(out.index("std::fs::write(&dest, csv)?"), out.index("let total_time"))
        with self.assertRaises(RuntimeError):
            patch_catapult(out)  # fail closed if applied twice

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(percentile([5, 5, 5], 99), 5)

    def test_profile_accounts_complete_request_and_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "L160.csv"
            with p.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=EXPECTED_FIELDS)
                writer.writeheader()
                for i in range(1000):
                    writer.writerow({
                        "query_index": 4000 + i,
                        "service_us": 200 + i,
                        "engine_us": 100 + i,
                        "remainder_us": 100,
                        "io_us": 10,
                        "pq_preprocess_us": 10,
                        "reported_cpu_us": 80,
                        "reads": 7,
                        "comparisons": 20,
                        "hops": 5,
                    })
            summary = profile_summary(p)
            self.assertEqual(summary["measured_queries"], 1000)
            self.assertEqual(summary["mean_remainder_us"], 100)
            self.assertAlmostEqual(summary["mean_service_us"], 699.5)
            self.assertEqual(summary["mean_reads"], 7)
            self.assertGreater(summary["service_p99_us"], summary["service_p50_us"])
            self.assertAlmostEqual(summary["serial_throughput_qps"], 1e6 / 699.5)
            with p.open("a") as f:
                f.write("extra")
            with self.assertRaises(ValueError):
                profile_summary(p)

    def test_no_extrapolation(self):
        points = {
            10: {"recall": 60.0, "mean_service_us": 100.0, "mean_engine_us": 80,
                 "mean_remainder_us": 20, "mean_io_us": 40, "mean_reads": 5},
            20: {"recall": 80.0, "mean_service_us": 200.0, "mean_engine_us": 160,
                 "mean_remainder_us": 40, "mean_io_us": 80, "mean_reads": 10},
        }
        self.assertIsNone(interpolate(points, 90))
        self.assertEqual(interpolate(points, 70)["mean_reads"], 7.5)
        self.assertEqual(len(frontier(points)), 2)

    def test_strict_source_anchors(self):
        with self.assertRaises(RuntimeError):
            once("x x", "x", "y", "ambiguous")
        with self.assertRaises(RuntimeError):
            once("abc", "x", "y", "missing")


if __name__ == "__main__":
    unittest.main()
