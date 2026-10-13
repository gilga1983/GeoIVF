#!/usr/bin/env python3
"""No-SSD regression tests for score controls and matched-recall accounting."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_navhints_score_controls import rank_from_trace
from qualify_navhints_review_gates import frontier, interpolate, summarize


class ReviewGateTests(unittest.TestCase):
    def test_rankers_consume_identical_candidate_universe(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.jsonl"
            rows = [
                {"query": 0, "ids": [0, 10, 11, 12, 12]},
                {"query": 1, "ids": [0, 10, 11, 13]},
                {"query": 2, "ids": [0, 12, 13, 14]},
            ]
            p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
            n, eligible, ranking = rank_from_trace(p, random_seed=7)
            self.assertEqual(n, 3)
            self.assertEqual(eligible, [10, 11, 12, 13, 14])
            for vals in ranking.values():
                self.assertEqual(set(vals), set(eligible))
                self.assertEqual(len(vals), len(set(vals)))
            self.assertEqual(ranking["position-sum"][0], 13)  # late + repeated
            self.assertEqual(ranking["frequency"][0], 10)  # tie by vertex ID
            self.assertEqual(ranking["random"],
                             rank_from_trace(p, random_seed=7)[2]["random"])

    def test_rankers_reject_noncontiguous_training(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trace.jsonl"
            p.write_text('{"query":1,"ids":[0,1]}\n')
            with self.assertRaises(ValueError):
                rank_from_trace(p, random_seed=1)

    def test_no_extrapolation_and_common_ref_anchors(self):
        base = {"20": {"recall": 20., "reads": 30., "latency_us": 100.},
                "40": {"recall": 40., "reads": 50., "latency_us": 200.},
                "80": {"recall": 60., "reads": 80., "latency_us": 300.},
                "160": {"recall": 80., "reads": 120., "latency_us": 400.}}
        competitor = {"20": {"recall": 25., "reads": 35., "latency_us": 110.},
                      "40": {"recall": 45., "reads": 65., "latency_us": 240.},
                      "80": {"recall": 65., "reads": 98., "latency_us": 350.},
                      "160": {"recall": 85., "reads": 145., "latency_us": 480.}}
        self.assertIsNone(interpolate(frontier(competitor), 24., "reads"))
        result = summarize([{"ref": base, "alt": competitor}],
                           [("ref", 10, 2, 8), ("alt", 10, 2, 8)], "ref")
        self.assertEqual(result["per_repetition"][0]["anchor_count"], 3)
        self.assertGreater(result["summary"]["alt"]["additional_read_percent"], 0)

    def test_frontier_removes_strictly_dominated_points(self):
        curve = {"20": {"recall": 40., "reads": 30., "latency_us": 100.},
                 "40": {"recall": 50., "reads": 25., "latency_us": 90.},
                 "80": {"recall": 70., "reads": 60., "latency_us": 200.}}
        f = frontier(curve)
        self.assertEqual([p["L"] for p in f], [40, 80])


if __name__ == "__main__":
    unittest.main()
