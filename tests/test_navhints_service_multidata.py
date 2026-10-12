#!/usr/bin/env python3
"""Static regression tests for the multiworkload causal service suite."""
from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qualify_navhints_service_multidata import (
    WORKLOADS, catapult_capacity, interpolated_across_repetitions,
    make_unscored_training_truth, shape, truth_shape,
)


class MultiDatasetTests(unittest.TestCase):
    def test_datasets_preserve_frozen_splits_and_metrics(self):
        self.assertEqual(set(WORKLOADS), {"coveo","text2image-10M","bigann-10M"})
        self.assertEqual(WORKLOADS["coveo"]["warmup"], 20000)
        self.assertEqual(WORKLOADS["coveo"]["measure"], 5000)
        self.assertEqual(WORKLOADS["bigann-10M"]["data_type"], "uint8")
        self.assertEqual(WORKLOADS["bigann-10M"]["distance"], "squared_l2")
        self.assertEqual(WORKLOADS["text2image-10M"]["dim"], 200)

    def test_payload_budget_never_undersizes_catapult(self):
        for payload,dim in ((93760,50),(102940,200),(102940,128),
                            (102940,768)):
            cap,actual=catapult_capacity(payload,dim)
            self.assertGreaterEqual(actual,payload)
            self.assertGreaterEqual(cap,80)
            self.assertLess(actual-payload,1024+8*dim*4)

    def test_training_truth_not_oracle_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt=Path(tmp)/"sentinel.gt"
            make_unscored_training_truth(gt)
            self.assertEqual(truth_shape(gt),(5000,10))
            with gt.open("rb") as f:
                f.seek(8)
                self.assertEqual(f.read(4),b"\0\0\0\0")

    def test_strict_xbin_and_gt_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"unit.u8bin"
            p.write_bytes(struct.pack("<II",5,3)+bytes(range(15)))
            self.assertEqual(shape(p,1),(5,3))
            with self.assertRaises(ValueError):
                shape(p,4)
            with self.assertRaises(ValueError):
                truth_shape(p)

    def test_no_fake_four_way_comparisons_when_catapult_absent(self):
        vals={}
        for l,rc in ((40,50.),(80,70.)):
            for arm,offset in (("baseline",0),("junctions",10),
                               ("navhints",20),("catapult-s0",30)):
                vals[f"r0-L{l}-{arm}"]={
                    "recall":rc if arm!="catapult-s0" else rc-40,
                    "mean_service_us":100+l+offset,
                    "mean_engine_us":80+l+offset,
                    "mean_remainder_us":20,
                    "mean_io_us":40,
                    "mean_reads":60,
                }
        three,four=interpolated_across_repetitions(vals,[40,80],1,[0])
        self.assertEqual(len(three),2)
        self.assertEqual(len(four),0)
        self.assertTrue(all(e["arms"]["catapult"] is None for e in three))


if __name__=="__main__":
    unittest.main()
