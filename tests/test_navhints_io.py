"""Binary-format and failed-copy checks using tiny synthetic replay inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "navhints_io", ROOT / "scripts/navhints_io.py"
)
binary_io = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binary_io)


class ReplayBinaryInputs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.bin"
        self.output = self.root / "output.bin"

    def test_float_vector_slice_preserves_raw_bytes(self):
        payload = struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 8)
        self.source.write_bytes(struct.pack("<II", 4, 2) + payload)
        binary_io.slice_vectors(self.source, self.output, 1, 2)
        self.assertEqual(
            self.output.read_bytes(), struct.pack("<II", 2, 2) + payload[8:24]
        )

    def test_uint8_vector_slice_preserves_raw_bytes(self):
        self.source.write_bytes(struct.pack("<II", 4, 2) + bytes(range(8)))
        binary_io.slice_vectors(self.source, self.output, 1, 2, itemsize=1)
        self.assertEqual(
            self.output.read_bytes(), struct.pack("<II", 2, 2) + bytes(range(2, 6))
        )

    def test_ground_truth_ids_only(self):
        ids = struct.pack("<8I", *range(8))
        self.source.write_bytes(struct.pack("<II", 4, 2) + ids)
        binary_io.slice_ground_truth(self.source, self.output, 1, 2)
        self.assertEqual(self.output.read_bytes(), struct.pack("<II", 2, 2) + ids[8:24])

    def test_ground_truth_copies_separate_id_and_distance_blocks(self):
        ids = struct.pack("<8I", *range(8))
        distances = struct.pack("<8f", *[n / 10 for n in range(8)])
        self.source.write_bytes(struct.pack("<II", 4, 2) + ids + distances)
        binary_io.slice_ground_truth(
            self.source, self.output, 1, 2, require_distances=True
        )
        self.assertEqual(
            self.output.read_bytes(),
            struct.pack("<II", 2, 2) + ids[8:24] + distances[8:24],
        )

    def test_short_header_reports_the_source_path(self):
        self.source.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "source.bin.*8-byte"):
            binary_io.read_xbin_shape(self.source)

    def test_empty_shape_is_rejected(self):
        self.source.write_bytes(struct.pack("<II", 0, 2))
        with self.assertRaisesRegex(ValueError, "empty shape"):
            binary_io.read_xbin_shape(self.source)

    def test_invalid_ranges_leave_existing_output_intact(self):
        self.source.write_bytes(struct.pack("<II", 4, 2) + bytes(range(8)))
        for start, count in ((-1, 1), (0, 0), (3, 2)):
            with self.subTest(start=start, count=count):
                self.output.write_bytes(b"keep this result")
                with self.assertRaisesRegex(ValueError, "requested rows"):
                    binary_io.slice_vectors(self.source, self.output, start, count, 1)
                self.assertEqual(self.output.read_bytes(), b"keep this result")

    def test_truncated_vector_copy_does_not_replace_output(self):
        self.source.write_bytes(struct.pack("<II", 4, 2) + b"ab")
        self.output.write_bytes(b"keep this result")
        with self.assertRaisesRegex(ValueError, "truncated payload"):
            binary_io.slice_vectors(self.source, self.output, 0, 4, 1)
        self.assertEqual(self.output.read_bytes(), b"keep this result")
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["output.bin", "source.bin"]
        )

    def test_malformed_ground_truth_does_not_replace_output(self):
        self.source.write_bytes(struct.pack("<II", 4, 2) + b"incomplete")
        self.output.write_bytes(b"keep this result")
        with self.assertRaisesRegex(ValueError, "unsupported ground-truth size"):
            binary_io.slice_ground_truth(self.source, self.output, 0, 1)
        self.assertEqual(self.output.read_bytes(), b"keep this result")

    def test_coveo_contract_requires_distances(self):
        self.source.write_bytes(
            struct.pack("<II", 4, 2) + struct.pack("<8I", *range(8))
        )
        with self.assertRaisesRegex(ValueError, "requires.*distance block"):
            binary_io.slice_ground_truth(
                self.source, self.output, 0, 1, require_distances=True
            )
        self.assertFalse(self.output.exists())

    def test_input_cannot_be_its_own_output(self):
        original = struct.pack("<II", 4, 2) + bytes(range(8))
        self.source.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "must differ"):
            binary_io.slice_vectors(self.source, self.source, 0, 1, 1)
        self.assertEqual(self.source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
