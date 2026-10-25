"""Check useful diagnostics when regenerated plot CSVs are malformed."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load_repro_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "repro" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = load_repro_module("verify_review_bundle")
with mock.patch.dict(sys.modules, {"verify_review_bundle": checker}):
    exporter = load_repro_module("export_paper_results")


class PaperCsvReview(unittest.TestCase):
    def check_csv(self, expected, actual):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper = root / "paper"
            references = paper / "reference-data"
            output = root / "output"
            references.mkdir(parents=True)
            output.mkdir()
            if expected is not None:
                (references / "result.csv").write_text(expected)
            (output / "result.csv").write_text(actual)
            with mock.patch.object(exporter, "PAPER", paper):
                with contextlib.redirect_stdout(io.StringIO()):
                    return exporter.compare_reference_csvs(output)

    def test_rounded_values_and_matched_nan_gaps(self):
        expected = "recall,cost\n50.0,1.2345\n60.0,nan\n"
        actual = "recall,cost\n50,1.234501\n60,nan\n"
        self.assertEqual(self.check_csv(expected, actual), 1)

    def test_missing_cell_identifies_row_and_column(self):
        with self.assertRaisesRegex(
            ValueError, "result.csv: row 2, column cost: missing cell"
        ):
            self.check_csv("recall,cost\n50,2\n", "recall,cost\n50\n")

    def test_changed_column_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "column names or order"):
            self.check_csv("recall,cost\n50,2\n", "cost,recall\n2,50\n")

    def test_changed_value_identifies_row_and_column(self):
        with self.assertRaisesRegex(ValueError, "row 2, column cost: 3.0 differs"):
            self.check_csv("recall,cost\n50,2\n", "recall,cost\n50,3\n")

    def test_source_export_omits_interpreter_caches(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paper, output = root / "paper", root / "output"
            output.mkdir()
            for name in ("figures", "tables", "scripts"):
                (paper / name).mkdir(parents=True)
            script = paper / "scripts/derive.py"
            script.write_text("print('original derivation')\n")
            cache = paper / "scripts/__pycache__"
            cache.mkdir()
            (cache / "derive.cpython.pyc").write_bytes(b"host-specific bytecode")
            for name in ("macros.tex", "figures.tex", "SOURCE.json"):
                (paper / name).write_text(name)
            with mock.patch.object(exporter, "PAPER", paper):
                exporter.copy_paper_sources(output)
            self.assertEqual(
                (output / "scripts/derive.py").read_bytes(), script.read_bytes()
            )
            self.assertFalse(any(output.rglob("__pycache__")))
            self.assertFalse(any(output.rglob("*.pyc")))

    def test_missing_reference_set_cannot_pass_as_zero_checks(self):
        with self.assertRaisesRegex(ValueError, "No frozen reference CSVs"):
            self.check_csv(None, "recall,cost\n50,2\n")


if __name__ == "__main__":
    unittest.main()
