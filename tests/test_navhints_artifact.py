"""Offline, synthetic safety/contract tests of the artifact-review verifier."""
from __future__ import annotations

import hashlib
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "navhints_artifact_verifier", ROOT / "repro/verify_review_bundle.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReviewerArtifact(unittest.TestCase):
    def test_public_four_manifest_inventory(self):
        digests = checker.check_source_manifests()
        self.assertEqual(len(digests), 32)
        self.assertEqual(len(set(digests)), len(digests))

    def make_zip(self, folder, *, corrupt=False, traversal=False):
        output = Path(folder) / "bundle.zip"
        contents = json.dumps({"metric": "logical-ssd-reads", "value": 123}).encode()
        digest = hashlib.sha256(contents).hexdigest()
        pathname = "../escape.json" if traversal else "evidence/paper/result.json"
        source_sha = "a" * 64
        manifest = {
            "format": "navhints-sanitized-summary-v1",
            "public_release_approved": False,
            "sources": [{"original_zip_sha256": source_sha, "retained": [{
                "published_member": "evidence/paper/result.json",
                "sanitized_bytes": len(contents),
                "sanitized_sha256": "b" * 64 if corrupt else digest,
            }]}],
            "results_members": 1,
        }
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("MANIFEST.json", json.dumps(manifest))
            z.writestr(pathname, contents)
        return output

    def test_rejects_tampered_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(self.make_zip(folder, corrupt=True)) as z:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    checker.archive_checks(z, {})

    def test_rejects_zip_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(self.make_zip(folder, traversal=True)) as z:
                with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                    checker.archive_checks(z, {})

    def test_valid_small_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(self.make_zip(folder)) as z:
                manifest = checker.archive_checks(z, {})
                self.assertEqual(manifest["results_members"], 1)

    def test_manifest_only_does_not_invoke_benchmarks(self):
        self.assertFalse(any(token in Path(ROOT / "repro/verify_review_bundle.py").read_text()
                             for token in ("subprocess.run(", "subprocess.call(", "os.system(")))

    def test_bundled_table5_independent_frontiers(self):
        with zipfile.ZipFile(ROOT / "repro/navhints-reviewer-evidence.zip") as z:
            meta = checker.archive_checks(z, checker.check_source_manifests())
            self.assertEqual(meta["results_members"], 48)
            with contextlib.redirect_stdout(io.StringIO()):
                rows = checker.destination_controls(z)
        self.assertEqual(len(rows), 9)
        self.assertLess(rows[0]["read_savings_pct"], 0)
        self.assertAlmostEqual(rows[2]["read_savings_pct"], rows[6]["read_savings_pct"])

    def test_table5_rejects_missing_recall_support(self):
        with zipfile.ZipFile(ROOT / "repro/navhints-reviewer-evidence.zip") as z:
            source = json.loads(z.read("evidence/destination-equal-work/demand-vs-random.json"))
            capacity = z.read("evidence/destination-capacity/reviewer-defense.json")
        for point in source["summary"]["randomdb512"].values():
            point["recall_percent"] = 90.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsupported.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("evidence/destination-equal-work/demand-vs-random.json", json.dumps(source))
                z.writestr("evidence/destination-capacity/reviewer-defense.json", capacity)
            with zipfile.ZipFile(path) as z:
                with self.assertRaisesRegex(ValueError, "outside common recall support"):
                    checker.destination_controls(z)

    def test_public_export_matches_manuscript_without_private_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / "repro/export_paper_results.py"),
                                     "--output", tmp], cwd=tmp, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((Path(tmp) / "REPRODUCTION.json").read_text())
            self.assertEqual(report["regenerated_reference_csvs"], 20)
            self.assertFalse(report["ssd_measurements_rerun"])


if __name__ == "__main__":
    unittest.main()
