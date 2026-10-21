"""Offline, synthetic safety/contract tests of the artifact-review verifier."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "navhints_artifact_verifier", ROOT / "repro/verify_review_bundle.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ReviewerArtifact(unittest.TestCase):
    def test_public_three_manifest_inventory(self):
        digests = checker.check_source_manifests()
        self.assertEqual(len(digests), 30)
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


if __name__ == "__main__":
    unittest.main()
