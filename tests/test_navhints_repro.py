"""Unit/contract tests for the paper reproduction interface.

All tests use synthetic manifests/results and standard-library mocks; they do
not download data, compile DiskANN or use the SSD. Real artifact validation is
additionally checked against the archived ten successful campaigns.
"""

from __future__ import annotations
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "navhints_repro_launcher", ROOT / "repro" / "run.py"
)
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)


class ManifestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = repro.manifest()

    def test_supported_final_core(self):
        core = self.m["controller"]
        self.assertEqual(
            (
                core["entry_budget"],
                core["destination_capacity"],
                core["ranked_extra_starts"],
            ),
            (14464, 2048, 8),
        )
        self.assertEqual(len(self.m["runs"]), 14)
        self.assertEqual(set(x["id"] for x in self.m["runs"]), set(repro.ARCHIVE_IDS))
        for record in self.m["runs"]:
            self.assertRegex(record["source_sha"], r"^[a-f0-9]{40}$")
            self.assertTrue(record["data_member"].endswith(".json"))
            self.assertRegex(record["original_artifact_sha256"], r"^[a-f0-9]{64}$")

    def test_source_dependencies_present(self):
        repro.check_mode_sources(self.m, "all")

    def test_list_and_dry_run_are_side_effect_free(self):
        log = io.StringIO()
        with mock.patch.object(
            repro.subprocess,
            "call",
            side_effect=AssertionError(
                "A no-execute command attempted to run an SSD benchmark."
            ),
        ):
            with contextlib.redirect_stdout(log):
                self.assertEqual(repro.run_list(self.m), 0)
                self.assertEqual(
                    repro.run_experiment(self.m, "pubmed", False, False, False), 0
                )
                self.assertEqual(
                    repro.run_experiment(self.m, "starling-coveo", False, False, False),
                    0,
                )
        self.assertIn("Dry run only", log.getvalue())

    def test_fetch_preview_has_no_gh_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "not-created"
            with mock.patch.object(
                repro.subprocess,
                "run",
                side_effect=AssertionError("A preview attempted a download."),
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        repro.fetch(self.m, "all", target, False, False), 0
                    )
            self.assertFalse(
                target.exists(), "A dry-run must not create an archive directory"
            )

    def test_paper_preview_is_safe(self):
        with mock.patch.object(
            repro.subprocess,
            "run",
            side_effect=AssertionError("A preview attempted to build a paper."),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    repro.paper(
                        self.m,
                        ROOT.parent / "nonexistent-paper",
                        ROOT / "nonexistent-archives",
                        False,
                    ),
                    0,
                )

    def test_starling_coveo_uses_private_coveo_requirements(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(repro, "DATA_ROOT", Path(tmp)),
        ):
            checks = repro.present_checks(self.m, "starling-coveo")
        self.assertTrue(checks)
        self.assertTrue(all("coveo-real-demand-v1" in p for p, _ in checks))


class ArtifactValidation(unittest.TestCase):
    def make_artifact(self, record, main_payload, extra=None, sha=None):
        d = tempfile.TemporaryDirectory()
        path = Path(d.name) / "test.zip"
        sha = sha or record["source_sha"]
        source = record.get(
            "source_member",
            (
                "geoivf-revision.txt"
                if record["id"].startswith("starling-")
                else "geoivf-commit.txt"
            ),
        )
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(record["data_member"], json.dumps(main_payload))
            for k, v in (extra or {}).items():
                z.writestr(k, json.dumps(v))
            z.writestr(source, sha + "\n")
        return d, path

    @classmethod
    def setUpClass(cls):
        cls.recs = {q["id"]: q for q in repro.manifest()["runs"]}

    def test_persistence_record_and_commit(self):
        rec = self.recs["persistence"]
        case = {"controller": {"B": 14464, "C": 2048, "k": 8}, "comparison": {}}
        d, path = self.make_artifact(rec, case)
        try:
            repro.check_artifact(path, rec)
        finally:
            d.cleanup()

        d, path = self.make_artifact(rec, case, sha="0" * 40)
        try:
            with self.assertRaisesRegex(ValueError, "wrong source SHA"):
                repro.check_artifact(path, rec)
        finally:
            d.cleanup()

    def test_sensitivity_uses_core_not_controller(self):
        rec = self.recs["sensitivity"]
        c = {
            "core": {
                "junction_budget": 14464,
                "destination_capacity": 2048,
                "starts": 8,
            }
        }
        extras = {p: c for p in rec["additional_members"]}
        d, path = self.make_artifact(rec, c, extras)
        try:
            repro.check_artifact(path, rec)
        finally:
            d.cleanup()

    def test_catapult_exact_model_metadata(self):
        rec = self.recs["catapult"]
        p = {
            "auxiliary_payload_navhints": {
                "core_reallocated": {
                    "landmarks": 14464,
                    "recent_ids_capacity": 2048,
                    "injected_starts": 8,
                    "payload_bytes": 102940,
                }
            }
        }
        d, path = self.make_artifact(rec, p)
        try:
            repro.check_artifact(path, rec)
        finally:
            d.cleanup()

        p["auxiliary_payload_navhints"]["core_reallocated"]["injected_starts"] = 1
        d, path = self.make_artifact(rec, p)
        try:
            with self.assertRaisesRegex(
                ValueError, "Catapult/Core allocation mismatch"
            ):
                repro.check_artifact(path, rec)
        finally:
            d.cleanup()

    def test_service_latencies_are_archived_and_causal(self):
        for name in repro.SERVICE_ARCHIVES:
            record = self.recs[name]
            if name == "service-pubmed":
                payload = {
                    "controller": {"junctions": 14464, "recent": 2048, "starts": 8},
                    "repetitions": 3,
                    "catapult_seeds": [0, 1, 2],
                    "widths": [10],
                    "diagnostic": False,
                }
            else:
                payload = {
                    "dataset": {
                        "service-coveo": "coveo",
                        "service-text2image": "text2image-10M",
                        "service-bigann": "bigann-10M",
                    }[name],
                    "core": {
                        "actual_junctions": 12169 if name == "service-coveo" else 14464,
                        "recent": 2048,
                        "starts": 8,
                        "accounted_payload_bytes": 102940,
                    },
                    "catapult": {"seeds": [0, 1, 2], "accounted_payload_bytes": 102940},
                    "protocol": {
                        "rotated_repetitions": 3,
                        "in_flight_queries": 1,
                        "widths": [10],
                    },
                    "diagnostic": False,
                }
            payload["measurements"] = {
                f"r{rep}-L10-{arm}": {"recall": 50.0, "mean_service_us": 100.0}
                for rep in range(3)
                for arm in (
                    "baseline",
                    "navhints",
                    "catapult-s0",
                    "catapult-s1",
                    "catapult-s2",
                )
            }
            d, path = self.make_artifact(record, payload)
            try:
                repro.check_artifact(path, record)
                payload["diagnostic"] = True
                with zipfile.ZipFile(path, "w") as z:
                    z.writestr(record["data_member"], json.dumps(payload))
                    z.writestr(
                        record.get("source_member", "geoivf-commit.txt"),
                        record["source_sha"] + "\n",
                    )
                with self.assertRaisesRegex(
                    ValueError, "wrong final-Core service protocol"
                ):
                    repro.check_artifact(path, record)
            finally:
                d.cleanup()

    def test_original_zip_digest_check_is_separate_from_content_validation(self):
        record = self.recs["persistence"]
        payload = {"controller": {"B": 14464, "C": 2048, "k": 8}, "comparison": {}}
        d, path = self.make_artifact(record, payload)
        try:
            repro.check_artifact(path, record)
            self.assertNotEqual(
                repro.original_sha256(path), record["original_artifact_sha256"]
            )
        finally:
            d.cleanup()

    def test_final_four_arm_requires_exact_config(self):
        rec = self.recs["pubmed"]
        p = {
            "controller": {
                "entry_ids": 14464,
                "cache_ids": 2048,
                "additional_starts": 8,
            },
            "summary": {name: {} for name in ("baseline", "ivf", "core", "full")},
        }
        d, path = self.make_artifact(rec, p)
        try:
            repro.check_artifact(path, rec)
        finally:
            d.cleanup()

        p["controller"]["cache_ids"] = 512
        d, path = self.make_artifact(rec, p)
        try:
            with self.assertRaisesRegex(ValueError, "wrong destination capacity"):
                repro.check_artifact(path, rec)
        finally:
            d.cleanup()


if __name__ == "__main__":
    unittest.main()
