"""Protect maintained replay dependencies during repository cleanup.

These checks parse sources and generate native-Starling shell scripts only.
They do not install dependencies, fetch engines, read datasets or run SSD work.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MaintainedRepository(unittest.TestCase):
    def test_local_script_imports_resolve(self):
        prefixes = ("patch_", "qualify_", "summarize_", "learn_", "prepare_", "build_")
        for path in (ROOT / "scripts").glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names = (
                    [node.module]
                    if isinstance(node, ast.ImportFrom)
                    else (
                        [alias.name for alias in node.names]
                        if isinstance(node, ast.Import)
                        else []
                    )
                )
                for name in names:
                    if name and name.startswith(prefixes):
                        self.assertTrue(
                            (ROOT / "scripts" / (name + ".py")).is_file(),
                            f"{path.name} imports missing local helper {name}",
                        )

    def test_workflow_script_targets_exist(self):
        for workflow in (ROOT / ".github/workflows").glob("*.yml"):
            for name in re.findall(r"scripts/[\w.-]+\.(?:py|sh)", workflow.read_text()):
                if ".generated.sh" not in name:
                    self.assertTrue(
                        (ROOT / name).is_file(), f"{workflow.name}: missing {name}"
                    )

    @unittest.skipUnless(
        shutil.which("bash"), "Native replay source generation requires Bash"
    )
    def test_final_starling_replay_generation_matches_frozen_source(self):
        reference = json.loads(
            (ROOT / "repro/starling-replay-reference.json").read_text()
        )
        with tempfile.TemporaryDirectory(prefix="navhints-replay-source-") as folder:
            work = Path(folder)
            shutil.copytree(
                ROOT / "scripts",
                work / "scripts",
                ignore=shutil.ignore_patterns("__pycache__", "*.generated.sh"),
            )
            for expected in reference["datasets"]:
                dataset = expected["dataset"]
                generator = (
                    "generate_starling_final_core_missing.py"
                    if dataset in ("pubmed", "text2image")
                    else "generate_starling_final_core.py"
                )
                result = subprocess.run(
                    [sys.executable, "scripts/" + generator, dataset],
                    cwd=work,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = (
                    work / f"scripts/run_starling_final_core_{dataset}.generated.sh"
                )
                payload = output.read_bytes()
                self.assertEqual(len(payload), expected["bytes"], dataset)
                self.assertEqual(
                    hashlib.sha256(payload).hexdigest(), expected["sha256"], dataset
                )
                subprocess.run(["bash", "-n", str(output)], check=True)


if __name__ == "__main__":
    unittest.main()
