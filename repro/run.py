#!/usr/bin/env python3
"""NavHints paper reproduction launcher. Safe by default; explicit --execute.

Examples:
  python3 repro/run.py list
  python3 repro/run.py doctor --mode pubmed
  python3 repro/run.py run pubmed --execute
  python3 repro/run.py fetch all --execute
  python3 repro/run.py verify all
  python3 repro/run.py paper --execute --paper-dir ../DemmandAwareANN

Experiments are expensive: the launcher NEVER starts an SSD measurement or
downloads public datasets without explicit approval via --execute.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "repro" / "manifest.json"
DATA_ROOT = Path.home() / ".cache" / "geoivf"
REPRO_RUNS = ("coveo","pubmed","text2image","bigann","catapult","memory",
              "sensitivity","persistence","starling-coveo","starling-bigann")
SERVICE_ARCHIVES = ("service-pubmed","service-coveo","service-text2image","service-bigann")
ARCHIVE_IDS = REPRO_RUNS + SERVICE_ARCHIVES
REQUIRED_TOOLS = ("bash","git","python3","flock","sha256sum","cargo","rustup")


def manifest() -> dict:
    m = json.loads(MANIFEST_PATH.read_text())
    if m.get("schema_version") != 1:
        raise RuntimeError("Unknown reproducibility manifest version")
    if len(m["runs"]) != len({x["id"] for x in m["runs"]}):
        raise ValueError("Duplicate run IDs in manifest")
    if set(x["id"] for x in m["runs"]) != set(ARCHIVE_IDS):
        raise ValueError("Frozen evidence inventory does not cover all fourteen artifacts")
    for x in m["runs"]:
        if len(x["source_sha"]) != 40 or len(x["original_artifact_sha256"]) != 64 \
                or any(ch not in "0123456789abcdef" for ch in
                       x["source_sha"] + x["original_artifact_sha256"]):
            raise ValueError(x["id"] + ": invalid provenance hash")
    if set(m["replay_commands"]) != set(REPRO_RUNS):
        raise ValueError("Missing or unknown final-Core experiment commands")
    if m["controller"]["entry_budget"] != 14464 \
            or m["controller"]["destination_capacity"] != 2048 \
            or m["controller"]["ranked_extra_starts"] != 8:
        raise ValueError("Not the approved final NavHints Core")
    return m


def present_checks(m: dict, mode: str) -> list[tuple[str, bool]]:
    selected = (mode,) if mode != "all" else tuple(m["paths"])
    selected = tuple("bigann" if k == "starling-bigann" else
                     "coveo" if k == "starling-coveo" else
                     "pubmed" if k in ("catapult","memory","sensitivity","persistence") else k
                     for k in selected)
    result = []
    for dataset in dict.fromkeys(selected):
        for name in m["paths"][dataset]:
            path = DATA_ROOT / name
            result.append((str(path), path.is_file() and path.stat().st_size > 0))
    return result


def check_mode_sources(m: dict, mode: str) -> None:
    for name in (mode,) if mode != "all" else REPRO_RUNS:
        command = m["replay_commands"][name]
        if len(command) < 2:
            raise RuntimeError(f"{name}: missing launch command")
        script = ROOT / command[1]
        if not script.is_file():
            raise FileNotFoundError(f"{name}: missing version-controlled script {script}")


def run_list(m: dict) -> int:
    print("Final NavHints Core: B<=14464, Recent2048, up to 8 starts")
    for name in REPRO_RUNS:
        a = next((e for e in m["runs"] if e["id"] == name), None)
        status = f"verified run #{a['run_id']} artifact #{a['artifact_id']}" if a else "no artifact"
        print(f"{name:18} {' '.join(m['replay_commands'][name]):68} {status}")
    print("\nEvidence-only complete-service latency archives (no automatic SSD replay):")
    for name in SERVICE_ARCHIVES:
        a = next(x for x in m["runs"] if x["id"] == name)
        print(f"{name:18} run #{a['run_id']} artifact #{a['artifact_id']}")
    return 0


def doctor(m: dict, mode: str) -> int:
    check_mode_sources(m, mode)
    print("Source scripts: PASS")
    print("Operating system:", platform.platform())
    print("Repository root:", ROOT)
    print("Required external program checks:")
    programs = [(n, bool(shutil.which(n))) for n in REQUIRED_TOOLS]
    for name, good in programs:
        print(f"  {'PASS' if good else 'MISSING'} {name}")
    print("Dataset cache root (frozen runner layout):", DATA_ROOT)
    files = present_checks(m, mode)
    for name, good in files:
        print(f"  {'PASS' if good else 'MISSING'} {name}")
    print("Offline dataset readiness:", sum(ok for _, ok in files), "/", len(files))
    # doctor is a diagnostic command, not a failure just because dataset
    # payloads and Rust toolchain are absent on a new laptop.
    return 0 if all(good for _, good in programs) and all(good for _, good in files) else 2


def run_experiment(m: dict, name: str, execute: bool,
                   allow_public_download: bool, overwrite: bool) -> int:
    check_mode_sources(m, name)
    cmd = m["replay_commands"][name]
    print("Frozen code command:", " ".join(cmd))
    print("Code repository:", m["code_repository"])
    print("Pinned DiskANN:", m["engines"]["diskann"]["commit"])
    if name.startswith("starling"):
        print("Pinned original Starling:", m["engines"]["starling"]["commit"])
    if name == "catapult":
        print("Catapult author source audit:", m["engines"]["catapult"]["source_commit"])
    if not execute:
        print("Dry run only. Add --execute to start an actual SSD experiment.")
        return 0
    if platform.system() != "Linux":
        raise RuntimeError("The measured SSD protocol requires Linux self-hosted hardware")
    if any(not shutil.which(t) for t in REQUIRED_TOOLS):
        raise RuntimeError("Missing required executable. Run doctor first.")
    missing = [p for p, good in present_checks(m, name) if not good]
    if missing and not (allow_public_download and name in ("text2image","bigann")):
        raise RuntimeError("Frozen inputs not present. Do not substitute other queries:\n" +
                           "\n".join(missing))
    if name in ("text2image","bigann") and missing and allow_public_download:
        print("Explicit public-dataset download approved; the original data "
              "preparer will verify official source and input format.")
    output = {
        "coveo": "artifacts/final-core-coveo",
        "pubmed": "artifacts/final-core-pubmed",
        "text2image": "artifacts/final-core-public-10m/text2image-10M",
        "bigann": "artifacts/final-core-public-10m/bigann-10M",
        "catapult": "artifacts/navhints-optimal-core-vs-catapult",
        "memory": "artifacts/navhints-fixed-memory-tradeoff",
        "sensitivity": "artifacts/final-core-sensitivity",
        "persistence": "artifacts/final-core-write-sweep",
        "starling-coveo": "artifacts/starling-final-core-coveo",
        "starling-bigann": "artifacts/starling-final-core-bigann"
    }[name]
    destination = ROOT / output
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise RuntimeError(f"Refusing to overwrite existing experiment evidence: {destination}\n"
                           "Archive/remove old output or pass --overwrite explicitly.")
    env = dict(os.environ)
    env.setdefault("RUNNER_TEMP", tempfile.gettempdir())
    env.setdefault("GITHUB_RUN_ID", "manual-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + str(os.getpid()))
    env.setdefault("GITHUB_RUN_ATTEMPT", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print("Executing on this host; benchmark scripts serialize measured SSD work.")
    return subprocess.call(cmd, cwd=ROOT, env=env)


def zip_contains(z: zipfile.ZipFile, member: str) -> bool:
    return sum(name == member or name.endswith("/" + member) for name in z.namelist()) == 1


def check_artifact(path: Path, record: dict) -> None:
    with zipfile.ZipFile(path) as z:
        members = (record["data_member"], *record.get("additional_members", []))
        for name in members:
            options = [q for q in z.namelist() if q == name or q.endswith("/" + name)]
            if len(options) != 1:
                raise ValueError(f"{record['id']}: expected unique {name}, got {options}")
            data = json.loads(z.read(options[0]))
            if not isinstance(data, dict):
                raise ValueError(f"{record['id']}: invalid JSON result object")
            c = data.get("controller", data.get("core", {}))
            if record["id"] in ("coveo","pubmed","text2image","bigann"):
                if c.get("recent_result_hints", c.get("cache_ids")) != 2048:
                    raise ValueError(f"{record['id']}: wrong destination capacity")
                if c.get("extra_ranked_starts", c.get("additional_starts")) != 8:
                    raise ValueError(f"{record['id']}: wrong additional starts")
                if not all(q in data["summary"] for q in ("baseline","ivf","core","full")):
                    raise ValueError(f"{record['id']}: missing complete four-arm baseline")
            if record["id"] == "persistence" and (c.get("B"),c.get("C"),c.get("k")) != (14464,2048,8):
                raise ValueError("Persistence artifact uses wrong final-Core allocation")
            if record["id"] == "sensitivity":
                # The three PubMed sensitivity outputs use the "core" key,
                # not "controller"; all three must carry B/C/k=14464/2048/8.
                if (c.get("junction_budget"), c.get("destination_capacity"),
                        c.get("starts")) != (14464, 2048, 8):
                    raise ValueError("Wrong final-Core sensitivity allocation")
            if record["id"].startswith("starling") and "final_core" not in data.get("modes", {}):
                raise ValueError("Starling result lacks final-Core mode")
            if record["id"] == "memory" and data.get("accounted_byte_budget") != 102940:
                raise ValueError("Incorrect fixed-payload ablation")
            if record["id"] == "catapult":
                nav = data.get("auxiliary_payload_navhints", {}).get("core_reallocated", {})
                observed = tuple(nav.get(k) for k in (
                    "landmarks", "recent_ids_capacity", "injected_starts", "payload_bytes"))
                if observed != (14464, 2048, 8, 102940):
                    raise ValueError("Catapult/Core allocation mismatch: " + str(observed))
            if record["id"] in SERVICE_ARCHIVES:
                if record["id"] == "service-pubmed":
                    control = data.get("controller", {})
                    budget = control.get("junctions")
                    capacity, starts = control.get("recent"), control.get("starts")
                    reps, seeds = data.get("repetitions"), data.get("catapult_seeds")
                    widths = data.get("widths", [])
                else:
                    expected_dataset = {
                        "service-coveo": "coveo",
                        "service-text2image": "text2image-10M",
                        "service-bigann": "bigann-10M",
                    }[record["id"]]
                    if data.get("dataset") != expected_dataset:
                        raise ValueError(record["id"] + ": wrong service workload")
                    control = data.get("core", {})
                    budget = control.get("actual_junctions")
                    capacity, starts = control.get("recent"), control.get("starts")
                    protocol = data.get("protocol", {})
                    reps, seeds = protocol.get("rotated_repetitions"), data.get("catapult", {}).get("seeds")
                    widths = protocol.get("widths", [])
                    if protocol.get("in_flight_queries") != 1:
                        raise ValueError(record["id"] + ": non-causal service replay")
                    if data.get("catapult", {}).get("accounted_payload_bytes", -1) < control.get("accounted_payload_bytes", 0):
                        raise ValueError(record["id"] + ": Catapult memory below NavHints")
                if (data.get("diagnostic") or not isinstance(budget, int)
                        or not 0 < budget <= 14464 or (capacity, starts) != (2048, 8)
                        or reps != 3 or sorted(seeds or []) != [0, 1, 2] or not widths):
                    raise ValueError(record["id"] + ": wrong final-Core service protocol")
                observations = data.get("measurements", {})
                for rep in range(3):
                    for arm in ("baseline", "navhints", "catapult-s0", "catapult-s1", "catapult-s2"):
                        if f"r{rep}-L{widths[0]}-{arm}" not in observations:
                            raise ValueError(record["id"] + ": missing service arm " + arm)


        # Check the producing source snapshot, not only the JSON payload.
        source_name = record.get("source_member",
            "geoivf-revision.txt" if record["id"].startswith("starling-")
            else "geoivf-commit.txt")
        source_members = [n for n in z.namelist()
                          if n == source_name or n.endswith("/" + source_name)]
        if len(source_members) != 1:
            raise ValueError(record["id"] + ": missing source commit marker")
        observed_sha = z.read(source_members[0]).decode("ascii").strip()
        if observed_sha != record["source_sha"]:
            raise ValueError(record["id"] + ": wrong source SHA " + observed_sha)


def original_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(m: dict, target: str, dest: Path, original_zip: bool = False) -> int:
    cases = m["runs"] if target == "all" else [next(x for x in m["runs"] if x["id"] == target)]
    failed = []
    for case in cases:
        file = dest / (case["id"] + ".zip")
        try:
            check_artifact(file, case)
            if original_zip and original_sha256(file) != case["original_artifact_sha256"]:
                raise ValueError("original GitHub Actions ZIP SHA-256 mismatch")
            suffix = " original ZIP digest confirmed" if original_zip else " content verified"
            print(f"PASS {case['id']} run={case['run_id']} artifact={case['artifact_id']} ({file})" + suffix)
        except (OSError, ValueError, zipfile.BadZipFile, KeyError) as exc:
            print(f"FAIL {case['id']} ({file}): {exc}")
            failed.append(case["id"])
    return 1 if failed else 0


def fetch(m: dict, target: str, dest: Path, execute: bool, replace: bool) -> int:
    cases = m["runs"] if target == "all" else [next(x for x in m["runs"] if x["id"] == target)]
    if execute:
        dest.mkdir(parents=True, exist_ok=True)
    for case in cases:
        destzip = dest / (case["id"] + ".zip")
        command = ["gh", "run", "download", str(case["run_id"]),
                   "--repo", m["code_repository"], "--name", case["artifact_name"]]
        if not execute:
            print(f"Dry run: {' '.join(command)} -> {destzip}")
            continue
        if destzip.exists() and not replace:
            check_artifact(destzip, case)
            print(f"Already verified: {destzip}; refusing replacement without --replace")
            continue
        if not shutil.which("gh"):
            raise RuntimeError("Install and authenticate GitHub CLI (gh auth login)")
        with tempfile.TemporaryDirectory(prefix="navhints-gh-") as tmp:
            command.extend(["--dir", tmp])
            subprocess.run(command, check=True)
            sources = [p for p in Path(tmp).rglob("*") if p.is_file()]
            if not sources:
                raise RuntimeError(f"Empty GitHub artifact {case['artifact_id']}")
            with tempfile.NamedTemporaryFile(
                prefix=f".{case['id']}-", suffix=".zip", dir=dest, delete=False
            ) as staged_stream:
                staged = Path(staged_stream.name)
            try:
                with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                    for f in sorted(sources):
                        z.write(f, f.relative_to(tmp).as_posix())
                check_artifact(staged, case)
                staged.replace(destzip)
            finally:
                staged.unlink(missing_ok=True)
        print("Fetched and validated:", destzip)
    return 0


def paper(m: dict, paper_dir: Path, archives: Path, execute: bool) -> int:
    paper_dir = paper_dir.resolve()
    primary = [archives / (n + ".zip") for n in ("coveo", "pubmed", "text2image", "bigann")]
    sensitivity = [archives / (n + ".zip") for n in ("sensitivity", "persistence")]
    service = [archives / ("service-" + n + ".zip")
               for n in ("coveo", "pubmed", "text2image", "bigann")]
    cmd1 = ["python3", "scripts/derive_final_core_eval.py",
            "--coveo", str(primary[0]), "--pubmed", str(primary[1]),
            "--text2image", str(primary[2]), "--bigann", str(primary[3])]
    cmd2 = ["python3", "scripts/derive_final_core_sensitivity.py",
            "--warmup", str(sensitivity[0]), "--locality", str(sensitivity[0]),
            "--shift", str(sensitivity[0]), "--writes", str(sensitivity[1])]
    cmd3 = ["python3", "scripts/derive_service_latency.py",
            "--coveo", str(service[0]), "--pubmed", str(service[1]),
            "--text2image", str(service[2]), "--bigann", str(service[3]),
            "--out-data", "figures/data",
            "--out-table", "tables/eval_service_latency_table.tex",
            "--out-summary", str(ROOT / "artifacts" / "service-reproduction-summary.json")]
    for command in (cmd1, cmd2, cmd3, ["make", "verify-latex"]):
        print("Paper:", " ".join(command))
    if not execute:
        print("Dry run only. Run with --execute after reviewing the source artifacts.")
        return 0
    required = {"coveo", "pubmed", "text2image", "bigann",
                "sensitivity", "persistence", *SERVICE_ARCHIVES}
    for case in m["runs"]:
        if case["id"] in required:
            check_artifact(archives / (case["id"] + ".zip"), case)
    if not (paper_dir / "main.tex").is_file():
        raise FileNotFoundError(f"Not a paper checkout: {paper_dir}")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    for command in (cmd1, cmd2, cmd3, ["make", "verify-latex"]):
        subprocess.run(command, cwd=paper_dir, check=True)
    print("Paper data regenerated and clean build attempted; inspect the new results "
          "and .latex-verify/result.json before publication.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action",required=True)
    sub.add_parser("list")
    dr = sub.add_parser("doctor")
    dr.add_argument("--mode",choices=("all",*REPRO_RUNS),default="all")
    rn = sub.add_parser("run")
    rn.add_argument("mode",choices=REPRO_RUNS)
    rn.add_argument("--execute",action="store_true")
    rn.add_argument("--allow-public-download",action="store_true")
    rn.add_argument("--overwrite",action="store_true")
    ft = sub.add_parser("fetch")
    ft.add_argument("target", choices=("all", *ARCHIVE_IDS), nargs="?", default="all")
    ft.add_argument("--execute",action="store_true")
    ft.add_argument("--replace",action="store_true")
    ft.add_argument("--dir",type=Path,default=ROOT/"artifacts"/"frozen")
    vf = sub.add_parser("verify")
    vf.add_argument("target", choices=("all", *ARCHIVE_IDS))
    vf.add_argument("--original-zip", action="store_true",
                    help="Require exact archived GitHub Actions ZIP digest, not a gh-repacked ZIP")
    vf.add_argument("--dir",type=Path,default=ROOT/"artifacts"/"frozen")
    rv = sub.add_parser("review", help="Verify public manifests or a sanitized reviewer bundle offline")
    rv.add_argument("--bundle", type=Path, help="Locally obtained sanitized reviewer ZIP")
    pdf = sub.add_parser("paper")
    pdf.add_argument("--paper-dir",type=Path,default=ROOT.parent/"DemmandAwareANN")
    pdf.add_argument("--archives",type=Path,default=ROOT/"artifacts"/"frozen")
    pdf.add_argument("--execute",action="store_true")
    a = p.parse_args()
    m = manifest()
    if a.action=="list":return run_list(m)
    if a.action=="doctor":return doctor(m,a.mode)
    if a.action=="run":return run_experiment(m,a.mode,a.execute,a.allow_public_download,a.overwrite)
    if a.action=="fetch":return fetch(m,a.target,a.dir.resolve(),a.execute,a.replace)
    if a.action=="verify":return verify(m,a.target,a.dir.resolve(),a.original_zip)
    if a.action=="review":
        cmd=[sys.executable,str(ROOT/"repro"/"verify_review_bundle.py")]
        if a.bundle is not None:cmd.extend(["--bundle",str(a.bundle.resolve())])
        return subprocess.call(cmd,cwd=ROOT)
    if a.action=="paper":return paper(m,a.paper_dir,a.archives.resolve(),a.execute)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    try:raise SystemExit(main())
    except (RuntimeError,ValueError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        print(f"NAVHINTS REPRO ERROR: {exc}",file=sys.stderr)
        raise SystemExit(2)
