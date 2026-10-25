#!/usr/bin/env python3
"""Generate first-time PubMed or Text2Image native Starling final-Core benchmark.

Inherits the *validated* original-author Starling pipeline and exact source
patch sequence of generate_starling_final_core.py bigann. The only changed
inputs are the native float32/MIPS dataset, index identity, split, and L grid.
Coveo/BigANN published native experiments are never rerun.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import subprocess

REV = "17dc3e8a011533a62374445f53963e951b72883a"
LS = "20 40 80 160 320 640"


def once(s, old, new, name):
    got = s.count(old)
    if got != 1:
        raise RuntimeError(f"{name}: expected one exact source anchor, found {got}")
    return s.replace(old, new, 1)


def replace_n(s, old, new, n, name):
    if s.count(old) != n:
        raise RuntimeError(f"{name}: expected {n} anchors, found {s.count(old)}")
    return s.replace(old, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=("pubmed", "text2image"))
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    subprocess.run(
        ["python3", "scripts/generate_starling_final_core.py", "bigann"], check=True
    )
    base = Path("scripts/run_starling_final_core_bigann.generated.sh")
    s = base.read_text()
    label = a.dataset
    if label == "pubmed":
        source = "$HOME/.cache/geoivf/medrag-zipf-qwen3b-v1/medrag-zipf-10k-medcpt.fbin"
        rows, dim = 1_000_000, 768
        prefix = "pubmed_native_medcpt"
    else:
        source = (
            "$HOME/.cache/geoivf/public-eval/text2image-10M/data/dataset.manifest.json"
        )
        rows, dim = 10_000_000, 200
        prefix = "text2image_native_10m"
    s = once(
        s,
        'ART="$PWD/artifacts/starling-final-core-bigann"',
        f'ART="$PWD/artifacts/starling-final-core-{label}"',
        "distinct artifacts",
    )
    s = once(
        s,
        'WORK="$RUNNER_TEMP/starling-final-core-bigann-$GITHUB_RUN_ID"',
        f'WORK="$RUNNER_TEMP/starling-final-core-{label}-$GITHUB_RUN_ID"',
        "distinct temporary build",
    )
    s = once(
        s,
        'SRC="$HOME/.cache/geoivf/public-eval/bigann-10M/data/dataset.manifest.json"',
        f'SRC="{source}"',
        "frozen dataset source",
    )
    start = 'python3 - "$SRC" "$DATA" <<\'PY\'\n'
    end = '\nPY\ncp "$DATA/bigann10m.manifest.json" "$ART/"'
    lo = s.index(start)
    hi = s.index(end, lo) + len(end)
    preparation = r'''python3 - "$SRC" "$DATA" "__LABEL__" <<'PY_PREPARE_FLOAT_MIPS'
from pathlib import Path
import hashlib,json,struct,sys
src=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
name=sys.argv[3]
root=Path.home()/".cache/geoivf"
if name=="pubmed":
    base=root/"catapult-paper-pubmed1m/pubmed-medcpt-first1m.fbin"
    allq=src
    gt=root/"medrag-zipf-qwen3b-v1/heldout5000-pubmed1m-top16-ip.gt"
    n,d=1_000_000,768
    qrows=10_000
else:
    manifest=json.loads(src.read_text())
    assert manifest["dataset"]=="text2image-10M"
    assert (manifest["rows"],manifest["dim"],manifest["data_type"],manifest["metric"])==(10_000_000,200,"float32","inner_product")
    files=manifest["files"]
    base=Path(files["base"]).resolve()
    allq=Path(files["queries"]).resolve()
    gt=Path(files["heldout5000_gt"]).resolve()
    n,d=10_000_000,200
    qrows=100_000
def header(f):
    with f.open("rb") as stream:raw=stream.read(8)
    if len(raw)!=8:raise RuntimeError("invalid vector/GT header "+str(f))
    return struct.unpack("<II",raw)
def check(f,r,k,datatype_bytes=4):
    if not f.is_file() or header(f)!=(r,k) or f.stat().st_size!=8+r*k*datatype_bytes:
        raise RuntimeError(f"unexpected prepared float32 source {f}: expected {(r,k)}")
check(base,n,d)
check(allq,qrows,d)
gr,gk=header(gt)
if (gr,gk)[0]<5000 or gk<10:raise RuntimeError("ground truth too short")
expected_gt_bytes=8+gr*gk*4
if gt.stat().st_size not in (expected_gt_bytes,expected_gt_bytes+gr*gk*4):
    raise RuntimeError("ground truth has unsupported layout")
def link(target,source):
    if target.exists() or target.is_symlink():target.unlink()
    target.symlink_to(source)
    if not target.is_file():raise RuntimeError("missing linked native input "+str(target))
def slice_vectors(source,target,start,count):
    with source.open("rb") as fi,target.open("wb") as fo:
        fo.write(struct.pack("<II",count,d))
        fi.seek(8+start*d*4)
        remaining=count*d*4
        while remaining:
            block=fi.read(min(16<<20,remaining))
            if not block:raise RuntimeError("truncated query input")
            fo.write(block);remaining-=len(block)
def slice_gt(source,target,start,count):
    r,k=header(source)
    with source.open("rb") as fi,target.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        fi.seek(8+start*k*4)
        ids=fi.read(count*k*4)
        if len(ids)!=count*k*4:raise RuntimeError("truncated GT IDs")
        fo.write(ids)
        if source.stat().st_size==8+r*k*8:
            fi.seek(8+r*k*4+start*k*4)
            distances=fi.read(count*k*4)
            if len(distances)!=count*k*4:raise RuntimeError("truncated GT distances")
            fo.write(distances)
        else:
            # Native Starling expects distance payload; IDs remain unmodified.
            fo.write(bytes(count*k*4))
link(out/"base.fbin",base)
if name=="pubmed":
    slice_vectors(allq,out/"train5000.fbin",0,5000)
    slice_vectors(allq,out/"heldout5000.fbin",5000,5000)
else:
    link(out/"train5000.fbin",Path(files["train5000"]).resolve())
    link(out/"heldout5000.fbin",Path(files["heldout5000"]).resolve())
slice_gt(gt,out/"heldout5000.gt",0,5000)
slice_vectors(out/"heldout5000.fbin",out/"warm4000.fbin",0,4000)
slice_vectors(out/"heldout5000.fbin",out/"eval1000.fbin",4000,1000)
slice_gt(out/"heldout5000.gt",out/"warm4000.gt",0,4000)
slice_gt(out/"heldout5000.gt",out/"eval1000.gt",4000,1000)
for file,count in (("train5000.fbin",5000),("heldout5000.fbin",5000),
                   ("warm4000.fbin",4000),("eval1000.fbin",1000)):
    check(out/file,count,d)
record={"dataset":name,"base_rows":n,"dim":d,"dtype":"float32","distance":"mips",
        "frozen_source":str(src),"base_file":str(base),"groundtruth_file":str(gt),
        "query_split":"5K training, heldout 4K causal warmup and 1K measurement",
        "original_gt_ids_preserved":True,
        "original_starling_source":"__REV__"}
(out/"native-float-mips.manifest.json").write_text(json.dumps(record,indent=2)+"\n")
print("STARLING_NATIVE_FLOAT_MIPS_INPUT_VERIFIED",name,n,d,flush=True)
PY_PREPARE_FLOAT_MIPS
cp "$DATA/native-float-mips.manifest.json" "$ART/"'''
    preparation = preparation.replace("__LABEL__", label).replace("__REV__", REV)
    s = s[:lo] + preparation + s[hi:]
    # Starling original-native input split is prepared and checked above,
    # not regenerated from the source manifest for BigANN.
    first = 'python3 - "$SRC" "$DATA" <<\'PY_NATIVE_SPLIT\'\n'
    last = "\nPY_NATIVE_SPLIT\n"
    low = s.index(first)
    high = s.index(last, low) + len(last)
    s = (
        s[:low]
        + 'echo "STARLING_NATIVE_DISJOINT_FLOAT_QUERY_SPLIT_VERIFIED"\n'
        + s[high:]
    )
    # Keep native author dataset type and metric faithful; no L2 substitution.
    s = s.replace(".u8bin", ".fbin")
    s = replace_n(
        s, "DATA_TYPE=uint8", "DATA_TYPE=float", 1, "author native float type"
    )
    s = replace_n(s, "DIST_FN=l2", "DIST_FN=mips", 1, "native original MIPS")
    s = replace_n(s, "DATA_DIM=128", f"DATA_DIM={dim}", 1, "native dimensionality")
    s = replace_n(s, "DATA_N=10000000", f"DATA_N={rows}", 1, "full native base size")
    s = once(
        s,
        'PREFIX="bigann_native_10m"',
        f'PREFIX="{prefix}"',
        "native separate index namespace",
    )
    s = s.replace("bigann10m.manifest.json", "native-float-mips.manifest.json")
    # The training-only ID directory and selector use original index PQ.
    count = s.count("--base-data-type uint8")
    if count != 2:
        raise RuntimeError(f"expected both native IVF builders, found {count}")
    s = s.replace("--base-data-type uint8", "--base-data-type float32")
    count = s.count("--partition l2")
    if count < 1:
        raise RuntimeError("native IVF partition selector missing")
    s = s.replace("--partition l2", "--partition spherical")
    s = replace_n(s, 'LS="20 40 80 160"', f'LS="{LS}"', 2, "expand native recall sweep")
    s = once(
        s,
        '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 4',
        '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 6',
        "validate six native widths",
    )
    s = once(
        s,
        "scripts/summarize_starling_final_core.py",
        "scripts/summarize_starling_final_core_missing.py",
        "dataset aware native final-Core report",
    )
    s = once(
        s,
        "STARLING_FINAL_BIGANN_SUCCESS",
        "STARLING_FINAL_" + label.upper() + "_SUCCESS",
        "distinct native completion marker",
    )
    dest = Path(f"scripts/run_starling_final_core_{label}.generated.sh")
    dest.write_text(s)
    dest.chmod(0o755)
    subprocess.run(["bash", "-n", str(dest)], check=True)
    if "DIST_FN=mips" not in s or "DATA_TYPE=float" not in s:
        raise RuntimeError("native float/MIPS contract violated")
    print(
        "STARLING_MISSING_NATIVE_GENERATED",
        label,
        "train=5000 warm=4000 eval=1000",
        "mips=1 original_index=1 original_navigator=1",
        flush=True,
    )
    if a.run:
        os.execvp("bash", ["bash", str(dest)])


if __name__ == "__main__":
    main()
