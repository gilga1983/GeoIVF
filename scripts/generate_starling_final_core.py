#!/usr/bin/env python3
"""Generate selected Core native-Starling replay from validated generator shells.

Source patches are applied to a temporary original pinned Starling checkout,
preserving original native index, page layout, memory navigator and search.
"""
from pathlib import Path
import argparse
import os
import subprocess


def one(s, old, new, label):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 anchor, got {n}")
    return s.replace(old, new, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=("bigann", "coveo"))
    p.add_argument("--run", action="store_true")
    a = p.parse_args()
    if a.dataset == "bigann":
        generator = "scripts/generate_starling_fast_bigann.py"
        old_art = "starling-fast-bigann10m"
        patch = "scripts/patch_starling_fast_bigann.py"
        summary = "scripts/summarize_starling_fast_bigann.py"
        done = "STARLING_FAST_BIGANN_SUCCESS"
        src_base = "$DATA/base.u8bin"
        dtype = "uint8"
        partition = "l2"
    else:
        generator = "scripts/generate_starling_coveo_fast_core.py"
        old_art = "starling-coveo-fast-core"
        patch = "scripts/patch_starling_fast_core.py"
        summary = "scripts/summarize_starling_fast_core.py"
        done = "STARLING_COVEO_FAST_CORE_SUCCESS"
        src_base = "$DATA/base.fbin"
        dtype = "float32"
        partition = "spherical"
    subprocess.run(["python3", generator], check=True)
    script = Path("scripts/run_" + old_art.replace("-", "_") + ".generated.sh")
    # Original generator filenames do not follow hyphen replacement.
    script = Path(
        "scripts/run_starling_fast_bigann.generated.sh"
        if a.dataset == "bigann"
        else "scripts/run_starling_coveo_fast_core.generated.sh"
    )
    s = script.read_text()
    final_art = "starling-final-core-" + a.dataset
    s = one(
        s,
        f'ART="$PWD/artifacts/{old_art}"',
        f'ART="$PWD/artifacts/{final_art}"',
        "archive",
    )
    s = one(
        s,
        f'WORK="$RUNNER_TEMP/{old_art}-$GITHUB_RUN_ID"',
        f'WORK="$RUNNER_TEMP/{final_art}-$GITHUB_RUN_ID"',
        "scratch",
    )
    needle = f'python3 {patch} "$root"'
    s = one(
        s,
        needle,
        needle + '\n        python3 scripts/patch_starling_final_core.py "$root"',
        "patch native source with B14464 / C2048 / k8",
    )
    for old, new in (
        (
            "0) order=(baseline recent512 fast_recent512 core16k_recent512 fast_core) ;;",
            "0) order=(baseline fast_core final_core) ;;",
        ),
        (
            "1) order=(fast_core core16k_recent512 baseline fast_recent512 recent512) ;;",
            "1) order=(final_core baseline fast_core) ;;",
        ),
        (
            "2) order=(core16k_recent512 fast_recent512 fast_core recent512 baseline) ;;",
            "2) order=(fast_core final_core baseline) ;;",
        ),
    ):
        s = one(s, old, new, "balanced native benchmark arm order")

    # Native Starling's original training-only top16K export is retained for
    # compact-control arms. Final Core trains a separate 14464/512 directory
    # from the FIRST 14464 IDs of the SAME original native training output.
    add = r"""
STARLING_FINAL_IDS="$HINT_DIR/starling-top14464.bin"
STARLING_FINAL_IVF="$HINT_DIR/starling-hints14464-512regions.bin"
python3 - "$STARLING_IDS" "$STARLING_FINAL_IDS" <<'PY_FINAL_IDS'
from pathlib import Path
import struct,sys
src,dst=map(Path,sys.argv[1:])
raw=src.read_bytes()
assert raw[:8]==b"GIDST001" and struct.unpack_from("<II",raw,8)==(16000,0)
assert len(raw)==16+16000*4
count=14464
dst.write_bytes(b"GIDST001"+struct.pack("<II",count,0)+raw[16:16+4*count])
assert dst.stat().st_size==16+4*count
print("STARLING_FINAL_TRAIN_ONLY_TOP14464_VERIFIED",flush=True)
PY_FINAL_IDS
"$WORK/venv/bin/python" scripts/build_hint_ivf_generic.py \
  --base "__BASE__" --base-data-type __DTYPE__ \
  --hints "$STARLING_FINAL_IDS" --out "$STARLING_FINAL_IVF" \
  --partition __PART__ --nlist 512 --iterations 5 --batch 1024 \
  --seed 20261005 > "$ART/starling-final-ivf-build.log" 2>&1
cp "$STARLING_FINAL_IVF.manifest.json" "$ART/starling-final-ivf.manifest.json"
sha256sum "$STARLING_IDS" "$STARLING_IVF" "$STARLING_FINAL_IDS" "$STARLING_FINAL_IVF" \
   > "$ART/native-training-sha256.txt"
"""
    add = (
        add.replace("__BASE__", src_base)
        .replace("__DTYPE__", dtype)
        .replace("__PART__", partition)
    )
    anchor = (
        'cp "$STARLING_IVF.manifest.json" "$ART/starling-native-hint-ivf.manifest.json"'
    )
    s = one(s, anchor, anchor + "\n" + add, "build final trained ID-only directory")
    launch = '    (cd "$source_dir/scripts" && STARLING_NAVHINTS_EVAL="$mode" \\'
    s = one(
        s,
        launch,
        '    if [ "$mode" = final_core ]; then MODE_IVF="$STARLING_FINAL_IVF"; else MODE_IVF="$STARLING_IVF"; fi\n'
        + launch,
        "select correctly trained directory by method",
    )
    s = one(
        s,
        'STARLING_NAVHINTS_IVF_FILE="$STARLING_IVF"',
        'STARLING_NAVHINTS_IVF_FILE="$MODE_IVF"',
        "route with correct directory",
    )
    s = one(
        s,
        summary,
        "scripts/summarize_starling_final_core.py",
        "final-Core changed-route matched recall analysis",
    )
    s = one(
        s,
        done,
        "STARLING_FINAL_" + a.dataset.upper() + "_SUCCESS",
        "preserve successful complete native run status",
    )
    output = Path(f"scripts/run_starling_final_core_{a.dataset}.generated.sh")
    output.write_text(s)
    output.chmod(0o755)
    subprocess.run(["bash", "-n", str(output)], check=True)
    print(
        f"STARLING_FINAL_CORE_GENERATED dataset={a.dataset} "
        "train5000 modes=baseline,fast_core,final_core reps=3",
        flush=True,
    )
    if a.run:
        os.execvp("bash", ["bash", str(output)])


if __name__ == "__main__":
    main()
