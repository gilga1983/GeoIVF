#!/usr/bin/env python3
"""Generate native-Starling component ablation or small-budget replay.

Reuses the validated original-author Starling index, causal splits, native
metric, rotated repetitions and provenance from the frozen final-Core runs.
Only new source change is patch_starling_prune_budget.py, applied AFTER the
validated final-Core patch. No paper reference results are overwritten.

Example:
 python3 scripts/generate_starling_prune_budget.py coveo --stage components
 python3 scripts/generate_starling_prune_budget.py bigann --stage compact --run
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess


MODES = {
    "components": (
        "baseline", "final_core", "junction_only", "destination_only"
    ),
    "compact": (
        "baseline", "final_core", "junction_only", "destination_only",
        "junction_compact", "destination_compact", "joint_compact"
    ),
}


def once(src: str, old: str, new: str, label: str) -> str:
    count = src.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one exact anchor, got {count}")
    return src.replace(old, new, 1)


def small_directory_shell(src: str) -> str:
    # Reuse the source generator's proven dataset/metric-specific directory
    # builder, adjusting only the first 4096 training IDs and the destination.
    pat = (r'("\$WORK/venv/bin/python" scripts/build_hint_ivf_generic.py '
           r'\\' + '\n' +
           r'[\s\S]*?--seed 20261005 > '
           r'"\$ART/starling-final-ivf-build.log" 2>&1)')
    found = re.search(pat, src)
    if found is None:
        raise RuntimeError("Could not locate existing native final-Core IVF builder")
    original = found.group(1)
    builder = original
    builder = once(builder, '--hints "$STARLING_FINAL_IDS"',
                   '--hints "$STARLING_SMALL_IDS"', "small directory IDs")
    builder = once(builder, '--out "$STARLING_FINAL_IVF"',
                   '--out "$STARLING_SMALL_IVF"', "small directory output")
    builder = once(builder, '"$ART/starling-final-ivf-build.log"',
                   '"$ART/starling-small-ivf-build.log"', "small build log")
    return r'''
STARLING_SMALL_IDS="$HINT_DIR/starling-top4096.bin"
STARLING_SMALL_IVF="$HINT_DIR/starling-hints4096-512regions.bin"
python3 - "$STARLING_IDS" "$STARLING_SMALL_IDS" <<'PY_STARLING_SMALL_IDS'
from pathlib import Path
import struct,sys
src,dst=map(Path,sys.argv[1:])
raw=src.read_bytes()
assert raw[:8]==b"GIDST001"
assert struct.unpack_from("<II",raw,8)==(16000,0)
assert len(raw)==16+16000*4
count=4096
dst.write_bytes(b"GIDST001"+struct.pack("<II",count,0)+raw[16:16+4*count])
assert dst.stat().st_size==16+count*4
print("STARLING_COMPACT_TRAIN_ONLY_TOP4096_VERIFIED",flush=True)
PY_STARLING_SMALL_IDS
''' + builder + r'''
cp "$STARLING_SMALL_IVF.manifest.json" "$ART/starling-small-ivf.manifest.json"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("coveo","bigann","pubmed","text2image"))
    parser.add_argument("--stage", choices=tuple(MODES), default="components")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.dataset in ("coveo", "bigann"):
        generator = "scripts/generate_starling_final_core.py"
    else:
        generator = "scripts/generate_starling_final_core_missing.py"
    subprocess.run(["python3", generator, args.dataset], check=True)
    old = Path(f"scripts/run_starling_final_core_{args.dataset}.generated.sh")
    if not old.is_file():
        raise FileNotFoundError(old)
    shell = old.read_text()

    source_patch = 'python3 scripts/patch_starling_final_core.py "$root"'
    shell = once(
        shell, source_patch,
        source_patch + '\n        python3 scripts/patch_starling_prune_budget.py "$root"',
        "apply optional-source component ablations")

    modes = MODES[args.stage]
    orders = (
        list(modes),
        list(modes[1::2]) + list(modes[::2]),
        list(reversed(modes)),
    )
    for rep, order in enumerate(orders):
        old_order = (
            f"{rep}) order=(" +
            (
                "baseline fast_core final_core" if rep == 0
                else "final_core baseline fast_core" if rep == 1
                else "fast_core final_core baseline"
            ) + ") ;;"
        )
        new_order = f"{rep}) order=({' '.join(order)}) ;;"
        shell = once(shell, old_order, new_order, f"rotate repetition {rep}")

    old_art = f"starling-final-core-{args.dataset}"
    new_art = f"starling-prune-budget-{args.dataset}-{args.stage}"
    shell = once(shell, f'ART="$PWD/artifacts/{old_art}"',
                 f'ART="$PWD/artifacts/{new_art}"', "isolated artifact path")
    shell = once(shell, f'WORK="$RUNNER_TEMP/{old_art}-$GITHUB_RUN_ID"',
                 f'WORK="$RUNNER_TEMP/{new_art}-$GITHUB_RUN_ID"',
                 "isolated native build path")
    shell = once(shell,
        '    if [ "$mode" = final_core ]; then MODE_IVF="$STARLING_FINAL_IVF"; '
        'else MODE_IVF="$STARLING_IVF"; fi',
        '    if [ "$mode" = junction_compact ] || [ "$mode" = joint_compact ]; then '
        'MODE_IVF="$STARLING_SMALL_IVF"; '
        'elif [ "$mode" = final_core ] || [ "$mode" = junction_only ] || '
        '[ "$mode" = destination_only ] || [ "$mode" = destination_compact ]; then '
        'MODE_IVF="$STARLING_FINAL_IVF"; '
        'else MODE_IVF="$STARLING_IVF"; fi',
        "choose correctly trained directory for each variant")

    if args.stage == "compact":
        build = small_directory_shell(shell)
        anchor = 'cp "$STARLING_FINAL_IVF.manifest.json" "$ART/starling-final-ivf.manifest.json"'
        shell = once(shell, anchor, anchor + '\n' + build,
                     "build compact 4096-ID directory once per native index")

    old_summarizer = ("scripts/summarize_starling_final_core.py"
                      if args.dataset in ("bigann","coveo")
                      else "scripts/summarize_starling_final_core_missing.py")
    shell = once(shell, old_summarizer, "scripts/summarize_starling_prune_budget.py",
                 "strict component/size ablation report")
    summary_args = ('scripts/summarize_starling_prune_budget.py '
                    '"$ART/diagnostics" "$ART/diagnostic-summary.json" '
                    '"$ART/diagnostic-summary.md"')
    shell = once(shell, summary_args, summary_args+f' --stage {args.stage}',
                 "set exact tested mode set for summarizer")
    shell = once(shell,
                 f"STARLING_FINAL_{args.dataset.upper()}_SUCCESS",
                 f"STARLING_PRUNE_{args.dataset.upper()}_{args.stage.upper()}_SUCCESS",
                 "unique success marker")

    output = Path(
        f"scripts/run_starling_prune_budget_{args.dataset}_{args.stage}.generated.sh"
    )
    output.write_text(shell)
    output.chmod(0o755)
    subprocess.run(["bash", "-n", str(output)], check=True)
    for marker in (
        "patch_starling_prune_budget.py", "STARLING_NAVHINTS_EVAL",
        "starling-final-ivf.manifest.json", "summarize_starling_prune_budget.py"
    ):
        if marker not in shell:
            raise RuntimeError(f"native generated shell missing {marker}")
    print("STARLING_PRUNE_BUDGET_GENERATION_OK",
          f"dataset={args.dataset}", f"stage={args.stage}",
          "modes="+",".join(modes), "original_starling_native=1", flush=True)
    if args.run:
        os.execvp("bash", ["bash", str(output)])


if __name__ == "__main__":
    main()
