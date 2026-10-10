#!/usr/bin/env python3
"""Derive a native Starling Coveo replay from verified, pinned Starling CI.

The source data are the existing frozen/previously authorized 31,950-product
Coveo benchmark. This does not download or publish licensed Coveo records.
Supports only correct disjoint 5K static-train + 20K warm + 5K measured.
"""
from pathlib import Path
import os
import subprocess
import sys


def once(s,old,new,name):
    n=s.count(old)
    if n!=1:raise RuntimeError(f"{name}: expected 1 anchor, got {n}")
    return s.replace(old,new,1)


def nrep(s,old,new,count,name):
    n=s.count(old)
    if n!=count:raise RuntimeError(f"{name}: expected {count}, got {n}")
    return s.replace(old,new)


def main():
    subprocess.run(["python3","scripts/generate_starling_selective_diverse.py"],
                   check=True)
    s=Path("scripts/run_starling_selective_diverse.generated.sh").read_text()
    s=once(s,'ART="$PWD/artifacts/starling-selective-diverse"',
           'ART="$PWD/artifacts/starling-coveo-coverage"',
           "dedicated Coveo artifact folder")
    s=once(s,'WORK="$RUNNER_TEMP/starling-selective-diverse-$GITHUB_RUN_ID"',
           'WORK="$RUNNER_TEMP/starling-coveo-coverage-$GITHUB_RUN_ID"',
           "isolated Coveo temporary build")
    s=once(s,
       'SRC="$HOME/.cache/geoivf/public-eval/bigann-10M/data/dataset.manifest.json"',
       'SRC="$HOME/.cache/geoivf/coveo-real-demand-v1/split/dataset.manifest.json"',
       "frozen Coveo production event manifest")
    start='python3 - "$SRC" "$DATA" <<\'PY\'\n'
    end='\nPY\ncp "$DATA/bigann10m.manifest.json" "$ART/"'
    a=s.index(start);b=s.index(end,a)+len(end)
    pre=r'''python3 - "$SRC" "$DATA" <<'PY'
import json,struct,sys,hashlib
from pathlib import Path
src=Path(sys.argv[1]).resolve()
dst=Path(sys.argv[2]).resolve()
m=json.loads(src.read_text())
assert m["dataset"]=="coveo-sigir-ecom-2021-real-demand", m.get("dataset")
assert m["rows"]==31950 and m["dim"]==50 and m["data_type"]=="float32"
assert m["metric"]=="inner_product" and "L2-normalized" in m["normalization"]
split=m["split"]
assert split["static_train"]==[0,5000]
assert split["online_replay"]==[5000,30000]
assert split["online_warmup_rows"]==20000 and split["measured_rows"]==5000
src_files=m["files"]
base=Path(src_files["base"]).resolve()
train=Path(src_files["train"]).resolve()
replay=Path(src_files["replay"]).resolve()
gt=(src.parent/"replay.gt").resolve()
def shape(p):
    if not p.is_file():raise FileNotFoundError(f"Missing prepared licensed Coveo artifact {p}")
    with p.open("rb") as f:h=f.read(8)
    if len(h)!=8:raise RuntimeError(f"Invalid Coveo binary header: {p}")
    return struct.unpack("<II",h)
assert shape(base)==(31950,50),shape(base)
assert shape(train)==(5000,50),shape(train)
assert shape(replay)==(25000,50),shape(replay)
assert shape(gt)[0]==25000 and shape(gt)[1]>=10,shape(gt)
for source,target in [
    (base,dst/"base.fbin"),
    (train,dst/"train5000.fbin"),
    (replay,dst/"heldout25000.fbin"),
    (gt,dst/"heldout25000.gt"),
    (Path(src_files["eval"]).resolve(),dst/"eval5000.fbin"),
    (src.parent/"eval.gt",dst/"eval5000.gt"),
]:
    if not source.is_file():raise FileNotFoundError(source)
    target.symlink_to(source)
manifest={
 "dataset":"Coveo SIGIR eCom 2021 production search embeddings",
 "rows":31950,"dim":50,"data_type":"float32",
 "metric":"Starling native squared L2 (rank-equivalent to original cosine for normalized vectors)",
 "GT":"original exact cosine/IP product IDs; rank-equivalent to L2^2; never used as routing feedback",
 "source_split_manifest":str(src),
 "source_prep_manifest":m["prepared_manifest"],
 "split":{"static_train":[0,5000],"warmup":[5000,25000],"eval":[25000,30000]},
 "catalog_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
 "query_replay_sha256":hashlib.sha256(replay.read_bytes()).hexdigest(),
 "gt_replay_sha256":hashlib.sha256(gt.read_bytes()).hexdigest(),
 "chronological":True,
 "source_time_range":m.get("time",{})
}
(dst/"coveo-native.manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("STARLING_COVEO_FROZEN_SPLIT_VERIFIED n=31950 d=50 train=5000 warm=20000 eval=5000",flush=True)
PY
cp "$DATA/coveo-native.manifest.json" "$ART/"'''
    s=s[:a]+pre+s[b:]
    # Replace all remaining neutralized filenames, not the actual binary data.
    names={
      "base.u8bin":"base.fbin",
      "train5000.u8bin":"train5000.fbin",
      "heldout5000.u8bin":"heldout25000.fbin",
      "heldout5000.gt":"heldout25000.gt",
      "eval1000.u8bin":"eval5000.fbin",
      "eval1000.gt":"eval5000.gt",
      "warm4000.u8bin":"warm20000.fbin",
      "warm4000.gt":"warm20000.gt",
      "bigann10m.manifest.json":"coveo-native.manifest.json",
    }
    for old,new in names.items():
        s=s.replace(old,new)
    s=nrep(s,'DATA_TYPE=uint8','DATA_TYPE=float',2,"native float32 author cli")
    s=nrep(s,'DIST_FN=l2','DIST_FN=l2',2,"native squared L2 preserved")
    s=nrep(s,'DATA_DIM=128','DATA_DIM=50',2,"Coveo dimensions")
    s=nrep(s,'DATA_N=10000000','DATA_N=31950',2,"Coveo catalog size")
    s=nrep(s,'PREFIX="bigann_native_10m"','PREFIX="coveo_native_real"',1,
           "Coveo index path")
    s=once(s,'--base-data-type uint8','--base-data-type float32',
           "float32 hint-region partition input")
    # Original native binary patching, now also add the stronger coverage arm.
    s=once(s,'python3 scripts/patch_starling_selective_diverse.py "$root"',
           'python3 scripts/patch_starling_selective_diverse.py "$root"\n'
           '        python3 scripts/patch_starling_coveo_coverage.py "$root"',
           "float32 Coveo causal replay and complementary objective")
    s=nrep(s,'LS="20 40 80"','LS="12 20 40 80"',2,"Coveo high-recall frontier")
    s=nrep(s,'(20|40|80|160|320|640)','(12|20|40|80)',1,
           "native summary regex")
    s=once(s,
       '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 3',
       '"$ART/starling-rep$rep-$mode-summary.txt")" -ge 4',
       "four native recall search widths")
    # Any unused Gorgeous launcher is never invoked. This experiment uses
    # only the original Starling index; it does not pretend to compare systems.
    oldsplit='python3 - "$SRC" "$DATA" <<\'PY_NATIVE_SPLIT\'\n'
    oldend='\nPY_NATIVE_SPLIT\n'
    a=s.index(oldsplit);b=s.index(oldend,a)+len(oldend)
    s=s[:a]+'echo "STARLING_COVEO_PRESERVED_STATIC_TRAIN_5K_AND_CHRONO_25K"\n'+s[b:]
    # Native Starling candidate-export phases reuse the frozen 5K static prefix.
    # The author 5K online replay path is now 25K, with a 20K warmup.
    s=once(s,
       '0) order=(baseline core16k_recent512 diverse_core gate50_core gate50_diverse gate25_diverse) ;;',
       '0) order=(baseline recent512 core16k_recent512 diverse_core maxmin_core cover_core gate50_cover) ;;',
       "rep0 seven-arm order")
    s=once(s,
       '1) order=(gate50_diverse gate25_diverse baseline diverse_core gate50_core core16k_recent512) ;;',
       '1) order=(cover_core gate50_cover baseline maxmin_core recent512 diverse_core core16k_recent512) ;;',
       "rep1 rotated seven arms")
    s=once(s,
       '2) order=(gate50_core diverse_core core16k_recent512 gate25_diverse baseline gate50_diverse) ;;',
       '2) order=(recent512 maxmin_core core16k_recent512 gate50_cover baseline cover_core diverse_core) ;;',
       "rep2 rotated seven arms")
    s=once(s,'STARLING_SELECTIVE_DIVERSE_SUCCESS',
           'STARLING_COVEO_COVERAGE_SUCCESS',"unique experiment success marker")
    s=once(s,
       'python3 scripts/summarize_starling_selective_diverse.py',
       'python3 scripts/summarize_starling_coveo_coverage.py',
       "Coveo seven-arm 20K/5K evaluator")
    p=Path("scripts/run_starling_coveo_coverage.generated.sh")
    p.write_text(s)
    p.chmod(0o755)
    subprocess.run(["bash","-n",str(p)],check=True)
    print("STARLING_COVEO_RUN_SCRIPT_GENERATED n=31950 qtrain=5000 "
          "warm=20000 eval=5000 arms=7 widths=12,20,40,80 native-original-only",flush=True)
    if "--run" in sys.argv:
        os.execvp("bash",["bash",str(p)])

if __name__=="__main__":main()
