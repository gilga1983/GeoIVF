#!/usr/bin/env python3
"""Offline macro-region mapping of the *same* 512 NavHints PQ-routing cells.

This does NOT modify the original Hint-IVF, its medoids/children, or the
search entry selection. It clusters their full-precision medoid vectors
(available only offline) into G=16/32/64/128/256 stable macro regions and
stores 512 little-endian u16 labels. Runtime queries already PQ-score the
512 medoids. Their selected coarse cells are mapped through 1,040B table.

The map clusters coarse medoids, not test-query vectors. There is no access
to held-out queries or measured results during training.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import numpy as np

from build_hint_ivf import (
    load_fbin_memmap,
    normalize_rows,
    lloyd_spherical,
)

MAGIC=b"GHMAC001"
COUNTS=(16,32,64,128,256)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,required=True)
    ap.add_argument("--ivf",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=20261011)
    a=ap.parse_args()
    a.base=a.base.resolve()
    a.ivf=a.ivf.resolve()
    a.out_dir=a.out_dir.resolve()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    raw=a.ivf.read_bytes()
    if raw[:8]!=b"GHIVF001" or len(raw)<24:
        raise ValueError("unexpected original Hint-IVF file format")
    nlist,children,total,reserved=struct.unpack("<IIII",raw[8:24])
    if nlist!=512 or total!=16000 or reserved!=0:
        raise ValueError("not original 16K/512 Hint-IVF directory")
    if len(raw)!=24+4*nlist+4*(nlist+1)+4*children:
        raise ValueError("truncated static Hint-IVF")
    ids=np.frombuffer(raw,dtype="<u4",count=nlist,offset=24).copy()
    base,n,dim=load_fbin_memmap(a.base)
    if ids.max(initial=0)>=n:
        raise ValueError("coarse medoid beyond base corpus")
    x=normalize_rows(np.asarray(base[ids.astype(np.int64)],dtype=np.float32))
    if x.shape!=(nlist,dim):
        raise ValueError("bad coarse medoid embeddings")
    outputs={}
    for groups in COUNTS:
        _, labels=lloyd_spherical(x,groups,6,512,a.seed+groups)
        labels=labels.astype("<u2")
        if labels.min()<0 or labels.max()>=groups:
            raise ValueError("invalid macro cell labels")
        dest=a.out_dir/f"macro-n{groups}.bin"
        payload=MAGIC+struct.pack("<II",nlist,groups)+labels.tobytes()
        if len(payload)!=16+2*nlist:
            raise ValueError("macro group payload should be 1040 bytes")
        dest.write_bytes(payload)
        counts=np.bincount(labels.astype(np.int32),minlength=groups)
        outputs[str(groups)]={
            "file":dest.name,
            "macro_regions_requested":groups,
            "occupied_macro_regions":int(np.count_nonzero(counts)),
            "largest_macro_cells":int(counts.max()),
            "median_macro_cells":float(np.median(counts)),
            "smallest_macro_cells":int(counts.min()),
            "map_bytes":len(payload),
            "sha256":hashlib.sha256(payload).hexdigest()
        }
    manifest={
        "source":"unchanged frozen 16,000-Hint-IVF graph router",
        "original_ivf_sha256":hashlib.sha256(raw).hexdigest(),
        "medoid_count":nlist,
        "base_dimensions":dim,
        "train_source":"512 coarse medoid vectors from frozen train-only NavHints directory",
        "seed_base":a.seed,
        "algorithm":"6-iteration deterministic spherical k-means of existing medoid vectors",
        "macro_maps":outputs,
        "test_query_leakage":False,
    }
    (a.out_dir/"macro.manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print("NAVHINTS_MACRO_MAPS_BUILT",json.dumps(outputs,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
