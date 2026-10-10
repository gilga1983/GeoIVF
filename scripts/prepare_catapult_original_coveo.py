#!/usr/bin/env python3
"""Make native CatapultDB input by ZERO padding the frozen real Coveo vectors.

No new data or labels, no changed neighbor order. Output goes to temporary
self-hosted scratch, never committed or included in GitHub artifacts.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

def shape(p):
    with p.open("rb") as f: hdr=f.read(8)
    if len(hdr)!=8: raise ValueError("Truncated Coveo array")
    return struct.unpack("<II",hdr)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("frozen_manifest",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("artifact",type=Path)
    args=ap.parse_args()
    m=json.loads(args.frozen_manifest.read_text())
    assert m["dataset"]=="coveo-sigir-ecom-2021-real-demand"
    assert m["rows"]==31950 and m["dim"]==50 and m["data_type"]=="float32"
    assert m["split"]["static_train"]==[0,5000]
    assert m["split"]["online_replay"]==[5000,30000]
    assert m["split"]["online_warmup_rows"]==20000
    assert m["split"]["measured_rows"]==5000
    base=Path(m["files"]["base"]).resolve()
    replay=Path(m["files"]["replay"]).resolve()
    gt=(args.frozen_manifest.parent/"replay.gt").resolve()
    assert shape(base)==(31950,50) and shape(replay)==(25000,50)
    assert shape(gt)[0]==25000 and shape(gt)[1]>=10

    args.output.mkdir(parents=True,exist_ok=True)
    B=np.memmap(base,dtype="<f4",mode="r",offset=8,shape=(31950,50))
    Q=np.memmap(replay,dtype="<f4",mode="r",offset=8,shape=(25000,50))
    assert np.all(np.isfinite(B)) and np.all(np.isfinite(Q))
    assert np.max(np.abs(np.linalg.norm(B,axis=1)-1.))<2e-5
    assert np.max(np.abs(np.linalg.norm(Q,axis=1)-1.))<2e-5

    p=args.output/"coveo-64d-base.fbin"
    with p.open("wb") as f:
        f.write(struct.pack("<II",31950,64))
        for lo in range(0,31950,2048):
            hi=min(lo+2048,31950)
            chunk=np.zeros((hi-lo,64),dtype="<f4")
            chunk[:,:50]=B[lo:hi]
            chunk.tofile(f)

    qpath=args.output/"coveo-64d-replay.npy"
    arr=np.lib.format.open_memmap(qpath,mode="w+",dtype="<f4",shape=(25000,64))
    arr[:,:50]=Q
    arr[:,50:]=0.
    del arr
    (args.output/"replay.gt").symlink_to(gt)
    summary={
      "dataset":"Coveo SIGIR eCom 2021 production search",
      "base_count":31950,"original_dim":50,"zero_padded_dim":64,
      "normalization":"existing L2 unit normalized; no renormalization",
      "query_source_order":"production chronological events [5000,30000]",
      "warmup":20000,"measured":5000,
      "source_manifest":str(args.frozen_manifest.resolve()),
      "source_gt":str(gt),
      "original_base_sha256":hashlib.sha256(base.read_bytes()).hexdigest(),
      "original_replay_sha256":hashlib.sha256(replay.read_bytes()).hexdigest(),
      "metric":"native squared L2, same ranking as cosine on unit vectors",
      "graph":"author-linked C++ DiskANN in-memory Vamana graph",
      "limitations":"CatapultDB original implementation is in-RAM; no SSD reads measured"
    }
    args.artifact.mkdir(parents=True,exist_ok=True)
    (args.artifact/"author-coveo-input-manifest.json").write_text(
        json.dumps(summary,indent=2)+"\n")
    print("CATAPULT_AUTHOR_COVEO_INPUT_VERIFIED n=31950 d64 warm20K test5K",flush=True)

if __name__=="__main__":main()
