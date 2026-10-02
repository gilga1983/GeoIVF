#!/usr/bin/env python3
"""Cache and validate dense ANN-Benchmarks datasets without silent substitution."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, subprocess, tempfile, time
from pathlib import Path
import h5py
import numpy as np

DATASETS = {
    "glove-200-angular": dict(url="https://ann-benchmarks.com/glove-200-angular.hdf5", dim=200, distance="angular"),
    "nytimes-256-angular": dict(url="https://ann-benchmarks.com/nytimes-256-angular.hdf5", dim=256, distance="angular"),
    "fashion-mnist-784-euclidean": dict(url="https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5", dim=784, distance="euclidean"),
    "gist-960-euclidean": dict(url="https://ann-benchmarks.com/gist-960-euclidean.hdf5", dim=960, distance="euclidean"),
}

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8<<20),b""): h.update(chunk)
    return h.hexdigest()

def validate(path: Path, spec: dict) -> dict:
    with h5py.File(path,"r") as f:
        for key in ("train","test","neighbors","distances"):
            if key not in f: raise ValueError(f"{path.name}: missing {key}")
        train,test,neighbors,distances=(f[k] for k in ("train","test","neighbors","distances"))
        if train.ndim!=2 or test.ndim!=2 or train.shape[1]!=spec["dim"] or test.shape[1]!=spec["dim"]:
            raise ValueError("dimension mismatch")
        if neighbors.shape!=distances.shape or neighbors.shape[0]!=test.shape[0] or neighbors.shape[1]<10:
            raise ValueError("ground-truth shape mismatch")
        distance=f.attrs.get("distance","")
        if isinstance(distance,bytes): distance=distance.decode()
        if distance!=spec["distance"]: raise ValueError(f"distance mismatch: {distance}")
        point_type=f.attrs.get("point_type","")
        if isinstance(point_type,bytes): point_type=point_type.decode()
        if point_type not in ("float","float32",""): raise ValueError(f"unsupported point type {point_type}")
        for arr,name in ((train,"train"),(test,"test")):
            for s in range(0,arr.shape[0],32768):
                block=np.asarray(arr[s:s+32768],dtype=np.float32)
                if not np.isfinite(block).all(): raise ValueError(f"nonfinite {name}")
        for s in range(0,neighbors.shape[0],32768):
            ids=np.asarray(neighbors[s:s+32768],dtype=np.int64)
            if ids.min()<0 or ids.max()>=train.shape[0]: raise ValueError("ground-truth ID out of range")
        return dict(
            train_rows=int(train.shape[0]), test_rows=int(test.shape[0]), dimension=int(train.shape[1]),
            neighbors=int(neighbors.shape[1]), distance=distance, point_type=point_type or "float",
            bytes=path.stat().st_size, sha256=digest(path),
        )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset",choices=sorted(DATASETS))
    ap.add_argument("--cache",type=Path,default=Path.home()/".cache/geoivf/datasets")
    ap.add_argument("--report",type=Path)
    a=ap.parse_args();a.cache.mkdir(parents=True,exist_ok=True)
    spec=DATASETS[a.dataset];target=a.cache/(a.dataset+".hdf5")
    with (a.cache/(a.dataset+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            with tempfile.NamedTemporaryFile(prefix=a.dataset+"-",suffix=".hdf5",dir=a.cache,delete=False) as tmp:
                temp=Path(tmp.name)
            try:
                subprocess.run(["curl","-fL","--connect-timeout","20","--retry","3","--retry-delay","2",
                    "--output",str(temp),spec["url"]],check=True)
                report=validate(temp,spec)
                os.replace(temp,target)
            finally:
                if temp.exists(): temp.unlink()
        else:
            report=validate(target,spec)
        out=dict(dataset=a.dataset,source=spec["url"],validated_unix=time.time(),files=report,
                 checksum_status="observed SHA256; ANN-Benchmarks does not publish a signed checksum here")
        print(json.dumps(out,indent=2),flush=True)
        if a.report:
            a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__": main()
