#!/usr/bin/env python3
"""Freeze a chronological Coveo paper split from the converted real-demand stream."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def shape(path: Path):
    with path.open("rb") as f:
        raw=f.read(8)
    if len(raw)!=8:
        raise ValueError("bad fbin header")
    return struct.unpack("<II",raw)


def slice_fbin(src: Path,dst: Path,start:int,count:int):
    rows,dim=shape(src)
    if start<0 or count<=0 or start+count>rows:
        raise ValueError("invalid fbin slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim*4)
        fo.write(struct.pack("<II",count,dim))
        rem=count*dim*4
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated fbin")
            fo.write(b); rem-=len(b)


def slice_jsonl(src: Path,dst: Path,start:int,count:int):
    rows=src.read_text().splitlines()
    if start<0 or count<=0 or start+count>len(rows):
        raise ValueError("invalid metadata slice")
    dst.write_text("\n".join(rows[start:start+count])+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepared-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--start",type=int,default=0)
    ap.add_argument("--static-train",type=int,default=5000)
    ap.add_argument("--online-warmup",type=int,default=20000)
    ap.add_argument("--eval",type=int,default=5000)
    args=ap.parse_args()

    src=args.prepared_dir.resolve()
    out=args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    base=src/"coveo-products-cosine.fbin"
    queries=src/"coveo-search-chronological-cosine.fbin"
    meta=src/"coveo-search-chronological.jsonl"
    prep=json.loads((src/"coveo-ann.manifest.json").read_text())
    if not base.is_file() or not queries.is_file() or not meta.is_file():
        raise FileNotFoundError("prepared Coveo artifacts missing")

    total=args.static_train+args.online_warmup+args.eval
    qrows,dim=shape(queries)
    if args.start+total>qrows:
        raise ValueError(f"need {total} queries from start {args.start}, have {qrows}")

    trainq=out/"train.fbin"
    replayq=out/"replay.fbin"
    evalq=out/"eval.fbin"
    trainmeta=out/"train.jsonl"
    replaymeta=out/"replay.jsonl"
    evalmeta=out/"eval.jsonl"

    slice_fbin(queries,trainq,args.start,args.static_train)
    replay_start=args.start+args.static_train
    replay_count=args.online_warmup+args.eval
    slice_fbin(queries,replayq,replay_start,replay_count)
    slice_fbin(queries,evalq,replay_start+args.online_warmup,args.eval)

    slice_jsonl(meta,trainmeta,args.start,args.static_train)
    slice_jsonl(meta,replaymeta,replay_start,replay_count)
    slice_jsonl(meta,evalmeta,replay_start+args.online_warmup,args.eval)

    first=json.loads((out/"train.jsonl").read_text().splitlines()[0])
    last=json.loads((out/"eval.jsonl").read_text().splitlines()[-1])

    manifest={
      "dataset":"coveo-sigir-ecom-2021-real-demand",
      "source":"Coveo SIGIR eCom 2021 production search interactions",
      "rows":shape(base)[0],
      "dim":dim,
      "dtype":"float32",
      "data_type":"float32",
      "metric":"inner_product",
      "normalization":"L2-normalized; inner product equals cosine",
      "query_rows":qrows,
      "split":{
        "start":args.start,
        "static_train":[args.start,args.start+args.static_train],
        "online_replay":[replay_start,replay_start+replay_count],
        "online_warmup_rows":args.online_warmup,
        "measured_rows":args.eval,
      },
      "time":{
        "first_training_timestamp_ms":first["timestamp_ms"],
        "last_measured_timestamp_ms":last["timestamp_ms"],
      },
      "files":{
        "base":str(base),
        "all_queries":str(queries),
        "train":str(trainq),
        "replay":str(replayq),
        "eval":str(evalq),
        "train_metadata":str(trainmeta),
        "replay_metadata":str(replaymeta),
        "eval_metadata":str(evalmeta),
      },
      "prepared_manifest":prep,
    }
    (out/"dataset.manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
