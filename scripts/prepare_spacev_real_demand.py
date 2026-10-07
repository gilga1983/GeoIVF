#!/usr/bin/env python3
"""Prepare a fully public Microsoft SPACEV-10M real-demand NavHints workload.

SPACEV is released by Microsoft from the Bing web-vector-search scenario. The
release contains a separate historical query log (94,162 descriptors) and
29,316 public evaluation queries. We use:
  * first 10M document vectors as the disk-resident corpus;
  * first 5K historical query-log vectors for static NavHints learning;
  * first 25K disjoint public queries as the causal online replay;
  * first 20K replay queries as online history and final 5K for measurement.

The release does not document the public-query ordering as chronological, so
the manifest calls this a replay order rather than a temporal trace.
"""
from __future__ import annotations
import argparse, hashlib, json, struct, time, urllib.request
from pathlib import Path

ROWS=10_000_000
DIM=100
PUBLIC_ROWS=29_316
HISTORY_ROWS=94_162
TRAIN_ROWS=5_000
WARMUP_ROWS=20_000
EVAL_ROWS=5_000

BASE_URL="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/spacev1b_base.i8bin"
QUERY_URL="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/query.i8bin"
GT_URL="https://comp21storage.z5.web.core.windows.net/comp21/spacev1b/msspacev-gt-10M"
HISTORY_URL="https://huggingface.co/datasets/jkhe/spacev1b/resolve/main/query_log.bin"


def sha256(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        while b:=f.read(8<<20): h.update(b)
    return h.hexdigest()


def download(url:str,dst:Path,max_bytes:int|None=None):
    if dst.is_file() and (max_bytes is None or dst.stat().st_size==max_bytes):
        return
    tmp=dst.with_suffix(dst.suffix+".partial")
    tmp.parent.mkdir(parents=True,exist_ok=True)
    tmp.unlink(missing_ok=True)
    total=0; t=time.time()
    print(f"download {url} -> {dst}",flush=True)
    with urllib.request.urlopen(url,timeout=180) as src,tmp.open("wb") as out:
        while True:
            need=(8<<20) if max_bytes is None else min(8<<20,max_bytes-total)
            if need<=0: break
            b=src.read(need)
            if not b: break
            out.write(b); total+=len(b)
            if total and total%(256<<20)<(8<<20):
                print(f"  {total/2**30:.2f} GiB",flush=True)
    if max_bytes is not None and total!=max_bytes:
        raise RuntimeError(f"cropped download got {total}, expected {max_bytes}")
    tmp.replace(dst)


def header(p:Path):
    with p.open("rb") as f: raw=f.read(8)
    if len(raw)!=8: raise ValueError(f"bad header {p}")
    return struct.unpack("<II",raw)


def validate_xbin(p:Path,rows:int,dim:int):
    r,d=header(p)
    if (r,d)!=(rows,dim): raise ValueError(f"{p}: {(r,d)} != {(rows,dim)}")
    exp=8+rows*dim
    if p.stat().st_size!=exp: raise ValueError(f"{p}: bad byte length")


def slice_xbin(src:Path,dst:Path,start:int,count:int):
    rows,dim=header(src)
    if start<0 or count<=0 or start+count>rows: raise ValueError("bad xbin slice")
    with src.open("rb") as fi,dst.open("wb") as fo:
        fi.seek(8+start*dim)
        fo.write(struct.pack("<II",count,dim))
        rem=count*dim
        while rem:
            b=fi.read(min(16<<20,rem))
            if not b: raise ValueError("truncated xbin")
            fo.write(b); rem-=len(b)


def validate_gt(p:Path,min_rows:int):
    with p.open("rb") as f: raw=f.read(8)
    if len(raw)!=8: raise ValueError("bad GT header")
    rows,k=struct.unpack("<II",raw)
    if rows<min_rows or k<10: raise ValueError(f"GT too small {(rows,k)}")
    if p.stat().st_size!=8+rows*k*8: raise ValueError("GT size mismatch")
    return rows,k


def slice_gt(src:Path,dst:Path,start:int,count:int):
    rows,k=validate_gt(src,start+count)
    rb=k*4; ids_bytes=rows*rb
    with src.open("rb") as fi,dst.open("wb") as fo:
        fo.write(struct.pack("<II",count,k))
        fi.seek(8+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated GT ids")
            fo.write(b); rem-=len(b)
        fi.seek(8+ids_bytes+start*rb); rem=count*rb
        while rem:
            b=fi.read(min(8<<20,rem))
            if not b: raise ValueError("truncated GT dists")
            fo.write(b); rem-=len(b)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)

    base=out/"base.10M.i8bin"
    query=out/"query.29316.i8bin"
    history=out/"query_log.94162.i8bin"
    gt=out/"groundtruth.29316.gt"

    wanted=8+ROWS*DIM
    if not base.is_file() or base.stat().st_size!=wanted or header(base)!=(ROWS,DIM):
        raw=out/"base.source-prefix.i8bin"
        download(BASE_URL,raw,max_bytes=wanted)
        src_rows,src_dim=header(raw)
        if src_dim!=DIM or src_rows<ROWS:
            raise ValueError(f"unexpected SPACEV source header {(src_rows,src_dim)}")
        raw.replace(base)
        with base.open("r+b") as f:f.write(struct.pack("<II",ROWS,DIM))
    validate_xbin(base,ROWS,DIM)

    download(QUERY_URL,query)
    validate_xbin(query,PUBLIC_ROWS,DIM)
    download(HISTORY_URL,history)
    validate_xbin(history,HISTORY_ROWS,DIM)
    download(GT_URL,gt)
    gt_rows,gt_k=validate_gt(gt,PUBLIC_ROWS)

    train=out/"train-history5000.i8bin"
    replay=out/"replay-public25000.i8bin"
    replay_gt=out/"replay-public25000.gt"
    evalq=out/"eval-public5000.i8bin"
    evalgt=out/"eval-public5000.gt"
    slice_xbin(history,train,0,TRAIN_ROWS)
    slice_xbin(query,replay,0,WARMUP_ROWS+EVAL_ROWS)
    slice_gt(gt,replay_gt,0,WARMUP_ROWS+EVAL_ROWS)
    slice_xbin(query,evalq,WARMUP_ROWS,EVAL_ROWS)
    slice_gt(gt,evalgt,WARMUP_ROWS,EVAL_ROWS)

    result={
      "dataset":"msspacev-10M-real-demand",
      "source":"Microsoft SPACEV1B / Bing web vector search",
      "rows":ROWS,"dim":DIM,"dtype":"int8","data_type":"int8",
      "metric":"squared_l2","query_rows":PUBLIC_ROWS,
      "split":{
        "static_train_source":"query_log.bin historical query descriptors",
        "static_train_rows":TRAIN_ROWS,
        "online_replay_source":"public SPACEV query descriptors",
        "online_warmup_rows":WARMUP_ROWS,
        "measured_rows":EVAL_ROWS,
        "public_query_order_semantics":"release order; not claimed chronological",
      },
      "official_gt_rows":gt_rows,"official_gt_k":gt_k,
      "files":{
        "base":str(base),"queries":str(query),"history_queries":str(history),
        "groundtruth":str(gt),"train":str(train),"replay":str(replay),
        "replay_gt":str(replay_gt),"heldout5000":str(evalq),
        "heldout5000_gt":str(evalgt),
      },
      "source_urls":{"base":BASE_URL,"queries":QUERY_URL,"history_queries":HISTORY_URL,"groundtruth":GT_URL},
      "sha256":{p.name:sha256(p) for p in (base,query,history,gt,train,replay,replay_gt,evalq,evalgt)},
    }
    (out/"dataset.manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
