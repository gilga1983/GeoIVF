#!/usr/bin/env python3
"""Encode query embeddings with DiskANN's existing 64-byte PQ codebook."""
from __future__ import annotations
import argparse,struct
from pathlib import Path
import numpy as np
from analyze_pq_semantic_filter import fbin_memmap, load_pq
from compare_semantic_cache_query_keys import encode_queries_pq

MAGIC=b"QCPQ0001"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--queries",type=Path,required=True)
    ap.add_argument("--pq-pivots",type=Path,required=True)
    ap.add_argument("--pq-codes",type=Path,required=True)
    ap.add_argument("--start",type=int,default=9000)
    ap.add_argument("--rows",type=int,default=1000)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    q=fbin_memmap(args.queries.resolve())
    piv,offs,_=load_pq(args.pq_pivots.resolve(),args.pq_codes.resolve())
    x=np.asarray(q[args.start:args.start+args.rows],dtype=np.float32)
    if len(x)!=args.rows: raise ValueError("query range")
    codes=encode_queries_pq(x,piv,offs)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("wb") as f:
        f.write(MAGIC);f.write(struct.pack("<II",codes.shape[0],codes.shape[1]));codes.tofile(f)
    print(f"encoded rows={codes.shape[0]} chunks={codes.shape[1]} bytes={args.out.stat().st_size}")
if __name__=="__main__":main()
