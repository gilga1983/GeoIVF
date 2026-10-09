#!/usr/bin/env python3
"""Bound an author's unmodified benchmark with comparable *external* resource measurements.

Read Linux /proc process-tree I/O and VmHWM at 10-ms intervals and the
filesystem block-device counters before/after execution. Per-PID poll
totals are LOWER BOUNDS; device counters include untracked background
system activity. The method's own latency/IO columns remain authoritative
until validated against hardware instrumentation.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def read_file(path: Path) -> str:
    try: return path.read_text()
    except (OSError, PermissionError): return ""

def children_of(pid: int) -> list[int]:
    s = read_file(Path(f"/proc/{pid}/task/{pid}/children"))
    out=[]
    for x in s.split():
        try: out.append(int(x))
        except ValueError: pass
    return out

def gather_tree(pid: int) -> list[int]:
    seen=set(); stack=[pid]
    while stack:
        cur=stack.pop()
        if cur in seen: continue
        seen.add(cur)
        stack.extend(children_of(cur))
    return list(seen)

def proc_stats(pid: int) -> dict:
    raw=read_file(Path(f"/proc/{pid}/io"))
    parsed={}
    for line in raw.splitlines():
        if ":" not in line: continue
        key,val=line.split(":",1)
        try: parsed[key.strip()]=int(val.strip())
        except ValueError: pass
    info=read_file(Path(f"/proc/{pid}/status"))
    for line in info.splitlines():
        if line.startswith(("VmRSS:","VmHWM:","VmPeak:")):
            name,v=line.split(":",1)
            try: parsed[name]=int(v.split()[0])*1024
            except (IndexError,ValueError): pass
    return parsed

def device_stats(path: Path) -> dict:
    p=path.resolve()
    try:
        dev=os.stat(p).st_dev
        major,minor=os.major(dev),os.minor(dev)
        f=Path(f"/sys/dev/block/{major}:{minor}/stat")
        a=[int(x) for x in f.read_text().split()]
        if len(a)<7: return {}
        return {"filesystem_device":f"{major}:{minor}","sysfs":str(f),
                "read_ops":a[0],"read_sectors_512B":a[2],
                "write_ops":a[4],"write_sectors_512B":a[6]}
    except (OSError,ValueError): return {}

def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--cwd",type=Path,required=True)
    ap.add_argument("--device-path",type=Path,required=True)
    ap.add_argument("--interval-ms",type=int,default=10)
    ap.add_argument("argv",nargs=argparse.REMAINDER)
    args=ap.parse_args()
    argv=args.argv
    if argv and argv[0]=="--": argv=argv[1:]
    if not argv: ap.error("expected -- COMMAND [ARGS]")
    if args.interval_ms<5: ap.error("polling more often than every 5ms is not useful")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    before=device_stats(args.device_path)
    started=time.monotonic()
    p=subprocess.Popen(argv,cwd=args.cwd)
    observed={}
    intervals=0
    while p.poll() is None:
        for child in gather_tree(p.pid):
            current=proc_stats(child)
            if not current: continue
            prev=observed.setdefault(str(child),{})
            for name,value in current.items():
                prev[name]=max(value,prev.get(name,0))
        intervals+=1
        time.sleep(args.interval_ms/1000)
    ret=p.wait()
    for child in gather_tree(p.pid):
        current=proc_stats(child)
        prev=observed.setdefault(str(child),{})
        for name,value in current.items():
            prev[name]=max(value,prev.get(name,0))
    after=device_stats(args.device_path)
    device_delta={}
    if before.get("filesystem_device")==after.get("filesystem_device"):
        for k in ("read_ops","read_sectors_512B","write_ops","write_sectors_512B"):
            if k in before and k in after:
                device_delta[k]=max(0,after[k]-before[k])
        device_delta["read_bytes_512_sector_estimate"]=device_delta.get("read_sectors_512B",0)*512
        device_delta["write_bytes_512_sector_estimate"]=device_delta.get("write_sectors_512B",0)*512
    out={
        "cmd":argv,"cwd":str(args.cwd.resolve()),"elapsed_seconds":time.monotonic()-started,
        "exit_code":ret,"poll_intervals":intervals,
        "process_pid_samples":len(observed),"per_pid_observed_maxima":observed,
        "process_io_observed_lower_bound":{
            k:sum(d.get(k,0) for d in observed.values())
            for k in ("read_bytes","write_bytes","rchar","wchar","syscr","syscw")},
        "peak_per_process_rss_bytes":max((d.get("VmHWM",0) for d in observed.values()),default=0),
        "filesystem_device_before":before,"filesystem_device_after":after,
        "filesystem_device_delta":device_delta,
        "qualification":"Per-PID /proc I/O is sampled lower bound; underlying device counters include other processes. Peak RSS is maximum observed single process, not total multi-process sum.",
    }
    args.out.write_text(json.dumps(out,indent=2)+"\n")
    print(f"RESOURCE_SNAPSHOT path={args.out} status={ret} device={device_delta} "
          f"peak_rss={out['peak_per_process_rss_bytes']}",flush=True)
    return ret

if __name__=="__main__": raise SystemExit(main())
