#!/usr/bin/env python3
"""First measurements of Table 7 alternatives on Coveo and Text2Image only.

Never remeasure the already completed selected Core or PubMed/BigANN baselines.
Use same pinned disk index, disjoint static training and chronological warmup.
Cache read counts must be genuine uncached disk-provider loads.
"""
from __future__ import annotations
import argparse,fcntl,json,os,struct,subprocess
from pathlib import Path
import numpy as np
from qualify_final_core_public import slice_xbin,slice_gt,result_rows

LS=(10,20,40,80,160,320,640)
CORE_LS=(10,20,40,80,160,320)
SCALES=(1,4,16)

def save(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(data,indent=2)+"\n")

def shape(p):
 with Path(p).open("rb") as f:raw=f.read(8)
 if len(raw)!=8:raise ValueError("truncated vector header")
 return struct.unpack("<II",raw)

def curve(values):
 out=sorted((float(a),float(b)) for a,b in values)
 if not out or any(not(0<=r<=100 and c>0) for r,c in out):
  raise ValueError("bad recall or SSD reads")
 valid=[]
 for a in out:
  if valid and a[0]<=valid[-1][0]:
   if a[1]<valid[-1][1]:valid[-1]=a
  else:valid.append(a)
 return valid

def interp(points,r):
 if r<points[0][0]-1e-6 or r>points[-1][0]+1e-6:return None
 for x,c in points:
  if abs(x-r)<1e-6:return c
 for (a,b),(c,d) in zip(points,points[1:]):
  if a<r<c:return b+(r-a)*(d-b)/(c-a)
 return None

def execute(binary,index,query,gt,out,tag,kind,n,hot,qsev):
 job={"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{
 "source":{"disk-index-source":"Load","data_type":"float32","load_path":str(index)},
 "search_phase":{"queries":str(query),"groundtruth":str(gt),"search_list":list(LS),
 "beam_width":8,"recall_at":10,"num_threads":4,"is_flat_search":False,
 "distance":"inner_product","vector_filters_file":None,
 "num_nodes_to_cache":n if kind=="bfs" else None,
 "search_io_limit":None,"post_processor":None}}}]}
 inp=out/(tag+".input.json");res=out/(tag+".output.json");log=out/(tag+".log")
 save(inp,job)
 env={k:v for k,v in os.environ.items() if not k.startswith("DISKANN_")}
 if kind=="hot":env["DISKANN_STATIC_CACHE_IDS_FILE"]=str(hot)
 if kind=="qsev":env["DISKANN_QSEV_FILE"]=str(qsev)
 with log.open("w") as f:
  subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(res)],
                 env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rr=sorted(result_rows(json.loads(res.read_text())),key=lambda v:int(v["search_l"]))
 if [int(v["search_l"]) for v in rr]!=list(LS):raise ValueError("incomplete recall sweep: "+tag)
 if kind in ("hot","bfs"):
  if not any(float(v.get("cache_hit_percentage",0))>0 for v in rr):
   raise RuntimeError("zero cache hits: rejected uninstrumented comparator "+tag)
  if any(float(v["mean_ios"])>float(v["mean_hops"])+1e-5 for v in rr):
   raise RuntimeError("SSD read count exceeds graph expansions: "+tag)
 return rr

def main():
 p=argparse.ArgumentParser()
 for key in ("dataset","reference","index-prefix","queries","gt","ivf",
             "cache-binary","qsev-binary","hot-dir","qsev-dir","out"):
  p.add_argument("--"+key,required=True)
 p.add_argument("--reps",type=int,default=3)
 a=p.parse_args(); ref=json.loads(Path(a.reference).read_text())
 if a.dataset not in ("coveo","text2image-10M"):raise ValueError("Only missing datasets permitted")
 x=ref["datasets"][a.dataset]
 if a.reps!=3:raise ValueError("three repetitions required")
 warm_n=x["warmup"];test_n=x["test"]
 if shape(a.queries)!=(warm_n+test_n,x["dim"]):raise ValueError("mismatched frozen query split")
 with Path(a.gt).open("rb") as f: gt_rows,gt_k=struct.unpack("<II",f.read(8))
 if gt_rows!=warm_n+test_n or gt_k<10:raise ValueError("groundtruth mismatch")
 actual_budget=Path(a.ivf).stat().st_size+512*(64+4)+2048*4
 if actual_budget!=x["payload_bytes"]:raise ValueError("Core payload mismatch")

 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 warm=out/"warm.fbin";meas=out/"measured.fbin"
 warm_gt=out/"warm.gt";meas_gt=out/"measured.gt"
 slice_xbin(Path(a.queries),warm,0,warm_n,4)
 slice_xbin(Path(a.queries),meas,warm_n,test_n,4)
 slice_gt(Path(a.gt),warm_gt,0,warm_n)
 slice_gt(Path(a.gt),meas_gt,warm_n,test_n)
 budget=x["payload_bytes"];dim=x["dim"]
 cache_unit=dim*4+256;qsev_unit=dim*4+4
 cache_counts=[budget*s//cache_unit for s in SCALES]
 qsev_counts=[(budget*s-16)//qsev_unit for s in SCALES]
 methods=["baseline"]+[f"hot-{n}" for n in cache_counts]+[f"bfs-{n}" for n in cache_counts]+[f"qsev-{n}" for n in qsev_counts]
 methods=list(dict.fromkeys(methods))
 data={m:[] for m in methods}
 allowed=sorted(os.sched_getaffinity(0))
 if len(allowed)<4:raise RuntimeError("need four available cores")
 os.sched_setaffinity(0,set(allowed[:4]))
 lock=Path.home()/".cache/geoivf/speed-device.lock"
 lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open("w") as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  for rep in range(a.reps):
   for name in methods[rep:]+methods[:rep]:
    kind=name.split("-")[0];n=0 if kind=="baseline" else int(name.split("-")[1])
    binary=a.qsev_binary if kind=="qsev" else a.cache_binary
    hot=Path(a.hot_dir)/(f"hot-cache-n{n}.bin") if kind=="hot" else None
    qsev=Path(a.qsev_dir)/(f"qsev-{n}.bin") if kind=="qsev" else None
    # The deployed Core learned from the same 20K or 4K prefix.
    # Static controls replay this prefix only for NVMe cache warmth.
    execute(binary,a.index_prefix,warm,warm_gt,out,f"r{rep}-{name}-warm",kind,n,hot,qsev)
    rr=execute(binary,a.index_prefix,meas,meas_gt,out,f"r{rep}-{name}",kind,n,hot,qsev)
    data[name].append(rr);save(out/"runs.partial.json",data)
 summary={}
 for method,runs in data.items():
  summary[method]={}
  for L in LS:
   vs=[v for run in runs for v in run if int(v["search_l"])==L]
   if len(vs)!=a.reps:raise RuntimeError("missing repetitions")
   summary[method][str(L)]={"recall_percent":float(np.mean([float(v["recall"]) for v in vs])),
                            "mean_ios":float(np.mean([float(v["mean_ios"]) for v in vs]))}
 # Validate index/query identity against the archived selected-Core baseline.
 for L in CORE_LS:
  old=x["baseline"][str(L)];now=summary["baseline"][str(L)]
  if abs(now["recall_percent"]-old[0])>0.02 or abs(now["mean_ios"]-old[1])>0.1:
   raise RuntimeError(f"Frozen baseline mismatch at L={L}: {now} != {old}")
 core=curve(x["core"].values())
 anchors=[float(x["core"][str(L)][0]) for L in CORE_LS]
 matched={}
 for method in methods[1:]:
  vals=curve((v["recall_percent"],v["mean_ios"]) for v in summary[method].values())
  observed=[]
  for recall in anchors:
   left=interp(core,recall);right=interp(vals,recall)
   if left is not None and right is not None:
    observed.append({"recall":recall,"core_reads":left,"competitor_reads":right,
                     "saving_pct":100*(1-left/right)})
  matched[method]={"count":len(observed),"anchors":observed,
                   "saving_pct":float(np.mean([v["saving_pct"] for v in observed])) if len(observed)>=2 else None}
 save(out/"table7-first-time-results.json",{
  "dataset":a.dataset,"frozen_core_run":x["source_run"],"frozen_core_artifact":x["source_artifact"],
  "sampled_replays":a.reps,"metric":"inner_product","ssd_reads":"uncached graph-provider reads",
  "split":{"warmup":warm_n,"measured":test_n},
  "budget":{"core_bytes":budget,"tiers":list(SCALES),
            "hot_bfs_nodes":cache_counts,"qsev_entries":qsev_counts},
  "methods":methods,"summary":summary,"matched":matched,
  "boundary":"Results may have different mutually reachable recall ranges; never extrapolate or pool target sets."
 })
 print("FIRST_TIME_TABLE7_SUCCESS",a.dataset,flush=True)
if __name__=="__main__":main()
