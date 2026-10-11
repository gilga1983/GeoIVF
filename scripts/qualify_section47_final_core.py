#!/usr/bin/env python3
"""Section 4.7: selected final Core vs competitor RAM allocation, matched recall.
Only B=14464,C=2048,k=8 is a NavHints reference; never replay retired Core.
"""
import argparse,fcntl,json,os,subprocess,struct
from pathlib import Path
import numpy as np
from qualify_final_core_public import slice_xbin,slice_gt,result_rows,aggregate,CFG,LS

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+"\n")
def run(binary,prefix,q,gt,out,tag,dtype,metric,params,ivf=None,cache=None,qsev=None,capacity=0):
 phase={"queries":str(q),"groundtruth":str(gt),"search_list":list(LS),"beam_width":8,"recall_at":10,"num_threads":4,"is_flat_search":False,"distance":metric,"vector_filters_file":None,"num_nodes_to_cache":capacity or None,"search_io_limit":None,"post_processor":None}
 inp=out/(tag+".input.json");output=out/(tag+".output.json")
 save(inp,{"search_directories":[str(out)],"jobs":[{"type":"disk-index","content":{"source":{"disk-index-source":"Load","data_type":dtype,"load_path":str(prefix)},"search_phase":phase}}]})
 env={k:v for k,v in os.environ.items() if not k.startswith("DISKANN_")}
 if cache: env["DISKANN_STATIC_CACHE_IDS_FILE"]=str(cache)
 if qsev: env["DISKANN_QSEV_FILE"]=str(qsev)
 if params=="core":
  env.update({"DISKANN_HINT_IVF_FILE":str(ivf),"DISKANN_HINT_IVF_NPROBE":"8","DISKANN_HINT_IVF_MAX_STARTS":"1","DISKANN_EXPERIENCE_REPLAY":"1","DISKANN_EXPERIENCE_WARMUP":"4000","DISKANN_EXPERIENCE_CACHE_CAPACITY":"2048","DISKANN_EXPERIENCE_HUB_CAPACITY":"0","DISKANN_EXPERIENCE_SAMPLE_DENOMINATOR":"2","DISKANN_EXPERIENCE_FILL_FROM_CACHE":"0","DISKANN_LOCAL_CACHE_MODE":"fifo","DISKANN_LOCAL_RECENT_STARTS":"8"})
 with (out/(tag+".log")).open("w") as f:subprocess.run([str(binary),"run","--input-file",str(inp),"--output-file",str(output)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 rr=sorted(result_rows(json.loads(output.read_text())),key=lambda r:int(r["search_l"]))
 if [int(r["search_l"]) for r in rr]!=list(LS):raise RuntimeError("Incomplete search grid: "+tag)
 return rr

def interp(summary,target):
 vals=sorted(((float(v["recall_percent"]),float(v["mean_ios"])) for v in summary.values()))
 good=[]
 for v in vals:
  if not good or v[0]>good[-1][0]:good.append(v)
 if not (good[0][0]<=target<=good[-1][0]):return None
 for (r0,c0),(r1,c1) in zip(good,good[1:]):
  if r0<=target<=r1:return c0+(target-r0)*(c1-c0)/(r1-r0)
 if target==good[-1][0]:return good[-1][1]
 return None

def main():
 ap=argparse.ArgumentParser()
 for name in ("dataset","index-prefix","queries","gt","ivf","core-binary","cache-binary","hot-dir","out","data-type","metric"):ap.add_argument("--"+name,required=True)
 ap.add_argument("--qsev-binary");ap.add_argument("--qsev-dir");ap.add_argument("--reps",type=int,default=3)
 a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 q=Path(a.queries).resolve();gt=Path(a.gt).resolve();ivf=Path(a.ivf).resolve()
 if a.reps<2:raise ValueError("Need rotated repetitions")
 if a.metric!="inner_product" and a.qsev_binary:raise ValueError("Existing QSEV patch implements inner product only")
 if not ivf.exists():raise FileNotFoundError(ivf)
 dim=struct.unpack("<II",q.open("rb").read(8))[1]
 if a.dataset=="pubmed":counts=(30,120,480);qcounts=(32,128,512)
 elif a.dataset=="bigann-10M":counts=(134,536,2144);qcounts=()
 else:raise ValueError("unsupported dataset")
 record_bytes=dim*4+64*4
 for n in counts:
  if not (0<n*record_bytes<=102940*16+record_bytes):raise ValueError("unexpected RAM counts")
 methods=["core"]+["hot-"+str(x) for x in counts]+["bfs-"+str(x) for x in counts]
 if a.qsev_binary:methods+=["qsev-"+str(x) for x in qcounts]
 warm=out/"heldout5000.input";meas=out/"measured1000.input"
 # Public and PubMed manifest provide exactly a disjoint 5,000-request replay.
 rows,dim2=struct.unpack("<II",q.open("rb").read(8))
 if rows!=5000:raise ValueError("Expected 4k warm-up + 1k measured queries")
 slice_xbin(q,meas,4000,1000,4 if a.data_type=="float32" else 1)
 mgt=out/"measured1000.gt";slice_gt(gt,mgt,4000,1000)
 binaries={"core":a.core_binary};results={k:[] for k in methods}
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(allowed[:4]))
 lockp=Path.home()/".cache/geoivf/speed-device.lock";lockp.parent.mkdir(parents=True,exist_ok=True)
 with lockp.open("w") as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  for rep in range(a.reps):
   order=methods[rep%len(methods):]+methods[:rep%len(methods)]
   for method in order:
    kind=method.split("-")[0]
    binary=a.core_binary if kind=="core" else a.qsev_binary if kind=="qsev" else a.cache_binary
    x=q if kind=="core" else meas;y=gt if kind=="core" else mgt
    cache=Path(a.hot_dir)/("hot-cache-n"+method.split("-")[1]+".bin") if kind=="hot" else None
    qsev=Path(a.qsev_dir)/("qsev-"+method.split("-")[1]+".bin") if kind=="qsev" else None
    capacity=int(method.split("-")[1]) if kind=="bfs" else 0
    rr=run(binary,a.index_prefix,x,y,out,"r%d-%s"%(rep,method),a.data_type,a.metric,kind,ivf,cache,qsev,capacity)
    results[method].append(rr);save(out/"partial-raw.json",results)
 summary={m:aggregate(reps) for m,reps in results.items()}
 anchor=[float(v["recall_percent"]) for v in summary["core"].values()]
 matched={}
 for method in methods[1:]:
  common=[(r,interp(summary["core"],r),interp(summary[method],r)) for r in anchor]
  common=[x for x in common if None not in x[1:]]
  matched[method]={"count":len(common),"targets":[r for r,_,_ in common],"saving_pct":float(np.mean([100*(1-c/k) for _,c,k in common])) if common else None}
 save(out/"section47-final-core.json",{"dataset":a.dataset,"selected_core":{"B":14464,"C":2048,"k":8,"regions":512,"persistent":False},"method_budgets":{"hot_nodes":counts,"record_bytes_nominal":record_bytes,"qsev_entries":qcounts},"reps":a.reps,"summary":summary,"matched_core_anchor":matched,"note":"QSEV PubMed only; native BFS and hot-record budgets are nominal payload, not complete RSS. Repeated frozen replay, not independent workload confidence interval."})
 print("SECTION47_FINAL_CORE_PASS",a.dataset,flush=True)
if __name__=="__main__":main()
