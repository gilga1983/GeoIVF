#!/usr/bin/env python3
"""Learn greedily disjoint stage-specific NavHints vocabularies."""
from __future__ import annotations
import argparse, collections, json, struct
from pathlib import Path

MAGIC=b"GIDST001"

def read_ids(path: Path):
    raw=path.read_bytes()
    if len(raw)<16 or raw[:8]!=MAGIC: raise ValueError("bad ID file")
    n,res=struct.unpack("<II",raw[8:16])
    if res!=0 or len(raw)!=16+4*n: raise ValueError("bad ID file shape")
    return list(struct.unpack(f"<{n}I",raw[16:]))

def write_ids(path: Path, ids):
    if len(ids)!=len(set(ids)) or not ids: raise ValueError("IDs must be nonempty unique")
    with path.open("wb") as f:
        f.write(MAGIC); f.write(struct.pack("<II",len(ids),0))
        f.write(struct.pack(f"<{len(ids)}I",*ids))

def rank(records, stage):
    score=collections.defaultdict(int); support=collections.defaultdict(int)
    for rec in records:
        first={}
        for pos,raw in enumerate(rec["ids"]): first.setdefault(int(raw),pos)
        for v,pos in first.items():
            r=pos-stage
            if r>0: score[v]+=r; support[v]+=1
    return sorted(((score[v],support[v],v) for v in score),
                  key=lambda x:(-x[0],-x[1],x[2]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=Path,required=True)
    ap.add_argument("--stage0",type=Path,required=True)
    ap.add_argument("--stages",default="3,5,7,8")
    ap.add_argument("--budgets",default="4096,2048,1024,1024")
    ap.add_argument("--out-dir",type=Path,required=True)
    args=ap.parse_args()
    stages=[int(x) for x in args.stages.split(",")]
    budgets=[int(x) for x in args.budgets.split(",")]
    if len(stages)!=len(budgets): raise ValueError("stage/budget length mismatch")
    records=[json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    used=set(read_ids(args.stage0))
    args.out_dir.mkdir(parents=True,exist_ok=True)
    result={"training_queries":len(records),"stage0_ids":len(used),
            "stages":stages,"budgets":budgets,"variants":{}}
    for stage,budget in zip(stages,budgets):
        ranked=rank(records,stage)
        avail=[x for x in ranked if x[2] not in used]
        take=min(budget,len(avail))
        ids=[v for _,_,v in avail[:take]]
        if not ids: raise ValueError(f"stage {stage} has no disjoint candidates")
        path=args.out_dir/f"stage{stage}-b{take}.bin"
        write_ids(path,ids)
        mass=sum(s for s,_,_ in avail)
        chosen=sum(s for s,_,_ in avail[:take])
        result["variants"][str(stage)]={
            "requested_budget":budget,"actual_budget":take,
            "available_disjoint_before_selection":len(avail),
            "chosen_score_mass":chosen,"available_score_mass":mass,
            "chosen_fraction_of_available_mass":chosen/mass if mass else 0,
            "file":path.name,
            "top20":[{"vertex":v,"score":s,"support":sup} for s,sup,v in avail[:20]],
        }
        used.update(ids)
    result["total_ids_including_stage0"]=len(used)
    (args.out_dir/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
