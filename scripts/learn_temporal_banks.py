#!/usr/bin/env python3
"""Learn disjoint temporal specialist banks relative to an existing general vocabulary."""
from __future__ import annotations
import argparse, collections, json, struct
from pathlib import Path

MAGIC=b"GIDST001"

def read_ids(path: Path):
    raw=path.read_bytes()
    if len(raw)<16 or raw[:8]!=MAGIC: raise ValueError(f"bad ID file: {path}")
    n,res=struct.unpack("<II",raw[8:16])
    if res!=0 or len(raw)!=16+4*n: raise ValueError(f"bad ID file shape: {path}")
    return list(struct.unpack(f"<{n}I",raw[16:]))

def write_ids(path: Path, ids):
    if not ids or len(ids)!=len(set(ids)): raise ValueError("IDs must be nonempty unique")
    with path.open("wb") as f:
        f.write(MAGIC); f.write(struct.pack("<II",len(ids),0))
        f.write(struct.pack(f"<{len(ids)}I",*ids))

def rank(records,stage):
    score=collections.defaultdict(int); support=collections.defaultdict(int)
    for rec in records:
        first={}
        for pos,raw in enumerate(rec["ids"]): first.setdefault(int(raw),pos)
        for v,pos in first.items():
            r=pos-stage
            if r>0:
                score[v]+=r; support[v]+=1
    return sorted(((score[v],support[v],v) for v in score),
                  key=lambda x:(-x[0],-x[1],x[2]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=Path,required=True)
    ap.add_argument("--exclude-ids",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--stages",default="8,16,24")
    ap.add_argument("--budgets",default="4000,2000,2000")
    a=ap.parse_args()
    stages=[int(x) for x in a.stages.split(",") if x.strip()]
    budgets=[int(x) for x in a.budgets.split(",") if x.strip()]
    if len(stages)!=len(budgets): raise ValueError("stage/budget mismatch")
    records=[json.loads(x) for x in a.trace.read_text().splitlines() if x.strip()]
    if len(records)!=5000: raise ValueError(f"expected 5000 traces, got {len(records)}")
    a.out_dir.mkdir(parents=True,exist_ok=True)

    excluded=read_ids(a.exclude_ids)
    used=set(excluded)
    out={
        "training_queries":len(records),
        "excluded_general_ids":len(excluded),
        "definition":"S_s(v)=sum_q max(first_expansion_position-s,0); banks chosen greedily disjoint from general and earlier specialist banks",
        "stages":{},
    }
    for stage,budget in zip(stages,budgets):
        ranked=rank(records,stage)
        avail=[x for x in ranked if x[2] not in used]
        if len(avail)<budget:
            raise ValueError(f"stage {stage}: need {budget}, only {len(avail)} disjoint candidates")
        chosen=avail[:budget]
        ids=[v for _,_,v in chosen]
        path=a.out_dir/f"stage{stage}-b{budget}.bin"
        write_ids(path,ids)
        used.update(ids)
        total=sum(s for s,_,_ in ranked)
        novel=sum(s for s,_,_ in avail)
        mass=sum(s for s,_,_ in chosen)
        out["stages"][str(stage)]={
            "budget":budget,"eligible_vertices":len(ranked),"disjoint_available":len(avail),
            "total_score_mass":total,"disjoint_score_mass":novel,"chosen_score_mass":mass,
            "chosen_fraction_of_disjoint_mass":mass/novel if novel else 0.0,
            "file":path.name,
            "top20":[{"vertex":v,"score":s,"support":sup} for s,sup,v in chosen[:20]],
        }
    out["total_unique_ids_with_general"]=len(used)
    (a.out_dir/"manifest.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
