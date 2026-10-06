#!/usr/bin/env python3
"""Build an equal-memory temporally diverse NavHints vocabulary.

Keep the current 8K generalist prefix, then add disjoint residual specialists
trained for progressively later expansion ages. The output is one ordinary
GIDST001 ID file, so the deployed progressive router is otherwise unchanged.
"""
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
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC); f.write(struct.pack("<II",len(ids),0))
        f.write(struct.pack(f"<{len(ids)}I",*ids))

def rank(records, stage):
    score=collections.defaultdict(int); support=collections.defaultdict(int)
    for rec in records:
        first={}
        for pos,raw in enumerate(rec["ids"]):
            first.setdefault(int(raw),pos)
        for vid,pos in first.items():
            residual=pos-stage
            if residual>0:
                score[vid]+=residual
                support[vid]+=1
    return sorted(((score[v],support[v],v) for v in score),
                  key=lambda x:(-x[0],-x[1],x[2]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--trace",type=Path,required=True)
    ap.add_argument("--general-ids",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--stages",default="8,16,24")
    ap.add_argument("--budgets",default="4096,2048,2048")
    args=ap.parse_args()

    stages=[int(x) for x in args.stages.split(",") if x.strip()]
    budgets=[int(x) for x in args.budgets.split(",") if x.strip()]
    if len(stages)!=len(budgets): raise ValueError("stage/budget mismatch")

    records=[json.loads(x) for x in args.trace.read_text().splitlines() if x.strip()]
    if len(records)!=5000: raise ValueError(f"expected 5000 training traces, got {len(records)}")

    general=read_ids(args.general_ids)
    if len(general)!=8192: raise ValueError(f"expected 8192 general IDs, got {len(general)}")
    used=set(general); combined=list(general)
    manifest={
        "training_queries":len(records),
        "general_ids":len(general),
        "definition":"S_s(v)=sum_q max(first_expansion_position-s,0), selected greedily disjoint from earlier banks",
        "stages":{},
    }

    for stage,budget in zip(stages,budgets):
        ranked=rank(records,stage)
        available=[x for x in ranked if x[2] not in used]
        if len(available)<budget:
            raise ValueError(f"stage {stage}: only {len(available)} disjoint candidates for budget {budget}")
        chosen=available[:budget]
        ids=[v for _,_,v in chosen]
        combined.extend(ids); used.update(ids)
        total_mass=sum(s for s,_,_ in ranked)
        available_mass=sum(s for s,_,_ in available)
        chosen_mass=sum(s for s,_,_ in chosen)
        manifest["stages"][str(stage)]={
            "budget":budget,
            "eligible_vertices":len(ranked),
            "disjoint_available":len(available),
            "total_stage_score_mass":total_mass,
            "disjoint_score_mass":available_mass,
            "chosen_score_mass":chosen_mass,
            "chosen_fraction_of_disjoint_mass":chosen_mass/available_mass if available_mass else 0.0,
            "top10":[{"vertex":v,"score":s,"support":sup} for s,sup,v in chosen[:10]],
        }

    if len(combined)!=16384: raise ValueError(f"expected 16384 IDs, got {len(combined)}")
    write_ids(args.out,combined)
    manifest["total_ids"]=len(combined)
    manifest["state_bytes"]=args.out.stat().st_size
    mp=args.out.with_suffix(args.out.suffix+".manifest.json")
    mp.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
