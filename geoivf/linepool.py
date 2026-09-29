"""Line-first greedy grouping in 64-vector within-list pools.

A broader construction than adjacent-page reassignment, still a local heuristic.
Candidate lines come from endpoint pairs; all final page lines are fitted freely.
"""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from .index import Index
from .lines import digest, line_score


def pool_layout(source,out,x,projected,*,dims=64,passes=1):
    source,out=Path(source),Path(out)
    base=Index(source);m=base.meta;cap=m['capacity'];width=8*cap
    if cap!=8 or dims>projected.shape[1]:raise ValueError('eight-vector pages required')
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    if digest(source/'vectors.pages')!=m['layout_payload_sha256']:raise ValueError('payload changed')
    out.mkdir(parents=True,exist_ok=True)
    blocks=[]
    for first,last in base.ranges:
        end=int(last)-(int(last)>int(first) and base.valid[int(last)-1]<cap)
        blocks.extend((p,min(8,end-p)) for p in range(int(first),end,8))
    pairs=np.asarray(list(combinations(range(width),2)),dtype=int)
    take=np.random.default_rng(9182).choice(len(pairs),192,replace=False)
    pairs=np.unique(np.concatenate((pairs[take],np.array([(i,i+1) for i in range(width-1)]))),axis=0)
    ii,jj=pairs[:,0],pairs[:,1]
    ids=base.page_ids.copy();accepted=0;old_objective=0.;new_objective=0.
    for start in range(0,len(blocks),64):
        part=blocks[start:start+64];B=len(part)
        old=np.full((B,width),-1,dtype=np.int64);counts=np.array([n*cap for _,n in part])
        for b,(p,n) in enumerate(part):old[b,:n*cap]=ids[p:p+n].reshape(-1)
        available=np.arange(width)[None]<counts[:,None]
        z=np.asarray(projected[np.maximum(old,0),:dims],dtype=float)
        z-=((z*available[...,None]).sum(axis=1)/counts[:,None])[:,None]
        gram=z@z.transpose(0,2,1);norm=np.diagonal(gram,axis1=1,axis2=2)
        sep=np.maximum(0.,norm[:,ii]+norm[:,jj]-2*gram[:,ii,jj])
        axial=(gram[:,jj,:]-gram[:,ii,:]-gram[:,ii,jj,None]+norm[:,ii,None])/np.sqrt(np.maximum(sep,1e-30))[:,:,None]
        error=np.maximum(0.,norm[:,None,:]+norm[:,ii,None]-2*gram[:,ii,:]-axial**2)
        proposed=np.full_like(old,-1)
        for page in range(8):
            active=np.flatnonzero(counts>page*cap)
            masked=np.where(available[:,None],error,np.inf)
            selected=np.argpartition(masked,cap-1,axis=2)[:,:,:cap]
            ee=np.take_along_axis(masked,selected,axis=2)
            tt=np.take_along_axis(axial,selected,axis=2)
            score=ee.sum(axis=2)+ee.max(axis=2)+.1*(tt.max(axis=2)-tt.min(axis=2))**2
            allowed=available[:,ii]&available[:,jj]&(sep>1e-20)
            score[~allowed]=np.inf
            choice=np.argmin(score,axis=1)
            chosen=selected[np.arange(B),choice]
            bad=~np.isfinite(score[np.arange(B),choice])
            chosen[bad]=np.argsort(~available[bad],axis=1,kind='stable')[:,:cap]
            proposed[active,page*cap:(page+1)*cap]=np.take_along_axis(old[active],chosen[active],axis=1)
            available[active[:,None],chosen[active]]=False
        def objective(order):
            groups=order.reshape(B*8,cap);good=groups[:,0]>=0
            zz=np.asarray(projected[np.maximum(groups[good],0),:dims],dtype=float)
            score=np.zeros(B*8)
            score[good]=line_score(zz,np.ones(zz.shape[:2],bool))[0]
            return score.reshape(B,8).sum(axis=1)
        before,after=objective(old),objective(proposed)
        use=after<before-1e-10*np.maximum(1.,before)
        old_objective+=float(before.sum());new_objective+=float(np.where(use,after,before).sum())
        for b,(p,n) in enumerate(part):
            if use[b]:ids[p:p+n]=proposed[b,:n*cap].reshape(n,cap);accepted+=1
    for first,last in base.ranges:
        np.testing.assert_array_equal(np.sort(ids[first:last].ravel()),np.sort(base.page_ids[first:last].ravel()))
    with np.load(source/'directory.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    arrays['page_ids']=ids
    with (out/'vectors.pages').open('wb') as f:
        for li,(first,last) in enumerate(base.ranges):
            for p in range(int(first),int(last)):
                group=ids[p,:base.valid[p]];payload=x[group].astype('<f4',copy=False).tobytes()
                f.write(payload+bytes(m['page_size']-len(payload)))
                r=np.linalg.norm(x[group].astype(float)-base.centers[li],axis=1)
                arrays['radial'][p]=[np.nextafter(np.float32(r.min()),np.float32(-np.inf)),np.nextafter(np.float32(r.max()),np.float32(np.inf))]
    arrays['codes'][:]=0;arrays['radii'][:]=-1;arrays['radii'][:,0]=np.inf
    np.savez(out/'directory.npz',**arrays)
    info=dict(m,layout='line-pool64',layout_payload_sha256=digest(out/'vectors.pages'),
              layout_directory_sha256=digest(out/'directory.npz'),
              refinement=dict(pool_vectors=width,candidate_pairs=int(len(pairs)),blocks=len(blocks),accepted_blocks=accepted,
                              objective_before=old_objective,objective_after=new_objective,
                              changed_membership_pages=int(np.any(np.sort(ids,axis=1)!=np.sort(base.page_ids,axis=1),axis=1).sum()),
                              query_fitted=False,global_optimum=False))
    (out/'manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
