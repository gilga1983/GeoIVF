import numpy as np
from geoivf.linepool import pool_layout
from geoivf.lines import summarize_lines, LineIndex
from geoivf.index import Index
from test_lines import fixture

def test_line_pool_preserves_full_vectors_and_lists(tmp_path):
    x,labels,physical,basis=fixture(tmp_path,n=301)
    out=tmp_path/'pool'
    info=pool_layout(physical,out,x,x.astype(float))
    before,after=Index(physical),Index(out)
    assert info['refinement']['objective_after']<=info['refinement']['objective_before']
    assert info['refinement']['changed_membership_pages']>0
    for first,last in before.ranges:
        np.testing.assert_array_equal(np.sort(before.page_ids[first:last].ravel()),np.sort(after.page_ids[first:last].ravel()))
    raw=np.fromfile(out/'vectors.pages',dtype='<f4').reshape(-1,8,128)
    for p,n in enumerate(after.valid):
        np.testing.assert_array_equal(raw[p,:n],x[after.page_ids[p,:n]])
    summarize_lines(out,tmp_path/'sidecar',basis,x.astype(float))
    idx=LineIndex(out,tmp_path/'sidecar')
    for q in x[:3]:
        for li,(first,last) in enumerate(idx.ranges):
            pp=np.arange(first,last);ids=idx.page_ids[pp]
            exact=np.linalg.norm(x[np.maximum(ids,0)].astype(float)-q,axis=2)
            exact[ids<0]=np.inf
            assert np.all(idx.bounds(q,pp,li,'combined')<=exact.min(axis=1)+1e-9)
