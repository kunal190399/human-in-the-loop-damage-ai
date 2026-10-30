import numpy as np,pandas as pd
from hitl_ai.active import entropy,fit,review_queue,FEATURES
def test_entropy():
    assert entropy([[.5,.5]])[0] > entropy([[.99,.01]])[0]
def test_review_queue():
    rows=[]
    for i in range(40):
        rows.append(dict(area_ratio=i/40,edge_density=.4,contrast=.3,component_code=1,
        image_quality=.8,damage_label="dent" if i%2 else "scratch"))
    frame=pd.DataFrame(rows);model=fit(frame)
    unlabelled=frame[FEATURES].copy();unlabelled["case_id"]=[str(i) for i in range(40)]
    assert len(review_queue(model,unlabelled,5))==5
