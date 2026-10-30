from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(21)
centres={"scratch":[.08,.65,.25,1,.8],"dent":[.22,.35,.5,2,.8],"crack":[.32,.7,.65,3,.65],"broken_glass":[.48,.4,.8,4,.7]}
def generate(n,labelled):
    y=rng.choice(list(centres),n)
    x=np.array([np.array(centres[k])+rng.normal(0,[.08,.12,.15,.25,.1]) for k in y])
    d=pd.DataFrame(x,columns=["area_ratio","edge_density","contrast","component_code","image_quality"])
    d.insert(0,"case_id",[f"SYN-{i:05d}" for i in range(n)])
    if labelled:d["damage_label"]=y
    return d
Path("data/sample").mkdir(parents=True,exist_ok=True)
generate(700,True).to_csv("data/sample/labelled.csv",index=False)
generate(200,False).to_csv("data/sample/unlabelled.csv",index=False)
print("Synthetic records created for pipeline testing only.")
