import argparse,pandas as pd,joblib
from pathlib import Path
from hitl_ai.active import fit
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",default="artifacts/classifier.joblib");a=p.parse_args()
m=fit(pd.read_csv(a.data));Path(a.output).parent.mkdir(parents=True,exist_ok=True);joblib.dump(m,a.output);print("Saved",a.output)
