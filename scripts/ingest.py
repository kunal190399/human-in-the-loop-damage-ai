import argparse,pandas as pd
from pathlib import Path
from hitl_ai.active import FEATURES,LABEL
p=argparse.ArgumentParser();p.add_argument("--queue",required=True);p.add_argument("--output",default="data/processed/feedback.csv");a=p.parse_args()
df=pd.read_csv(a.queue)
if "expert_label" not in df: raise SystemExit("Missing expert_label column")
df=df[df.expert_label.notna() & (df.expert_label.astype(str).str.strip()!="")].copy()
if df.empty: raise SystemExit("No expert-reviewed labels found.")
df=df.rename(columns={"expert_label":LABEL})
missing=set(FEATURES+[LABEL])-set(df.columns)
if missing: raise SystemExit(f"Missing columns: {missing}")
Path(a.output).parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.output,index=False)
print("Ingested",len(df),"expert-labelled cases.")
