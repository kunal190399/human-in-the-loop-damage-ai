import argparse,pandas as pd,joblib
from pathlib import Path
from hitl_ai.active import review_queue
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--model",required=True)
p.add_argument("--n",type=int,default=50);p.add_argument("--output",default="data/processed/review_queue.csv");a=p.parse_args()
q=review_queue(joblib.load(a.model),pd.read_csv(a.data),a.n)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);q.to_csv(a.output,index=False);print("Wrote",len(q),"cases")
