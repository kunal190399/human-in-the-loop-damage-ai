import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
FEATURES=["area_ratio","edge_density","contrast","component_code","image_quality"]
LABEL="damage_label"
def build():
    return Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler()),
      ("clf",RandomForestClassifier(n_estimators=250,class_weight="balanced",random_state=42,n_jobs=-1))])
def entropy(probs):
    p=np.clip(np.asarray(probs,dtype=float),1e-12,1)
    return -(p*np.log(p)).sum(axis=1)
def fit(frame):
    missing=set(FEATURES+[LABEL])-set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    m=build();m.fit(frame[FEATURES],frame[LABEL].astype(str));return m
def review_queue(model,frame,n=50):
    prob=model.predict_proba(frame[FEATURES]);out=frame.copy()
    out["suggested_label"]=model.classes_[prob.argmax(axis=1)]
    out["uncertainty"]=entropy(prob)
    out["expert_label"]=""
    return out.sort_values("uncertainty",ascending=False).head(n)
