from pathlib import Path
import sys,pandas as pd,streamlit as st
st.title("Expert Review Queue")
st.caption("Prototype review console. Do not accept model suggestions without assessment.")
path=st.text_input("Queue CSV","data/processed/review_queue.csv")
if Path(path).exists():
    df=pd.read_csv(path)
    labels=["","scratch","dent","crack","broken_glass","not_damage","needs_adjudication"]
    i=st.number_input("Case row",0,max(0,len(df)-1),0)
    row=df.iloc[int(i)]
    st.subheader(row.get("case_id","Case"))
    st.write("Suggested label:",row.get("suggested_label"))
    st.write("Uncertainty:",row.get("uncertainty"))
    choice=st.selectbox("Expert label",labels)
    if st.button("Save expert label"):
        df.at[int(i),"expert_label"]=choice
        Path(path).parent.mkdir(parents=True,exist_ok=True);df.to_csv(path,index=False)
        st.success("Saved. Continue reviewing, then run scripts/ingest.py.")
else: st.info("Generate a review queue first; see README.")
