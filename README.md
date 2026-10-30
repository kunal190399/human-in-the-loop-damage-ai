# Human-in-the-Loop Vehicle Damage AI

An active-learning workflow that ranks uncertain cases for expert review, exports
a review queue, ingests expert labels and retrains a candidate classifier.

## Quick start
```bash
pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/train.py --data data/sample/labelled.csv
python scripts/queue.py --data data/sample/unlabelled.csv --model artifacts/classifier.joblib
streamlit run app/review.py
```
After filling `expert_label` in the review queue:
```bash
python scripts/ingest.py --queue data/processed/review_queue.csv
python scripts/retrain.py --data data/processed/feedback.csv
pytest
```

## Important data note
The generator creates synthetic numeric feature vectors, not vehicle photographs.
Use real expert-annotated records only after governance approval. Never treat model
suggestions as ground truth. Preserve annotator identity/access controls, label
provenance, model version and adjudication history.

## Method and evaluation
The starter uses predictive entropy for uncertainty sampling and a balanced
RandomForest classifier. Compare random sampling, least confidence, margin sampling
and entropy. Plot macro-F1 / per-class recall against number of labels acquired;
measure calibration, annotation time, expert disagreement and rare-class recall.
Use a fixed untouched test set for every model comparison.
