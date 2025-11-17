import joblib
import os
import numpy as np
from modules.pre_processing import preprocess_essay
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one folder (AI-Detection), then into model/
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "xgb_essay_numeric_pipeline.joblib"))

model = joblib.load(MODEL_PATH)

FEATURE_ORDER = [
    "punct_ratio",
    "avg_token_length",
    "type_token_ratio",
    "hapax_legomena_ratio",
    "pos_noun_ratio",
    "pos_punct_ratio",
    "pos_det_ratio",
    "pos_adj_ratio",
    "pos_adv_ratio",
    "pos_verb_ratio",
    "pos_pron_ratio",
    "sentence_token_entropy"
]

def prediction(user_input : dict):
    """
    user_input = { "text": "the essay goes here" }
    """

    essay_text = user_input.get("text", "")

    # Step 1: preprocess text → feature dict
    features = preprocess_essay(essay_text)

    # Step 2: convert dict → feature vector in correct order
    vector = np.array([features[f] for f in FEATURE_ORDER]).reshape(1, -1)

    # Step 3: model prediction
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]

    # Step 4: map label
    label = "AI Essay" if pred == 1 else "Human Essay"

    return {
        "label": label,
        "prediction": int(pred),
        "probability": float(prob)
    }