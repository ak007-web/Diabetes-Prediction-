# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import traceback

app = Flask(__name__)
CORS(app)

MODEL_PATHS = [
    "diabetes_rf_model.joblib",
    "diabetes_rf_model.pkl",
]

model = None

def load_model():
    global model
    # Try joblib first (preferred), then pickle
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                print(f"Loaded model from {p} (joblib.load)")
                return
            except Exception as e_job:
                print(f"joblib.load failed for {p}: {e_job}")
                try:
                    # fallback to pickle via joblib for older pickles:
                    import pickle
                    with open(p, "rb") as f:
                        model = pickle.load(f)
                    print(f"Loaded model from {p} using pickle")
                    return
                except Exception as e_pickle:
                    print(f"pickle.load failed for {p}: {e_pickle}")
    raise FileNotFoundError("No model file found. Place 'diabetes_rf_model.joblib' or 'diabetes_rf_model.pkl' next to app.py")

# Basic expected columns (match your frontend)
COLUMNS = [
  'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
  'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
  'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity'
]

def preprocess_input(data: dict) -> pd.DataFrame:
    # Ensure all columns present (fill missing with defaults)
    row = {}
    for c in COLUMNS:
        if c in data and data[c] != "":
            row[c] = data[c]
        else:
            # sensible defaults
            if c == "Age":
                row[c] = 0
            elif c == "Gender":
                row[c] = "Male"
            else:
                row[c] = "No"

    df = pd.DataFrame([row])

    # encode object columns: Yes->1, No->0, Male->1, Female->0
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # Ensure numeric dtype for Age
    try:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce").fillna(0).astype(int)
    except Exception:
        df["Age"] = 0

    return df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            load_model()
        payload = request.get_json(force=True)
        # payload expected to be a dict representing the form (same as frontend)
        df = preprocess_input(payload)

        # predict
        pred = model.predict(df)
        # try to get probability
        prob = None
        try:
            proba = model.predict_proba(df)
            # for binary classifiers scikit gives shape (n_samples, n_classes)
            # choose probability of positive class (class 1) if available
            if proba.shape[1] == 2:
                prob = float(proba[0,1])
            else:
                # fallback: max class prob
                prob = float(np.max(proba[0]))
        except Exception:
            prob = None

        label = None
        # model may predict {0,1} or strings; normalize
        if isinstance(pred[0], (int, float, np.integer, np.floating)):
            label = "Positive" if int(pred[0]) == 1 else "Negative"
        else:
            # handle string labels
            label = str(pred[0])

        return jsonify({
            "diabetic": label,
            "probability_positive": prob
        })
    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    try:
        load_model()
    except Exception as e:
        print("Model load warning:", e)
    app.run(host="0.0.0.0", port=5000, debug=True)
