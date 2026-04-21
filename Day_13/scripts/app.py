"""
app.py
======
Flask API — bridges meetmux_ui.html with the saved .joblib models.

Endpoints:
  POST /predict            — compatibility score for a user pair
  GET  /feature-importance — ranked RF feature importances
  GET  /metrics            — confusion matrix + accuracy/precision/recall/F1

Run:
    cd Day_13/scripts
    python app.py
Then open meetmux_ui.html in your browser.

Requirements:
    pip install flask flask-cors scikit-learn joblib numpy pandas
"""

import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# ── Make sure sibling scripts are importable ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    MODELS_DIR, FEATURE_NAMES,
    DOMAIN_GROUPS, GOAL_COMPAT, COMM_COMPAT,
    SCORE_GOOD, SCORE_MEDIUM,
)
from data_generator import generate_users, generate_pairs, make_pair_features
from preprocessor   import Preprocessor
from model          import MatcherModel
from evaluator      import Evaluator

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # allow the HTML file (file://) to call this API


# ── Load models once at startup ───────────────────────────────────────────────
print("[app] Loading models...")

try:
    preprocessor = Preprocessor.load()
    model        = MatcherModel.load()
    print("[app] Models loaded successfully.")
except Exception as e:
    print(f"[app] ERROR loading models: {e}")
    print("[app] Run 'python main.py' first to train and save the models.")
    preprocessor = None
    model        = None

# ── Cached metrics (computed once on startup) ─────────────────────────────────
_cached_metrics     = None
_cached_importances = None

def _compute_metrics_once():
    """Run evaluation on a fresh test split and cache the results."""
    global _cached_metrics, _cached_importances
    if _cached_metrics is not None:
        return

    print("[app] Computing evaluation metrics (one-time)...")
    users = generate_users()
    pairs = generate_pairs(users)

    prep_fresh = Preprocessor()
    _, X_test, _, y_test = prep_fresh.fit_transform(pairs)

    # Re-use the *loaded* preprocessor to transform (same poly+scaler as training)
    X_raw  = pairs[FEATURE_NAMES].values
    from sklearn.model_selection import train_test_split
    from config import TEST_SIZE, RANDOM_SEED
    _, X_raw_test, _, y_t = train_test_split(
        X_raw, pairs["label"].values,
        test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=pairs["label"].values,
    )
    X_scaled = preprocessor.transform(X_raw_test)
    y_pred   = model.predict(X_scaled)

    from sklearn.metrics import (
        confusion_matrix, accuracy_score,
        precision_score, recall_score, f1_score,
    )
    cm      = confusion_matrix(y_t, y_pred)
    tn, fp, fn, tp = cm.ravel()

    _cached_metrics = {
        "tp"       : int(tp),
        "tn"       : int(tn),
        "fp"       : int(fp),
        "fn"       : int(fn),
        "accuracy" : round(accuracy_score(y_t, y_pred) * 100, 1),
        "precision": round(precision_score(y_t, y_pred) * 100, 1),
        "recall"   : round(recall_score(y_t, y_pred) * 100, 1),
        "f1"       : round(f1_score(y_t, y_pred) * 100, 1),
    }

    fi_df = model.feature_importances(preprocessor.feature_names, top_n=10)
    _cached_importances = [
        {"feature": row["feature"], "importance": round(float(row["importance"]) * 100, 1)}
        for _, row in fi_df.iterrows()
    ]
    print("[app] Metrics ready.")


# ── Helper: raw features from profile dict ────────────────────────────────────
def _profile_to_features(a: dict, b: dict) -> np.ndarray:
    u1 = pd.Series(a)
    u2 = pd.Series(b)
    raw = make_pair_features(u1, u2, noise_std=0.0)
    return np.array([[raw[f] for f in FEATURE_NAMES]])


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "ok",
        "models" : "loaded" if model else "not loaded",
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
    {
      "user_a": { "domain":"AI/ML", "goal":"Find co-founder",
                  "experience":4, "comm_style":"Direct", "location":"Bangalore" },
      "user_b": { "domain":"Data Science", "goal":"Learn & grow",
                  "experience":3, "comm_style":"Collaborative", "location":"Bangalore" }
    }

    Response:
    {
      "score": 74.2,
      "verdict": "Good Match",
      "top_signal": "domain_sim",
      "features": { "domain_sim":1.0, "goal_sim":1.0, ... },
      "poly_terms": { "domain × goal":1.0, ... }
    }
    """
    if not model:
        return jsonify({"error": "Model not loaded. Run main.py first."}), 503

    data = request.get_json(silent=True)
    if not data or "user_a" not in data or "user_b" not in data:
        return jsonify({"error": "Provide user_a and user_b in JSON body."}), 400

    user_a = data["user_a"]
    user_b = data["user_b"]

    required = ["domain", "goal", "experience", "comm_style", "location"]
    for field in required:
        if field not in user_a or field not in user_b:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # Build features
    u1 = pd.Series(user_a)
    u2 = pd.Series(user_b)
    raw_feats = make_pair_features(u1, u2, noise_std=0.0)
    x_raw    = np.array([[raw_feats[f] for f in FEATURE_NAMES]])
    x_scaled = preprocessor.transform(x_raw)

    prob  = float(model.predict_proba(x_scaled)[0])
    score = round(prob * 100, 1)

    verdict = (
        "Good Match"        if score >= SCORE_GOOD   else
        "Possible Match"    if score >= SCORE_MEDIUM else
        "Low Compatibility"
    )

    top_signal = FEATURE_NAMES[
        int(np.argmax([raw_feats[f] for f in FEATURE_NAMES]))
    ]

    poly_terms = {
        "domain × goal" : round(raw_feats["domain_sim"] * raw_feats["goal_sim"], 3),
        "exp × comm"    : round(raw_feats["exp_sim"]    * raw_feats["comm_sim"], 3),
        "domain²"       : round(raw_feats["domain_sim"] ** 2, 3),
        "goal × comm"   : round(raw_feats["goal_sim"]   * raw_feats["comm_sim"], 3),
    }

    return jsonify({
        "score"      : score,
        "verdict"    : verdict,
        "top_signal" : top_signal,
        "features"   : {f: round(raw_feats[f], 3) for f in FEATURE_NAMES},
        "poly_terms" : poly_terms,
    })


@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    """
    GET /feature-importance
    Response: [ { "feature": "goal_sim", "importance": 22.1 }, ... ]
    """
    if not model:
        return jsonify({"error": "Model not loaded."}), 503

    _compute_metrics_once()
    return jsonify(_cached_importances)


@app.route("/metrics", methods=["GET"])
def metrics():
    """
    GET /metrics
    Response:
    {
      "tp":195, "tn":347, "fp":34, "fn":24,
      "accuracy":90.3, "precision":85.2, "recall":89.0, "f1":87.1
    }
    """
    if not model:
        return jsonify({"error": "Model not loaded."}), 503

    _compute_metrics_once()
    return jsonify(_cached_metrics)


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*52)
    print("  MeetMux Flask API")
    print("  http://localhost:5000")
    print("  Endpoints: /predict  /feature-importance  /metrics")
    print("═"*52 + "\n")
    app.run(debug=True, port=5000)
