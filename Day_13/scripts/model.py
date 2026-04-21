"""
model.py
========
Wraps the RandomForestClassifier:
  - Training
  - Prediction + probability scoring
  - Model persistence (save / load)
  - Feature importance extraction

Usage:
    from model import MatcherModel
    m = MatcherModel()
    m.train(X_train, y_train)
    preds = m.predict(X_test)
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from config import RF_PARAMS, MODELS_DIR, SCORE_GOOD, SCORE_MEDIUM


class MatcherModel:
    """
    Random Forest–based compatibility classifier for event attendees.

    Methods
    train(X, y)— fit the classifier
    predict(X)— return class labels (0 / 1)
    predict_proba(X) — return match probability (0–1)
    feature_importances(names) — return sorted DataFrame of importances
    save() / load()          — persist to disk
    """

    def __init__(self):
        self.clf     = RandomForestClassifier(**RF_PARAMS)
        self._trained = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Random Forest on training data."""
        print("\n-----raining RandomForestClassifier---- ")
        print(f"   n_estimators : {RF_PARAMS['n_estimators']}")
        print(f"   max_depth    : {RF_PARAMS['max_depth']}")
        print(f"   Train size   : {len(X_train)}")

        self.clf.fit(X_train, y_train)
        self._trained = True

        oob_possible = RF_PARAMS.get("oob_score", False)
        print(f"   Training done {'(OOB score enabled)' if oob_possible else ''}")

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_trained()
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of class 1 (Match) for each sample."""
        self._check_trained()
        return self.clf.predict_proba(X)[:, 1]

    def predict_pair_score(self, x_scaled: np.ndarray) -> dict:
        """
        High-level prediction for a single pair (already scaled).
        Returns a verdict dict with score (0-100), label, and colour hint.
        """
        prob  = float(self.predict_proba(x_scaled.reshape(1, -1))[0])
        score = round(prob * 100, 1)

        if score >= SCORE_GOOD:
            verdict, colour = "Good Match",       "green"
        elif score >= SCORE_MEDIUM:
            verdict, colour = "Possible Match",   "amber"
        else:
            verdict, colour = "Low Compatibility","red"

        return {"score": score, "verdict": verdict, "colour": colour}

    # ── Feature Importance ─────────────────────────────────────────────────────

    def feature_importances(self, feature_names: list,
                            top_n: int = 10) -> pd.DataFrame:
        """
        Return a sorted DataFrame of the top_n most important features.

        Parameters
        ----------
        feature_names : list of strings (from Preprocessor.feature_names)
        top_n         : how many features to return

        Returns
        -------
        pd.DataFrame with columns: feature, importance
        """
        self._check_trained()
        df = pd.DataFrame({
            "feature"   : feature_names,
            "importance": self.clf.feature_importances_,
        }).sort_values("importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    # Persistence

    def save(self, path: str = None) -> None:
        path = path or f"{MODELS_DIR}/random_forest.joblib"
        joblib.dump(self.clf, path)
        print(f"[model] Saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "MatcherModel":
        path = path or f"{MODELS_DIR}/random_forest.joblib"
        obj  = cls.__new__(cls)
        obj.clf      = joblib.load(path)
        obj._trained = True
        print(f"[model] Loaded ← {path}")
        return obj

    # Internal

    def _check_trained(self):
        if not self._trained:
            raise RuntimeError(
                "Model is not trained yet. Call .train(X, y) first."
            )

# Quick test

if __name__ == "__main__":
    from data_generator import generate_users, generate_pairs
    from preprocessor   import Preprocessor

    users = generate_users()
    pairs = generate_pairs(users)

    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(pairs)

    model = MatcherModel()
    model.train(X_train, y_train)

    preds = model.predict(X_test)
    print(f"\n   Sample predictions : {preds[:10]}")
    print(f"   Sample proba       : {model.predict_proba(X_test)[:5].round(2)}")

    fi = model.feature_importances(prep.feature_names)
    print(f"\n   Top features:\n{fi.head(5).to_string(index=False)}")

    model.save()
    prep.save()
