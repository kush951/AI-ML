"""
predictor.py
============
High-level interface for predicting compatibility between two attendees.
Loads saved model + preprocessor and exposes a clean predict() function.

Usage:
    from predictor import Predictor
    p = Predictor()
    result = p.predict(user_a, user_b)
    print(result)
"""

import numpy as np
import pandas as pd

from config        import FEATURE_NAMES, SCORE_GOOD, SCORE_MEDIUM
from data_generator import make_pair_features
from preprocessor  import Preprocessor
from model         import MatcherModel


class Predictor:
    """
    End-to-end compatibility predictor for a pair of attendees.

    Parameters
    ----------
    model        : trained MatcherModel  (or loads from disk if None)
    preprocessor : fitted Preprocessor  (or loads from disk if None)
    """

    def __init__(self, model: MatcherModel = None,
                 preprocessor: Preprocessor = None):
        self.model = model or MatcherModel.load()
        self.prep  = preprocessor or Preprocessor.load()

    # ── Main consol interface

    def predict(self, user_a: dict, user_b: dict) -> dict:
        """
        Predict compatibility for two attendee profiles.

        Parameters
        ----------
        user_a, user_b : dicts with keys:
            domain, goal, experience (int), comm_style, location

        Returns
        -------
        dict:
            score       — float 0-100
            verdict     — "Good Match" / "Possible Match" / "Low Compatibility"
            top_signal  — name of the highest-scoring feature
            features    — dict of individual feature scores
            poly_terms  — key interaction terms
        """
        u1 = pd.Series(user_a)
        u2 = pd.Series(user_b)

        raw_feats = make_pair_features(u1, u2, noise_std=0.0)
        feat_vals = np.array([[raw_feats[f] for f in FEATURE_NAMES]])

        x_scaled = self.prep.transform(feat_vals)
        prob      = float(self.model.predict_proba(x_scaled)[0])
        score     = round(prob * 100, 1)

        verdict = (
            "Good Match"       if score >= SCORE_GOOD   else
            "Possible Match"   if score >= SCORE_MEDIUM else
            "Low Compatibility"
        )

        top_signal = FEATURE_NAMES[
            int(np.argmax([raw_feats[f] for f in FEATURE_NAMES]))
        ]

        poly_terms = {
            "domain × goal"  : round(raw_feats["domain_sim"] * raw_feats["goal_sim"], 3),
            "exp × comm"     : round(raw_feats["exp_sim"]    * raw_feats["comm_sim"], 3),
            "domain²"        : round(raw_feats["domain_sim"] ** 2, 3),
            "goal × comm"    : round(raw_feats["goal_sim"]   * raw_feats["comm_sim"], 3),
        }

        return {
            "score"      : score,
            "verdict"    : verdict,
            "top_signal" : top_signal,
            "features"   : {f: round(raw_feats[f], 3) for f in FEATURE_NAMES},
            "poly_terms" : poly_terms,
        }

    def predict_batch(self, pairs: list[tuple[dict, dict]]) -> list[dict]:
        """Predict compatibility for a list of (user_a, user_b) tuples."""
        return [self.predict(a, b) for a, b in pairs]

    #  Pretty print

    @staticmethod
    def print_result(user_a: dict, user_b: dict, result: dict) -> None:
        width = 52
        line  = "─" * width

        print(f"\n┌{line}┐")
        print(f"│{'  MeetMux Compatibility Report':^{width}}│")
        print(f"├{line}┤")
        print(f"│  {'Attendee A':<20} {user_a.get('domain','?'):<28}│")
        print(f"│  {'Attendee B':<20} {user_b.get('domain','?'):<28}│")
        print(f"├{line}┤")
        print(f"│  Score    : {result['score']:>5.1f} / 100{'':<28}│")
        print(f"│  Verdict  : {result['verdict']:<39}│")
        print(f"│  Top signal: {result['top_signal']:<38}│")
        print(f"├{line}┤")
        print(f"│  Feature breakdown:{'':<32}│")
        for name, val in result["features"].items():
            bar = "█" * int(val * 15)
            print(f"│    {name:<15} {bar:<16} {val:.2f}          │")
        print(f"├{line}┤")
        print(f"│  Polynomial interaction terms:{'':<22}│")
        for term, val in result["poly_terms"].items():
            print(f"│    {term:<20} {val:.3f}{'':<21}│")
        print(f"└{line}┘")

# Quick test (run standalone — needs saved model + preprocessor)

if __name__ == "__main__":
    # Train fresh if no saved models exist
    try:
        p = Predictor()
    except Exception:
        print("No saved model found — training now...")
        from data_generator import generate_users, generate_pairs
        from preprocessor   import Preprocessor
        from model          import MatcherModel

        users = generate_users()
        pairs = generate_pairs(users)
        prep  = Preprocessor()
        X_tr, X_te, y_tr, y_te = prep.fit_transform(pairs)
        mdl   = MatcherModel()
        mdl.train(X_tr, y_tr)
        mdl.save(); prep.save()
        p = Predictor(mdl, prep)

    #  Example prediction
    alice = {"domain": "AI/ML",       "goal": "Find co-founder",
             "experience": 4,          "comm_style": "Direct",
             "location": "Bangalore"}

    ravi  = {"domain": "Data Science", "goal": "Learn & grow",
             "experience": 3,           "comm_style": "Collaborative",
             "location": "Bangalore"}

    result = p.predict(alice, ravi)
    Predictor.print_result(alice, ravi, result)
