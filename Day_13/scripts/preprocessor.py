"""
preprocessor.py
===============
Handles all data preprocessing steps:
  - Selecting feature columns
  - Applying PolynomialFeatures (non-linear interaction terms)
  - Scaling with StandardScaler
  - Train / test split

Usage:
    from preprocessor import Preprocessor
    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(pairs_df)
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_NAMES, POLY_DEGREE,
    TEST_SIZE, RANDOM_SEED, MODELS_DIR,
)


class Preprocessor:
    """
    Encapsulates feature engineering and splitting logic.

    Attributes
    ----------
    poly        : fitted PolynomialFeatures transformer
    scaler      : fitted StandardScaler
    feature_names : list of feature names after polynomial expansion
    """

    def __init__(self, degree: int = POLY_DEGREE):
        self.degree   = degree
        self.poly     = PolynomialFeatures(
            degree=degree,
            interaction_only=False,
            include_bias=False,
        )
        self.scaler   = StandardScaler()
        self.feature_names = None

    #  Main interface

    def fit_transform(self, pairs: pd.DataFrame):
        """
        Fit the pipeline on pairs data and return train/test splits.

        Parameters
        ----------
        pairs : pd.DataFrame — must contain FEATURE_NAMES + 'label'

        Returns
        -------
        X_train, X_test, y_train, y_test (all numpy arrays)
        """
        X_raw = pairs[FEATURE_NAMES].values
        y     = pairs["label"].values

        X_poly  = self.poly.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_poly)

        self.feature_names = list(
            self.poly.get_feature_names_out(FEATURE_NAMES)
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        self._print_summary(X_raw.shape[1], X_poly.shape[1],
                            len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Transform new data using already-fitted poly + scaler.
        Call this at inference time (not fit_transform).
        """
        X_poly   = self.poly.transform(X_raw)
        X_scaled = self.scaler.transform(X_poly)
        return X_scaled

    # Persistence

    def save(self, path: str = None) -> None:
        path = path or f"{MODELS_DIR}/preprocessor.joblib"
        joblib.dump({"poly": self.poly, "scaler": self.scaler,
                     "feature_names": self.feature_names}, path)
        print(f"[preprocessor] Saved → {path}")

    @classmethod
    def load(cls, path: str = None) -> "Preprocessor":
        path = path or f"{MODELS_DIR}/preprocessor.joblib"
        data = joblib.load(path)
        obj  = cls.__new__(cls)
        obj.poly          = data["poly"]
        obj.scaler        = data["scaler"]
        obj.feature_names = data["feature_names"]
        obj.degree        = obj.poly.degree
        print(f"[preprocessor] Loaded ← {path}")
        return obj

    # Internal

    @staticmethod
    def _print_summary(n_orig, n_poly, n_train, n_test):
        print("\n── Preprocessor ────────────────────────────────")
        print(f"   Original features   : {n_orig}")
        print(f"   After poly (deg={POLY_DEGREE})  : {n_poly}")
        print(f"   Train samples       : {n_train}")
        print(f"   Test  samples       : {n_test}")

# Quick test
if __name__ == "__main__":
    from data_generator import generate_users, generate_pairs

    users = generate_users()
    pairs = generate_pairs(users)

    prep = Preprocessor()
    X_train, X_test, y_train, y_test = prep.fit_transform(pairs)

    print(f"\n   X_train shape : {X_train.shape}")
    print(f"   Feature names : {prep.feature_names[:6]} ...")
    prep.save()
