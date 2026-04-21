"""
evaluator.py
============
Everything related to model evaluation:
  - Confusion Matrix (printed + plotted)
  - Classification Report
  - Accuracy / Precision / Recall / F1
  - Feature Importance bar chart

Usage:
    from evaluator import Evaluator
    ev = Evaluator(model, preprocessor)
    ev.evaluate(X_test, y_test)
    ev.plot_all()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from config import OUTPUTS_DIR, PALETTE


class Evaluator:
    """
    Evaluates a trained MatcherModel and produces reports + plots.

    Parameters
    ----------
    model        : trained MatcherModel instance
    preprocessor : fitted Preprocessor instance
    """

    def __init__(self, model, preprocessor):
        self.model  = model
        self.prep   = preprocessor
        self._metrics = {}   # populated after evaluate()
        self._y_test  = None
        self._y_pred  = None

    # ── Main entry point ───────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Run predictions on test set, print a full report, and
        store metrics for later plotting.

        Returns
        -------
        dict with accuracy, precision, recall, f1, confusion_matrix
        """
        self._y_test = y_test
        self._y_pred = self.model.predict(X_test)

        cm = confusion_matrix(y_test, self._y_pred)
        tn, fp, fn, tp = cm.ravel()

        self._metrics = {
            "accuracy" : round(accuracy_score(y_test, self._y_pred) * 100, 1),
            "precision": round(precision_score(y_test, self._y_pred) * 100, 1),
            "recall"   : round(recall_score(y_test, self._y_pred) * 100, 1),
            "f1"       : round(f1_score(y_test, self._y_pred) * 100, 1),
            "cm"       : cm, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        }

        self._print_report()
        return self._metrics

    #Plotting

    def plot_all(self, save: bool = True) -> None:
        """
        Produce a 3-panel figure:
          Panel 1 — Confusion Matrix heatmap
          Panel 2 — Feature Importance bar chart
          Panel 3 — Metric summary cards
        """
        if not self._metrics:
            raise RuntimeError("Call evaluate() before plot_all().")

        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(
            "MeetMux Networking Matcher — Model Evaluation",
            fontsize=15, fontweight="bold", y=1.02,
        )
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        self._plot_confusion_matrix(fig.add_subplot(gs[0]))
        self._plot_feature_importance(fig.add_subplot(gs[1]))
        self._plot_metric_cards(fig.add_subplot(gs[2]))

        plt.tight_layout()
        if save:
            path = f"{OUTPUTS_DIR}/evaluation.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[evaluator] Plot saved → {path}")
        plt.show()

    # Individual plots

    def _plot_confusion_matrix(self, ax):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=self._metrics["cm"],
            display_labels=["No Match", "Match"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", fontweight="bold")

        tn = self._metrics["TN"]
        fp = self._metrics["FP"]
        fn = self._metrics["FN"]
        tp = self._metrics["TP"]
        ax.set_xlabel(
            f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
            fontsize=9, color="gray",
        )

    def _plot_feature_importance(self, ax):
        fi = self.model.feature_importances(self.prep.feature_names, top_n=10)
        colors = [PALETTE["primary"]] * len(fi)

        ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors)
        ax.set_title("Top 10 Feature Importances", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    def _plot_metric_cards(self, ax):
        ax.axis("off")
        metrics = [
            ("Accuracy",  self._metrics["accuracy"],  PALETTE["primary"]),
            ("Precision", self._metrics["precision"], PALETTE["secondary"]),
            ("Recall",    self._metrics["recall"],    PALETTE["accent"]),
            ("F1 Score",  self._metrics["f1"],        PALETTE["warn"]),
        ]
        for i, (label, value, color) in enumerate(metrics):
            y_pos = 0.85 - i * 0.22
            ax.text(0.1, y_pos,       label,         fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.text(0.1, y_pos - 0.1, f"{value}%",  fontsize=22,
                    fontweight="bold", color=color,
                    transform=ax.transAxes)

        ax.set_title("Metrics Summary", fontweight="bold")

    #Console report

    def _print_report(self):
        m = self._metrics
        print("\n--------Evaluation Report-------")
        print(classification_report(
            self._y_test, self._y_pred,
            target_names=["No Match", "Match"],
        ))
        print(f"   TP : {m['TP']}   TN : {m['TN']}")
        print(f"   FP : {m['FP']}   FN : {m['FN']}")
        print(f"\n   Accuracy  : {m['accuracy']}%")
        print(f"   Precision : {m['precision']}%")
        print(f"   Recall    : {m['recall']}%")
        print(f"   F1 Score  : {m['f1']}%")

# Quick test
if __name__ == "__main__":
    from data_generator import generate_users, generate_pairs
    from preprocessor   import Preprocessor
    from model          import MatcherModel

    users = generate_users()
    pairs = generate_pairs(users)

    prep  = Preprocessor()
    X_tr, X_te, y_tr, y_te = prep.fit_transform(pairs)

    mdl = MatcherModel()
    mdl.train(X_tr, y_tr)

    ev = Evaluator(mdl, prep)
    ev.evaluate(X_te, y_te)
    ev.plot_all()
