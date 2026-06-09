"""
calibration.py — Probability Calibration for Ensemble Model (Task 3.6).

Applies two calibration methods to the raw ensemble probabilities:
    1. Platt Scaling      — fits a logistic regression on top of raw probas
    2. Isotonic Regression — non-parametric monotone fit (more flexible)

Calibration is trained on 2023 val set and evaluated on 2024+ holdout.
Calibration curves (reliability diagrams) are saved to results/.

Output:
    models/saved/ensemble_calibrated.joblib  — best calibrator + ensemble
    results/calibration_curves.png           — reliability diagrams
    results/model_comparison.csv             — updated with calibrated row

Usage:
    python -m src.models.calibration
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder

from src.features.encoder import MATCH_FEATURES_CSV
from src.models.ensemble import (
    CLASSES, HOLDOUT_YEAR, START_YEAR, VAL_YEARS,
    WeightedEnsemble, _DataBundle, _collect_probas, _load_bundles,
)
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonGoalModel
from src.models.xgboost_model import XGBoostOutcomeModel

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_CSV  = RESULTS_DIR / "model_comparison.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Calibrators
# ─────────────────────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Platt Scaling: fits one multinomial logistic regression using the raw
    ensemble probabilities as input features → calibrated probabilities.
    One-vs-rest: a separate logistic is fit per class, then normalised.
    """

    def __init__(self):
        self.models: list[LogisticRegression] = []
        self.classes = CLASSES
        self.is_fit = False

    def fit(self, proba_raw: np.ndarray, y_raw: np.ndarray) -> None:
        """
        proba_raw : (n, 3) raw probabilities
        y_raw     : (n,)  string labels in CLASSES
        """
        self.models = []
        for i, cls in enumerate(self.classes):
            binary = (y_raw == cls).astype(int)
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(proba_raw[:, i].reshape(-1, 1), binary)
            self.models.append(lr)
        self.is_fit = True

    def predict_proba(self, proba_raw: np.ndarray) -> np.ndarray:
        """Return calibrated (n, 3) probabilities, rows normalised."""
        cal = np.column_stack([
            m.predict_proba(proba_raw[:, i].reshape(-1, 1))[:, 1]
            for i, m in enumerate(self.models)
        ])
        cal = np.clip(cal, 1e-7, 1.0)
        return cal / cal.sum(axis=1, keepdims=True)


class IsotonicCalibrator:
    """
    Isotonic Regression calibration (one per class, then normalised).
    More flexible than Platt; can overfit on small val sets.
    """

    def __init__(self):
        self.models: list[IsotonicRegression] = []
        self.classes = CLASSES
        self.is_fit = False

    def fit(self, proba_raw: np.ndarray, y_raw: np.ndarray) -> None:
        self.models = []
        for i, cls in enumerate(self.classes):
            binary = (y_raw == cls).astype(float)
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(proba_raw[:, i], binary)
            self.models.append(ir)
        self.is_fit = True

    def predict_proba(self, proba_raw: np.ndarray) -> np.ndarray:
        cal = np.column_stack([
            m.transform(proba_raw[:, i]) for i, m in enumerate(self.models)
        ])
        cal = np.clip(cal, 1e-7, 1.0)
        return cal / cal.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated ensemble wrapper
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedEnsemble:
    """
    Wraps a WeightedEnsemble + one of the two calibrators.
    Exposes the same predict_proba interface as WeightedEnsemble.
    """

    def __init__(self, ensemble: WeightedEnsemble, calibrator, method: str):
        self.ensemble   = ensemble
        self.calibrator = calibrator
        self.method     = method     # "platt" or "isotonic"

    def predict_proba(self, bundle: _DataBundle) -> pd.DataFrame:
        raw_df   = self.ensemble.predict_proba(bundle)
        cal      = self.calibrator.predict_proba(raw_df.values)
        return pd.DataFrame(cal, columns=CLASSES)

    def save(self, path: Path | str | None = None) -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else MODEL_DIR / "ensemble_calibrated.joblib"
        joblib.dump({
            "ensemble_weights": self.ensemble.weights,
            "calibrator":       self.calibrator,
            "method":           self.method,
        }, out)
        log.info("Calibrated ensemble saved → %s", out)
        return out

    @classmethod
    def load(cls, ensemble: WeightedEnsemble,
             path: Path | str | None = None) -> "CalibratedEnsemble":
        src = Path(path) if path else MODEL_DIR / "ensemble_calibrated.joblib"
        data = joblib.load(src)
        ensemble.weights = data["ensemble_weights"]
        obj = cls(ensemble, data["calibrator"], data["method"])
        log.info("Calibrated ensemble loaded (method=%s)", data["method"])
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(proba: np.ndarray, y_enc: np.ndarray, y_raw: np.ndarray) -> dict:
    preds = np.argmax(proba, axis=1)
    acc   = accuracy_score(y_enc, preds)
    ll    = log_loss(y_enc, proba)
    brier = float(np.mean([
        brier_score_loss((y_raw == cls).astype(int), proba[:, i])
        for i, cls in enumerate(CLASSES)
    ]))
    return {"accuracy": round(acc, 4), "log_loss": round(ll, 4),
            "brier_score": round(brier, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# Calibration curve plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_calibration_curves(
    probas_dict: dict[str, np.ndarray],
    y_raw: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """
    Plot reliability diagrams for each outcome class and each model variant.
    probas_dict: {"Uncalibrated": array, "Platt": array, "Isotonic": array}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    class_names = {"A": "Away Win", "D": "Draw", "H": "Home Win"}
    colours = {"Uncalibrated": "#e74c3c", "Platt": "#2ecc71", "Isotonic": "#3498db"}
    styles  = {"Uncalibrated": "--",      "Platt": "-",        "Isotonic": "-."}

    for ax, (cls, cls_name) in zip(axes, class_names.items()):
        cls_idx = CLASSES.index(cls)
        binary  = (y_raw == cls).astype(int)

        for label, proba_arr in probas_dict.items():
            prob_cls = proba_arr[:, cls_idx]
            frac_pos, mean_pred = calibration_curve(binary, prob_cls,
                                                    n_bins=n_bins, strategy="uniform")
            ax.plot(mean_pred, frac_pos,
                    marker="o", linestyle=styles[label],
                    color=colours[label], label=label, linewidth=2)

        # Perfect diagonal
        ax.plot([0, 1], [0, 1], "k:", linewidth=1.5, label="Perfect")
        ax.set_title(f"Calibration — {cls_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    plt.suptitle("Reliability Diagrams — Ensemble Calibration (Holdout 2024+)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Calibration curves saved → %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration() -> CalibratedEnsemble:
    """
    Full calibration pipeline:
      1. Load val (2023) and holdout (2024+) bundles
      2. Load ensemble & component models
      3. Fit Platt and Isotonic calibrators on val probabilities
      4. Evaluate all variants on holdout
      5. Plot calibration curves
      6. Save the better calibrator as ensemble_calibrated.joblib
    """
    log.info("Loading data bundles...")
    val_bundle, hold_bundle, scaler, le = _load_bundles()
    log.info("Val: %d | Holdout: %d matches", len(val_bundle.mf), len(hold_bundle.mf))

    # Load ensemble (already optimised)
    ensemble = WeightedEnsemble.load()
    log.info("Ensemble weights: %s", ensemble.weights)

    # Load component models (needed for predict_proba)
    mf_full  = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    mf_train = mf_full[
        mf_full["is_played"] & mf_full["is_competitive"]
        & (mf_full["year"] >= START_YEAR)
        & (mf_full["year"] < VAL_YEARS[0])
    ].reset_index(drop=True)
    ensemble._load_component_models(mf_train)

    # ── Step 1: Get raw probabilities on val and holdout ──────────────────────
    log.info("Collecting raw ensemble probabilities on val set...")
    val_raw_df  = ensemble.predict_proba(val_bundle)
    val_raw     = val_raw_df.values
    val_y_raw   = val_bundle.y_raw

    log.info("Collecting raw ensemble probabilities on holdout...")
    hold_raw_df = ensemble.predict_proba(hold_bundle)
    hold_raw    = hold_raw_df.values
    hold_y_raw  = hold_bundle.y_raw
    hold_y_enc  = hold_bundle.y_enc

    # ── Step 2: Fit calibrators on val ───────────────────────────────────────
    log.info("Fitting Platt calibrator on val (n=%d)...", len(val_raw))
    platt = PlattCalibrator()
    platt.fit(val_raw, val_y_raw)

    log.info("Fitting Isotonic calibrator on val (n=%d)...", len(val_raw))
    isotonic = IsotonicCalibrator()
    isotonic.fit(val_raw, val_y_raw)

    # ── Step 3: Get calibrated holdout probabilities ──────────────────────────
    hold_platt    = platt.predict_proba(hold_raw)
    hold_isotonic = isotonic.predict_proba(hold_raw)

    # ── Step 4: Evaluate all variants ────────────────────────────────────────
    m_raw      = _metrics(hold_raw,      hold_y_enc, hold_y_raw)
    m_platt    = _metrics(hold_platt,    hold_y_enc, hold_y_raw)
    m_isotonic = _metrics(hold_isotonic, hold_y_enc, hold_y_raw)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON — Holdout 2024+")
    print("=" * 60)
    print(f"{'Method':<22} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>10}")
    print("-" * 60)
    for label, m in [("Uncalibrated", m_raw), ("Platt Scaling", m_platt),
                     ("Isotonic Reg.", m_isotonic)]:
        print(f"{label:<22} {m['accuracy']:>10.4f} {m['log_loss']:>10.4f} "
              f"{m['brier_score']:>10.4f}")
    print("=" * 60)

    # ── Step 5: Plot calibration curves ──────────────────────────────────────
    _plot_calibration_curves(
        {"Uncalibrated": hold_raw, "Platt": hold_platt, "Isotonic": hold_isotonic},
        hold_y_raw,
        save_path=RESULTS_DIR / "calibration_curves.png",
    )

    # ── Step 6: Pick the better calibrator (lower log loss) ──────────────────
    if m_platt["log_loss"] <= m_isotonic["log_loss"]:
        best_method, best_cal, best_m = "platt",    platt,    m_platt
    else:
        best_method, best_cal, best_m = "isotonic", isotonic, m_isotonic

    log.info("Best calibrator: %s (log_loss=%.4f)", best_method, best_m["log_loss"])

    # ── Step 7: Update model_comparison.csv ──────────────────────────────────
    new_row = pd.DataFrame([{
        "model":          "ensemble_calibrated",
        "accuracy":       best_m["accuracy"],
        "log_loss":       best_m["log_loss"],
        "brier_score":    best_m["brier_score"],
        "train_matches":  int((mf_full["year"] < VAL_YEARS[0]).sum()),
        "test_matches":   len(hold_bundle.mf),
        "holdout_year":   HOLDOUT_YEAR,
    }])
    existing = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    if not existing.empty:
        existing = existing[existing["model"] != "ensemble_calibrated"]
    results = pd.concat([existing, new_row], ignore_index=True).sort_values("log_loss")
    results.to_csv(RESULTS_CSV, index=False)
    log.info("Updated model_comparison.csv")

    # ── Step 8: Save ─────────────────────────────────────────────────────────
    cal_ensemble = CalibratedEnsemble(ensemble, best_cal, best_method)
    cal_ensemble.save()

    return cal_ensemble


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 3.6 — Probability Calibration")
    log.info("=" * 60)

    cal_ensemble = run_calibration()

    # Final comparison table
    results = pd.read_csv(RESULTS_CSV)
    print("\n" + "=" * 65)
    print("FULL MODEL COMPARISON (updated)")
    print("=" * 65)
    print(results[["model", "accuracy", "log_loss", "brier_score"]].to_string(index=False))
    print("=" * 65)
    print(f"\nCalibration curves → {RESULTS_DIR / 'calibration_curves.png'}")
    print(f"Calibrated model   → {MODEL_DIR / 'ensemble_calibrated.joblib'}")
    print("\nTask 3.6 complete.")


if __name__ == "__main__":
    main()
