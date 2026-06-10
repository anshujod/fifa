"""
metrics.py — Evaluation framework for the FIFA WC 2026 prediction system.

Metrics
-------
Accuracy      : percentage of correctly predicted top-1 outcomes
Log Loss      : cross-entropy; heavily penalises confident wrong predictions
Brier Score   : mean squared probability error averaged over all 3 outcomes
RPS           : Ranked Probability Score (standard metric in sports forecasting)
RPSS          : RPS Skill Score vs uniform-1/3 baseline (positive = better)

Holdout sets
------------
⚠️  IN-SAMPLE (leakage) — WC 2014, 2018, 2022 were included in model training.
    These numbers are optimistic (especially tree models). Use for reference only.

✅  OUT-OF-SAMPLE — All competitive matches 2023-2024 (~1,623 matches).
    Models were trained on data ≤ 2022; these are genuinely unseen.
    No WC matches exist yet; covers UEFA Euros, Copa América, Nations League, etc.

Benchmark targets (task spec)
------------------------------
Log Loss < 0.85   (ambitious; true OOS football ≈ 0.96–1.00)
RPS      < 0.19   (typical good model ≈ 0.20–0.22 on WC matches)

Models evaluated
----------------
ensemble            : XGB + LGBM + ELO blended, isotonically calibrated
xgboost             : standalone XGBClassifier
lightgbm            : standalone LGBMClassifier
logistic_regression : logistic regression on same 37-feature set
random_forest       : random forest on same 37-feature set
elo                 : ELO-only model (elo_diff → logistic)
baseline_uniform    : constant 1/3 per outcome
baseline_historical : empirical H/D/A frequencies from training data
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODEL_DIR     = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR   = PROJECT_ROOT / "results"
FEATURES_CSV  = PROJECT_ROOT / "data" / "processed" / "match_features.csv"

# Class ordering used throughout (alphabetical, matches LabelEncoder.classes_)
CLASSES_ADH = ["A", "D", "H"]   # probability column order
CLASSES_HDA = ["H", "D", "A"]   # display order

# Benchmark thresholds from task spec
BENCHMARK_LOG_LOSS = 0.85
BENCHMARK_RPS      = 0.19


# ─────────────────────────────────────────────────────────────────────────────
# Score functions
# ─────────────────────────────────────────────────────────────────────────────

def rps_single(probs_h: float, probs_d: float, probs_a: float,
               outcome: str) -> float:
    """
    Ranked Probability Score for a single match (3-outcome football).

    Convention: outcomes ordered H > D > A (home win, draw, away win).

    RPS = (1/(K-1)) * Σ_{j=1}^{K-1} (cum_p_j - cum_y_j)^2
        = 0.5 * [(p_H - y_H)^2 + ((p_H+p_D) - (y_H+y_D))^2]
    """
    y_h = 1.0 if outcome == "H" else 0.0
    y_d = 1.0 if outcome == "D" else 0.0
    cum_p1 = probs_h
    cum_p2 = probs_h + probs_d
    cum_y1 = y_h
    cum_y2 = y_h + y_d
    return 0.5 * ((cum_p1 - cum_y1) ** 2 + (cum_p2 - cum_y2) ** 2)


def rps_array(proba: np.ndarray, y_str: np.ndarray) -> np.ndarray:
    """
    Vectorised RPS for an array of predictions.

    Parameters
    ----------
    proba  : (N, 3) array of probabilities in [A, D, H] column order
    y_str  : (N,)  array of string outcomes ('A', 'D', 'H')

    Returns
    -------
    (N,) array of per-match RPS values
    """
    p_h = proba[:, 2]   # column 2 = H (alphabetical)
    p_d = proba[:, 1]   # column 1 = D
    p_a = proba[:, 0]   # column 0 = A

    y_h = (y_str == "H").astype(float)
    y_d = (y_str == "D").astype(float)

    cum_p1 = p_h
    cum_p2 = p_h + p_d
    cum_y1 = y_h
    cum_y2 = y_h + y_d

    return 0.5 * ((cum_p1 - cum_y1) ** 2 + (cum_p2 - cum_y2) ** 2)


def brier_score_multi(proba: np.ndarray, y_int: np.ndarray) -> float:
    """
    Multi-class Brier score: mean over matches of sum over classes of (p - y)^2.

    Parameters
    ----------
    proba  : (N, K) probability array
    y_int  : (N,)   integer class labels in [0, K-1]
    """
    n, k = proba.shape
    y_onehot = np.zeros((n, k), dtype=float)
    y_onehot[np.arange(n), y_int] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def rps_skill_score(rps: float, rps_ref: float) -> float:
    """RPSS = 1 - RPS / RPS_ref.  Positive = better than reference."""
    if rps_ref == 0:
        return 0.0
    return 1.0 - rps / rps_ref


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name:     str
    n_matches:      int
    accuracy:       float
    log_loss:       float
    brier_score:    float
    rps:            float
    rps_skill_score: float           # vs uniform baseline
    beats_log_loss_target: bool
    beats_rps_target:      bool
    # per-class metrics
    class_accuracy: dict[str, float] = field(default_factory=dict)
    # optional breakdown
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series({
            "Model":          self.model_name,
            "N Matches":      self.n_matches,
            "Accuracy":       f"{self.accuracy*100:.2f}%",
            "Log Loss":       f"{self.log_loss:.4f}",
            "Brier Score":    f"{self.brier_score:.4f}",
            "RPS":            f"{self.rps:.4f}",
            "RPSS":           f"{self.rps_skill_score:+.4f}",
            "✅ Log Loss":    "✅" if self.beats_log_loss_target else "❌",
            "✅ RPS":         "✅" if self.beats_rps_target else "❌",
        })


# ─────────────────────────────────────────────────────────────────────────────
# ModelEvaluator
# ─────────────────────────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Centralised evaluation framework for the FIFA WC 2026 prediction models.

    Usage
    -----
    >>> ev = ModelEvaluator()
    >>> X_test, y_str, meta = ev.load_holdout(years=[2014, 2018, 2022])
    >>> results = ev.evaluate_all(X_test, y_str)
    >>> ev.print_comparison_table(results)
    """

    # models loaded lazily
    _xgb_model   = None
    _lgbm_model  = None
    _lr_model    = None
    _rf_model    = None
    _elo_model   = None
    _calibrator  = None
    _ensemble_w  = None

    def __init__(self) -> None:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        self._le_classes: list[str] = CLASSES_ADH   # LabelEncoder order

    # ── lazy model loading ────────────────────────────────────────────────────

    def _ensure_models_loaded(self) -> None:
        """Load all component models if not already cached."""
        if self._xgb_model is not None:
            return

        import joblib
        log.info("Loading component models from %s", MODEL_DIR)

        xgb_d   = joblib.load(MODEL_DIR / "xgboost_outcome.joblib")
        lgbm_d  = joblib.load(MODEL_DIR / "lightgbm.joblib")
        lr_d    = joblib.load(MODEL_DIR / "logistic_regression.joblib")
        rf_d    = joblib.load(MODEL_DIR / "random_forest.joblib")
        ens_cal = joblib.load(MODEL_DIR / "ensemble_calibrated.joblib")
        ens_raw = joblib.load(MODEL_DIR / "ensemble.joblib")

        self._xgb_model  = xgb_d["model"]
        self._xgb_le     = xgb_d["label_encoder"]

        self._lgbm_model = lgbm_d["model"]
        self._lgbm_le    = lgbm_d["label_encoder"]

        self._lr_model   = lr_d["model"]
        self._lr_le      = lr_d["label_encoder"]
        self._lr_scaler  = lr_d["scaler"]

        self._rf_model   = rf_d["model"]
        self._rf_le      = rf_d["label_encoder"]

        self._calibrator = ens_cal["calibrator"]
        self._ensemble_w = ens_raw["weights"]   # {elo, xgb, lgbm, poisson}

        # ELO model — load via MatchPredictor
        from src.simulation.match_predictor import MatchPredictor
        mp = MatchPredictor.load()
        self._elo_model = mp.elo_model

        log.info("All models loaded.")

    # ── prediction ───────────────────────────────────────────────────────────

    def _reorder_proba(
        self, proba: np.ndarray, source_classes: list[str]
    ) -> np.ndarray:
        """Reorder columns so output is always [A, D, H] (CLASSES_ADH)."""
        col_order = [source_classes.index(c) for c in CLASSES_ADH]
        return proba[:, col_order]

    def predict(
        self, model_name: str, X: pd.DataFrame,
        historical_hda_rates: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """
        Predict outcome probabilities for model `model_name`.

        Returns
        -------
        (N, 3) array of probabilities, columns = [A, D, H]
        """
        self._ensure_models_loaded()
        n = len(X)

        if model_name == "xgboost":
            raw = self._xgb_model.predict_proba(X)
            return self._reorder_proba(raw, list(self._xgb_le.classes_))

        if model_name == "lightgbm":
            raw = self._lgbm_model.predict_proba(X)
            return self._reorder_proba(raw, list(self._lgbm_le.classes_))

        if model_name == "logistic_regression":
            X_sc = self._lr_scaler.transform(X)
            raw  = self._lr_model.predict_proba(X_sc)
            return self._reorder_proba(raw, list(self._lr_le.classes_))

        if model_name == "random_forest":
            raw = self._rf_model.predict_proba(X)
            return self._reorder_proba(raw, list(self._rf_le.classes_))

        if model_name == "elo":
            elo_diff_col = pd.DataFrame({"elo_diff": X["elo_diff"].values})
            raw = self._elo_model.predict_proba(elo_diff_col)
            return self._reorder_proba(raw, list(self._elo_model.calibrator.classes_))

        if model_name == "ensemble":
            return self._predict_ensemble(X)

        if model_name == "baseline_uniform":
            return np.full((n, 3), 1.0 / 3.0)

        if model_name == "baseline_historical":
            if historical_hda_rates is None:
                raise ValueError("Pass historical_hda_rates for the historical baseline.")
            p_h, p_d, p_a = historical_hda_rates
            return np.tile([p_a, p_d, p_h], (n, 1))

        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {', '.join(self.model_names)}")

    def _predict_ensemble(self, X: pd.DataFrame) -> np.ndarray:
        """
        Blend XGB + LGBM + ELO using saved weights, then apply isotonic calibration.
        Poisson component is skipped (requires live team data); weights are renormalised.
        """
        w = self._ensemble_w          # {elo, xgb, lgbm, poisson}
        w_total = w["elo"] + w["xgb"] + w["lgbm"]   # exclude poisson
        w_elo  = w["elo"]  / w_total
        w_xgb  = w["xgb"]  / w_total
        w_lgbm = w["lgbm"] / w_total

        p_xgb  = self.predict("xgboost",  X)    # (N, 3) A,D,H
        p_lgbm = self.predict("lightgbm", X)
        p_elo  = self.predict("elo",      X)

        blended = w_xgb * p_xgb + w_lgbm * p_lgbm + w_elo * p_elo
        blended = np.abs(blended) / np.abs(blended).sum(axis=1, keepdims=True)

        # Calibration — calibrator expects [A, D, H] order
        calibrated = self._calibrator.predict_proba(blended)
        calibrated = np.clip(calibrated, 1e-8, 1.0)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated

    # ── core evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self,
        model_name: str,
        X_test: pd.DataFrame,
        y_str: np.ndarray,
        historical_hda_rates: tuple[float, float, float] | None = None,
    ) -> EvalResult:
        """
        Evaluate a single model on a labelled dataset.

        Parameters
        ----------
        model_name           : one of self.model_names
        X_test               : feature DataFrame
        y_str                : outcome strings ('H', 'D', 'A')
        historical_hda_rates : (p_H, p_D, p_A) required for baseline_historical

        Returns
        -------
        EvalResult dataclass with all metrics
        """
        proba = self.predict(model_name, X_test,
                             historical_hda_rates=historical_hda_rates)

        # Integer labels (A=0, D=1, H=2)
        le_map = {c: i for i, c in enumerate(CLASSES_ADH)}
        y_int  = np.array([le_map[o] for o in y_str])

        # Top-1 predictions
        y_pred = np.array([CLASSES_ADH[i] for i in np.argmax(proba, axis=1)])

        acc   = float(accuracy_score(y_str, y_pred))
        ll    = float(log_loss(y_int, proba, labels=list(range(3))))
        brier = float(brier_score_multi(proba, y_int))
        rps_v = float(np.mean(rps_array(proba, y_str)))

        # RPS skill score vs uniform baseline
        unif  = np.full_like(proba, 1.0 / 3.0)
        rps_u = float(np.mean(rps_array(unif, y_str)))
        rpss  = rps_skill_score(rps_v, rps_u)

        # Per-outcome accuracy
        class_acc = {}
        for outcome in ["H", "D", "A"]:
            mask = y_str == outcome
            if mask.any():
                class_acc[outcome] = float(accuracy_score(
                    y_str[mask], y_pred[mask]
                ))

        return EvalResult(
            model_name     = model_name,
            n_matches      = len(y_str),
            accuracy       = acc,
            log_loss       = ll,
            brier_score    = brier,
            rps            = rps_v,
            rps_skill_score = rpss,
            beats_log_loss_target = ll < BENCHMARK_LOG_LOSS,
            beats_rps_target      = rps_v < BENCHMARK_RPS,
            class_accuracy = class_acc,
        )

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_str:  np.ndarray,
    ) -> list[EvalResult]:
        """Evaluate all models on the same holdout and return list of EvalResults."""
        # Compute historical base rates from pre-2014 competitive matches
        hda = self._historical_base_rates(end_year_excl=2014)

        results = []
        for name in self.model_names:
            log.info("Evaluating %s …", name)
            try:
                r = self.evaluate(name, X_test, y_str, historical_hda_rates=hda)
                results.append(r)
            except Exception as exc:
                log.warning("Skipping %s — %s", name, exc)
        return results

    # ── time-series evaluation ────────────────────────────────────────────────

    def evaluate_time_series(
        self,
        model_name: str = "ensemble",
        tournament_filter: str = "FIFA World Cup",
        start_year: int = 2010,
        end_year:   int = 2023,
    ) -> pd.DataFrame:
        """
        Year-by-year evaluation on all competitive matches (or filtered tournament).

        Returns a DataFrame with columns [year, n_matches, accuracy, log_loss, brier, rps].
        """
        self._ensure_models_loaded()
        rows = []
        for year in range(start_year, end_year + 1):
            try:
                X_y, y_y, meta = self.load_holdout(
                    tournament_type=tournament_filter,
                    years=[year],
                )
                if len(X_y) == 0:
                    continue
                r = self.evaluate(model_name, X_y, y_y)
                rows.append({
                    "Year":       year,
                    "N Matches":  r.n_matches,
                    "Accuracy":   round(r.accuracy * 100, 2),
                    "Log Loss":   round(r.log_loss, 4),
                    "Brier":      round(r.brier_score, 4),
                    "RPS":        round(r.rps, 4),
                    "RPSS":       round(r.rps_skill_score, 4),
                })
            except Exception as exc:
                log.debug("Year %d skipped: %s", year, exc)
        return pd.DataFrame(rows)

    # ── calibration analysis ──────────────────────────────────────────────────

    def calibration_analysis(
        self,
        model_name: str,
        X_test:  pd.DataFrame,
        y_str:   np.ndarray,
        n_bins:  int = 10,
    ) -> dict:
        """
        Reliability diagram data: compare predicted probabilities to observed
        frequencies in equal-width bins.

        Returns dict with keys: bins, mean_predicted, fraction_positive, counts
        (one entry per outcome class)
        """
        proba = self.predict(model_name, X_test)
        results: dict[str, dict] = {}

        for i, outcome in enumerate(CLASSES_ADH):
            p_o    = proba[:, i]
            actual = (y_str == outcome).astype(float)

            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            mean_pred  = []
            frac_pos   = []
            counts     = []

            for bin_i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                mask = (p_o >= lo) & (p_o < hi)
                if bin_i == n_bins - 1:   # include right edge in last bin
                    mask = (p_o >= lo) & (p_o <= hi)
                c = int(mask.sum())
                counts.append(c)
                mean_pred.append(float(p_o[mask].mean()) if c > 0 else (lo + hi) / 2)
                frac_pos.append(float(actual[mask].mean()) if c > 0 else float("nan"))

            results[outcome] = {
                "mean_predicted": mean_pred,
                "fraction_positive": frac_pos,
                "counts": counts,
                "bin_edges": list(bin_edges),
            }

        return results

    # ── confidence breakdown ──────────────────────────────────────────────────

    def confidence_breakdown(
        self,
        model_name: str,
        X_test:  pd.DataFrame,
        y_str:   np.ndarray,
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Accuracy stratified by prediction confidence (max probability).

        Shows: when the model is "confident" (high max prob), is it also accurate?
        """
        if thresholds is None:
            thresholds = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

        proba   = self.predict(model_name, X_test)
        max_p   = proba.max(axis=1)
        y_pred  = np.array([CLASSES_ADH[i] for i in np.argmax(proba, axis=1)])

        rows = []
        for thr in thresholds:
            mask = max_p >= thr
            n    = int(mask.sum())
            if n == 0:
                continue
            acc = float(accuracy_score(y_str[mask], y_pred[mask]))
            rows.append({
                "Min Confidence": f"≥{thr:.0%}",
                "N Matches":      n,
                "Coverage":       f"{n/len(y_str)*100:.1f}%",
                "Accuracy":       f"{acc*100:.1f}%",
            })

        return pd.DataFrame(rows)

    # ── full report ───────────────────────────────────────────────────────────

    def run_full_report(
        self,
        holdout_years:    list[int]  = None,
        tournament_type:  str        = "FIFA World Cup",
        save:             bool       = True,
        output_path:      Path | str | None = None,
    ) -> dict:
        """
        Run the complete evaluation suite and return (+ optionally save) a report.

        Parameters
        ----------
        holdout_years  : WC years to test on  (default: [2014, 2018, 2022])
        tournament_type: tournament filter
        save           : whether to save the report JSON
        output_path    : save location (default: results/evaluation_report.json)

        Returns
        -------
        dict with per-model metrics and breakdown tables
        """
        if holdout_years is None:
            holdout_years = [2014, 2018, 2022]

        self._ensure_models_loaded()

        log.info("=" * 60)
        log.info("TASK 6.1 — Evaluation Framework")
        log.info("=" * 60)
        log.info("Holdout: %s  %s", tournament_type, holdout_years)

        # ── 1. Combined WC holdout ────────────────────────────────────────────
        X_all, y_all, meta_all = self.load_holdout(
            tournament_type=tournament_type,
            years=holdout_years,
        )

        # Outcome distribution
        outcome_dist = {o: int((y_all == o).sum()) for o in ["H", "D", "A"]}

        # Historical base rates from pre-holdout competitive matches
        hda_base = self._historical_base_rates(end_year_excl=min(holdout_years))

        # ── 2. Evaluate all models ────────────────────────────────────────────
        all_results = self.evaluate_all(X_all, y_all)

        # ── 3. Per-tournament breakdown ───────────────────────────────────────
        per_tourn: dict[int, list[EvalResult]] = {}
        for yr in holdout_years:
            X_y, y_y, _ = self.load_holdout(tournament_type=tournament_type, years=[yr])
            per_tourn[yr] = [
                self.evaluate(r.model_name, X_y, y_y, historical_hda_rates=hda_base)
                for r in all_results
            ]

        # ── 4. Best model identification ──────────────────────────────────────
        best_rps = min(all_results, key=lambda r: r.rps)
        best_ll  = min(all_results, key=lambda r: r.log_loss)
        best_acc = max(all_results, key=lambda r: r.accuracy)

        # ── 5. Time-series for best model ─────────────────────────────────────
        ts_df = self.evaluate_time_series(
            model_name=best_rps.model_name,
            tournament_filter=tournament_type,
        )

        # ── 6. Calibration for ensemble ───────────────────────────────────────
        cal = self.calibration_analysis("ensemble", X_all, y_all)

        # ── 7. Confidence breakdown ───────────────────────────────────────────
        conf_df = self.confidence_breakdown("ensemble", X_all, y_all)

        # ── 8. Confusion matrix for best accuracy model ───────────────────────
        proba_best = self.predict(best_acc.model_name, X_all)
        y_pred_best = [CLASSES_ADH[i] for i in np.argmax(proba_best, axis=1)]
        cm = confusion_matrix(y_all, y_pred_best, labels=["H", "D", "A"])

        # ── 9. TRUE OOS evaluation (2023-2024) ────────────────────────────────
        log.info("Running true out-of-sample evaluation on 2023-2024 …")
        oos_results = self.evaluate_oos()

        # ── Print results ─────────────────────────────────────────────────────
        self._print_report(
            all_results, per_tourn, ts_df, conf_df, cm,
            outcome_dist, best_rps, best_ll, best_acc,
            oos_results=oos_results,
        )

        # ── Compile report dict ───────────────────────────────────────────────
        report = {
            "holdout": {
                "tournament_type": tournament_type,
                "years": holdout_years,
                "n_matches": len(X_all),
                "outcome_distribution": outcome_dist,
                "⚠️_leakage_warning": (
                    "WC 2014/2018/2022 matches were included in training data. "
                    "Metrics for tree models (LightGBM, XGBoost, Random Forest) "
                    "are in-sample and optimistically biased. "
                    "See 'oos_results' for genuine out-of-sample numbers."
                ),
            },
            "results": [r.as_dict() for r in all_results],
            "per_tournament": {
                str(yr): [r.as_dict() for r in results]
                for yr, results in per_tourn.items()
            },
            "oos_results": [r.as_dict() for r in oos_results],
            "best_model": {
                "by_rps":      best_rps.model_name,
                "by_log_loss": best_ll.model_name,
                "by_accuracy": best_acc.model_name,
            },
            "time_series": ts_df.to_dict(orient="records"),
            "calibration":  cal,
            "confidence_breakdown": conf_df.to_dict(orient="records"),
            "confusion_matrix": {
                "labels": ["H", "D", "A"],
                "matrix": cm.tolist(),
            },
            "benchmarks": {
                "log_loss_target": BENCHMARK_LOG_LOSS,
                "rps_target":      BENCHMARK_RPS,
            },
        }

        # ── Save ──────────────────────────────────────────────────────────────
        if save:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            dest = Path(output_path) if output_path else (
                RESULTS_DIR / "evaluation_report.json"
            )
            with open(dest, "w") as f:
                json.dump(report, f, indent=2, default=str)
            log.info("Report saved → %s", dest)

        return report

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def model_names(self) -> list[str]:
        return [
            "ensemble",
            "xgboost",
            "lightgbm",
            "logistic_regression",
            "random_forest",
            "elo",
            "baseline_uniform",
            "baseline_historical",
        ]

    # Cached full feature matrix — loaded once per evaluator instance
    _X_cache: "pd.DataFrame | None" = None
    _raw_cache: "pd.DataFrame | None" = None

    def _get_full_matrix(self) -> tuple["pd.DataFrame", "pd.DataFrame"]:
        """Load (and cache) the full feature matrix + aligned raw CSV."""
        if self._X_cache is not None:
            return self._X_cache, self._raw_cache  # type: ignore[return-value]

        from src.features.encoder import FeaturePipeline
        raw = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        raw = raw[raw["is_played"] & raw["is_competitive"] &
                  (raw["year"] >= 2010)].reset_index(drop=True)

        fp = FeaturePipeline()
        X_all, y_all, _, _ = fp.prepare_training_data(
            start_year=2010, competitive_only=True
        )
        # Attach outcome labels to raw for convenience
        raw = raw.iloc[:len(X_all)].copy()
        raw["_outcome"] = list(y_all)

        self._X_cache   = X_all.reset_index(drop=True)
        self._raw_cache = raw.reset_index(drop=True)
        return self._X_cache, self._raw_cache

    def load_holdout(
        self,
        tournament_type: str | list[str] = "FIFA World Cup",
        years: list[int] | None = None,
        start_year: int = 2010,
        competitive_only: bool = True,
    ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        Load a holdout evaluation set.

        ⚠️  If the requested years overlap with the model training period (≤2022),
        the resulting metrics are IN-SAMPLE and will be optimistically biased for
        tree-based models.  Use years=[2023, 2024] for true OOS evaluation.

        Returns (X_test, y_str, meta)
        """
        X_all, raw = self._get_full_matrix()

        if isinstance(tournament_type, str):
            tournament_type = [tournament_type]

        mask = raw["tournament_type"].isin(tournament_type)
        if years is not None:
            mask &= raw["year"].isin(years)

        idx     = mask.values
        X_test  = X_all[idx].reset_index(drop=True)
        y_str   = raw.loc[idx, "_outcome"].values
        meta_cols = [c for c in ["date", "home_team", "away_team", "year",
                                  "tournament_type"] if c in raw.columns]
        meta    = raw.loc[idx, meta_cols].reset_index(drop=True)

        log.info("Holdout: %d matches  (tournaments: %s, years: %s)",
                 len(X_test), tournament_type, years)
        return X_test, y_str, meta

    def _historical_base_rates(
        self,
        end_year_excl: int = 2014,
    ) -> tuple[float, float, float]:
        """
        Compute H/D/A base rates from competitive training data
        (years 2010 to end_year_excl-1) directly from raw CSV.
        """
        raw = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
        raw = raw[raw["is_played"] & raw["is_competitive"] &
                  (raw["year"] >= 2010) &
                  (raw["year"] < end_year_excl)].copy()
        if raw.empty:
            return 1 / 3, 1 / 3, 1 / 3
        outcomes = raw["result"]    # H / D / A
        return (
            float((outcomes == "H").mean()),
            float((outcomes == "D").mean()),
            float((outcomes == "A").mean()),
        )

    @staticmethod
    def _base_rates(y_str: np.ndarray) -> tuple[float, float, float]:
        """Return (p_H, p_D, p_A) base rates from a label array."""
        total = len(y_str)
        if total == 0:
            return 1 / 3, 1 / 3, 1 / 3
        return (
            float((y_str == "H").mean()),
            float((y_str == "D").mean()),
            float((y_str == "A").mean()),
        )

    def evaluate_oos(self) -> list[EvalResult]:
        """
        True out-of-sample evaluation on all competitive matches 2023-2024.
        Models were trained on data ≤ 2022; these matches are genuinely unseen.
        """
        self._ensure_models_loaded()
        X_oos, y_oos, _ = self.load_holdout(
            tournament_type=[
                "FIFA World Cup qualification",
                "UEFA Euro qualification",
                "UEFA Euro",
                "UEFA Nations League",
                "Copa América",
                "African Cup of Nations",
                "African Cup of Nations qualification",
                "AFC Asian Cup",
                "CONCACAF Nations League",
                "CONCACAF Gold Cup",
                "FIFA World Cup",
            ],
            years=[2023, 2024],
        )
        if len(X_oos) == 0:
            log.warning("No 2023-2024 competitive matches found.")
            return []

        hda = self._historical_base_rates(end_year_excl=2023)
        results = []
        for name in self.model_names:
            try:
                r = self.evaluate(name, X_oos, y_oos,
                                  historical_hda_rates=hda)
                results.append(r)
            except Exception as exc:
                log.warning("Skipping %s (OOS) — %s", name, exc)
        return results

    # ── pretty printing ───────────────────────────────────────────────────────

    @staticmethod
    def _print_report(
        all_results:  list[EvalResult],
        per_tourn:    dict,
        ts_df:        pd.DataFrame,
        conf_df:      pd.DataFrame,
        cm:           np.ndarray,
        outcome_dist: dict,
        best_rps:     EvalResult,
        best_ll:      EvalResult,
        best_acc:     EvalResult,
        oos_results:  list[EvalResult] | None = None,
    ) -> None:
        W = 80
        print("\n" + "═" * W)
        print("  FIFA WC 2026 — MODEL EVALUATION REPORT")
        print("═" * W)

        # ── DATA LEAKAGE WARNING ─────────────────────────────────────────────
        print(f"\n  ⚠️  DATA LEAKAGE WARNING")
        print(f"  {'─' * (W - 4)}")
        print(f"  WC 2014 / 2018 / 2022 matches were included in model training.")
        print(f"  Metrics below are IN-SAMPLE for tree models (LightGBM etc.).")
        print(f"  High accuracy / low log-loss for these models is an ARTIFACT.")
        print(f"  See the TRUE OOS section below for genuine out-of-sample results.")

        # ── IN-SAMPLE section header ─────────────────────────────────────────
        n = sum(outcome_dist.values())
        h, d, a = outcome_dist.get("H", 0), outcome_dist.get("D", 0), outcome_dist.get("A", 0)
        print(f"\n  ── IN-SAMPLE  (WC 2014 / 2018 / 2022  |  {n} matches) ──────────────────")
        print(f"     Outcome distribution: "
              f"H={h} ({h/n*100:.0f}%)  D={d} ({d/n*100:.0f}%)  A={a} ({a/n*100:.0f}%)")

        # Main comparison table
        print(f"\n  {'Model':<24} {'Acc':>6}  {'LogLoss':>8}  {'Brier':>7}  "
              f"{'RPS':>6}  {'RPSS':>6}  {'LL<.85':>6}  {'RPS<.19':>7}")
        print("  " + "─" * (W - 2))
        for r in all_results:
            ll_flag  = "  ✅" if r.beats_log_loss_target else "  ❌"
            rps_flag = "    ✅" if r.beats_rps_target else "    ❌"
            print(
                f"  {r.model_name:<24} "
                f"{r.accuracy*100:>5.1f}%  "
                f"{r.log_loss:>8.4f}  "
                f"{r.brier_score:>7.4f}  "
                f"{r.rps:>6.4f}  "
                f"{r.rps_skill_score:>+6.4f}"
                f"{ll_flag}{rps_flag}"
            )

        # Best models
        print(f"\n  Best by RPS:      {best_rps.model_name:<20} RPS={best_rps.rps:.4f}")
        print(f"  Best by Log Loss: {best_ll.model_name:<20} LL={best_ll.log_loss:.4f}")
        print(f"  Best by Accuracy: {best_acc.model_name:<20} Acc={best_acc.accuracy*100:.1f}%")

        # Per-tournament breakdown for ensemble
        ens_results = {yr: next(r for r in results if r.model_name == "ensemble")
                       for yr, results in per_tourn.items()}
        print(f"\n  {'Tournament':<20} {'Acc':>6}  {'LogLoss':>8}  {'RPS':>6}  {'N':>4}")
        print("  " + "─" * 50)
        for yr, r in sorted(ens_results.items()):
            print(f"  {'FIFA WC '+str(yr):<20} {r.accuracy*100:>5.1f}%  "
                  f"{r.log_loss:>8.4f}  {r.rps:>6.4f}  {r.n_matches:>4}")

        # Per-class accuracy for ensemble
        ens_all = next((r for r in all_results if r.model_name == "ensemble"), None)
        if ens_all:
            print(f"\n  Ensemble per-outcome accuracy (in-sample):")
            for outcome, acc_val in ens_all.class_accuracy.items():
                label = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[outcome]
                print(f"    {label:<12}: {acc_val*100:.1f}%")

        # Confidence breakdown
        if not conf_df.empty:
            print(f"\n  Confidence breakdown (ensemble):")
            print(conf_df.to_string(index=False, col_space=14))

        # Confusion matrix
        print(f"\n  Confusion Matrix — {best_acc.model_name}  (rows=true, cols=pred):")
        print(f"           H      D      A")
        for i, label in enumerate(["H", "D", "A"]):
            print(f"    {label}    {cm[i,0]:>4}   {cm[i,1]:>4}   {cm[i,2]:>4}")

        # Time-series (last 8 rows)
        if not ts_df.empty:
            print(f"\n  Year-by-year — {best_rps.model_name}:")
            print(ts_df.tail(8).to_string(index=False, col_space=10))

        # ── TRUE OOS SECTION (2023-2024) ──────────────────────────────────────
        if oos_results:
            n_oos = oos_results[0].n_matches if oos_results else 0
            print(f"\n  ── TRUE OUT-OF-SAMPLE  (2023-2024 competitive  |  {n_oos} matches) ──")
            print(f"     Models trained on data ≤ 2022. These matches are genuinely unseen.")
            print(f"\n  {'Model':<24} {'Acc':>6}  {'LogLoss':>8}  {'Brier':>7}  "
                  f"{'RPS':>6}  {'RPSS':>6}  {'LL<.85':>6}  {'RPS<.19':>7}")
            print("  " + "─" * (W - 2))
            for r in oos_results:
                ll_flag  = "  ✅" if r.beats_log_loss_target else "  ❌"
                rps_flag = "    ✅" if r.beats_rps_target else "    ❌"
                print(
                    f"  {r.model_name:<24} "
                    f"{r.accuracy*100:>5.1f}%  "
                    f"{r.log_loss:>8.4f}  "
                    f"{r.brier_score:>7.4f}  "
                    f"{r.rps:>6.4f}  "
                    f"{r.rps_skill_score:>+6.4f}"
                    f"{ll_flag}{rps_flag}"
                )

            # OOS benchmark verdict
            best_oos = min(
                [r for r in oos_results
                 if r.model_name not in ("baseline_uniform", "baseline_historical")],
                key=lambda r: r.rps + r.log_loss * 0.5,
                default=None,
            )
            if best_oos:
                print(f"\n  OOS Benchmark targets (best model = {best_oos.model_name}):")
                ll_met  = "✅ MET" if best_oos.log_loss < BENCHMARK_LOG_LOSS else "❌ NOT MET"
                rps_met = "✅ MET" if best_oos.rps      < BENCHMARK_RPS      else "❌ NOT MET"
                print(f"    Log Loss < {BENCHMARK_LOG_LOSS}:  {best_oos.log_loss:.4f}  → {ll_met}")
                print(f"    RPS      < {BENCHMARK_RPS}:  {best_oos.rps:.4f}  → {rps_met}")
        else:
            print(f"\n  (No 2023-2024 competitive data found for OOS evaluation)")

        # Benchmark verdict
        best_overall = min(all_results, key=lambda r: r.rps + r.log_loss * 0.5)
        print(f"\n  In-sample benchmark (best model = {best_overall.model_name}):")
        ll_met  = "✅ MET" if best_overall.log_loss  < BENCHMARK_LOG_LOSS else "❌ NOT MET"
        rps_met = "✅ MET" if best_overall.rps < BENCHMARK_RPS       else "❌ NOT MET"
        print(f"    Log Loss < {BENCHMARK_LOG_LOSS}:  {best_overall.log_loss:.4f}  → {ll_met}")
        print(f"    RPS      < {BENCHMARK_RPS}:  {best_overall.rps:.4f}  → {rps_met}")
        print(f"\n  ✅  Task 6.1 — Evaluation Framework complete.")
        print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone metric helpers (importable by other modules)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    proba: np.ndarray, y_str: np.ndarray, model_name: str = "model"
) -> EvalResult:
    """
    Compute all metrics given a probability array and string outcome labels.
    Convenience function that doesn't require loading any models.

    Parameters
    ----------
    proba      : (N, 3) array, columns = [A, D, H]
    y_str      : (N,) array of 'H', 'D', 'A' strings
    model_name : label for the result
    """
    le_map = {c: i for i, c in enumerate(CLASSES_ADH)}
    y_int  = np.array([le_map[o] for o in y_str])
    y_pred = np.array([CLASSES_ADH[i] for i in np.argmax(proba, axis=1)])

    acc   = float(accuracy_score(y_str, y_pred))
    ll    = float(log_loss(y_int, proba, labels=list(range(3))))
    brier = float(brier_score_multi(proba, y_int))
    rps_v = float(np.mean(rps_array(proba, y_str)))

    unif  = np.full_like(proba, 1.0 / 3.0)
    rps_u = float(np.mean(rps_array(unif, y_str)))
    rpss  = rps_skill_score(rps_v, rps_u)

    class_acc = {}
    for outcome in ["H", "D", "A"]:
        mask = y_str == outcome
        if mask.any():
            class_acc[outcome] = float(accuracy_score(y_str[mask], y_pred[mask]))

    return EvalResult(
        model_name     = model_name,
        n_matches      = len(y_str),
        accuracy       = acc,
        log_loss       = ll,
        brier_score    = brier,
        rps            = rps_v,
        rps_skill_score = rpss,
        beats_log_loss_target = ll < BENCHMARK_LOG_LOSS,
        beats_rps_target      = rps_v < BENCHMARK_RPS,
        class_accuracy = class_acc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="TASK 6.1 — Model Evaluation")
    p.add_argument("--years",    nargs="+", type=int,
                   default=[2014, 2018, 2022],
                   help="WC years to evaluate on")
    p.add_argument("--all-competitive", action="store_true",
                   help="Evaluate on all competitive matches instead of WC only")
    p.add_argument("--model", default="all",
                   help="Specific model to evaluate (or 'all')")
    p.add_argument("--no-save", action="store_true",
                   help="Skip saving the JSON report")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    ev = ModelEvaluator()

    if args.all_competitive:
        # Evaluate on all competitive matches 2019–2024 (out-of-sample period)
        X_test, y_str, meta = ev.load_holdout(
            tournament_type=[
                "FIFA World Cup", "UEFA Euro", "Copa América",
                "African Cup of Nations", "FIFA World Cup qualification",
            ],
            years=list(range(2019, 2025)),
        )
        log.info("All-competitive holdout: %d matches", len(X_test))
        ev.run_full_report(
            holdout_years=args.years,
            save=not args.no_save,
        )
    else:
        ev.run_full_report(
            holdout_years=args.years,
            save=not args.no_save,
        )


if __name__ == "__main__":
    main()
