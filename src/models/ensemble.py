from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.features.encoder import FeaturePipeline, MATCH_FEATURES_CSV
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonGoalModel
from src.models.xgboost_model import XGBoostOutcomeModel

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR   = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "model_comparison.csv"

CLASSES       = ["A", "D", "H"]
# Component models are trained on years < 2023, so val/holdout must be >= 2023
# to avoid evaluating on training data (data leakage).
VAL_YEARS     = (2023, 2023)   # weight optimisation — 702 unseen matches
HOLDOUT_YEAR  = 2024           # final evaluation   — 1,656 unseen matches
START_YEAR    = 2010
RANDOM_STATE  = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loader (shared across all models)
# ─────────────────────────────────────────────────────────────────────────────

class _DataBundle:
    """Holds aligned feature matrices and labels for a data slice."""

    def __init__(
        self,
        mf: pd.DataFrame,
        X_pipeline: pd.DataFrame,
        scaler,
        label_encoder: LabelEncoder,
    ):
        self.mf = mf                     # full match_features rows (for Elo & Poisson)
        self.X_pipeline = X_pipeline     # FeaturePipeline output (for XGB & LGBM)
        self.X_pipeline_sc = pd.DataFrame(
            scaler.transform(X_pipeline),
            columns=X_pipeline.columns,
            index=X_pipeline.index,
        )
        self.le = label_encoder
        self.y_raw = mf["result"].values
        self.y_enc = label_encoder.transform(self.y_raw)


def _load_bundles() -> tuple[_DataBundle, _DataBundle, StandardScaler, LabelEncoder]:
    """
    Build val (2023) and holdout (2024+) DataBundles from the feature store.
    A single StandardScaler and LabelEncoder are fit on the training set (< 2023)
    and applied consistently to all splits.
    """
    mf_full = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    mf_full = mf_full[mf_full["is_played"] & mf_full["is_competitive"]
                      & (mf_full["year"] >= START_YEAR)].reset_index(drop=True)

    # FeaturePipeline (fits encoders on the whole competitive set)
    pipeline = FeaturePipeline()
    X_all, y_all, _, _ = pipeline.prepare_training_data(
        start_year=START_YEAR, competitive_only=True
    )
    # X_all and mf_full are aligned (same filter)
    years = mf_full["year"].values

    train_mask = years < VAL_YEARS[0]
    val_mask   = (years >= VAL_YEARS[0]) & (years <= VAL_YEARS[1])
    hold_mask  = years >= HOLDOUT_YEAR

    # Fit scaler & label encoder on training portion only
    # Use boolean array indexing (iloc with bool masks is deprecated in pandas 3+)
    scaler = StandardScaler()
    scaler.fit(X_all[train_mask])

    le = LabelEncoder()
    le.fit(y_all[train_mask])

    def _bundle(mask):
        return _DataBundle(
            mf=mf_full[mask].reset_index(drop=True),
            X_pipeline=X_all[mask].reset_index(drop=True),
            scaler=scaler,
            label_encoder=le,
        )

    return _bundle(val_mask), _bundle(hold_mask), scaler, le


# ─────────────────────────────────────────────────────────────────────────────
# Per-model probability extraction
# ─────────────────────────────────────────────────────────────────────────────

def _proba_xgboost(bundle: _DataBundle, xgb_model: XGBoostOutcomeModel) -> np.ndarray:
    """(n, 3) array ordered [A, D, H]."""
    df = xgb_model.predict_proba(bundle.X_pipeline)
    return df[CLASSES].values


def _proba_lgbm(bundle: _DataBundle) -> np.ndarray:
    """(n, 3) array ordered [A, D, H]."""
    data = joblib.load(MODEL_DIR / "lightgbm.joblib")
    clf, le = data["model"], data["label_encoder"]
    raw = clf.predict_proba(bundle.X_pipeline)          # columns in le.classes_ order
    df = pd.DataFrame(raw, columns=list(le.classes_))
    return df[CLASSES].values


def _proba_poisson(bundle: _DataBundle, poi_model: PoissonGoalModel) -> np.ndarray:
    """(n, 3) array ordered [A, D, H]."""
    X_poi = bundle.mf[poi_model.feature_names].fillna(0).reset_index(drop=True)
    df = poi_model.predict_proba(X_poi)
    return df[CLASSES].values


def _proba_elo(bundle: _DataBundle, elo_model: EloModel) -> np.ndarray:
    """(n, 3) array ordered [A, D, H]."""
    raw = elo_model.predict_proba(bundle.mf[["elo_diff"]].reset_index(drop=True))
    df = pd.DataFrame(raw, columns=list(elo_model.calibrator.classes_))
    return df[CLASSES].values


def _collect_probas(
    bundle: _DataBundle,
    xgb_model: XGBoostOutcomeModel,
    poi_model: PoissonGoalModel,
    elo_model: EloModel,
) -> dict[str, np.ndarray]:
    return {
        "elo":     _proba_elo(bundle, elo_model),
        "xgb":     _proba_xgboost(bundle, xgb_model),
        "poisson": _proba_poisson(bundle, poi_model),
        "lgbm":    _proba_lgbm(bundle),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────────────────────

class WeightedEnsemble:
    """
    Weighted average of four model probability outputs.

    weights = [w_elo, w_xgb, w_poisson, w_lgbm]  (normalised to sum=1 internally)
    """

    MODEL_KEYS = ["elo", "xgb", "poisson", "lgbm"]
    INIT_WEIGHTS = [0.15, 0.40, 0.30, 0.15]   # suggested starting point

    def __init__(self):
        self.weights: dict[str, float] = dict(zip(self.MODEL_KEYS, self.INIT_WEIGHTS))
        self.xgb_model: XGBoostOutcomeModel | None = None
        self.poi_model: PoissonGoalModel | None     = None
        self.elo_model: EloModel | None             = None
        self.is_fit = False

    def _load_component_models(self, mf_train: pd.DataFrame) -> None:
        """Load / initialise all component models."""
        self.xgb_model = XGBoostOutcomeModel.load()

        self.poi_model = PoissonGoalModel.load()

        self.elo_model = EloModel()
        self.elo_model.fit_calibrator(
            mf_train[mf_train["result"].notna() & mf_train["elo_diff"].notna()]
        )

    def _blend(self, probas: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
        """Weighted average; w is raw (will be normalised)."""
        w = np.array(w, dtype=float)
        w = np.abs(w) / np.abs(w).sum()   # enforce positive & sum-to-one
        blended = sum(w[i] * probas[k] for i, k in enumerate(self.MODEL_KEYS))
        # row-normalise for numerical safety
        return blended / blended.sum(axis=1, keepdims=True)

    # ── Optimisation ──────────────────────────────────────────────────────────

    def optimise_weights(
        self,
        val_bundle: _DataBundle,
        n_restarts: int = 5,
    ) -> dict[str, float]:
        """
        Find weights that minimise log loss on the validation bundle using
        Nelder-Mead. Multiple random restarts guard against local minima.
        """
        log.info("Collecting val probabilities from 4 component models...")
        val_probas = _collect_probas(
            val_bundle, self.xgb_model, self.poi_model, self.elo_model
        )
        y_val = val_bundle.y_enc

        def objective(w):
            blended = self._blend(val_probas, w)
            return log_loss(y_val, blended)

        best_ll  = np.inf
        best_w   = np.array(self.INIT_WEIGHTS)

        # Always include the suggested starting point as one of the restarts
        starts = [np.array(self.INIT_WEIGHTS)]
        rng = np.random.default_rng(RANDOM_STATE)
        for _ in range(n_restarts - 1):
            raw = rng.dirichlet(np.ones(4))
            starts.append(raw)

        for i, w0 in enumerate(starts):
            res = minimize(
                objective, w0,
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
            )
            if res.fun < best_ll:
                best_ll = res.fun
                best_w  = res.x
            log.info("  Restart %d/%d  val log_loss=%.4f", i + 1, n_restarts, res.fun)

        # Normalise
        best_w = np.abs(best_w) / np.abs(best_w).sum()
        self.weights = dict(zip(self.MODEL_KEYS, best_w.round(4).tolist()))

        log.info("Optimised weights: %s", self.weights)
        log.info("Val log loss with optimised weights: %.4f", best_ll)
        self.is_fit = True
        return self.weights

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba_from_probas(
        self, probas: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Blend pre-computed probability arrays using current weights."""
        w = [self.weights[k] for k in self.MODEL_KEYS]
        return self._blend(probas, w)

    def predict_proba(self, bundle: _DataBundle) -> pd.DataFrame:
        """
        End-to-end prediction for a DataBundle. Returns DataFrame [A, D, H].
        """
        probas = _collect_probas(
            bundle, self.xgb_model, self.poi_model, self.elo_model
        )
        blended = self.predict_proba_from_probas(probas)
        return pd.DataFrame(blended, columns=CLASSES)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, bundle: _DataBundle, label: str = "holdout") -> dict:
        """Compute accuracy, log loss, Brier on a DataBundle."""
        proba_df = self.predict_proba(bundle)
        proba    = proba_df.values
        preds    = np.argmax(proba, axis=1)

        acc   = accuracy_score(bundle.y_enc, preds)
        ll    = log_loss(bundle.y_enc, proba)
        brier = float(np.mean([
            brier_score_loss((bundle.y_raw == cls).astype(int), proba[:, i])
            for i, cls in enumerate(CLASSES)
        ]))

        log.info("[%s]  acc=%.4f  log_loss=%.4f  brier=%.4f", label, acc, ll, brier)
        return {"accuracy": round(acc, 4), "log_loss": round(ll, 4),
                "brier_score": round(brier, 4)}

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else MODEL_DIR / "ensemble.joblib"
        joblib.dump({"weights": self.weights, "is_fit": self.is_fit}, out)
        log.info("Ensemble saved → %s", out)
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> "WeightedEnsemble":
        src = Path(path) if path else MODEL_DIR / "ensemble.joblib"
        if not src.exists():
            raise FileNotFoundError(f"No saved ensemble at {src}.")
        data = joblib.load(src)
        obj = cls()
        obj.weights = data["weights"]
        obj.is_fit  = data["is_fit"]
        log.info("Ensemble loaded. Weights: %s", obj.weights)
        return obj


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
    log.info("TASK 3.5 — Weighted Ensemble")
    log.info("=" * 60)

    # 1. Load data
    log.info("Building val (%d-%d) and holdout (%d+) bundles...",
             VAL_YEARS[0], VAL_YEARS[1], HOLDOUT_YEAR)
    val_bundle, hold_bundle, scaler, le = _load_bundles()
    log.info("Val: %d matches | Holdout: %d matches",
             len(val_bundle.mf), len(hold_bundle.mf))

    # Training set needed for Elo calibration
    mf_full = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    mf_train = mf_full[
        mf_full["is_played"] & mf_full["is_competitive"]
        & (mf_full["year"] >= START_YEAR)
        & (mf_full["year"] < VAL_YEARS[0])
    ].reset_index(drop=True)

    # 2. Build ensemble & load component models
    ensemble = WeightedEnsemble()
    ensemble._load_component_models(mf_train)

    # 3. Optimise weights on validation set
    log.info("Optimising weights on %d--%d validation set...", *VAL_YEARS)
    weights = ensemble.optimise_weights(val_bundle, n_restarts=8)

    print("\n" + "=" * 55)
    print("OPTIMISED ENSEMBLE WEIGHTS")
    print("=" * 55)
    for model, w in weights.items():
        bar = "█" * int(w * 40)
        print(f"  {model:<10} {w:.4f}  {bar}")
    print("=" * 55)

    # 4. Evaluate on 2023+ holdout
    log.info("Evaluating ensemble on 2023+ holdout...")
    hold_metrics = ensemble.evaluate(hold_bundle, label="holdout 2023+")

    # 5. Compare against individual models
    existing = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()

    ensemble_row = pd.DataFrame([{
        "model":          "ensemble",
        "accuracy":       hold_metrics["accuracy"],
        "log_loss":       hold_metrics["log_loss"],
        "brier_score":    hold_metrics["brier_score"],
        "train_matches":  len(mf_train),
        "test_matches":   len(hold_bundle.mf),
        "holdout_year":   HOLDOUT_YEAR,
    }])

    # Remove old ensemble row if rerunning
    if not existing.empty:
        existing = existing[existing["model"] != "ensemble"]
    results = pd.concat([existing, ensemble_row], ignore_index=True)
    results = results.sort_values("log_loss").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 65)
    print("FULL MODEL COMPARISON — Holdout 2023+")
    print("=" * 65)
    print(results[["model", "accuracy", "log_loss", "brier_score"]].to_string(index=False))
    print("=" * 65)

    # 6. Confirm ensemble beats all individuals
    ensemble_ll = hold_metrics["log_loss"]
    best_individual = results[results["model"] != "ensemble"]["log_loss"].min()
    if ensemble_ll < best_individual:
        print(f"\n✅  Ensemble log_loss ({ensemble_ll:.4f}) beats best individual "
              f"({best_individual:.4f}) by {best_individual - ensemble_ll:.4f}")
    else:
        print(f"\n⚠️  Ensemble ({ensemble_ll:.4f}) did not beat best individual "
              f"({best_individual:.4f})")

    # 7. Save
    saved = ensemble.save()
    print(f"\nEnsemble saved → {saved}")
    print("\nTask 3.5 complete.")


if __name__ == "__main__":
    main()
