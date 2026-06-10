from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMClassifier

from src.features.encoder import FeaturePipeline, MATCH_FEATURES_CSV

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_CSV  = RESULTS_DIR / "model_comparison.csv"

HOLDOUT_YEAR = 2023
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray,
                      classes: list[str]) -> float:
    """Average Brier score across all three outcome classes."""
    scores = []
    for i, cls in enumerate(classes):
        binary = (y_true == cls).astype(int)
        scores.append(brier_score_loss(binary, y_proba[:, i]))
    return float(np.mean(scores))


def _load_data(start_year: int = 2010) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Return (X, y_encoded, y_raw, label_encoder, years) from FeaturePipeline."""
    pipeline = FeaturePipeline()
    X, y_raw, _, _ = pipeline.prepare_training_data(
        start_year=start_year, competitive_only=True
    )
    df_years = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    df_years = df_years[df_years["is_played"] & (df_years["year"] >= start_year)
                        & df_years["is_competitive"]].reset_index(drop=True)
    years = df_years["year"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, y_raw, le, years


def _time_split(X, y, years, holdout_year=HOLDOUT_YEAR):
    train = years < holdout_year
    test  = years >= holdout_year
    return (X.iloc[train], X.iloc[test],
            y[train],      y[test],
            train, test)


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def _build_models() -> dict:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(start_year: int = 2010) -> pd.DataFrame:
    """
    Train all three models on a time-series split, evaluate on the holdout,
    and return a DataFrame of metrics.
    """
    log.info("Loading feature matrix (start_year=%d, competitive_only=True)", start_year)
    X, y, y_raw, le, years = _load_data(start_year)

    X_train, X_test, y_train, y_test, train_mask, test_mask = _time_split(X, y, years)
    y_test_raw = y_raw.iloc[test_mask].reset_index(drop=True)

    log.info("Train: %d matches | Holdout (2023+): %d matches",
             len(X_train), len(X_test))

    # Logistic Regression needs scaled features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = _build_models()
    records = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        log.info("Training %s...", name)

        if name == "logistic_regression":
            clf.fit(X_train_sc, y_train)
            proba = clf.predict_proba(X_test_sc)
            preds = clf.predict(X_test_sc)
        else:
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)
            preds = clf.predict(X_test)

        acc    = accuracy_score(y_test, preds)
        ll     = log_loss(y_test, proba, labels=list(range(len(le.classes_))))
        brier  = _brier_multiclass(y_test_raw.values, proba, list(le.classes_))

        log.info("%-22s  acc=%.4f  log_loss=%.4f  brier=%.4f", name, acc, ll, brier)
        records.append({
            "model":        name,
            "accuracy":     round(acc,   4),
            "log_loss":     round(ll,    4),
            "brier_score":  round(brier, 4),
            "train_matches": int(train_mask.sum()),
            "test_matches":  int(test_mask.sum()),
            "holdout_year":  HOLDOUT_YEAR,
        })

        # Persist
        save_payload = {"model": clf, "label_encoder": le}
        if name == "logistic_regression":
            save_payload["scaler"] = scaler
        out = MODEL_DIR / f"{name}.joblib"
        joblib.dump(save_payload, out)
        log.info("Saved → %s", out)

    # ── Also add XGBoost & Elo scores so we have a single comparison table ──
    _append_existing_models(X_test, y_test, y_test_raw, le, records)

    results_df = pd.DataFrame(records).sort_values("log_loss").reset_index(drop=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    log.info("Comparison table saved → %s", RESULTS_CSV)

    return results_df


def _append_existing_models(X_test, y_test, y_test_raw, le, records: list) -> None:
    """Load already-trained XGBoost and Elo models and add their holdout scores."""
    # XGBoost
    xgb_path = MODEL_DIR / "xgboost_outcome.joblib"
    if xgb_path.exists():
        try:
            from src.models.xgboost_model import XGBoostOutcomeModel
            xgb_model = XGBoostOutcomeModel.load()
            proba_df = xgb_model.predict_proba(X_test)
            proba = proba_df[list(le.classes_)].values
            preds = np.argmax(proba, axis=1)
            acc   = accuracy_score(y_test, preds)
            ll    = log_loss(y_test, proba)
            brier = _brier_multiclass(y_test_raw.values, proba, list(le.classes_))
            records.append({
                "model": "xgboost",
                "accuracy": round(acc, 4),
                "log_loss": round(ll, 4),
                "brier_score": round(brier, 4),
                "train_matches": records[0]["train_matches"],
                "test_matches":  records[0]["test_matches"],
                "holdout_year":  HOLDOUT_YEAR,
            })
            log.info("%-22s  acc=%.4f  log_loss=%.4f  brier=%.4f", "xgboost", acc, ll, brier)
        except Exception as e:
            log.warning("Could not evaluate XGBoost: %s", e)

    # Elo baseline (uses elo_diff feature)
    try:
        from src.models.elo_model import EloModel
        import pandas as pd
        mf = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
        played = mf[mf["is_played"] == True].copy()
        train_elo = played[played["year"] < HOLDOUT_YEAR]
        test_elo  = played[played["year"] >= HOLDOUT_YEAR].reset_index(drop=True)

        elo_m = EloModel()
        elo_m.fit_calibrator(train_elo)
        elo_proba_arr = elo_m.predict_proba(test_elo[["elo_diff"]].reset_index(drop=True))
        # elo_model returns ndarray ordered by calibrator.classes_ (['A','D','H'])
        elo_classes = list(elo_m.calibrator.classes_)
        elo_proba_df = pd.DataFrame(elo_proba_arr, columns=elo_classes)
        elo_proba = elo_proba_df[list(le.classes_)].values
        elo_preds_idx = np.argmax(elo_proba, axis=1)
        y_test_elo = le.transform(test_elo["result"].values)
        acc   = accuracy_score(y_test_elo, elo_preds_idx)
        ll    = log_loss(y_test_elo, elo_proba)
        brier = _brier_multiclass(test_elo["result"].values, elo_proba, list(le.classes_))
        records.append({
            "model": "elo_baseline",
            "accuracy": round(acc, 4),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "train_matches": records[0]["train_matches"],
            "test_matches":  records[0]["test_matches"],
            "holdout_year":  HOLDOUT_YEAR,
        })
        log.info("%-22s  acc=%.4f  log_loss=%.4f  brier=%.4f", "elo_baseline", acc, ll, brier)
    except Exception as e:
        log.warning("Could not evaluate Elo baseline: %s", e)


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
    log.info("TASK 3.4 — Additional Models (RF, LightGBM, LR)")
    log.info("=" * 60)

    results = train_and_evaluate(start_year=2010)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON — Holdout 2023+")
    print("=" * 65)
    print(results[["model", "accuracy", "log_loss", "brier_score"]].to_string(index=False))
    print("=" * 65)
    print(f"\nResults saved → {RESULTS_CSV}")
    print("\nTask 3.4 complete.")


if __name__ == "__main__":
    main()
