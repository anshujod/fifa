from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score

from src.features.encoder import FeaturePipeline, MATCH_FEATURES_CSV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

log = logging.getLogger(__name__)


class XGBoostOutcomeModel:
    def __init__(self):
        self.pipeline = FeaturePipeline()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.best_params = None

    def _get_aligned_years(
            self, start_year: int = 2014,
            competitive_only: bool = True) -> pd.Series:
        """Helper to get the 'year' column aligned with the pipeline's output."""
        df = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
        df = df[df["is_played"]].copy()
        df = df[df["year"] >= start_year].copy()

        if competitive_only:
            df = df[df["is_competitive"]].copy()

        df = df.reset_index(drop=True)
        return df["year"]

    def tune_and_train(self, n_trials: int = 30):
        """
        Tunes hyperparameters using time-series cross-validation and trains
        the final model on the best parameters.
        """
        # 1. Load Data
        log.info("Loading features from FeaturePipeline...")
        X, y_outcome, _, _ = self.pipeline.prepare_training_data(
            start_year=2010, competitive_only=True
        )

        years = self._get_aligned_years(start_year=2010, competitive_only=True)

        # 2. Encode Target
        # XGBoost requires labels in [0, num_classes-1]
        y = self.label_encoder.fit_transform(y_outcome)
        log.info(
            "Classes mapping: %s -> %s",
            self.label_encoder.classes_,
            self.label_encoder.transform(self.label_encoder.classes_))

        # 3. Define Optuna Objective for Time-Series CV
        def objective(trial):
            params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    50,
                    300),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    2,
                    6),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    1e-3,
                    0.1,
                    log=True),
                "subsample": trial.suggest_float(
                    "subsample",
                    0.5,
                    1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree",
                    0.5,
                    1.0),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha",
                    1e-3,
                    10.0,
                    log=True),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    10.0,
                    log=True),
                "eval_metric": "mlogloss",
                "verbosity": 0,
                "random_state": 42}

            # Time-series CV: Train on <= Y, Val on Y+1
            # We'll test on the last few years (e.g. 2019 to 2022)
            # skip 2020 due to covid sparsity
            val_years = [2018, 2019, 2021, 2022]
            cv_scores = []

            for vy in val_years:
                train_idx = years[years < vy].index
                val_idx = years[years == vy].index

                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue

                X_tr, y_tr = X.iloc[train_idx], y[train_idx]
                X_val, y_val = X.iloc[val_idx], y[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr)

                preds = model.predict_proba(X_val)
                score = log_loss(y_val, preds)
                cv_scores.append(score)

            return np.mean(cv_scores)

        # 4. Run Optuna  (lazy import — not needed at inference time)
        import optuna  # noqa: PLC0415
        log.info("Starting Optuna tuning (%d trials)...", n_trials)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        self.best_params = study.best_params
        self.best_params["objective"] = "multi:softprob"
        self.best_params["num_class"] = 3
        self.best_params["random_state"] = 42

        log.info("Best CV Log Loss: %.4f", study.best_value)
        log.info("Best Params: %s", self.best_params)

        # 5. Train Final Model on Train/Val up to 2022, evaluate on 2023+
        train_idx = years[years <= 2022].index
        test_idx = years[years > 2022].index

        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]

        log.info("Training final model on %d matches...", len(X_train))
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(X_train, y_train)

        log.info("Evaluating on holdout set (2023+): %d matches", len(X_test))
        preds_proba = self.model.predict_proba(X_test)
        preds_class = self.model.predict(X_test)

        ll = log_loss(y_test, preds_proba)
        acc = accuracy_score(y_test, preds_class)

        print("\n" + "=" * 40)
        print("🏆 XGBOOST CLASSIFIER PERFORMANCE")
        print("=" * 40)
        print(f"Holdout Accuracy (2023+): {acc:.4f}")
        print(f"Holdout Log Loss (2023+): {ll:.4f}")
        print("=" * 40)

        if ll < 0.85:
            log.info("Target log loss of < 0.85 achieved (%.4f)", ll)
        else:
            log.warning("Missed target log loss (%.4f >= 0.85)", ll)

        # Print top feature importances
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Features by Importance:")
        print(importance.head(10).to_string(index=False))

        return {"log_loss": ll, "accuracy": acc, "best_params": self.best_params}

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the trained model and label encoder to disk."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call tune_and_train() first.")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else MODEL_DIR / "xgboost_outcome.joblib"
        joblib.dump({"model": self.model, "label_encoder": self.label_encoder,
                     "best_params": self.best_params, "feature_names": list(self.pipeline.feature_names_)
                     if hasattr(self.pipeline, "feature_names_") else None},
                    out)
        log.info("Model saved → %s", out)
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> "XGBoostOutcomeModel":
        """Load a previously saved model from disk."""
        src = Path(path) if path else MODEL_DIR / "xgboost_outcome.joblib"
        if not src.exists():
            raise FileNotFoundError(f"No saved model at {src}. Train first.")
        data = joblib.load(src)
        obj = cls()
        obj.model = data["model"]
        obj.label_encoder = data["label_encoder"]
        obj.best_params = data["best_params"]
        log.info("Model loaded from %s", src)
        return obj

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame with columns [A, D, H] probabilities."""
        if self.model is None:
            raise RuntimeError("Model not trained/loaded.")
        proba = self.model.predict_proba(X)
        return pd.DataFrame(proba, columns=self.label_encoder.classes_, index=X.index)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 3.2 — XGBoost Match Outcome Classifier")
    log.info("=" * 60)

    model = XGBoostOutcomeModel()
    model.tune_and_train(n_trials=30)
    saved_path = model.save()

    print(f"\nModel saved to {saved_path}")
    print("Task 3.2 complete.")


if __name__ == "__main__":
    main()
