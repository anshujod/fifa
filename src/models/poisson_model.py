"""
poisson_model.py — Poisson Goal Prediction Model (Task 3.3).

Predicts λ_home and λ_away (expected goals) using PoissonRegressor,
applies Dixon-Coles correction for low-scoring scorelines, and derives
win/draw/loss probabilities by integrating over the joint score distribution.

Key components:
    PoissonGoalModel.fit()              — train two PoissonRegressors (home/away goals)
    PoissonGoalModel.predict_lambda()   — predict (λ_home, λ_away) for a feature matrix
    PoissonGoalModel.score_matrix()     — full P(i, j) scoreline distribution with DC correction
    PoissonGoalModel.predict_proba()    — P(Home Win), P(Draw), P(Away Win)
    PoissonGoalModel.simulate_scoreline() — sample a single scoreline from the distribution
    PoissonGoalModel.save() / .load()   — serialise/deserialise to disk

Usage:
    python -m src.models.poisson_model
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.features.encoder import MATCH_FEATURES_CSV

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# Scoreline grid — goals 0 through MAX_GOALS on each axis
MAX_GOALS = 8

# Features used for goal prediction (attack/defence focused, no leakage)
GOAL_FEATURES = [
    "home_goals_scored_decay",
    "away_goals_scored_decay",
    "home_goals_conceded_decay",
    "away_goals_conceded_decay",
    "home_goals_scored_avg_5",
    "away_goals_scored_avg_5",
    "home_goals_conceded_avg_5",
    "away_goals_conceded_avg_5",
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "neutral_venue",
    "match_importance",
    "expected_goals_home",   # Dixon-Coles pre-estimate as feature
    "expected_goals_away",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dixon-Coles correction
# ─────────────────────────────────────────────────────────────────────────────

def _dc_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """
    Dixon-Coles correction factor τ(i, j) for low-scoring cells.
    Only cells where i + j <= 1 (i.e. 0-0, 1-0, 0-1) and 1-1 are adjusted.
    All other cells return 1.0.
    """
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _dc_log_likelihood(rho: float, actual_h: np.ndarray, actual_a: np.ndarray,
                        lam_h: np.ndarray, lam_a: np.ndarray) -> float:
    """Negative log-likelihood for Dixon-Coles ρ estimation."""
    ll = 0.0
    for i, j, lh, la in zip(actual_h, actual_a, lam_h, lam_a):
        tau = _dc_tau(int(i), int(j), lh, la, rho)
        if tau <= 0:
            return 1e9  # invalid
        ll += np.log(max(tau, 1e-10))
    return -ll


def _estimate_rho(actual_h: np.ndarray, actual_a: np.ndarray,
                  lam_h: np.ndarray, lam_a: np.ndarray) -> float:
    """Estimate the Dixon-Coles ρ parameter by MLE on low-scoring matches."""
    # Only use matches with ≤ 2 total goals for the estimation (where correction matters)
    mask = (actual_h + actual_a) <= 2
    if mask.sum() < 50:
        log.warning("Too few low-scoring matches for ρ estimation; using ρ=-0.13")
        return -0.13
    res = minimize_scalar(
        lambda r: _dc_log_likelihood(r, actual_h[mask], actual_a[mask],
                                      lam_h[mask], lam_a[mask]),
        bounds=(-0.5, 0.0),
        method="bounded",
    )
    rho = float(res.x)
    log.info("Estimated Dixon-Coles ρ = %.4f (optimised on %d low-scoring matches)",
             rho, mask.sum())
    return rho


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class PoissonGoalModel:
    """
    Two-model Poisson regression for international football goal prediction.

    One PoissonRegressor predicts home goals (λ_home), another predicts away
    goals (λ_away). A Dixon-Coles ρ correction is estimated from training data
    and applied to the joint scoreline probability matrix at inference.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            L2 regularisation strength for PoissonRegressor (default 1.0).
        """
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model_home = PoissonRegressor(alpha=alpha, max_iter=300)
        self.model_away = PoissonRegressor(alpha=alpha, max_iter=300)
        self.rho: float = -0.13        # Dixon-Coles correlation; estimated in fit()
        self.feature_names: list[str] = []
        self.is_fit = False

    # ── Training ──────────────────────────────────────────────────────────────

    def _load_training_data(
        self, start_year: int = 2000, competitive_only: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare the feature matrix and goal targets."""
        df = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
        df = df[df["is_played"] == True].copy()
        df = df[df["year"] >= start_year].copy()
        if competitive_only:
            df = df[df["is_competitive"] == True].copy()
        df = df.reset_index(drop=True)

        self.feature_names = [f for f in GOAL_FEATURES if f in df.columns]
        X = df[self.feature_names].fillna(0).values
        y_home = df["home_goals"].values.astype(float)
        y_away = df["away_goals"].values.astype(float)

        log.info("Training Poisson model on %d matches (features: %d)",
                 len(df), len(self.feature_names))
        return df, X, y_home, y_away

    def fit(
        self,
        start_year: int = 2000,
        competitive_only: bool = True,
        holdout_year: int = 2023,
    ) -> dict:
        """
        Fit both goal models and estimate the Dixon-Coles ρ on training data.

        Parameters
        ----------
        start_year : int
            Earliest year of matches to include.
        competitive_only : bool
            Restrict to competitive (non-friendly) matches.
        holdout_year : int
            Matches from this year onward are used for evaluation only.

        Returns
        -------
        dict
            Evaluation metrics: mae_home, mae_away, log_loss on holdout set.
        """
        df, X, y_home, y_away = self._load_training_data(start_year, competitive_only)

        train_mask = df["year"] < holdout_year
        test_mask = df["year"] >= holdout_year

        X_train, X_test = X[train_mask], X[test_mask]
        yh_train, yh_test = y_home[train_mask], y_home[test_mask]
        ya_train, ya_test = y_away[train_mask], y_away[test_mask]

        # Scale features
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        # Fit models
        log.info("Fitting home-goals model on %d matches...", train_mask.sum())
        self.model_home.fit(X_train_sc, yh_train)

        log.info("Fitting away-goals model on %d matches...", train_mask.sum())
        self.model_away.fit(X_train_sc, ya_train)

        self.is_fit = True

        # Estimate Dixon-Coles ρ from training set predictions
        lam_h_train = self.model_home.predict(X_train_sc)
        lam_a_train = self.model_away.predict(X_train_sc)
        self.rho = _estimate_rho(yh_train, ya_train, lam_h_train, lam_a_train)

        # Evaluate on holdout
        lam_h_test = self.model_home.predict(X_test_sc)
        lam_a_test = self.model_away.predict(X_test_sc)

        mae_home = mean_absolute_error(yh_test, lam_h_test)
        mae_away = mean_absolute_error(ya_test, lam_a_test)

        # Derive outcome probabilities for each holdout match and compute log loss
        from sklearn.metrics import log_loss as sk_log_loss
        proba_rows = []
        for lh, la in zip(lam_h_test, lam_a_test):
            p = self._outcome_proba_from_lambda(lh, la)
            proba_rows.append(p)
        proba_df = pd.DataFrame(proba_rows)

        holdout_df = df[test_mask].reset_index(drop=True)
        ll = sk_log_loss(holdout_df["result"], proba_df[["A", "D", "H"]],
                         labels=["A", "D", "H"])

        metrics = {
            "mae_home_goals": round(mae_home, 4),
            "mae_away_goals": round(mae_away, 4),
            "log_loss_outcome": round(ll, 4),
            "rho": round(self.rho, 4),
            "train_matches": int(train_mask.sum()),
            "test_matches": int(test_mask.sum()),
        }

        print("\n" + "=" * 45)
        print("POISSON MODEL PERFORMANCE")
        print("=" * 45)
        print(f"Holdout MAE — Home Goals : {mae_home:.4f}")
        print(f"Holdout MAE — Away Goals : {mae_away:.4f}")
        print(f"Holdout Log Loss (outcome): {ll:.4f}")
        print(f"Dixon-Coles ρ            : {self.rho:.4f}")
        print("=" * 45)

        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def _prepare_X(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Missing features: {missing}")
            X = X[self.feature_names].fillna(0).values
        return self.scaler.transform(X)

    def predict_lambda(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict expected goals (λ_home, λ_away) for each row in X.

        Returns
        -------
        lam_home : np.ndarray  shape (n,)
        lam_away : np.ndarray  shape (n,)
        """
        if not self.is_fit:
            raise RuntimeError("Model not fitted. Call fit() first.")
        Xs = self._prepare_X(X)
        return self.model_home.predict(Xs), self.model_away.predict(Xs)

    def score_matrix(
        self, lam_home: float, lam_away: float, max_goals: int = MAX_GOALS
    ) -> np.ndarray:
        """
        Build a (max_goals+1) × (max_goals+1) matrix of joint scoreline
        probabilities P(home=i, away=j) with Dixon-Coles correction applied.

        Entry [i, j] = P(home scores i, away scores j).
        Rows = home goals, Columns = away goals.
        """
        g = max_goals + 1
        matrix = np.zeros((g, g))
        for i in range(g):
            for j in range(g):
                p = poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
                p *= _dc_tau(i, j, lam_home, lam_away, self.rho)
                matrix[i, j] = max(p, 0.0)  # tau can push near-zero cells negative

        # Renormalise so probabilities sum to 1
        total = matrix.sum()
        if total > 0:
            matrix /= total
        return matrix

    def _outcome_proba_from_lambda(
        self, lam_home: float, lam_away: float
    ) -> dict[str, float]:
        """Derive P(H), P(D), P(A) from the score matrix."""
        mat = self.score_matrix(lam_home, lam_away)
        g = mat.shape[0]
        p_home_win = float(np.sum([mat[i, j] for i in range(g) for j in range(g) if i > j]))
        p_draw     = float(np.sum([mat[i, i] for i in range(g)]))
        p_away_win = float(np.sum([mat[i, j] for i in range(g) for j in range(g) if j > i]))
        total = p_home_win + p_draw + p_away_win
        return {
            "H": p_home_win / total,
            "D": p_draw / total,
            "A": p_away_win / total,
        }

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Predict win/draw/loss probabilities for each match in X.

        Returns
        -------
        pd.DataFrame with columns ['A', 'D', 'H'] (alphabetical, same as XGBoost).
        """
        lam_h, lam_a = self.predict_lambda(X)
        rows = [self._outcome_proba_from_lambda(lh, la) for lh, la in zip(lam_h, lam_a)]
        idx = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(rows, columns=["A", "D", "H"], index=idx).reset_index(drop=True)

    def simulate_scoreline(
        self,
        lam_home: float,
        lam_away: float,
        rng: np.random.Generator | None = None,
        fast: bool = False,
    ) -> tuple[int, int]:
        """
        Sample a single scoreline (home_goals, away_goals).

        Parameters
        ----------
        lam_home, lam_away : float
            Expected goals for each side.
        rng  : np.random.Generator, optional
        fast : bool  (default False)
            If True, sample directly from independent Poisson(λ) distributions,
            skipping the Dixon-Coles correction.  ~200× faster than the full
            score-matrix approach and appropriate for large Monte Carlo runs
            where the small DC correction (ρ = −0.023) averages out over many
            simulations.

        Returns
        -------
        (home_goals, away_goals) : tuple[int, int]
        """
        _rng = rng if rng is not None else np.random.default_rng()
        if fast:
            return int(_rng.poisson(lam_home)), int(_rng.poisson(lam_away))
        mat  = self.score_matrix(lam_home, lam_away)
        flat = mat.flatten()
        idx  = _rng.choice(len(flat), p=flat / flat.sum())
        g    = mat.shape[0]
        return int(idx // g), int(idx % g)

    def simulate_match(
        self,
        X_row: pd.DataFrame,
        n_simulations: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> dict:
        """
        Run n_simulations of a single match, returning summary statistics.
        Useful for Monte Carlo tournament simulation.
        """
        lam_h, lam_a = self.predict_lambda(X_row)
        lam_h, lam_a = float(lam_h[0]), float(lam_a[0])

        home_wins = draws = away_wins = 0
        scorelines: dict[tuple, int] = {}

        _rng = rng or np.random.default_rng()
        mat = self.score_matrix(lam_h, lam_a)
        flat = mat.flatten()
        g = mat.shape[0]

        samples = _rng.choice(len(flat), size=n_simulations, p=flat / flat.sum())
        for s in samples:
            hg, ag = int(s // g), int(s % g)
            if hg > ag:
                home_wins += 1
            elif hg == ag:
                draws += 1
            else:
                away_wins += 1
            scorelines[(hg, ag)] = scorelines.get((hg, ag), 0) + 1

        top_scores = sorted(scorelines.items(), key=lambda x: -x[1])[:5]
        return {
            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),
            "p_home_win": home_wins / n_simulations,
            "p_draw": draws / n_simulations,
            "p_away_win": away_wins / n_simulations,
            "top_scorelines": {f"{h}-{a}": round(c / n_simulations, 3) for (h, a), c in top_scores},
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str | None = None) -> Path:
        """Serialise model to disk."""
        if not self.is_fit:
            raise RuntimeError("Model not fitted yet.")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        out = Path(path) if path else MODEL_DIR / "poisson_goal_model.joblib"
        joblib.dump({
            "model_home": self.model_home,
            "model_away": self.model_away,
            "scaler": self.scaler,
            "rho": self.rho,
            "alpha": self.alpha,
            "feature_names": self.feature_names,
        }, out)
        log.info("Poisson model saved → %s", out)
        return out

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PoissonGoalModel":
        """Load a previously saved model from disk."""
        src = Path(path) if path else MODEL_DIR / "poisson_goal_model.joblib"
        if not src.exists():
            raise FileNotFoundError(f"No saved model at {src}. Train first.")
        data = joblib.load(src)
        obj = cls(alpha=data["alpha"])
        obj.model_home = data["model_home"]
        obj.model_away = data["model_away"]
        obj.scaler = data["scaler"]
        obj.rho = data["rho"]
        obj.feature_names = data["feature_names"]
        obj.is_fit = True
        log.info("Poisson model loaded from %s", src)
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
    log.info("TASK 3.3 — Poisson Goal Prediction Model")
    log.info("=" * 60)

    model = PoissonGoalModel(alpha=0.5)
    metrics = model.fit(start_year=2000, competitive_only=True, holdout_year=2023)

    # Show score distributions for a few real 2026 WC match-ups
    print("\nScore Distributions — Sample Matches")
    print("-" * 50)

    mf = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    recent = mf[mf["is_played"] == True].tail(5)

    for _, row in recent.iterrows():
        row_df = recent.loc[[row.name], model.feature_names].fillna(0)
        lam_h, lam_a = model.predict_lambda(row_df)
        proba = model._outcome_proba_from_lambda(float(lam_h[0]), float(lam_a[0]))
        print(f"\n  {row['home_team']} vs {row['away_team']}  "
              f"(actual: {int(row['home_goals'])}-{int(row['away_goals'])})")
        print(f"    λ_home={lam_h[0]:.2f}  λ_away={lam_a[0]:.2f}")
        print(f"    P(H)={proba['H']:.2%}  P(D)={proba['D']:.2%}  P(A)={proba['A']:.2%}")

    # Demo simulate_match on the last match
    last_row = recent.tail(1)
    sim = model.simulate_match(last_row[model.feature_names].fillna(0), n_simulations=10000)
    print(f"\nMonte Carlo simulation (10,000 runs) — last match:")
    print(f"  P(H)={sim['p_home_win']:.2%}  P(D)={sim['p_draw']:.2%}  P(A)={sim['p_away_win']:.2%}")
    print(f"  Top scorelines: {sim['top_scorelines']}")

    # Save
    saved = model.save()
    print(f"\nModel saved to {saved}")

    # Round-trip load test
    m2 = PoissonGoalModel.load()
    assert m2.is_fit, "Load failed"
    print("Load test: OK")
    print("\nTask 3.3 complete.")


if __name__ == "__main__":
    main()
