"""
elo_model.py — Baseline Elo Model for FIFA World Cup 2026 Predictor.

Computes Elo ratings dynamically through historical matches.
Serves as the benchmark model for Phase 3 (Machine Learning).

Features:
- Expected win probability: P(home) = 1 / (1 + 10^((elo_away - elo_home)/400))
- Variable K-factors: K=32 (World Cup), K=20 (Qualifiers/Continental), K=10 (Friendlies)
- Uses Logistic Regression on `elo_diff` to map to H/D/A probabilities for log-loss benchmarking.

Usage:
    python -m src.models.elo_model
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_RESULTS_CSV = PROCESSED_DIR / "clean_results.csv"

log = logging.getLogger(__name__)


class EloModel:
    """
    Elo rating system and baseline predictor for international matches.
    """

    def __init__(self, initial_elo: float = 1500.0):
        self.initial_elo = initial_elo
        self.ratings: dict[str, float] = defaultdict(lambda: initial_elo)
        
        self.calibrator = LogisticRegression(max_iter=1000)
        self.is_fit = False

    def _get_k_factor(self, tournament: str) -> float:
        """Determine the K-factor based on tournament prestige."""
        t_lower = tournament.lower()
        if "world cup" in t_lower and "qualification" not in t_lower:
            return 32.0
        elif "qualification" in t_lower:
            return 20.0
        elif "friendly" in t_lower:
            return 10.0
        else:
            # Default for continental cups (Euro, Copa America, etc.)
            return 20.0

    def _expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score (win probability) for team A vs team B."""
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def _actual_score(self, goals_a: float, goals_b: float) -> float:
        """Convert goals to Elo actual score (1=Win, 0.5=Draw, 0=Loss)."""
        if goals_a > goals_b:
            return 1.0
        elif goals_a == goals_b:
            return 0.5
        else:
            return 0.0

    def calculate_historical_elos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterate through matches sequentially, calculating pre-match Elos
        and updating them post-match. Returns a DataFrame with Elo features.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        
        home_elos = []
        away_elos = []
        
        for idx, row in df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            
            e_home = self.ratings[home]
            e_away = self.ratings[away]
            
            home_elos.append(e_home)
            away_elos.append(e_away)
            
            # If match was played, update ratings
            # Support both raw (home_score) and processed (home_goals) column names
            score_col = "home_score" if "home_score" in row.index else "home_goals"
            away_score_col = "away_score" if "away_score" in row.index else "away_goals"
            tournament_col = "tournament" if "tournament" in row.index else "tournament_type"
            if row.get("is_played", True) and pd.notna(row[score_col]):
                k = self._get_k_factor(row[tournament_col])

                # Incorporate home advantage into expected score
                # Typically home advantage is worth ~100 Elo points
                home_adv = 100.0 if not row.get("neutral_venue", False) else 0.0

                exp_home = self._expected_score(e_home + home_adv, e_away)
                exp_away = 1.0 - exp_home

                act_home = self._actual_score(row[score_col], row[away_score_col])
                act_away = 1.0 - act_home
                
                # Update
                self.ratings[home] = e_home + k * (act_home - exp_home)
                self.ratings[away] = e_away + k * (act_away - exp_away)
                
        df["elo_home"] = home_elos
        df["elo_away"] = away_elos
        df["elo_diff"] = df["elo_home"] - df["elo_away"]
        
        return df

    def fit_calibrator(self, df: pd.DataFrame):
        """Fit a multinomial logistic regression to map elo_diff to P(H), P(D), P(A)."""
        # Ensure we have elo_diff
        if "elo_diff" not in df.columns:
            df = self.calculate_historical_elos(df)
            
        played = df[df["is_played"] == True].copy()
        
        # Prepare X and y
        X = played[["elo_diff"]]
        y = played["result"]  # 'H', 'D', 'A'
        
        self.calibrator.fit(X, y)
        self.is_fit = True
        log.info("Calibrator fit. Classes: %s", self.calibrator.classes_)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for H, D, A."""
        if not self.is_fit:
            raise ValueError("Must call fit_calibrator before predict_proba.")
            
        if "elo_diff" not in df.columns:
            df = self.calculate_historical_elos(df)
            
        return self.calibrator.predict_proba(df[["elo_diff"]])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict discrete class."""
        if not self.is_fit:
            raise ValueError("Must call fit_calibrator before predict.")
            
        if "elo_diff" not in df.columns:
            df = self.calculate_historical_elos(df)
            
        return self.calibrator.predict(df[["elo_diff"]])


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 3.1 — Elo Baseline Model")
    log.info("=" * 60)

    # 1. Load Clean Results
    log.info("Loading clean match data from %s...", CLEAN_RESULTS_CSV.name)
    df = pd.read_csv(CLEAN_RESULTS_CSV, parse_dates=["date"])
    
    # Optional: Fill neutral venue if missing
    if "neutral_venue" not in df.columns:
        df["neutral_venue"] = df["tournament"].str.contains("World Cup|Euro|Copa|Asian Cup|Africa Cup", case=False)

    # 2. Compute Elo Ratings sequentially
    model = EloModel(initial_elo=1500.0)
    df_elo = model.calculate_historical_elos(df)
    
    # 3. Filter for modern era evaluation (e.g. 2014 onwards, competitive only)
    df_eval = df_elo[
        (df_elo["is_played"] == True) &
        (df_elo["date"].dt.year >= 2014) &
        (df_elo["is_competitive"] == True)
    ].copy().reset_index(drop=True)
    
    log.info("Evaluation set: %d competitive matches (2014-present)", len(df_eval))
    
    # 4. Train-Test Split (temporal)
    # Train calibrator on 2014-2022, Evaluate on 2023+
    train_mask = df_eval["date"].dt.year <= 2022
    test_mask = df_eval["date"].dt.year > 2022
    
    train_df = df_eval[train_mask]
    test_df = df_eval[test_mask]
    
    log.info("Calibrating on %d matches (<=2022)...", len(train_df))
    model.fit_calibrator(train_df)
    
    log.info("Evaluating on %d matches (>2022)...", len(test_df))
    
    # 5. Evaluate
    y_test = test_df["result"]
    y_pred = model.predict(test_df)
    y_proba = model.predict_proba(test_df)
    
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=model.calibrator.classes_)
    
    print("\n" + "="*40)
    print("📈 ELO BASELINE MODEL PERFORMANCE")
    print("="*40)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Log Loss: {ll:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=model.calibrator.classes_))
    
    print("\nTop 10 Teams by Current Elo:")
    current_elos = pd.Series(model.ratings).sort_values(ascending=False).head(10)
    print(current_elos.to_string())
    print("="*40)
    
    # Check if we hit the baseline target
    if ll < 0.98:
        log.info("✅ Baseline log-loss of ~0.95 achieved (%.4f)!", ll)
    else:
        log.warning("⚠️ Log loss is higher than expected (%.4f)", ll)


if __name__ == "__main__":
    main()
