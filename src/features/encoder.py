from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MATCH_FEATURES_CSV = PROCESSED_DIR / "match_features.csv"
SQUAD_FEATURES_CSV = PROCESSED_DIR / "squad_features.csv"

log = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Assembles ML-ready feature vectors for matches.
    """
    
    def __init__(self):
        self.confederation_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fit = False
        
        # Load static squad features for 2026 World Cup teams
        # Used for future match simulation
        if SQUAD_FEATURES_CSV.exists():
            self.squad_features = pd.read_csv(SQUAD_FEATURES_CSV)
        else:
            self.squad_features = None

    def _fit_encoders(self, df: pd.DataFrame):
        """Fit the confederation one-hot encoder."""
        confeds = pd.concat([
            df[["home_confederation"]].rename(columns={"home_confederation": "confederation"}),
            df[["away_confederation"]].rename(columns={"away_confederation": "confederation"})
        ])
        # Handle potential missing or NaN
        confeds["confederation"] = confeds["confederation"].fillna("OTHER")
        self.confederation_encoder.fit(confeds)
        self.is_fit = True

    def _encode_confederations(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode home and away confederations."""
        if not self.is_fit:
            raise ValueError("Encoders not fit yet. Call prepare_training_data first.")
            
        home_conf = df[["home_confederation"]].fillna("OTHER").rename(columns={"home_confederation": "confederation"})
        away_conf = df[["away_confederation"]].fillna("OTHER").rename(columns={"away_confederation": "confederation"})
        
        home_encoded = self.confederation_encoder.transform(home_conf)
        away_encoded = self.confederation_encoder.transform(away_conf)
        
        categories = self.confederation_encoder.categories_[0]
        home_cols = [f"home_conf_{c}" for c in categories]
        away_cols = [f"away_conf_{c}" for c in categories]
        
        home_df = pd.DataFrame(home_encoded, columns=home_cols, index=df.index)
        away_df = pd.DataFrame(away_encoded, columns=away_cols, index=df.index)
        
        return pd.concat([home_df, away_df], axis=1)

    def prepare_training_data(
        self,
        start_year: int = 2010,
        competitive_only: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads match_features.csv, applies encodings, and returns X and y datasets.
        
        Returns:
            X: Feature matrix
            y_outcome: Match outcome (H, D, A)
            y_home_goals: Home goals scored
            y_away_goals: Away goals scored
        """
        log.info("Loading training data from %s...", MATCH_FEATURES_CSV)
        df = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
        
        # Filter training set
        df = df[df["is_played"] == True].copy()
        df = df[df["year"] >= start_year].copy()
        
        if competitive_only:
            df = df[df["is_competitive"] == True].copy()
            
        df = df.reset_index(drop=True)
        log.info("Training set size: %d matches (start_year=%d, competitive_only=%s)", 
                 len(df), start_year, competitive_only)
        
        # Fit categorical encoders
        self._fit_encoders(df)
        
        # 1. Base continuous features (Elo, Rolling stats, Poisson, H2H, Contextual)
        base_features = [
            # Elo Embedding
            "home_elo_before", "away_elo_before", "elo_diff",
            "confederation_elo_diff",
            
            # Poisson
            "home_attack_strength", "away_attack_strength",
            "home_defence_weakness", "away_defence_weakness",
            "expected_goals_home", "expected_goals_away",
            
            # Rolling Form / TeamStrengthEncoder (Decay stats preferred for ML)
            "home_form_score_decay", "away_form_score_decay", "feat_form_decay_diff",
            "home_goals_scored_decay", "away_goals_scored_decay",
            "home_goals_conceded_decay", "away_goals_conceded_decay",
            "home_win_pct_decay", "away_win_pct_decay",
            
            # Fixed-window as backup context
            "home_goals_scored_avg_5", "away_goals_scored_avg_5",
            "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
            
            # H2H
            "h2h_matches", "h2h_win_pct", "h2h_goals_diff",
            
            # Context
            "neutral_venue", "match_importance",
            "home_wc_experience_norm", "away_wc_experience_norm",
            "wc_experience_diff",
        ]
        
        # Filter base_features to only those that exist in df
        base_features = [f for f in base_features if f in df.columns]
        X_base = df[base_features].copy()
        
        # 2. Add categorical encodings
        X_cat = self._encode_confederations(df)
        
        # Combine X
        X = pd.concat([X_base, X_cat], axis=1)
        
        # Impute any remaining NaNs with 0 (e.g. fresh H2H)
        X = X.fillna(0)
        
        # Prepare targets
        y_outcome = df["result"]  # 'H', 'D', 'A'
        y_home_goals = df["home_goals"]
        y_away_goals = df["away_goals"]
        
        log.info("Prepared feature matrix X: %s", X.shape)
        
        return X, y_outcome, y_home_goals, y_away_goals


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 2.5 — Team Encoding & Representation")
    log.info("=" * 60)

    pipeline = FeaturePipeline()
    X, y_outcome, y_home_goals, y_away_goals = pipeline.prepare_training_data(
        start_year=2014, 
        competitive_only=True
    )
    
    print("\n📊 Feature Pipeline Summary:")
    print(f"   Matches: {len(X)}")
    print(f"   Features: {X.shape[1]}")
    
    print("\n🔍 Sample Feature Matrix (X):")
    sample_cols = [
        "home_elo_before", "away_elo_before", "expected_goals_home",
        "home_form_score_decay", "h2h_win_pct", "home_conf_UEFA", "away_conf_CONMEBOL"
    ]
    # filter columns that actually exist just in case
    show_cols = [c for c in sample_cols if c in X.columns]
    print(X[show_cols].head(5).to_string(index=False))
    
    print("\n🎯 Sample Targets:")
    targets = pd.DataFrame({
        "result (Class)": y_outcome,
        "home_goals (Poisson)": y_home_goals,
        "away_goals (Poisson)": y_away_goals
    })
    print(targets.head(5).to_string(index=False))
    
    print("\n✅ Team Encoding & Representation complete.")


if __name__ == "__main__":
    main()
