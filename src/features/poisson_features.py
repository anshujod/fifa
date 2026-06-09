"""
poisson_features.py — Poisson Goal Prediction Features (Task 2.4).

Computes attack strength and defence weakness for both home and away teams,
and calculates expected goals (xG) based on a simplified Dixon-Coles model.

Features:
    - home_attack_strength, away_attack_strength
    - home_defence_weakness, away_defence_weakness
    - expected_goals_home, expected_goals_away

Formulas (per match, using rolling historical stats):
    attack_strength = team_avg_goals_scored / global_avg_goals
    defence_weakness = team_avg_goals_conceded / global_avg_goals
    expected_goals_home = home_attack * away_defence * global_avg_home_goals
    expected_goals_away = away_attack * home_defence * global_avg_away_goals

Usage:
    python -m src.features.poisson_features
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_CSV = PROCESSED_DIR / "match_features.csv"

log = logging.getLogger(__name__)


def compute_poisson_features(
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute Dixon-Coles Poisson features for all matches.

    Uses the previously computed time-decay features (e.g., home_goals_scored_decay)
    for team averages to reflect current form. Global averages are computed
    as expanding windows to avoid data leakage.
    """
    if df is None:
        log.info("Loading match features from %s", FEATURES_CSV)
        df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])

    log.info("Computing Poisson features for %d matches", len(df))

    # We only use played matches to compute expanding global averages
    played = df[df["is_played"] == True].copy().sort_values("date").reset_index(drop=True)
    unplayed = df[df["is_played"] == False].copy()

    # Calculate global expanding averages (shifting by 1 to avoid data leakage)
    # i.e., average goals scored across all international matches BEFORE the current one
    played["global_home_goals_avg"] = played["home_goals"].expanding().mean().shift(1)
    played["global_away_goals_avg"] = played["away_goals"].expanding().mean().shift(1)

    # For the very first match, we don't have a prior average, so fill with reasonable defaults (e.g., historical averages)
    played["global_home_goals_avg"] = played["global_home_goals_avg"].fillna(1.5)
    played["global_away_goals_avg"] = played["global_away_goals_avg"].fillna(1.1)

    # The global avg goals per team (regardless of home/away)
    played["global_goals_avg"] = (played["global_home_goals_avg"] + played["global_away_goals_avg"]) / 2

    # We will use the time-decayed goals scored/conceded as the "team average"
    # This responds faster to team form changes than an all-time average.
    
    # 1. Attack Strength & Defence Weakness
    # team_avg_scored / global_avg
    played["home_attack_strength"] = played["home_goals_scored_decay"] / played["global_goals_avg"]
    played["away_attack_strength"] = played["away_goals_scored_decay"] / played["global_goals_avg"]

    played["home_defence_weakness"] = played["home_goals_conceded_decay"] / played["global_goals_avg"]
    played["away_defence_weakness"] = played["away_goals_conceded_decay"] / played["global_goals_avg"]

    # Handle missing/zero division gracefully
    for col in ["home_attack_strength", "away_attack_strength", "home_defence_weakness", "away_defence_weakness"]:
        played[col] = played[col].fillna(1.0).replace([float('inf'), -float('inf')], 1.0)
        played[col] = played[col].round(3)

    # 2. Expected Goals (Dixon-Coles style)
    # home_xG = home_attack * away_defence * global_home_avg
    # If neutral venue, we use global_goals_avg instead of home/away specific advantage
    
    is_neutral = played["neutral_venue"] == 1

    # Base expectations depending on venue
    base_home_xg = played["global_home_goals_avg"].where(~is_neutral, played["global_goals_avg"])
    base_away_xg = played["global_away_goals_avg"].where(~is_neutral, played["global_goals_avg"])

    played["expected_goals_home"] = (
        played["home_attack_strength"] * played["away_defence_weakness"] * base_home_xg
    ).round(3)

    played["expected_goals_away"] = (
        played["away_attack_strength"] * played["home_defence_weakness"] * base_away_xg
    ).round(3)

    # Recombine with unplayed matches
    # For unplayed matches, we use the final global averages from the played dataset
    final_home_avg = played["global_home_goals_avg"].iloc[-1]
    final_away_avg = played["global_away_goals_avg"].iloc[-1]
    final_global_avg = played["global_goals_avg"].iloc[-1]

    unplayed["global_home_goals_avg"] = final_home_avg
    unplayed["global_away_goals_avg"] = final_away_avg
    unplayed["global_goals_avg"] = final_global_avg

    unplayed["home_attack_strength"] = (unplayed["home_goals_scored_decay"] / final_global_avg).round(3)
    unplayed["away_attack_strength"] = (unplayed["away_goals_scored_decay"] / final_global_avg).round(3)
    unplayed["home_defence_weakness"] = (unplayed["home_goals_conceded_decay"] / final_global_avg).round(3)
    unplayed["away_defence_weakness"] = (unplayed["away_goals_conceded_decay"] / final_global_avg).round(3)

    unplayed_is_neutral = unplayed["neutral_venue"] == 1
    u_base_home_xg = pd.Series(final_home_avg, index=unplayed.index).where(~unplayed_is_neutral, final_global_avg)
    u_base_away_xg = pd.Series(final_away_avg, index=unplayed.index).where(~unplayed_is_neutral, final_global_avg)

    unplayed["expected_goals_home"] = (
        unplayed["home_attack_strength"] * unplayed["away_defence_weakness"] * u_base_home_xg
    ).round(3)

    unplayed["expected_goals_away"] = (
        unplayed["away_attack_strength"] * unplayed["home_defence_weakness"] * u_base_away_xg
    ).round(3)

    # Combine back
    result = pd.concat([played, unplayed], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    new_cols = [
        "home_attack_strength", "away_attack_strength",
        "home_defence_weakness", "away_defence_weakness",
        "expected_goals_home", "expected_goals_away"
    ]
    
    log.info("Added %d Poisson feature columns", len(new_cols))

    if save:
        result.to_csv(FEATURES_CSV, index=False)
        log.info("Saved → %s (%.1f MB)", FEATURES_CSV, FEATURES_CSV.stat().st_size / 1e6)

    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 2.4 — Poisson Features (Goal Prediction)")
    log.info("=" * 60)

    result = compute_poisson_features()

    print("\n📊 Poisson Features Summary")
    print(f"   Total matches:      {len(result):,}")

    played = result[result["is_played"] == True]  # noqa
    recent = played.tail(10)

    print("\n📈 Attack & Defence Strengths (last 5 played):")
    cols = ["date", "home_team", "away_team", "home_attack_strength", "away_defence_weakness", "expected_goals_home"]
    print(recent[cols].tail(5).to_string(index=False))

    print("\n🔮 Expected Goals vs Actual (last 5 played):")
    xg_cols = ["date", "home_team", "away_team", "expected_goals_home", "home_goals", "expected_goals_away", "away_goals"]
    print(recent[xg_cols].tail(5).to_string(index=False))
    
    print("\n✅ Poisson features complete.")


if __name__ == "__main__":
    main()
