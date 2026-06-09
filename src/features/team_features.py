"""
team_features.py — Rolling performance features for FIFA WC2026 Predictor.

For each match, computes rolling statistics for both home and away teams
using their previous N matches. Features include:

    - goals_scored_avg_5 / _10       — average goals scored
    - goals_conceded_avg_5 / _10     — average goals conceded
    - win_pct_last_5 / _10           — win percentage
    - clean_sheet_ratio_last_10      — proportion of clean sheets
    - form_score                     — weighted W=3, D=1, L=0 (recent 2x)
    - Exponential time-decay variants (λ = 0.005)
    - Separate competitive vs friendly stats

Usage:
    python -m src.features.team_features
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA_DECAY = 0.005  # exponential time-decay parameter
FORM_POINTS = {"W": 3, "D": 1, "L": 0}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Core: extract per-team match record from a row                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _team_perspective(row: dict, team: str) -> dict:
    """
    Convert a match row into a record from a specific team's perspective.

    Returns dict with: date, goals_for, goals_against, result (W/D/L),
    is_competitive, match_importance.
    """
    is_home = row["home_team"] == team
    gf = row["home_goals"] if is_home else row["away_goals"]
    ga = row["away_goals"] if is_home else row["home_goals"]
    match_result = row.get("result", "")

    if is_home:
        result = "W" if match_result == "H" else ("D" if match_result == "D" else "L")
    else:
        result = "W" if match_result == "A" else ("D" if match_result == "D" else "L")

    return {
        "date": row["date"],
        "goals_for": gf,
        "goals_against": ga,
        "result": result,
        "is_competitive": row.get("is_competitive", True),
        "match_importance": row.get("match_importance", 0.5),
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Rolling feature computation for a single team's history                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _compute_rolling_features(
    history: list[dict],
    match_date: pd.Timestamp,
    prefix: str = "",
) -> dict:
    """
    Compute rolling features from a team's match history up to match_date.

    Parameters
    ----------
    history : list of dict
        Previous matches in chronological order (oldest first).
        Each dict has: date, goals_for, goals_against, result,
        is_competitive, match_importance.
    match_date : Timestamp
        The date of the current match (used for time-decay).
    prefix : str
        Column prefix ('home_' or 'away_').

    Returns
    -------
    dict of feature_name → value
    """
    p = prefix

    # Default features when no history
    defaults = {
        f"{p}goals_scored_avg_5": 0.0,
        f"{p}goals_scored_avg_10": 0.0,
        f"{p}goals_conceded_avg_5": 0.0,
        f"{p}goals_conceded_avg_10": 0.0,
        f"{p}win_pct_last_5": 0.0,
        f"{p}win_pct_last_10": 0.0,
        f"{p}clean_sheet_ratio_last_10": 0.0,
        f"{p}form_score": 0.0,
        f"{p}goals_scored_decay": 0.0,
        f"{p}goals_conceded_decay": 0.0,
        f"{p}win_pct_decay": 0.0,
        f"{p}form_score_decay": 0.0,
        f"{p}matches_played": 0,
        f"{p}competitive_goals_scored_avg_5": 0.0,
        f"{p}competitive_goals_conceded_avg_5": 0.0,
        f"{p}competitive_win_pct_last_5": 0.0,
        f"{p}friendly_goals_scored_avg_5": 0.0,
        f"{p}friendly_win_pct_last_5": 0.0,
        f"{p}days_since_last_match": 999,
    }

    if not history:
        return defaults

    # --- Last N subsets ---
    last_5 = history[-5:]
    last_10 = history[-10:]
    all_h = history

    # --- Competitive / Friendly splits (last 5 of each type) ---
    comp_hist = [m for m in history if m["is_competitive"]]
    friendly_hist = [m for m in history if not m["is_competitive"]]
    comp_last_5 = comp_hist[-5:] if comp_hist else []
    friendly_last_5 = friendly_hist[-5:] if friendly_hist else []

    # --- Basic rolling stats ---
    def _avg(matches, key):
        if not matches:
            return 0.0
        return round(np.mean([m[key] for m in matches]), 3)

    def _win_pct(matches):
        if not matches:
            return 0.0
        wins = sum(1 for m in matches if m["result"] == "W")
        return round(wins / len(matches) * 100, 1)

    def _clean_sheet_ratio(matches):
        if not matches:
            return 0.0
        cs = sum(1 for m in matches if m["goals_against"] == 0)
        return round(cs / len(matches) * 100, 1)

    # --- Form score: W=3, D=1, L=0 — recent 2 matches weighted 2x ---
    def _form_score(matches):
        if not matches:
            return 0.0
        vals = [FORM_POINTS.get(m["result"], 0) for m in matches]
        n = len(vals)
        # Weights: older matches get 1.0, most recent 2 get 2.0
        weights = [1.0] * max(0, n - 2) + [2.0] * min(2, n)
        weights = weights[-n:]
        return round(float(np.average(vals, weights=weights)), 2)

    # --- Exponential time-decay features ---
    def _decay_weighted(matches, key, is_binary=False):
        """
        Compute time-decay weighted average.
        weight = exp(-λ * days_since_match)
        """
        if not matches:
            return 0.0
        weights = []
        values = []
        for m in matches:
            days = (match_date - pd.Timestamp(m["date"])).days
            w = np.exp(-LAMBDA_DECAY * max(days, 0))
            weights.append(w)
            if is_binary:
                values.append(1.0 if m["result"] == "W" else 0.0)
            else:
                values.append(float(m[key]))

        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return round(sum(v * w for v, w in zip(values, weights)) / total_w, 3)

    def _decay_form(matches):
        if not matches:
            return 0.0
        weights = []
        values = []
        for m in matches:
            days = (match_date - pd.Timestamp(m["date"])).days
            w = np.exp(-LAMBDA_DECAY * max(days, 0))
            weights.append(w)
            values.append(FORM_POINTS.get(m["result"], 0))
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return round(sum(v * w for v, w in zip(values, weights)) / total_w, 2)

    # Days since last match
    last_match_date = pd.Timestamp(history[-1]["date"])
    days_since = max((match_date - last_match_date).days, 0)

    features = {
        # --- Standard rolling ---
        f"{p}goals_scored_avg_5": _avg(last_5, "goals_for"),
        f"{p}goals_scored_avg_10": _avg(last_10, "goals_for"),
        f"{p}goals_conceded_avg_5": _avg(last_5, "goals_against"),
        f"{p}goals_conceded_avg_10": _avg(last_10, "goals_against"),
        f"{p}win_pct_last_5": _win_pct(last_5),
        f"{p}win_pct_last_10": _win_pct(last_10),
        f"{p}clean_sheet_ratio_last_10": _clean_sheet_ratio(last_10),
        f"{p}form_score": _form_score(last_5),

        # --- Time-decay weighted (all history used) ---
        f"{p}goals_scored_decay": _decay_weighted(all_h, "goals_for"),
        f"{p}goals_conceded_decay": _decay_weighted(all_h, "goals_against"),
        f"{p}win_pct_decay": _decay_weighted(all_h, None, is_binary=True),
        f"{p}form_score_decay": _decay_form(all_h),

        # --- Meta ---
        f"{p}matches_played": len(history),
        f"{p}days_since_last_match": days_since,

        # --- Competitive split ---
        f"{p}competitive_goals_scored_avg_5": _avg(comp_last_5, "goals_for"),
        f"{p}competitive_goals_conceded_avg_5": _avg(comp_last_5, "goals_against"),
        f"{p}competitive_win_pct_last_5": _win_pct(comp_last_5),

        # --- Friendly split ---
        f"{p}friendly_goals_scored_avg_5": _avg(friendly_last_5, "goals_for"),
        f"{p}friendly_win_pct_last_5": _win_pct(friendly_last_5),
    }

    return features


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main pipeline: compute features for every match                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_team_features(
    df: pd.DataFrame | None = None,
    save: bool = True,
    max_history: int = 50,
) -> pd.DataFrame:
    """
    Compute rolling team features for every match in the dataset.

    For each match row, looks up the home and away team's previous matches
    and computes rolling stats with both fixed-window and time-decay methods.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Match records. Loaded from match_records.csv if None.
    save : bool
        Save output to data/processed/match_features.csv
    max_history : int
        Maximum number of previous matches to retain per team (for memory).

    Returns
    -------
    pd.DataFrame
        Original match records with 38 new feature columns appended.
    """
    if df is None:
        path = PROCESSED_DIR / "match_records.csv"
        log.info("Loading match records from %s", path)
        df = pd.read_csv(path, parse_dates=["date"])

    # Only process played matches (have scores)
    played_mask = df["is_played"] == True  # noqa: E712
    played = df[played_mask].copy().sort_values("date").reset_index(drop=True)
    unplayed = df[~played_mask].copy()

    log.info("Computing features for %d played matches "
             "(%d unplayed held aside)", len(played), len(unplayed))

    # --- Build team histories incrementally ---
    team_history: dict[str, list[dict]] = defaultdict(list)
    feature_rows: list[dict] = []

    total = len(played)
    log_interval = max(total // 20, 1)  # log every ~5%

    for idx, row in played.iterrows():
        match_date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        # Compute features from current history (BEFORE this match)
        home_feats = _compute_rolling_features(
            team_history[home], match_date, prefix="home_"
        )
        away_feats = _compute_rolling_features(
            team_history[away], match_date, prefix="away_"
        )

        # Combine
        combined = {**home_feats, **away_feats}
        feature_rows.append(combined)

        # Update history with this match (AFTER computing features)
        home_record = _team_perspective(row.to_dict(), home)
        away_record = _team_perspective(row.to_dict(), away)

        team_history[home].append(home_record)
        team_history[away].append(away_record)

        # Trim history to max_history for memory
        if len(team_history[home]) > max_history:
            team_history[home] = team_history[home][-max_history:]
        if len(team_history[away]) > max_history:
            team_history[away] = team_history[away][-max_history:]

        if (idx + 1) % log_interval == 0:
            pct = (idx + 1) / total * 100
            log.info("  Progress: %d/%d (%.0f%%)", idx + 1, total, pct)

    # --- Join features onto played matches ---
    features_df = pd.DataFrame(feature_rows)
    played = pd.concat(
        [played.reset_index(drop=True), features_df], axis=1
    )

    # --- Add diff features (home - away) ---
    played["feat_goals_scored_diff_5"] = (
        played["home_goals_scored_avg_5"] - played["away_goals_scored_avg_5"]
    )
    played["feat_goals_conceded_diff_5"] = (
        played["home_goals_conceded_avg_5"] - played["away_goals_conceded_avg_5"]
    )
    played["feat_win_pct_diff_5"] = (
        played["home_win_pct_last_5"] - played["away_win_pct_last_5"]
    )
    played["feat_form_diff"] = (
        played["home_form_score"] - played["away_form_score"]
    )
    played["feat_form_decay_diff"] = (
        played["home_form_score_decay"] - played["away_form_score_decay"]
    )

    # --- Recombine with unplayed ---
    result = pd.concat([played, unplayed], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in result.columns if c.startswith(("home_goals_scored",
        "home_goals_conceded", "home_win_pct", "home_clean_sheet",
        "home_form", "home_matches", "home_days", "home_competitive",
        "home_friendly", "away_goals_scored", "away_goals_conceded",
        "away_win_pct", "away_clean_sheet", "away_form", "away_matches",
        "away_days", "away_competitive", "away_friendly", "feat_"))]

    log.info("Generated %d feature columns for %d matches",
             len(feature_cols), len(played))

    if save:
        out = PROCESSED_DIR / "match_features.csv"
        result.to_csv(out, index=False)
        log.info("Saved → %s (%.1f MB)", out, out.stat().st_size / 1e6)

    return result


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 2.1 — Rolling Performance Features")
    log.info("=" * 60)

    result = compute_team_features()

    # --- Display sample ---
    played = result[result["is_played"] == True]  # noqa
    recent = played.tail(10)

    print("\n📊 Feature Engineering Summary")
    print(f"   Total matches:    {len(result):,}")
    print(f"   Feature columns:  {len([c for c in result.columns if c.startswith(('home_goals_scored','home_win','home_form','away_goals_scored','away_win','away_form','feat_'))])}")

    print("\n🔍 Recent match features (last 5):")
    sample_cols = [
        "date", "home_team", "away_team",
        "home_goals_scored_avg_5", "away_goals_scored_avg_5",
        "home_win_pct_last_5", "away_win_pct_last_5",
        "home_form_score", "away_form_score",
        "home_form_score_decay", "away_form_score_decay",
        "feat_form_diff",
    ]
    existing_cols = [c for c in sample_cols if c in recent.columns]
    print(recent[existing_cols].tail(5).to_string(index=False))

    print("\n📈 Competitive vs Friendly split (last 5):")
    split_cols = [
        "date", "home_team",
        "home_competitive_goals_scored_avg_5",
        "home_competitive_win_pct_last_5",
        "home_friendly_goals_scored_avg_5",
        "home_friendly_win_pct_last_5",
    ]
    existing_split = [c for c in split_cols if c in recent.columns]
    print(recent[existing_split].tail(5).to_string(index=False))

    print("\n⏱️  Time-decay features (last 5):")
    decay_cols = [
        "date", "home_team",
        "home_goals_scored_avg_5", "home_goals_scored_decay",
        "home_win_pct_last_5", "home_win_pct_decay",
    ]
    existing_decay = [c for c in decay_cols if c in recent.columns]
    print(recent[existing_decay].tail(5).to_string(index=False))

    print("\n✅ Feature engineering complete.")


if __name__ == "__main__":
    main()
