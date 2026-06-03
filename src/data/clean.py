"""
clean.py — Data cleaning pipeline for FIFA World Cup 2026 Predictor.

Functions:
    standardize_team_names(df)     → Normalise all team name variants
    remove_duplicates(df)          → Drop duplicate fixtures
    handle_missing_scores(df)      → Flag/remove abandoned or unplayed matches
    filter_competitive_matches(df) → Split competitive vs friendly matches
    add_match_metadata(df)         → Enrich with derived columns
    run_cleaning_pipeline(df)      → Orchestrate the full pipeline

Output:
    data/processed/clean_results.csv        — Full cleaned dataset
    data/processed/competitive_results.csv  — Competitive matches only
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.fetch_data import (
    _standardise_team_name,
    TEAM_NAME_MAP,
    PROCESSED_DIR,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tournament classification
# ---------------------------------------------------------------------------
# Competitive tournaments (non-friendly, non-exhibition)
MAJOR_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "Copa América",
    "Copa América qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "Gold Cup",
    "Gold Cup qualification",
    "CONCACAF Nations League",
    "UEFA Nations League",
    "Confederations Cup",
    "OFC Nations Cup",
    "OFC Nations Cup qualification",
}

COMPETITIVE_KEYWORDS = [
    "World Cup",
    "qualification",
    "Euro ",
    "Nations League",
    "Nations Cup",
    "Cup of Nations",
    "Copa América",
    "Gold Cup",
    "Asian Cup",
    "Confederations Cup",
    "Intercontinental Cup",
    "COSAFA Cup",
    "CECAFA Cup",
    "CFU Caribbean Cup",
    "AFF Championship",
    "SAFF Championship",
    "Gulf Cup",
    "Arab Cup",
    "Baltic Cup",
    "CONCACAF Championship",
]

FRIENDLY_KEYWORDS = [
    "Friendly",
    "friendly",
]


def _is_competitive(tournament: str) -> bool:
    """Return True if the tournament is competitive (not a friendly)."""
    if tournament in MAJOR_TOURNAMENTS:
        return True
    return any(kw in tournament for kw in COMPETITIVE_KEYWORDS)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Pipeline functions                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise team name aliases across the dataset.

    Applies the canonical mapping from fetch_data.TEAM_NAME_MAP plus
    handles the former_names.csv mapping from the Kaggle dataset.

    Returns a copy with standardised team columns.
    """
    df = df.copy()
    df["home_team"] = df["home_team"].apply(_standardise_team_name)
    df["away_team"] = df["away_team"].apply(_standardise_team_name)

    # Also load former_names.csv if available
    former_path = Path(__file__).resolve().parents[2] / "former_names.csv"
    if former_path.exists():
        former = pd.read_csv(former_path)
        former_map = dict(zip(former["former"], former["current"]))
        # Only apply mappings that don't conflict with our primary map
        for old, new in former_map.items():
            if old not in TEAM_NAME_MAP:
                df["home_team"] = df["home_team"].replace(old, new)
                df["away_team"] = df["away_team"].replace(old, new)
        log.info(
            "Applied %d former-name mappings from former_names.csv",
            len(former_map),
        )

    log.info("Team names standardised — %d unique teams", 
             len(set(df["home_team"]) | set(df["away_team"])))
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and drop duplicate fixtures (same date + teams).

    Keeps the first occurrence.
    """
    before = len(df)
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    )
    dropped = before - len(df)
    if dropped > 0:
        log.warning("Removed %d duplicate fixtures", dropped)
    else:
        log.info("No duplicate fixtures found")
    return df


def handle_missing_scores(
    df: pd.DataFrame,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Flag and optionally remove matches with missing scores.

    Adds a boolean ``is_played`` column.
    Future/scheduled matches (no score) are flagged but kept by default.

    Parameters
    ----------
    drop_missing : bool
        If True, remove rows where scores are NaN.
    """
    df = df.copy()
    df["is_played"] = df["home_score"].notna() & df["away_score"].notna()

    unplayed = (~df["is_played"]).sum()
    log.info(
        "Matches: %d played, %d unplayed/missing scores",
        df["is_played"].sum(),
        unplayed,
    )

    if drop_missing and unplayed > 0:
        df = df[df["is_played"]].copy()
        log.info("Dropped %d unplayed matches (drop_missing=True)", unplayed)

    return df


def filter_competitive_matches(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into competitive and friendly matches.

    Returns
    -------
    competitive : pd.DataFrame
        Matches from competitive tournaments.
    friendly : pd.DataFrame
        Friendly / exhibition matches.
    """
    df = df.copy()
    df["is_competitive"] = df["tournament"].apply(_is_competitive)

    competitive = df[df["is_competitive"]].copy()
    friendly = df[~df["is_competitive"]].copy()

    log.info(
        "Split: %d competitive, %d friendly/exhibition",
        len(competitive),
        len(friendly),
    )

    return competitive, friendly


def add_match_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the dataset with derived columns useful for feature engineering.

    New columns:
        year, month          — from date
        decade               — e.g. 2020
        is_competitive       — boolean
        match_importance      — weight: WC=1.0, WC qual=0.8, continental=0.7,
                                 nations league=0.6, friendly=0.3
        result               — H/D/A (for played matches)
        goal_diff            — home_score - away_score
        total_goals          — sum of scores
    """
    df = df.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["decade"] = (df["year"] // 10) * 10

    # Tournament classification
    df["is_competitive"] = df["tournament"].apply(_is_competitive)

    # Match importance weighting
    importance_map = {
        "FIFA World Cup": 1.0,
        "Copa América": 0.75,
        "UEFA Euro": 0.75,
        "African Cup of Nations": 0.70,
        "AFC Asian Cup": 0.70,
        "Gold Cup": 0.70,
        "Confederations Cup": 0.70,
        "CONCACAF Nations League": 0.60,
        "UEFA Nations League": 0.60,
        "OFC Nations Cup": 0.60,
    }

    def _get_importance(tournament: str) -> float:
        if tournament in importance_map:
            return importance_map[tournament]
        if "World Cup qualification" in tournament:
            return 0.80
        if "qualification" in tournament:
            return 0.65
        if any(kw in tournament for kw in COMPETITIVE_KEYWORDS):
            return 0.55
        return 0.30  # friendlies

    df["match_importance"] = df["tournament"].apply(_get_importance)

    # Result and goal features (only for played matches)
    played = df["home_score"].notna() & df["away_score"].notna()

    if "result" not in df.columns:
        df["result"] = np.nan

    df.loc[played, "result"] = np.select(
        [
            df.loc[played, "home_score"] > df.loc[played, "away_score"],
            df.loc[played, "home_score"] == df.loc[played, "away_score"],
        ],
        ["H", "D"],
        default="A",
    )

    if "goal_diff" not in df.columns:
        df["goal_diff"] = np.nan
    if "total_goals" not in df.columns:
        df["total_goals"] = np.nan

    df.loc[played, "goal_diff"] = (
        df.loc[played, "home_score"] - df.loc[played, "away_score"]
    )
    df.loc[played, "total_goals"] = (
        df.loc[played, "home_score"] + df.loc[played, "away_score"]
    )

    log.info("Added match metadata columns: year, month, decade, "
             "is_competitive, match_importance")

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Full cleaning pipeline                                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_cleaning_pipeline(
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Run the full data cleaning pipeline.

    Steps:
        1. Standardise team names
        2. Remove duplicate fixtures
        3. Flag missing scores
        4. Add match metadata (importance, time features)
        5. Split competitive vs friendly

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input match results. If None, loads from processed/match_results.csv
    save : bool
        Persist output CSVs to data/processed/

    Returns
    -------
    dict with keys:
        'all'          — Full cleaned dataset
        'competitive'  — Competitive matches only
        'friendly'     — Friendly matches only
    """
    if df is None:
        path = PROCESSED_DIR / "match_results.csv"
        log.info("Loading from %s", path)
        df = pd.read_csv(path, parse_dates=["date"])

    log.info("Starting cleaning pipeline — %d rows", len(df))

    # Step 1 — Team names
    df = standardize_team_names(df)

    # Step 2 — Duplicates
    df = remove_duplicates(df)

    # Step 3 — Missing scores
    df = handle_missing_scores(df)

    # Step 4 — Metadata
    df = add_match_metadata(df)

    # Step 5 — Competitive split
    competitive, friendly = filter_competitive_matches(df)

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)
    competitive = competitive.sort_values("date").reset_index(drop=True)
    friendly = friendly.sort_values("date").reset_index(drop=True)

    # Persist
    if save:
        df.to_csv(PROCESSED_DIR / "clean_results.csv", index=False)
        competitive.to_csv(PROCESSED_DIR / "competitive_results.csv", index=False)
        friendly.to_csv(PROCESSED_DIR / "friendly_results.csv", index=False)
        log.info("Saved: clean_results.csv, competitive_results.csv, "
                 "friendly_results.csv → %s", PROCESSED_DIR)

    result = {
        "all": df,
        "competitive": competitive,
        "friendly": friendly,
    }

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
    log.info("FIFA WC2026 — Data Cleaning Pipeline")
    log.info("=" * 60)

    result = run_cleaning_pipeline()

    df = result["all"]
    comp = result["competitive"]

    print("\n📊 Cleaning Summary")
    print(f"  Total matches:       {len(df):,}")
    print(f"  Played:              {df['is_played'].sum():,}")
    print(f"  Upcoming:            {(~df['is_played']).sum():,}")
    print(f"  Competitive:         {len(comp):,}")
    print(f"  Friendly:            {len(result['friendly']):,}")
    print(f"  Unique teams:        {len(set(df['home_team']) | set(df['away_team']))}")
    print(f"  Date range:          {df['date'].min().date()} → {df['date'].max().date()}")

    print("\n🏆 Tournament importance distribution:")
    print(
        df.groupby("match_importance")["tournament"]
        .apply(lambda x: f"{len(x):,} matches ({x.iloc[0]}...)")
        .to_string()
    )

    print("\n📈 Competitive matches by tournament type (top 15):")
    print(comp["tournament"].value_counts().head(15).to_string())

    print("\n🕐 Matches by decade:")
    print(df.groupby("decade").size().tail(10).to_string())

    print("\n✅ Cleaning pipeline complete.")


if __name__ == "__main__":
    main()
