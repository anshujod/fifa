"""
fetch_data.py — Data ingestion pipeline for FIFA World Cup 2026 Predictor.

Functions:
    get_elo_ratings()       → Load, clean, and return Elo ratings DataFrame
    get_fifa_rankings()     → Derive FIFA-style rankings from Elo snapshot data
    get_match_results()     → Load cleaned international match results DataFrame

All outputs conform to lowercase snake_case column naming and are ready for
feature engineering, merging with match datasets, and ML model training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import date

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

# Ensure output dirs exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team-name standardisation
# ---------------------------------------------------------------------------
# Mapping from common alternate names → canonical name used in results.csv
TEAM_NAME_MAP: dict[str, str] = {
    # Elo site names that differ from results.csv
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Côte d'Ivoire": "Ivory Coast",  # handle if encountered
    "Cote d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "IR Iran": "Iran",
    "China PR": "China",
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "Timor Leste": "Timor-Leste",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "Cape Verde": "Cabo Verde",
    "Cabo Verde": "Cape Verde",  # results.csv uses Cape Verde
    "Republic of Ireland": "Ireland",
    "Eswatini": "Eswatini",
    "Swaziland": "Eswatini",
    "North Macedonia": "North Macedonia",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
    "Czechia": "Czech Republic",
    "Congo": "Congo Republic",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
}


def _standardise_team_name(name: str) -> str:
    """Normalise a team name to the canonical form used across datasets."""
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)


def _fix_unicode_minus(series: pd.Series) -> pd.Series:
    """Replace Unicode minus (U+2212 '−') with ASCII hyphen-minus '-'."""
    return series.astype(str).str.replace("\u2212", "-", regex=False)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  get_elo_ratings                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def get_elo_ratings(
    raw_path: Path | str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load and return a cleaned Elo ratings DataFrame.

    The raw CSV is the output of the eloratings.net scraper (testt.py).
    It has 16 unnamed numeric columns scraped from the SlickGrid table.

    Parameters
    ----------
    raw_path : Path or str, optional
        Override for the raw CSV location.
        Default: ``data/raw/world_football_elo.csv``
    save : bool
        If True, persist the cleaned DataFrame to
        ``data/processed/elo_ratings.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns::

            rank              – current world rank (int)
            team              – standardised team name (str)
            elo_rating        – current Elo rating (int)
            highest_rank      – all-time highest rank (int)
            highest_elo       – all-time highest Elo rating (int)
            rank_change_1y    – rank change over the last 1 year (int)
            elo_change_1y     – Elo change over the last 1 year (int)
            matches_total     – total matches played (int)
            wins_home         – home wins (int)
            wins_away         – away wins (int)
            draws_home        – home draws (int – split by venue)
            draws_away        – away draws (int)
            losses_home       – home losses (int)
            losses_away       – away losses (int)
            goals_for         – total goals scored (int)
            goals_against     – total goals conceded (int)
            date              – snapshot date (date string, ISO format)
    """
    raw_path = Path(raw_path) if raw_path else RAW_DIR / "world_football_elo.csv"
    log.info("Loading raw Elo ratings from %s", raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Elo CSV not found at {raw_path}. "
            "Run testt.py (Selenium scraper) first to generate it."
        )

    df = pd.read_csv(raw_path, dtype=str)  # read everything as str first

    # ------------------------------------------------------------------
    # Column mapping (scraped column index → meaningful name)
    # ------------------------------------------------------------------
    # The SlickGrid columns from eloratings.net are:
    #   0  = rank
    #   1  = team name
    #   2  = current Elo rating
    #   3  = highest rank achieved
    #   4  = highest Elo rating achieved
    #   5  = rank change over last 1 year
    #   6  = Elo change over last 1 year
    #   7  = total matches played
    #   8  = home wins
    #   9  = away wins
    #  10  = home draws
    #  11  = away draws (using: total draws minus home draws approximation)
    #  12  = home losses
    #  13  = away losses
    #  14  = total goals for
    #  15  = total goals against
    COLUMN_MAP = {
        "0": "rank",
        "1": "team",
        "2": "elo_rating",
        "3": "highest_rank",
        "4": "highest_elo",
        "5": "rank_change_1y",
        "6": "elo_change_1y",
        "7": "matches_total",
        "8": "wins_home",
        "9": "wins_away",
        "10": "draws_home",
        "11": "draws_away",
        "12": "losses_home",
        "13": "losses_away",
        "14": "goals_for",
        "15": "goals_against",
    }
    df = df.rename(columns=COLUMN_MAP)

    log.info("Raw shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Fix Unicode minus signs in numeric columns
    # ------------------------------------------------------------------
    numeric_cols = [c for c in df.columns if c != "team"]
    for col in numeric_cols:
        df[col] = _fix_unicode_minus(df[col])

    # ------------------------------------------------------------------
    # Strip leading + from change columns and coerce to numeric
    # ------------------------------------------------------------------
    for col in numeric_cols:
        df[col] = df[col].str.replace("+", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Standardise team names
    # ------------------------------------------------------------------
    df["team"] = df["team"].apply(_standardise_team_name)

    # ------------------------------------------------------------------
    # Handle missing values
    # ------------------------------------------------------------------
    # Drop rows where team name is missing (scraped artifacts)
    df = df.dropna(subset=["team"])

    # Drop rows where elo_rating is NaN (incomplete scrape rows)
    df = df.dropna(subset=["elo_rating"])

    # Fill remaining NaN in numeric columns with 0
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ------------------------------------------------------------------
    # Cast numeric columns to int
    # ------------------------------------------------------------------
    int_cols = [c for c in numeric_cols if c in df.columns]
    df[int_cols] = df[int_cols].astype(int)

    # ------------------------------------------------------------------
    # Remove duplicate teams (keep first occurrence = highest rank)
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(subset=["team"], keep="first")
    if (dropped := before - len(df)) > 0:
        log.warning("Dropped %d duplicate team rows", dropped)

    # ------------------------------------------------------------------
    # Derived features
    # ------------------------------------------------------------------
    df["wins_total"] = df["wins_home"] + df["wins_away"]
    df["draws_total"] = df["draws_home"] + df["draws_away"]
    df["losses_total"] = df["losses_home"] + df["losses_away"]
    df["goal_diff"] = df["goals_for"] - df["goals_against"]

    # Win percentage (avoid division by zero)
    df["win_pct"] = np.where(
        df["matches_total"] > 0,
        (df["wins_total"] / df["matches_total"] * 100).round(1),
        0.0,
    )

    # ------------------------------------------------------------------
    # Add snapshot date (date the data was scraped)
    # ------------------------------------------------------------------
    df["date"] = date.today().isoformat()

    # ------------------------------------------------------------------
    # Sort by rank and reset index
    # ------------------------------------------------------------------
    df = df.sort_values("rank").reset_index(drop=True)

    log.info(
        "Cleaned Elo ratings: %d teams, columns=%s",
        len(df),
        list(df.columns),
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if save:
        out_path = PROCESSED_DIR / "elo_ratings.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved processed Elo ratings → %s", out_path)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  get_fifa_rankings                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def get_fifa_rankings(
    elo_path: Path | str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Derive a FIFA-style rankings snapshot from the Elo ratings data.

    The eloratings.net snapshot already contains an ordinal rank (column 0)
    and Elo points per team.  This function reshapes that into the standard
    FIFA rankings schema and optionally saves it as
    ``data/raw/fifa_rankings.csv``.

    Parameters
    ----------
    elo_path : Path or str, optional
        Path to the processed Elo ratings CSV.
        Default: ``data/processed/elo_ratings.csv``
    save : bool
        If True, persist the result to ``data/raw/fifa_rankings.csv``.

    Returns
    -------
    pd.DataFrame
        Columns::

            rank        – ordinal rank (1 = best)
            team        – standardised team name
            points      – Elo rating used as ranking points
            rank_change – rank change over the last year (from Elo snapshot)
            date        – snapshot date (ISO string)
    """
    elo_path = Path(elo_path) if elo_path else PROCESSED_DIR / "elo_ratings.csv"

    if not elo_path.exists():
        log.info("Processed Elo not found; calling get_elo_ratings() first.")
        get_elo_ratings(save=True)

    log.info("Deriving FIFA rankings from %s", elo_path)
    elo_df = pd.read_csv(elo_path)

    rankings = pd.DataFrame({
        "rank":        elo_df["rank"],
        "team":        elo_df["team"],
        "points":      elo_df["elo_rating"],
        "rank_change": elo_df["rank_change_1y"],
        "date":        elo_df["date"],
    })

    rankings = rankings.sort_values("rank").reset_index(drop=True)

    log.info("FIFA rankings derived: %d teams", len(rankings))

    if save:
        out_path = RAW_DIR / "fifa_rankings.csv"
        rankings.to_csv(out_path, index=False)
        log.info("Saved FIFA rankings → %s", out_path)

    return rankings


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  get_match_results                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def get_match_results(
    raw_path: Path | str | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Load and return the cleaned international match results DataFrame.

    Parameters
    ----------
    raw_path : Path or str, optional
        Override for the raw CSV location.
        Default: ``data/raw/international_results.csv``
    save : bool
        If True, persist the cleaned DataFrame to
        ``data/processed/match_results.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns::

            date            – match date (datetime64[ns])
            home_team       – standardised home team name
            away_team       – standardised away team name
            home_score      – goals scored by home team (int, NaN for future)
            away_score      – goals scored by away team (int, NaN for future)
            tournament      – competition name
            neutral         – whether venue is neutral (bool)
            result          – 'H' / 'D' / 'A' (NaN for future matches)
            goal_diff       – home_score - away_score (NaN for future)
            total_goals     – home_score + away_score (NaN for future)
    """
    raw_path = Path(raw_path) if raw_path else RAW_DIR / "international_results.csv"
    log.info("Loading raw match results from %s", raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(f"Results CSV not found at {raw_path}.")

    df = pd.read_csv(raw_path)
    log.info("Raw results shape: %s", df.shape)

    # ------------------------------------------------------------------
    # Lowercase column names
    # ------------------------------------------------------------------
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ------------------------------------------------------------------
    # Date parsing
    # ------------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # ------------------------------------------------------------------
    # Standardise team names
    # ------------------------------------------------------------------
    df["home_team"] = df["home_team"].apply(_standardise_team_name)
    df["away_team"] = df["away_team"].apply(_standardise_team_name)

    # ------------------------------------------------------------------
    # Score columns — coerce to numeric (future matches have 'NA')
    # ------------------------------------------------------------------
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # ------------------------------------------------------------------
    # Neutral venue → boolean
    # ------------------------------------------------------------------
    df["neutral"] = df["neutral"].astype(str).str.upper().map(
        {"TRUE": True, "FALSE": False, "1": True, "0": False}
    )
    df["neutral"] = df["neutral"].fillna(False)

    # ------------------------------------------------------------------
    # Derived columns (only for played matches)
    # ------------------------------------------------------------------
    played = df["home_score"].notna() & df["away_score"].notna()

    df.loc[played, "result"] = np.select(
        [
            df.loc[played, "home_score"] > df.loc[played, "away_score"],
            df.loc[played, "home_score"] == df.loc[played, "away_score"],
            df.loc[played, "home_score"] < df.loc[played, "away_score"],
        ],
        ["H", "D", "A"],
        default=None,
    )

    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    # ------------------------------------------------------------------
    # Remove duplicate fixtures
    # ------------------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="first"
    )
    if (dropped := before - len(df)) > 0:
        log.warning("Dropped %d duplicate fixtures", dropped)

    # ------------------------------------------------------------------
    # Sort chronologically
    # ------------------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        "Cleaned results: %d matches (%d played, %d upcoming)",
        len(df),
        played.sum(),
        (~played).sum(),
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if save:
        out_path = PROCESSED_DIR / "match_results.csv"
        df.to_csv(out_path, index=False)
        log.info("Saved processed results → %s", out_path)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI entry point                                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def main() -> None:
    """Run the full data ingestion pipeline."""
    log.info("=" * 60)
    log.info("FIFA WC2026 — Data Ingestion Pipeline")
    log.info("=" * 60)

    # 1. Match results
    results_df = get_match_results()
    print("\n📊 Match Results (sample):")
    print(results_df.head(10).to_string(index=False))
    print(f"\n  → {len(results_df):,} total matches")
    print(f"  → Date range: {results_df['date'].min().date()} to {results_df['date'].max().date()}")
    print(f"  → Tournaments: {results_df['tournament'].nunique()}")

    # 2. Elo ratings
    elo_df = get_elo_ratings()
    print("\n⚽ Elo Ratings (top 20):")
    print(
        elo_df[["rank", "team", "elo_rating", "win_pct", "matches_total"]]
        .head(20)
        .to_string(index=False)
    )
    print(f"\n  → {len(elo_df)} teams rated")

    # 3. FIFA rankings
    rankings_df = get_fifa_rankings()
    print("\n🏆 FIFA Rankings (top 20):")
    print(rankings_df.head(20).to_string(index=False))
    print(f"\n  → {len(rankings_df)} teams ranked")

    # 3. Quick merge demo
    print("\n🔗 Merge demo (latest Elo for each match team):")
    recent = results_df[results_df["home_score"].notna()].tail(5).copy()
    elo_lookup = elo_df.set_index("team")["elo_rating"]
    recent["home_elo"] = recent["home_team"].map(elo_lookup)
    recent["away_elo"] = recent["away_team"].map(elo_lookup)
    print(
        recent[
            ["date", "home_team", "away_team", "home_score", "away_score",
             "result", "home_elo", "away_elo"]
        ].to_string(index=False)
    )

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    main()
