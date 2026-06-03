"""
schema.py — Unified data schemas for FIFA World Cup 2026 Predictor.

Defines three core schemas and the functions to materialise them:

1. MatchRecord  — One row per match with Elo ratings joined
2. TeamSnapshot — Rolling stats for any team at any given date
3. SquadProfile — Aggregate squad-level metrics per team (2026 tournament)

Usage:
    python -m src.data.schema          # Build all schemas and save to data/processed/
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SQUAD_DIR = RAW_DIR / "squads_2026"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. MATCH RECORD SCHEMA                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Master schema for every international match.
#
# Columns:
#   match_id          str    – deterministic ID: YYYYMMDD_HomeTeam_AwayTeam
#   date              date   – match date
#   home_team         str    – standardised home team name
#   away_team         str    – standardised away team name
#   home_goals        int    – goals scored by home team (NaN if unplayed)
#   away_goals        int    – goals scored by away team (NaN if unplayed)
#   tournament_type   str    – tournament name
#   neutral_venue     bool   – True if played on neutral ground
#   home_elo_before   int    – home team Elo rating before the match
#   away_elo_before   int    – away team Elo rating before the match
#   elo_diff          int    – home_elo_before − away_elo_before
#   result            str    – 'H' (home win) / 'D' (draw) / 'A' (away win)
#   goal_diff         int    – home_goals − away_goals
#   total_goals       int    – home_goals + away_goals
#   is_competitive    bool   – competitive match (not a friendly)
#   match_importance  float  – weight: WC=1.0, qualifiers=0.8, friendly=0.3
#   year              int    – year extracted from date
#   month             int    – month extracted from date
#   decade            int    – decade (e.g. 2020)
#   is_played         bool   – whether the match has been played
# ═══════════════════════════════════════════════════════════════════════════

MATCH_RECORD_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "tournament_type",
    "neutral_venue",
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "result",
    "goal_diff",
    "total_goals",
    "is_competitive",
    "match_importance",
    "year",
    "month",
    "decade",
    "is_played",
]


def _generate_match_id(row: pd.Series) -> str:
    """Create a deterministic match ID from date + teams."""
    d = pd.Timestamp(row["date"]).strftime("%Y%m%d")
    home = row["home_team"].replace(" ", "")
    away = row["away_team"].replace(" ", "")
    return f"{d}_{home}_v_{away}"


def build_match_records(save: bool = True) -> pd.DataFrame:
    """
    Build the unified match_record table by joining clean results
    with Elo ratings.

    Returns a DataFrame conforming to the MATCH_RECORD schema.
    """
    # Load cleaned results
    results_path = PROCESSED_DIR / "clean_results.csv"
    log.info("Loading clean results from %s", results_path)
    df = pd.read_csv(results_path, parse_dates=["date"])

    # Load Elo ratings (current snapshot — lookup table)
    elo_path = PROCESSED_DIR / "elo_ratings.csv"
    log.info("Loading Elo ratings from %s", elo_path)
    elo = pd.read_csv(elo_path)
    elo_lookup = elo.set_index("team")["elo_rating"].to_dict()

    # --- Rename columns to match schema ---
    df = df.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
        "tournament": "tournament_type",
        "neutral": "neutral_venue",
    })

    # --- Join Elo ratings ---
    df["home_elo_before"] = df["home_team"].map(elo_lookup)
    df["away_elo_before"] = df["away_team"].map(elo_lookup)
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]

    # Fill missing Elo with a baseline (1500 = default new team)
    for col in ["home_elo_before", "away_elo_before"]:
        df[col] = df[col].fillna(1500).astype(int)
    df["elo_diff"] = df["elo_diff"].fillna(0).astype(int)

    # --- Generate match_id ---
    df["match_id"] = df.apply(_generate_match_id, axis=1)

    # --- Ensure all schema columns exist ---
    for col in MATCH_RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # --- Select and order columns ---
    df = df[MATCH_RECORD_COLUMNS].copy()

    # --- Sort ---
    df = df.sort_values("date").reset_index(drop=True)

    log.info("Built match_record table: %d rows, %d columns",
             len(df), len(df.columns))

    if save:
        out = PROCESSED_DIR / "match_records.csv"
        df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. TEAM SNAPSHOT SCHEMA                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Rolling statistics for a team computed at any given date.
# Used for feature engineering — one snapshot per team per match.
#
# Columns:
#   team                       str   – team name
#   snapshot_date              date  – the date of the snapshot
#   elo_rating                 int   – Elo at snapshot date
#   matches_played_total       int   – total matches played to date
#   matches_played_competitive int   – competitive matches to date
#   wins_last_5                int   – wins in last 5 matches
#   wins_last_10               int   – wins in last 10 matches
#   draws_last_5               int   – draws in last 5
#   draws_last_10              int   – draws in last 10
#   losses_last_5              int   – losses in last 5
#   losses_last_10             int   – losses in last 10
#   goals_scored_avg_5         float – avg goals scored last 5
#   goals_scored_avg_10        float – avg goals scored last 10
#   goals_conceded_avg_5       float – avg goals conceded last 5
#   goals_conceded_avg_10      float – avg goals conceded last 10
#   clean_sheets_last_10       int   – clean sheets in last 10
#   win_pct_last_5             float – win % in last 5
#   win_pct_last_10            float – win % in last 10
#   form_score                 float – weighted form: W=3, D=1, L=0 (recent 2x)
#   goal_diff_avg_10           float – avg goal diff last 10
# ═══════════════════════════════════════════════════════════════════════════

TEAM_SNAPSHOT_COLUMNS = [
    "team",
    "snapshot_date",
    "elo_rating",
    "matches_played_total",
    "matches_played_competitive",
    "wins_last_5",
    "wins_last_10",
    "draws_last_5",
    "draws_last_10",
    "losses_last_5",
    "losses_last_10",
    "goals_scored_avg_5",
    "goals_scored_avg_10",
    "goals_conceded_avg_5",
    "goals_conceded_avg_10",
    "clean_sheets_last_10",
    "win_pct_last_5",
    "win_pct_last_10",
    "form_score",
    "goal_diff_avg_10",
]


def build_team_snapshot(
    team: str,
    as_of_date: str | pd.Timestamp,
    match_records: pd.DataFrame | None = None,
) -> dict:
    """
    Compute rolling stats for a team up to (but not including) a given date.

    Parameters
    ----------
    team : str
        Team name (canonical).
    as_of_date : str or Timestamp
        Snapshot cutoff date.
    match_records : DataFrame, optional
        Pre-loaded match_records table. Loaded from CSV if None.

    Returns
    -------
    dict conforming to TEAM_SNAPSHOT_COLUMNS.
    """
    if match_records is None:
        match_records = pd.read_csv(
            PROCESSED_DIR / "match_records.csv", parse_dates=["date"]
        )

    as_of = pd.Timestamp(as_of_date)

    # Filter: all played matches for this team before the snapshot date
    mask = (
        (match_records["date"] < as_of)
        & match_records["is_played"]
        & (
            (match_records["home_team"] == team)
            | (match_records["away_team"] == team)
        )
    )
    history = match_records.loc[mask].sort_values("date")

    if history.empty:
        return {col: (team if col == "team"
                       else as_of_date if col == "snapshot_date"
                       else 0)
                for col in TEAM_SNAPSHOT_COLUMNS}

    # --- Compute per-match stats from team's perspective ---
    records = []
    for _, row in history.iterrows():
        is_home = row["home_team"] == team
        records.append({
            "date": row["date"],
            "goals_for": row["home_goals"] if is_home else row["away_goals"],
            "goals_against": row["away_goals"] if is_home else row["home_goals"],
            "result": (
                "W" if (is_home and row["result"] == "H")
                     or (not is_home and row["result"] == "A")
                else "D" if row["result"] == "D"
                else "L"
            ),
            "is_competitive": row["is_competitive"],
        })
    hist = pd.DataFrame(records)

    last_5 = hist.tail(5)
    last_10 = hist.tail(10)

    # Form score: W=3, D=1, L=0 — recent matches weighted 2x
    form_points = {"W": 3, "D": 1, "L": 0}
    form_vals = last_5["result"].map(form_points).values
    n = len(form_vals)
    if n > 0:
        weights = np.array([1.0] * max(0, n - 2) + [2.0] * min(2, n))
        weights = weights[-n:]  # trim to actual length
        form_score = float(np.average(form_vals, weights=weights))
    else:
        form_score = 0.0

    # Elo lookup
    elo_df = pd.read_csv(PROCESSED_DIR / "elo_ratings.csv")
    elo_map = elo_df.set_index("team")["elo_rating"].to_dict()
    elo_val = elo_map.get(team, 1500)

    snapshot = {
        "team": team,
        "snapshot_date": str(as_of.date()),
        "elo_rating": elo_val,
        "matches_played_total": len(hist),
        "matches_played_competitive": int(hist["is_competitive"].sum()),
        "wins_last_5": int((last_5["result"] == "W").sum()),
        "wins_last_10": int((last_10["result"] == "W").sum()),
        "draws_last_5": int((last_5["result"] == "D").sum()),
        "draws_last_10": int((last_10["result"] == "D").sum()),
        "losses_last_5": int((last_5["result"] == "L").sum()),
        "losses_last_10": int((last_10["result"] == "L").sum()),
        "goals_scored_avg_5": round(last_5["goals_for"].mean(), 2),
        "goals_scored_avg_10": round(last_10["goals_for"].mean(), 2),
        "goals_conceded_avg_5": round(last_5["goals_against"].mean(), 2),
        "goals_conceded_avg_10": round(last_10["goals_against"].mean(), 2),
        "clean_sheets_last_10": int((last_10["goals_against"] == 0).sum()),
        "win_pct_last_5": round(
            (last_5["result"] == "W").sum() / len(last_5) * 100, 1
        ),
        "win_pct_last_10": round(
            (last_10["result"] == "W").sum() / len(last_10) * 100, 1
        ),
        "form_score": round(form_score, 2),
        "goal_diff_avg_10": round(
            (last_10["goals_for"] - last_10["goals_against"]).mean(), 2
        ),
    }

    return snapshot


def build_all_team_snapshots(
    as_of_date: str = "2026-06-11",
    save: bool = True,
) -> pd.DataFrame:
    """
    Build team snapshots for all teams at a given date (default: WC2026 start).

    Returns DataFrame with one row per team.
    """
    match_records = pd.read_csv(
        PROCESSED_DIR / "match_records.csv", parse_dates=["date"]
    )
    # Get all unique teams
    teams = sorted(
        set(match_records["home_team"]) | set(match_records["away_team"])
    )
    log.info("Building snapshots for %d teams as of %s", len(teams), as_of_date)

    snapshots = [
        build_team_snapshot(team, as_of_date, match_records)
        for team in teams
    ]
    df = pd.DataFrame(snapshots)
    df = df.sort_values("elo_rating", ascending=False).reset_index(drop=True)

    if save:
        out = PROCESSED_DIR / "team_snapshots.csv"
        df.to_csv(out, index=False)
        log.info("Saved → %s (%d teams)", out, len(df))

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. SQUAD PROFILE SCHEMA                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# Aggregate squad-level metrics per team for the 2026 World Cup.
# Built from the per-player JSON files in data/raw/squads_2026/.
#
# Columns:
#   team                      str   – team name
#   squad_size                int   – number of players in squad
#   avg_age                   float – average player age
#   avg_caps                  float – average international caps
#   avg_goals                 float – average international goals
#   total_caps                int   – sum of all player caps
#   total_goals               int   – sum of all player goals
#   squad_market_value_eur    float – total squad market value (€M)
#   avg_market_value_eur      float – average player market value (€M)
#   star_player_value_eur     float – top 3 players combined value (€M)
#   gk_count                  int   – number of goalkeepers
#   df_count                  int   – number of defenders
#   mf_count                  int   – number of midfielders
#   fw_count                  int   – number of forwards
#   experience_score          float – weighted experience metric
#   youth_ratio               float – % of players aged ≤ 23
#   veteran_ratio             float – % of players aged ≥ 32
# ═══════════════════════════════════════════════════════════════════════════

SQUAD_PROFILE_COLUMNS = [
    "team",
    "squad_size",
    "avg_age",
    "avg_caps",
    "avg_goals",
    "total_caps",
    "total_goals",
    "squad_market_value_eur",
    "avg_market_value_eur",
    "star_player_value_eur",
    "gk_count",
    "df_count",
    "mf_count",
    "fw_count",
    "experience_score",
    "youth_ratio",
    "veteran_ratio",
]


def _parse_market_value(mv_str: str | None) -> float:
    """
    Parse market value strings like '€18.00m', '€500k' to float (in millions).

    Returns 0.0 if unparseable.
    """
    if not mv_str or not isinstance(mv_str, str):
        return 0.0

    mv_str = mv_str.strip().lower()
    # Remove currency symbols
    mv_str = re.sub(r"[€$£]", "", mv_str).strip()

    try:
        if "m" in mv_str:
            return float(mv_str.replace("m", "").strip())
        elif "k" in mv_str:
            return float(mv_str.replace("k", "").strip()) / 1000.0
        elif "bn" in mv_str:
            return float(mv_str.replace("bn", "").strip()) * 1000.0
        else:
            return float(mv_str)
    except (ValueError, TypeError):
        return 0.0


def _normalise_position(pos: str) -> str:
    """Map position abbreviations to standard groups."""
    pos = pos.strip().upper()
    if pos in ("GK",):
        return "GK"
    if pos in ("DF", "CB", "LB", "RB", "LWB", "RWB", "SW"):
        return "DF"
    if pos in ("MF", "CM", "AM", "DM", "LM", "RM", "CDM", "CAM"):
        return "MF"
    if pos in ("FW", "CF", "LW", "RW", "ST", "SS"):
        return "FW"
    return pos


def build_squad_profile(team_name: str, squad: list[dict]) -> dict:
    """
    Compute aggregate squad-level metrics from a player list.

    Parameters
    ----------
    team_name : str
        Canonical team name.
    squad : list of dict
        Player records with keys: name, position, club, age, caps, goals,
        market_value.

    Returns
    -------
    dict conforming to SQUAD_PROFILE_COLUMNS.
    """
    n = len(squad)
    if n == 0:
        return {col: (team_name if col == "team" else 0) for col in SQUAD_PROFILE_COLUMNS}

    ages = [p.get("age") or 0 for p in squad]
    caps = [p.get("caps") or 0 for p in squad]
    goals = [p.get("goals") or 0 for p in squad]
    values = [_parse_market_value(p.get("market_value")) for p in squad]
    positions = [_normalise_position(p.get("position", "")) for p in squad]

    # Top 3 player values
    top3_values = sorted(values, reverse=True)[:3]

    # Experience score: weighted sum of caps (higher caps → more experience)
    # Normalised to 0-100 scale
    exp_raw = sum(min(c, 150) for c in caps)  # cap each at 150
    experience_score = round(exp_raw / (n * 150) * 100, 1)

    valid_ages = [a for a in ages if a > 0]

    profile = {
        "team": team_name,
        "squad_size": n,
        "avg_age": round(np.mean(valid_ages), 1) if valid_ages else 0.0,
        "avg_caps": round(np.mean(caps), 1),
        "avg_goals": round(np.mean(goals), 1),
        "total_caps": int(sum(caps)),
        "total_goals": int(sum(goals)),
        "squad_market_value_eur": round(sum(values), 2),
        "avg_market_value_eur": round(np.mean(values), 2) if values else 0.0,
        "star_player_value_eur": round(sum(top3_values), 2),
        "gk_count": positions.count("GK"),
        "df_count": positions.count("DF"),
        "mf_count": positions.count("MF"),
        "fw_count": positions.count("FW"),
        "experience_score": experience_score,
        "youth_ratio": round(
            sum(1 for a in valid_ages if a <= 23) / max(len(valid_ages), 1) * 100, 1
        ),
        "veteran_ratio": round(
            sum(1 for a in valid_ages if a >= 32) / max(len(valid_ages), 1) * 100, 1
        ),
    }
    return profile


def build_all_squad_profiles(save: bool = True) -> pd.DataFrame:
    """
    Build squad profiles for all 48 WC2026 teams from JSON files.

    Reads from data/raw/squads_2026/*.json.
    """
    if not SQUAD_DIR.exists():
        log.warning("Squad directory not found: %s", SQUAD_DIR)
        return pd.DataFrame(columns=SQUAD_PROFILE_COLUMNS)

    profiles = []
    for json_file in sorted(SQUAD_DIR.glob("*.json")):
        team_name = json_file.stem.replace("_", " ").title()
        # Fix common casing issues
        team_name = (team_name
                     .replace("Dr ", "DR ")
                     .replace("Curacao", "Curaçao")
                     .replace("Curaçao", "Curaçao"))

        with open(json_file, "r", encoding="utf-8") as f:
            squad = json.load(f)

        profile = build_squad_profile(team_name, squad)
        profiles.append(profile)

    df = pd.DataFrame(profiles)
    df = df.sort_values("squad_market_value_eur", ascending=False).reset_index(drop=True)

    log.info("Built squad profiles for %d teams", len(df))

    if save:
        out = PROCESSED_DIR / "squad_profiles.csv"
        df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return df


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI — Build all schemas                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 1.4 — Building Unified Data Schemas")
    log.info("=" * 60)

    # 1. Match Records
    print("\n" + "─" * 60)
    print("1️⃣  MATCH RECORD SCHEMA")
    print("─" * 60)
    mr = build_match_records()
    print(f"   Rows: {len(mr):,}  |  Columns: {len(mr.columns)}")
    print(f"   Schema: {list(mr.columns)}")
    print(f"   Sample:")
    print(mr.tail(5).to_string(index=False))

    # 2. Team Snapshots (WC2026-specific subset: 48 teams)
    print("\n" + "─" * 60)
    print("2️⃣  TEAM SNAPSHOT SCHEMA (as of 2026-06-11)")
    print("─" * 60)
    wc_teams = [
        "Spain", "Argentina", "France", "England", "Brazil", "Portugal",
        "Colombia", "Netherlands", "Germany", "Norway", "Japan", "Croatia",
        "Uruguay", "Denmark", "Mexico", "Senegal", "Belgium", "Italy",
        "Turkey", "Morocco", "Canada", "Australia", "Ecuador", "Nigeria",
        "South Korea", "United States", "Paraguay", "Egypt", "Iran",
        "Serbia", "Switzerland", "Saudi Arabia", "Ghana", "Cameroon",
        "Panama", "DR Congo", "Uzbekistan", "Albania", "Costa Rica",
        "Honduras", "Slovenia", "Chile", "Bolivia", "Peru",
        "Wales", "New Zealand", "Indonesia", "Qatar",
    ]
    ts = build_all_team_snapshots(as_of_date="2026-06-11")
    ts_wc = ts[ts["team"].isin(wc_teams)].head(20)
    print(f"   Total teams: {len(ts)}  |  WC2026 teams: {len(ts[ts['team'].isin(wc_teams)])}")
    print(f"   Schema: {list(ts.columns)}")
    print(f"   Top 20 WC2026 teams by Elo:")
    print(
        ts_wc[["team", "elo_rating", "matches_played_total",
               "wins_last_10", "goals_scored_avg_5", "form_score"]]
        .to_string(index=False)
    )

    # 3. Squad Profiles
    print("\n" + "─" * 60)
    print("3️⃣  SQUAD PROFILE SCHEMA (2026 WC)")
    print("─" * 60)
    sp = build_all_squad_profiles()
    print(f"   Squads: {len(sp)}")
    print(f"   Schema: {list(sp.columns)}")
    print(f"   Top 10 by market value:")
    print(
        sp[["team", "squad_size", "avg_age", "squad_market_value_eur",
            "star_player_value_eur", "experience_score"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n✅ All three schemas built and saved to data/processed/")


if __name__ == "__main__":
    main()
