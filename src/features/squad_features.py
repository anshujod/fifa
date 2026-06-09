"""
squad_features.py — Squad-Level Features for 2026 World Cup (Task 2.3).

Tournament-level static features computed from the 48 WC2026 rosters.
These features capture squad quality, depth, experience, and star power.

Features per team:
    squad_market_value_total   — total squad market value (€M)
    avg_player_age             — average age across the squad
    star_player_market_value   — top 3 players combined value (€M)
    key_player_caps            — avg caps of top 11 players (by value)
    top_scorer_goals           — goals tally of primary striker
    injury_flag                — binary: key player absent (manual input)
    squad_depth_score          — positional balance metric
    experience_score           — normalised squad experience

Usage:
    python -m src.features.squad_features
"""

from __future__ import annotations

import json
import logging
import re
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

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known injury / absence data (manually curated as of June 2026)
# Update this dict as tournament approaches.
# Key = team name (lowercase), value = list of absent key players
# ---------------------------------------------------------------------------
KNOWN_ABSENCES: dict[str, list[str]] = {
    # Example entries — update with real data before tournament:
    # "france": ["N'Golo Kanté"],
    # "brazil": ["Neymar"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_market_value(mv_str: str | None) -> float:
    """Parse '€18.00m', '€500k' → float in millions."""
    if not mv_str or not isinstance(mv_str, str):
        return 0.0
    mv_str = mv_str.strip().lower()
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


def _team_name_from_filename(filename: str) -> str:
    """Convert filename to display team name."""
    name = filename.replace("_", " ").title()
    # Fix common casing
    name = (name
            .replace("Dr ", "DR ")
            .replace("Curacao", "Curaçao")
            .replace("Curaçao", "Curaçao")
            .replace("And ", "and ")
            .replace("Of ", "of "))
    return name


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Build squad features for a single team                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_single_squad_features(
    team_name: str,
    squad: list[dict],
) -> dict:
    """
    Compute all squad-level features for a single team.

    Parameters
    ----------
    team_name : str
        Canonical team name.
    squad : list of dict
        Player records with keys: name, position, club, age, caps, goals,
        market_value.

    Returns
    -------
    dict of feature_name → value.
    """
    n = len(squad)
    if n == 0:
        return {"team": team_name}

    # --- Parse values ---
    ages = [p.get("age") or 0 for p in squad]
    caps = [p.get("caps") or 0 for p in squad]
    goals = [p.get("goals") or 0 for p in squad]
    values = [_parse_market_value(p.get("market_value")) for p in squad]
    positions = [_normalise_position(p.get("position", "")) for p in squad]
    names = [p.get("name", "") for p in squad]

    valid_ages = [a for a in ages if a > 0]

    # --- 1. squad_market_value_total (€M) ---
    squad_value = round(sum(values), 2)

    # --- 2. avg_player_age ---
    avg_age = round(np.mean(valid_ages), 1) if valid_ages else 0.0

    # --- 3. star_player_market_value (top 3 combined) ---
    top3_values = sorted(values, reverse=True)[:3]
    star_value = round(sum(top3_values), 2)

    # --- 4. key_player_caps (avg caps of top 11 by market value) ---
    # Sort players by value descending, take top 11 (approx starting XI)
    player_data = list(zip(values, caps, goals, names, positions))
    player_data.sort(key=lambda x: x[0], reverse=True)
    top11 = player_data[:min(11, n)]
    key_player_caps = round(np.mean([p[1] for p in top11]), 1)

    # --- 5. top_scorer_goals ---
    # Find the forward/striker with the most international goals
    forwards = [(p[2], p[3]) for p in player_data if p[4] == "FW"]
    if forwards:
        top_scorer = max(forwards, key=lambda x: x[0])
        top_scorer_goals = top_scorer[0]
        top_scorer_name = top_scorer[1]
    else:
        # Fallback: highest goal scorer in any position
        all_scorers = [(p[2], p[3]) for p in player_data]
        top_scorer = max(all_scorers, key=lambda x: x[0])
        top_scorer_goals = top_scorer[0]
        top_scorer_name = top_scorer[1]

    # --- 6. injury_flag ---
    team_key = team_name.lower()
    absent_players = KNOWN_ABSENCES.get(team_key, [])
    injury_flag = 1 if absent_players else 0

    # --- 7. squad_depth_score (positional balance: ideal is 3 GK, 8 DF, 8 MF, 7 FW) ---
    gk_count = positions.count("GK")
    df_count = positions.count("DF")
    mf_count = positions.count("MF")
    fw_count = positions.count("FW")

    # Penalise deviation from ideal distribution
    ideal = {"GK": 3, "DF": 8, "MF": 8, "FW": 7}
    deviation = (
        abs(gk_count - ideal["GK"])
        + abs(df_count - ideal["DF"])
        + abs(mf_count - ideal["MF"])
        + abs(fw_count - ideal["FW"])
    )
    squad_depth_score = round(max(0, 100 - deviation * 10), 1)

    # --- 8. experience_score (normalised: avg caps / 150, capped at 1.0) ---
    avg_caps = np.mean(caps) if caps else 0
    experience_score = round(min(avg_caps / 80, 1.0), 3)

    # --- 9. avg_market_value (per player) ---
    avg_value = round(np.mean(values), 2) if values else 0.0

    # --- 10. youth_ratio (% players ≤ 23) ---
    youth_ratio = round(
        sum(1 for a in valid_ages if a <= 23) / max(len(valid_ages), 1) * 100, 1
    )

    # --- 11. veteran_ratio (% players ≥ 32) ---
    veteran_ratio = round(
        sum(1 for a in valid_ages if a >= 32) / max(len(valid_ages), 1) * 100, 1
    )

    # --- 12. star_concentration (top 3 value / total value) ---
    star_concentration = round(
        star_value / squad_value * 100, 1
    ) if squad_value > 0 else 0.0

    # --- 13. goals_per_cap (squad efficiency) ---
    total_caps = sum(caps)
    total_goals = sum(goals)
    goals_per_cap = round(total_goals / max(total_caps, 1), 3)

    return {
        "team": team_name,
        "squad_size": n,
        "squad_market_value_total": squad_value,
        "avg_player_age": avg_age,
        "star_player_market_value": star_value,
        "key_player_caps": key_player_caps,
        "top_scorer_goals": top_scorer_goals,
        "top_scorer_name": top_scorer_name,
        "injury_flag": injury_flag,
        "squad_depth_score": squad_depth_score,
        "experience_score": experience_score,
        "avg_market_value": avg_value,
        "youth_ratio": youth_ratio,
        "veteran_ratio": veteran_ratio,
        "star_concentration": star_concentration,
        "goals_per_cap": goals_per_cap,
        "gk_count": gk_count,
        "df_count": df_count,
        "mf_count": mf_count,
        "fw_count": fw_count,
        "total_caps": total_caps,
        "total_goals": total_goals,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Build features for all 48 teams                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_all_squad_features(save: bool = True) -> pd.DataFrame:
    """
    Compute squad-level features for all 48 WC2026 teams.

    Reads from data/raw/squads_2026/*.json.

    Returns a DataFrame with one row per team and 22 feature columns.
    """
    if not SQUAD_DIR.exists():
        log.error("Squad directory not found: %s", SQUAD_DIR)
        return pd.DataFrame()

    features = []
    for json_file in sorted(SQUAD_DIR.glob("*.json")):
        team_name = _team_name_from_filename(json_file.stem)

        with open(json_file, "r", encoding="utf-8") as f:
            squad = json.load(f)

        feats = compute_single_squad_features(team_name, squad)
        features.append(feats)
        log.debug("  %s: %d players, €%.1fM", team_name, len(squad), feats["squad_market_value_total"])

    df = pd.DataFrame(features)
    df = df.sort_values("squad_market_value_total", ascending=False).reset_index(drop=True)

    log.info("Built squad features for %d teams", len(df))

    if save:
        out = PROCESSED_DIR / "squad_features.csv"
        df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return df


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
    log.info("TASK 2.3 — Squad-Level Features from 2026 Rosters")
    log.info("=" * 60)

    df = compute_all_squad_features()

    print(f"\n📊 Squad Features Summary — {len(df)} teams")
    print(f"   Feature columns: {len(df.columns)}")

    print("\n💰 Top 10 by Squad Market Value:")
    top_cols = [
        "team", "squad_market_value_total", "star_player_market_value",
        "avg_player_age", "key_player_caps", "experience_score",
    ]
    print(df[top_cols].head(10).to_string(index=False))

    print("\n⚽ Top Scorers by Team:")
    scorer_cols = ["team", "top_scorer_name", "top_scorer_goals", "goals_per_cap"]
    top_scorers = df.sort_values("top_scorer_goals", ascending=False)
    print(top_scorers[scorer_cols].head(15).to_string(index=False))

    print("\n👶 Youth vs Veterans:")
    age_cols = ["team", "avg_player_age", "youth_ratio", "veteran_ratio"]
    youngest = df.sort_values("avg_player_age")
    print("  Youngest squads:")
    print(youngest[age_cols].head(5).to_string(index=False))
    print("  Oldest squads:")
    print(youngest[age_cols].tail(5).to_string(index=False))

    print("\n⭐ Star Concentration (top 3 as % of total value):")
    star_cols = ["team", "squad_market_value_total", "star_player_market_value", "star_concentration"]
    print(df.sort_values("star_concentration", ascending=False)[star_cols].head(10).to_string(index=False))

    print("\n🏗️ Squad Depth Score (positional balance):")
    depth_cols = ["team", "gk_count", "df_count", "mf_count", "fw_count", "squad_depth_score"]
    print(df.sort_values("squad_depth_score", ascending=False)[depth_cols].head(10).to_string(index=False))

    print("\n✅ Squad features complete.")


if __name__ == "__main__":
    main()
