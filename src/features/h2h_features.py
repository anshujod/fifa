"""
h2h_features.py — Head-to-Head & Contextual Features (Task 2.2).

Computes per-match features that capture:
    - Direct H2H history between the two teams
    - Elo ratings (already present — re-exported for completeness)
    - Continental/confederation strength
    - World Cup experience
    - Neutral venue flag

Usage:
    python -m src.features.h2h_features
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
# Confederation mapping (FIFA member associations)
# ---------------------------------------------------------------------------
CONFEDERATION: dict[str, str] = {
    # UEFA (Europe)
    **{t: "UEFA" for t in [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Czechia", "Denmark", "England", "Estonia",
        "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar",
        "Greece", "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan",
        "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg",
        "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
        "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal",
        "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "Wales",
    ]},
    # CONMEBOL (South America)
    **{t: "CONMEBOL" for t in [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ]},
    # CONCACAF (North/Central America & Caribbean)
    **{t: "CONCACAF" for t in [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Bermuda",
        "Canada", "Cayman Islands", "Costa Rica", "Cuba", "Curaçao",
        "Dominica", "Dominican Republic", "El Salvador", "Grenada",
        "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico",
        "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia",
        "Saint Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "Turks and Caicos Islands",
        "United States", "US Virgin Islands",
    ]},
    # CAF (Africa)
    **{t: "CAF" for t in [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad", "Comoros",
        "Congo Republic", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon",
        "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya",
        "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
        "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal",
        "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
        "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    ]},
    # AFC (Asia)
    **{t: "AFC" for t in [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China", "Chinese Taipei", "Guam",
        "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
        "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia",
        "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea", "Oman",
        "Pakistan", "Palestine", "Philippines", "Qatar", "Saudi Arabia",
        "Singapore", "South Korea", "Sri Lanka", "Syria", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates",
        "Uzbekistan", "Vietnam", "Yemen",
    ]},
    # OFC (Oceania)
    **{t: "OFC" for t in [
        "American Samoa", "Cook Islands", "Fiji", "Kiribati",
        "New Caledonia", "New Zealand", "Papua New Guinea", "Samoa",
        "Solomon Islands", "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
    ]},
}


def _get_confederation(team: str) -> str:
    """Return the confederation for a team, or 'OTHER' if unknown."""
    return CONFEDERATION.get(team, "OTHER")


# ---------------------------------------------------------------------------
# World Cup appearances (counted from our dataset)
# ---------------------------------------------------------------------------

def _build_wc_appearances(df: pd.DataFrame) -> dict[str, int]:
    """
    Count unique World Cup tournaments each team has appeared in.

    Returns dict: team → number of WC tournaments.
    """
    wc = df[
        (df["tournament_type"] == "FIFA World Cup")
        & (df["is_played"] == True)  # noqa
    ].copy()

    if wc.empty:
        return {}

    wc["wc_year"] = pd.to_datetime(wc["date"]).dt.year

    appearances: dict[str, set] = defaultdict(set)
    for _, row in wc.iterrows():
        appearances[row["home_team"]].add(row["wc_year"])
        appearances[row["away_team"]].add(row["wc_year"])

    return {team: len(years) for team, years in appearances.items()}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  H2H feature computation                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _compute_h2h_features(
    h2h_history: list[dict],
    home_team: str,
) -> dict:
    """
    Compute head-to-head features from the history of encounters
    between two teams.

    Parameters
    ----------
    h2h_history : list of dict
        Previous H2H matches (chronological). Each has:
        home_team, away_team, home_goals, away_goals, result.
    home_team : str
        The team for which win% is computed as "home" perspective.

    Returns
    -------
    dict with h2h_* features.
    """
    if not h2h_history:
        return {
            "h2h_matches": 0,
            "h2h_win_pct": 50.0,       # neutral prior when no history
            "h2h_goals_diff": 0.0,
            "h2h_wins": 0,
            "h2h_draws": 0,
            "h2h_losses": 0,
        }

    wins = 0
    draws = 0
    losses = 0
    goal_diffs = []

    for match in h2h_history:
        is_home = match["home_team"] == home_team
        if is_home:
            gf = match["home_goals"]
            ga = match["away_goals"]
        else:
            gf = match["away_goals"]
            ga = match["home_goals"]

        gd = gf - ga
        goal_diffs.append(gd)

        if gd > 0:
            wins += 1
        elif gd == 0:
            draws += 1
        else:
            losses += 1

    total = len(h2h_history)

    return {
        "h2h_matches": total,
        "h2h_win_pct": round(wins / total * 100, 1),
        "h2h_goals_diff": round(np.mean(goal_diffs), 2),
        "h2h_wins": wins,
        "h2h_draws": draws,
        "h2h_losses": losses,
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main pipeline                                                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def compute_h2h_contextual_features(
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute H2H and contextual features for every match.

    New columns added:
        h2h_matches, h2h_win_pct, h2h_goals_diff,
        h2h_wins, h2h_draws, h2h_losses,
        home_confederation, away_confederation,
        home_confederation_avg_elo, away_confederation_avg_elo,
        confederation_elo_diff,
        home_wc_appearances, away_wc_appearances,
        wc_experience_diff,
        neutral_venue (already present — ensured as int)
    """
    if df is None:
        path = PROCESSED_DIR / "match_features.csv"
        log.info("Loading match features from %s", path)
        df = pd.read_csv(path, parse_dates=["date"])

    log.info("Computing H2H & contextual features for %d matches", len(df))

    # --- Drop any previously computed H2H/contextual columns to avoid duplicates ---
    _h2h_cols = [
        "h2h_matches", "h2h_win_pct", "h2h_goals_diff", "h2h_wins", "h2h_draws",
        "h2h_losses", "home_confederation", "away_confederation",
        "home_confederation_avg_elo", "away_confederation_avg_elo",
        "confederation_elo_diff", "home_wc_appearances", "away_wc_appearances",
        "home_wc_experience_norm", "away_wc_experience_norm", "wc_experience_diff",
        "home_fifa_rank", "away_fifa_rank", "ranking_diff",
    ]
    df = df.drop(columns=[c for c in _h2h_cols if c in df.columns])

    # --- Pre-compute lookups ---

    # 0. Dynamic FIFA-style rank per team per year (derived from Elo before each match).
    #    For each year, collect all (team, elo_before) pairs then rank teams by their
    #    mean Elo that year.  Lower rank = stronger team (rank 1 = best).
    elo_cols = ["year", "home_team", "away_team", "home_elo_before", "away_elo_before"]
    if all(c in df.columns for c in elo_cols):
        played_elo = df[df["is_played"] == True][elo_cols].copy()  # noqa: E712
        home_elo = played_elo[["year", "home_team", "home_elo_before"]].rename(
            columns={"home_team": "team", "home_elo_before": "elo"}
        )
        away_elo = played_elo[["year", "away_team", "away_elo_before"]].rename(
            columns={"away_team": "team", "away_elo_before": "elo"}
        )
        team_year_elo = pd.concat([home_elo, away_elo]).groupby(["year", "team"])["elo"].mean()
        year_rank: dict[tuple[int, str], int] = {}
        for yr, group in team_year_elo.groupby(level="year"):
            ranked = group.droplevel("year").rank(ascending=False, method="min").astype(int)
            for team_name, r in ranked.items():
                year_rank[(int(yr), team_name)] = int(r)
        log.info("Pre-computed Elo-based FIFA rank for %d (year, team) pairs", len(year_rank))
    else:
        year_rank = {}
        log.warning("Elo columns missing — FIFA rank features will be NaN")

    # 1. Elo ratings for confederation strength
    elo_path = PROCESSED_DIR / "elo_ratings.csv"
    elo_df = pd.read_csv(elo_path)
    elo_map = elo_df.set_index("team")["elo_rating"].to_dict()

    # Compute average Elo per confederation
    conf_elos: dict[str, list[int]] = defaultdict(list)
    for team, elo in elo_map.items():
        conf = _get_confederation(team)
        conf_elos[conf].append(elo)
    conf_avg_elo = {
        conf: round(np.mean(elos))
        for conf, elos in conf_elos.items()
    }
    log.info("Confederation avg Elo: %s", conf_avg_elo)

    # 2. World Cup appearances
    wc_apps = _build_wc_appearances(df)
    max_wc = max(wc_apps.values()) if wc_apps else 1
    log.info("WC appearances computed for %d teams (max=%d)", len(wc_apps), max_wc)

    # --- Process matches ---
    played_mask = df["is_played"] == True  # noqa
    played = df[played_mask].copy().sort_values("date").reset_index(drop=True)
    unplayed = df[~played_mask].copy()

    # Build H2H history incrementally (keyed by sorted team pair)
    h2h_history: dict[tuple[str, str], list[dict]] = defaultdict(list)
    feature_rows: list[dict] = []

    total = len(played)
    log_interval = max(total // 10, 1)

    for idx, row in played.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        pair_key = tuple(sorted([home, away]))

        # --- H2H features (BEFORE this match) ---
        h2h_feats = _compute_h2h_features(h2h_history[pair_key], home)

        # --- Confederation ---
        home_conf = _get_confederation(home)
        away_conf = _get_confederation(away)
        home_conf_elo = conf_avg_elo.get(home_conf, 1500)
        away_conf_elo = conf_avg_elo.get(away_conf, 1500)

        # --- WC Experience (normalised 0-1) ---
        home_wc = wc_apps.get(home, 0)
        away_wc = wc_apps.get(away, 0)

        yr = int(row["year"]) if "year" in row.index else row["date"].year
        home_rank = year_rank.get((yr, home), np.nan)
        away_rank = year_rank.get((yr, away), np.nan)
        ranking_diff = (
            (home_rank - away_rank)
            if not (np.isnan(home_rank) or np.isnan(away_rank))
            else np.nan
        )

        feats = {
            **h2h_feats,
            "home_confederation": home_conf,
            "away_confederation": away_conf,
            "home_confederation_avg_elo": home_conf_elo,
            "away_confederation_avg_elo": away_conf_elo,
            "confederation_elo_diff": home_conf_elo - away_conf_elo,
            "home_wc_appearances": home_wc,
            "away_wc_appearances": away_wc,
            "home_wc_experience_norm": round(home_wc / max_wc, 3) if max_wc > 0 else 0,
            "away_wc_experience_norm": round(away_wc / max_wc, 3) if max_wc > 0 else 0,
            "wc_experience_diff": home_wc - away_wc,
            "home_fifa_rank": home_rank,
            "away_fifa_rank": away_rank,
            "ranking_diff": ranking_diff,
        }
        feature_rows.append(feats)

        # --- Update H2H history AFTER computing features ---
        h2h_history[pair_key].append({
            "home_team": home,
            "away_team": away,
            "home_goals": row["home_goals"],
            "away_goals": row["away_goals"],
            "result": row["result"],
        })

        if (idx + 1) % log_interval == 0:
            log.info("  Progress: %d/%d (%.0f%%)", idx + 1, total, (idx + 1) / total * 100)

    # --- Join features onto played matches ---
    features_df = pd.DataFrame(feature_rows)
    played = pd.concat(
        [played.reset_index(drop=True), features_df], axis=1
    )

    # --- Ensure neutral_venue is int ---
    played["neutral_venue"] = played["neutral_venue"].astype(int)

    # --- Recombine ---
    # Add placeholder columns to unplayed
    for col in features_df.columns:
        if col not in unplayed.columns:
            unplayed[col] = np.nan

    if "neutral_venue" in unplayed.columns:
        unplayed["neutral_venue"] = unplayed["neutral_venue"].fillna(0).astype(int)

    result = pd.concat([played, unplayed], ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    new_cols = list(features_df.columns)
    log.info("Added %d H2H/contextual feature columns", len(new_cols))

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
    log.info("TASK 2.2 — Head-to-Head & Contextual Features")
    log.info("=" * 60)

    result = compute_h2h_contextual_features()

    played = result[result["is_played"] == True]  # noqa
    recent = played.tail(10)

    print("\n📊 H2H & Contextual Features Summary")
    print(f"   Total matches:      {len(result):,}")
    new_cols = [c for c in result.columns if c.startswith(("h2h_", "home_conf",
                "away_conf", "conf", "home_wc", "away_wc", "wc_experience"))]
    print(f"   New feature columns: {len(new_cols)}")

    print("\n🤝 Head-to-Head features (last 5 played):")
    h2h_cols = ["date", "home_team", "away_team",
                "h2h_matches", "h2h_win_pct", "h2h_goals_diff"]
    print(recent[[c for c in h2h_cols if c in recent.columns]].tail(5).to_string(index=False))

    print("\n🌍 Confederation features (last 5):")
    conf_cols = ["date", "home_team", "home_confederation",
                 "home_confederation_avg_elo", "away_confederation_avg_elo",
                 "confederation_elo_diff"]
    print(recent[[c for c in conf_cols if c in recent.columns]].tail(5).to_string(index=False))

    print("\n🏆 World Cup experience (last 5):")
    wc_cols = ["date", "home_team", "away_team",
               "home_wc_appearances", "away_wc_appearances", "wc_experience_diff"]
    print(recent[[c for c in wc_cols if c in recent.columns]].tail(5).to_string(index=False))

    # Show classic rivalries
    print("\n⚔️  Classic rivalry H2H samples:")
    rivalries = [
        ("Argentina", "Brazil"),
        ("England", "Germany"),
        ("Spain", "France"),
    ]
    for t1, t2 in rivalries:
        mask = (
            ((played["home_team"] == t1) & (played["away_team"] == t2)) |
            ((played["home_team"] == t2) & (played["away_team"] == t1))
        )
        last = played[mask].tail(1)
        if not last.empty:
            r = last.iloc[0]
            print(f"   {t1} vs {t2}: {int(r.get('h2h_matches', 0))} matches, "
                  f"win%={r.get('h2h_win_pct', 'N/A')} (for {r['home_team']}), "
                  f"avg GD={r.get('h2h_goals_diff', 'N/A')}")

    print("\n✅ H2H & contextual features complete.")


if __name__ == "__main__":
    main()
