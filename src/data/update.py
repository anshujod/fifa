"""
update.py — Auto-update pipeline for international match results.

Fetches the latest international football results from external APIs
and appends new rows to data/raw/international_results.csv without
overwriting existing history.

Data Sources:
    1. football-data.org  — Free tier, covers major international competitions
    2. API-Football       — RapidAPI, broader coverage of recent matches

Environment Variables:
    FOOTBALL_DATA_API_KEY  — API key for football-data.org (free at https://www.football-data.org/client/register)
    RAPIDAPI_KEY           — API key for API-Football on RapidAPI

Usage:
    # Set API keys
    export FOOTBALL_DATA_API_KEY="your_key_here"
    export RAPIDAPI_KEY="your_key_here"        # optional

    # Run update
    python -m src.data.update

    # Or with explicit date range
    python -m src.data.update --from-date 2026-01-01 --to-date 2026-06-04
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from src.data.fetch_data import _standardise_team_name

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_CSV = RAW_DIR / "international_results.csv"

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
# API Configuration
# ---------------------------------------------------------------------------
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
RAPIDAPI_BASE = "https://api-football-v1.p.rapidapi.com/v3"

# International competition IDs on football-data.org
FOOTBALL_DATA_COMPETITIONS = {
    "WC": "FIFA World Cup",
    "EC": "UEFA Euro",
    "CLI": "Copa Libertadores",  # placeholder — internationals
}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Source 1: football-data.org                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fetch_football_data_org(
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch recent international match results from football-data.org.

    Parameters
    ----------
    from_date : str, optional
        Start date (YYYY-MM-DD). Defaults to 30 days ago.
    to_date : str, optional
        End date (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame
        New match results with columns:
        date, home_team, away_team, home_score, away_score, tournament, neutral
    """
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        log.warning(
            "FOOTBALL_DATA_API_KEY not set. "
            "Get a free key at https://www.football-data.org/client/register"
        )
        return pd.DataFrame()

    if not from_date:
        from_date = (date.today() - timedelta(days=30)).isoformat()
    if not to_date:
        to_date = date.today().isoformat()

    headers = {"X-Auth-Token": api_key}
    all_matches = []

    # Fetch from the /matches endpoint (covers all competitions)
    url = f"{FOOTBALL_DATA_BASE}/matches"
    params = {
        "dateFrom": from_date,
        "dateTo": to_date,
        "status": "FINISHED",
    }

    log.info("Fetching football-data.org: %s to %s", from_date, to_date)

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        matches = data.get("matches", [])
        log.info("football-data.org returned %d matches", len(matches))

        for m in matches:
            # Only include international team matches (not club)
            area_type = m.get("area", {}).get("type", "")
            # Filter: we want national team matches
            competition = m.get("competition", {}).get("name", "")

            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            score = m.get("score", {})
            ft = score.get("fullTime", {})

            if not home or not away:
                continue

            all_matches.append({
                "date": m.get("utcDate", "")[:10],
                "home_team": _standardise_team_name(home),
                "away_team": _standardise_team_name(away),
                "home_score": ft.get("home"),
                "away_score": ft.get("away"),
                "tournament": competition,
                "neutral": False,  # API doesn't always provide this
            })

    except requests.exceptions.RequestException as e:
        log.error("football-data.org API error: %s", e)
        return pd.DataFrame()

    if all_matches:
        df = pd.DataFrame(all_matches)
        log.info("Parsed %d matches from football-data.org", len(df))
        return df

    return pd.DataFrame()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Source 2: API-Football (RapidAPI)                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def fetch_api_football(
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """
    Fetch recent international match results from API-Football (RapidAPI).

    Parameters
    ----------
    from_date : str, optional
        Start date (YYYY-MM-DD). Defaults to 30 days ago.
    to_date : str, optional
        End date (YYYY-MM-DD). Defaults to today.

    Returns
    -------
    pd.DataFrame
        New match results with standard columns.
    """
    api_key = os.environ.get("RAPIDAPI_KEY")
    if not api_key:
        log.warning(
            "RAPIDAPI_KEY not set. "
            "Get a key at https://rapidapi.com/api-sports/api/api-football"
        )
        return pd.DataFrame()

    if not from_date:
        from_date = (date.today() - timedelta(days=30)).isoformat()
    if not to_date:
        to_date = date.today().isoformat()

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
    }

    # International fixtures: league_id for World Cup, qualifiers, friendlies
    # 1 = World Cup, 4 = Euro, 9 = Copa America, 10 = Friendlies
    international_leagues = [1, 4, 9, 10, 29, 30, 31, 32, 33, 34]

    all_matches = []

    for league_id in international_leagues:
        url = f"{RAPIDAPI_BASE}/fixtures"
        params = {
            "league": league_id,
            "from": from_date,
            "to": to_date,
            "status": "FT",  # Full Time only
            "season": datetime.now().year,
        }

        log.info(
            "Fetching API-Football league=%d: %s to %s",
            league_id, from_date, to_date,
        )

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            fixtures = data.get("response", [])
            log.info("API-Football league=%d returned %d fixtures", league_id, len(fixtures))

            for fix in fixtures:
                teams = fix.get("teams", {})
                goals = fix.get("goals", {})
                league = fix.get("league", {})
                fixture_info = fix.get("fixture", {})

                home_name = teams.get("home", {}).get("name", "")
                away_name = teams.get("away", {}).get("name", "")

                if not home_name or not away_name:
                    continue

                venue = fixture_info.get("venue", {})
                fixture_date = fixture_info.get("date", "")[:10]

                all_matches.append({
                    "date": fixture_date,
                    "home_team": _standardise_team_name(home_name),
                    "away_team": _standardise_team_name(away_name),
                    "home_score": goals.get("home"),
                    "away_score": goals.get("away"),
                    "tournament": league.get("name", "Unknown"),
                    "neutral": False,
                })

        except requests.exceptions.RequestException as e:
            log.error("API-Football error (league=%d): %s", league_id, e)
            continue

    if all_matches:
        df = pd.DataFrame(all_matches)
        # Deduplicate across leagues
        df = df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
        log.info("Parsed %d unique matches from API-Football", len(df))
        return df

    return pd.DataFrame()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Append Logic                                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_existing_results() -> pd.DataFrame:
    """Load the existing results CSV."""
    if not RESULTS_CSV.exists():
        log.warning("No existing results file at %s", RESULTS_CSV)
        return pd.DataFrame(
            columns=["date", "home_team", "away_team",
                      "home_score", "away_score", "tournament", "neutral"]
        )
    df = pd.read_csv(RESULTS_CSV, dtype=str)
    log.info("Loaded %d existing results from %s", len(df), RESULTS_CSV)
    return df


def append_new_results(
    existing: pd.DataFrame,
    new: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Append new results to existing dataset, avoiding duplicates.

    Deduplicates on (date, home_team, away_team).

    Returns
    -------
    tuple of (combined DataFrame, count of new rows added)
    """
    if new.empty:
        log.info("No new results to append")
        return existing, 0

    # Ensure consistent column types
    for col in ["home_score", "away_score"]:
        if col in new.columns:
            new[col] = new[col].astype(str)

    new["neutral"] = new["neutral"].astype(str).str.upper()

    # Standardise team names in new data
    new = new.copy()
    new["home_team"] = new["home_team"].apply(_standardise_team_name)
    new["away_team"] = new["away_team"].apply(_standardise_team_name)

    # Build set of existing match keys for fast lookup
    existing_keys = set(
        zip(existing["date"], existing["home_team"], existing["away_team"])
    )

    # Filter to only genuinely new matches
    new["_key"] = list(zip(new["date"], new["home_team"], new["away_team"]))
    genuinely_new = new[~new["_key"].isin(existing_keys)].drop(columns=["_key"])

    new_count = len(genuinely_new)

    if new_count > 0:
        combined = pd.concat([existing, genuinely_new], ignore_index=True)
        log.info("Appended %d genuinely new results", new_count)
    else:
        combined = existing
        log.info("No new results — all %d fetched matches already exist", len(new))

    return combined, new_count


def save_results(df: pd.DataFrame) -> None:
    """Save the updated results CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    log.info("Saved %d results → %s", len(df), RESULTS_CSV)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Main update pipeline                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_update(
    from_date: str | None = None,
    to_date: str | None = None,
) -> int:
    """
    Run the full update pipeline.

    1. Load existing results
    2. Fetch new data from all available sources
    3. Merge and deduplicate
    4. Save updated CSV

    Returns the number of new rows added.
    """
    log.info("=" * 60)
    log.info("FIFA WC2026 — Data Update Pipeline")
    log.info("=" * 60)

    # Load existing
    existing = load_existing_results()

    # Fetch from all sources
    new_frames = []

    # Source 1: football-data.org
    fd_df = fetch_football_data_org(from_date, to_date)
    if not fd_df.empty:
        new_frames.append(fd_df)
        log.info("football-data.org: %d matches fetched", len(fd_df))

    # Source 2: API-Football (RapidAPI)
    af_df = fetch_api_football(from_date, to_date)
    if not af_df.empty:
        new_frames.append(af_df)
        log.info("API-Football: %d matches fetched", len(af_df))

    # Combine all new data
    if new_frames:
        all_new = pd.concat(new_frames, ignore_index=True)
        # Deduplicate across sources
        all_new = all_new.drop_duplicates(
            subset=["date", "home_team", "away_team"], keep="first"
        )
        log.info("Total new matches (deduplicated): %d", len(all_new))
    else:
        all_new = pd.DataFrame()
        log.warning(
            "No data fetched from any source. "
            "Check your API keys: FOOTBALL_DATA_API_KEY, RAPIDAPI_KEY"
        )

    # Append and save
    combined, new_count = append_new_results(existing, all_new)

    if new_count > 0:
        save_results(combined)
        log.info("✅ Update complete — %d new matches added", new_count)
    else:
        log.info("✅ No new matches to add — dataset is up to date")

    return new_count


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update international football results from external APIs."
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 30 days ago.",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    args = parser.parse_args()

    new_count = run_update(args.from_date, args.to_date)

    print(f"\n📊 Update Summary:")
    print(f"   New matches added: {new_count}")

    existing = load_existing_results()
    print(f"   Total dataset size: {len(existing):,} matches")
    print(f"   Date range: {existing['date'].min()} → {existing['date'].max()}")


if __name__ == "__main__":
    main()
