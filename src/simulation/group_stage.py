from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.simulation.match_predictor import MatchPredictor

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Official FIFA World Cup 2026 Groups (derived from confirmed fixture schedule)
# ─────────────────────────────────────────────────────────────────────────────

WC2026_GROUPS: dict[str, list[str]] = {
    # Official draw – December 2025, Washington D.C.
    # Host nations: Mexico (A1), Canada (B1), United States (D1)
    "A": ["Czech Republic",  "Mexico",       "South Africa", "South Korea"],
    "B": ["Bosnia and Herzegovina", "Canada", "Qatar",       "Switzerland"],
    "C": ["Brazil",          "Haiti",        "Morocco",      "Scotland"],
    "D": ["Australia",       "Paraguay",     "Turkey",       "United States"],
    "E": ["Curaçao",         "Ecuador",      "Germany",      "Ivory Coast"],
    "F": ["Japan",           "Netherlands",  "Sweden",       "Tunisia"],
    "G": ["Belgium",         "Egypt",        "Iran",         "New Zealand"],
    "H": ["Cape Verde",      "Saudi Arabia", "Spain",        "Uruguay"],
    "I": ["France",          "Iraq",         "Norway",       "Senegal"],
    "J": ["Algeria",         "Argentina",    "Austria",      "Jordan"],
    "K": ["Colombia",        "DR Congo",     "Portugal",     "Uzbekistan"],
    "L": ["Croatia",         "England",      "Ghana",        "Panama"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamRecord:
    """Running tallies for one team through the group stage."""
    team:        str
    played:      int = 0
    wins:        int = 0
    draws:       int = 0
    losses:      int = 0
    goals_for:   int = 0
    goals_against: int = 0
    yellows:     int = 0
    reds:        int = 0

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def fair_play(self) -> int:
        """Lower is worse (more cards). Used as tiebreaker — sort ascending."""
        return -(self.yellows * 1 + self.reds * 3)


@dataclass
class MatchResult:
    home:        str
    away:        str
    home_goals:  int
    away_goals:  int

    @property
    def outcome(self) -> str:
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals == self.away_goals:
            return "D"
        return "A"


@dataclass
class GroupStandings:
    group:    str
    teams:    list[str]
    records:  dict[str, TeamRecord]
    results:  list[MatchResult]
    ranking:  list[str]  # final ordered list (1st → 4th)


# ─────────────────────────────────────────────────────────────────────────────
# FIFA tiebreaker logic
# ─────────────────────────────────────────────────────────────────────────────

def _h2h_record(
    teams: list[str], results: list[MatchResult]
) -> dict[str, dict]:
    """
    Compute head-to-head stats (points, GD, GF) among a subset of teams.
    Only matches where BOTH teams are in the subset count.
    """
    team_set = set(teams)
    h2h: dict[str, dict] = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    for r in results:
        if r.home not in team_set or r.away not in team_set:
            continue
        if r.outcome == "H":
            h2h[r.home]["pts"] += 3
        elif r.outcome == "D":
            h2h[r.home]["pts"] += 1
            h2h[r.away]["pts"] += 1
        else:
            h2h[r.away]["pts"] += 3
        h2h[r.home]["gd"] += r.home_goals - r.away_goals
        h2h[r.away]["gd"] += r.away_goals - r.home_goals
        h2h[r.home]["gf"] += r.home_goals
        h2h[r.away]["gf"] += r.away_goals
    return h2h


def _sort_key(team: str, records: dict[str, TeamRecord],
              results: list[MatchResult], tied_group: list[str]) -> tuple:
    """
    Returns a sort key tuple for one team within a tied group.
    Python's sort is stable and ascending, so negate values where higher=better.
    Tiebreaker order (FIFA 2026):
        1. points  2. GD  3. GF
        4. H2H pts  5. H2H GD  6. H2H GF
        7. fair play  8. random (handled separately)
    """
    rec = records[team]
    h2h = _h2h_record(tied_group, results)
    return (
        -rec.points,
        -rec.goal_diff,
        -rec.goals_for,
        -h2h[team]["pts"],
        -h2h[team]["gd"],
        -h2h[team]["gf"],
        -rec.fair_play,   # fair_play already negative; negate → more cards = worse
    )


def rank_group(
    teams: list[str],
    records: dict[str, TeamRecord],
    results: list[MatchResult],
    rng: Optional[np.random.Generator] = None,
) -> list[str]:
    """
    Rank teams in a group using the full FIFA 2026 tiebreaker chain.
    Returns teams in order 1st → 4th.
    """
    # Step 1: sort by points → GD → GF → H2H → fair play
    sorted_teams = sorted(teams, key=lambda t: _sort_key(t, records, results, teams))

    # Step 2: find any remaining ties after all deterministic criteria
    # and resolve by drawing of lots (random shuffle within still-tied clusters)
    resolved = []
    i = 0
    while i < len(sorted_teams):
        j = i + 1
        while j < len(sorted_teams):
            t1, t2 = sorted_teams[i], sorted_teams[j]
            tied_sub = [t for t in sorted_teams[i:] if
                        _sort_key(t, records, results, sorted_teams[i:]) ==
                        _sort_key(t1, records, results, sorted_teams[i:])]
            if t2 in tied_sub:
                j += 1
            else:
                break
        cluster = sorted_teams[i:j]
        if len(cluster) > 1:
            if rng is not None:
                rng.shuffle(cluster)
            else:
                random.shuffle(cluster)
        resolved.extend(cluster)
        i = j
    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# Card simulation (fair play proxy)
# ─────────────────────────────────────────────────────────────────────────────

_YELLOW_RATE = 3.2   # avg yellows per match in international football
_RED_RATE    = 0.12  # avg reds per match


def _simulate_cards(rng: np.random.Generator) -> tuple[int, int]:
    """Sample (yellows, reds) for one team in one match."""
    yellows = int(rng.poisson(_YELLOW_RATE / 2))   # per team
    reds    = int(rng.poisson(_RED_RATE / 2))
    return yellows, reds


# ─────────────────────────────────────────────────────────────────────────────
# Single group simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_group(
    group_name: str,
    teams: list[str],
    predictor: MatchPredictor,
    rng: Optional[np.random.Generator] = None,
) -> GroupStandings:
    """
    Simulate a single group stage round-robin (6 matches for 4 teams).

    Parameters
    ----------
    group_name : str   e.g. "A"
    teams      : list  4 team names
    predictor  : MatchPredictor  wraps the calibrated ensemble + Poisson model
    rng        : np.random.Generator  for reproducibility

    Returns
    -------
    GroupStandings with final rankings applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    records = {t: TeamRecord(team=t) for t in teams}
    results: list[MatchResult] = []

    # All 6 fixtures (C(4,2))
    for home, away in combinations(teams, 2):
        hg, ag = predictor.simulate_scoreline(home, away, rng=rng)

        # Update records
        home_rec = records[home]
        away_rec = records[away]

        home_rec.played += 1
        away_rec.played += 1
        home_rec.goals_for     += hg
        home_rec.goals_against += ag
        away_rec.goals_for     += ag
        away_rec.goals_against += hg

        if hg > ag:
            home_rec.wins   += 1
            away_rec.losses += 1
        elif hg == ag:
            home_rec.draws += 1
            away_rec.draws += 1
        else:
            home_rec.losses += 1
            away_rec.wins   += 1

        # Fair play cards
        hy, hr = _simulate_cards(rng)
        ay, ar = _simulate_cards(rng)
        home_rec.yellows += hy;  home_rec.reds += hr
        away_rec.yellows += ay;  away_rec.reds += ar

        results.append(MatchResult(home=home, away=away,
                                   home_goals=hg, away_goals=ag))

    ranking = rank_group(teams, records, results, rng=rng)

    return GroupStandings(
        group=group_name,
        teams=teams,
        records=records,
        results=results,
        ranking=ranking,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulate all 12 groups
# ─────────────────────────────────────────────────────────────────────────────

def simulate_all_groups(
    predictor: MatchPredictor,
    groups: dict[str, list[str]] | None = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, GroupStandings]:
    """
    Simulate all 12 World Cup groups.

    Returns
    -------
    dict mapping group letter → GroupStandings
    """
    if groups is None:
        groups = WC2026_GROUPS
    if rng is None:
        rng = np.random.default_rng()

    standings: dict[str, GroupStandings] = {}
    for group_name, teams in groups.items():
        standings[group_name] = simulate_group(group_name, teams, predictor, rng=rng)
    return standings


def standings_to_dataframe(all_standings: dict[str, GroupStandings]) -> pd.DataFrame:
    """
    Convert all group standings to a tidy DataFrame for display / serialisation.
    """
    rows = []
    for grp, gs in all_standings.items():
        for pos, team in enumerate(gs.ranking, start=1):
            rec = gs.records[team]
            rows.append({
                "group":    grp,
                "position": pos,
                "team":     team,
                "played":   rec.played,
                "wins":     rec.wins,
                "draws":    rec.draws,
                "losses":   rec.losses,
                "gf":       rec.goals_for,
                "ga":       rec.goals_against,
                "gd":       rec.goal_diff,
                "points":   rec.points,
                "yellows":  rec.yellows,
                "reds":     rec.reds,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=" * 60)
    log.info("TASK 4.1 — Group Stage Simulator")
    log.info("=" * 60)

    predictor = MatchPredictor.load()
    rng = np.random.default_rng(42)

    log.info("Simulating all 12 groups...")
    all_standings = simulate_all_groups(predictor, rng=rng)

    df = standings_to_dataframe(all_standings)

    print("\n" + "=" * 70)
    print("FIFA WORLD CUP 2026 — SIMULATED GROUP STANDINGS")
    print("=" * 70)
    for grp in sorted(all_standings.keys()):
        gs = all_standings[grp]
        print(f"\n  GROUP {grp}  ({' | '.join(gs.teams)})")
        print(f"  {'Pos':<4} {'Team':<28} {'Pld':>3} {'W':>3} {'D':>3} {'L':>3} "
              f"{'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}")
        print("  " + "-" * 66)
        for pos, team in enumerate(gs.ranking, 1):
            rec = gs.records[team]
            qual = "✓" if pos <= 2 else " "
            print(f"  {qual}{pos:<3} {team:<28} {rec.played:>3} {rec.wins:>3} "
                  f"{rec.draws:>3} {rec.losses:>3} {rec.goals_for:>4} "
                  f"{rec.goals_against:>4} {rec.goal_diff:>+4} {rec.points:>4}")

    # Show match results for Group A
    print(f"\n  GROUP A — MATCH RESULTS")
    for r in all_standings["A"].results:
        print(f"    {r.home:<22} {r.home_goals}–{r.away_goals}  {r.away}")

    print("\n✅ Group stage simulation complete.")
    print(f"   48 teams, 12 groups, 72 matches simulated.")


if __name__ == "__main__":
    main()
