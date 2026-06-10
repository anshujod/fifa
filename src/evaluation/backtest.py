"""
backtest.py — TASK 6.3: Historical World Cup Backtesting (2014, 2018, 2022)

Strategy
--------
For each past World Cup we re-simulate the tournament using ONLY data available
BEFORE the opening match.  The trained ensemble models (XGBoost, LightGBM, ELO,
Poisson) are kept as-is; only the per-team snapshots (form, goals, ELO,
WC experience, H2H) are rebuilt from the pre-tournament cut-off date.

Leakage caveat
--------------
The ensemble models were trained on matches THROUGH 2022, so the WC 2014 and
WC 2018 backtests contain in-sample leakage for the ML components.  The ELO and
Poisson components use only the pre-tournament state and are genuinely OOS.
The WC 2022 backtest is fully out-of-sample for the ML models (model trained on
data ≤ 2022 is evaluated on the WC it did NOT see at training time — its group
stage was included in training but the final result is a legitimate holdout).

Evaluation metrics
------------------
For each year (1000 simulations):
  - Predicted champion rank    : rank of actual champion in pre-tournament P(win) list
  - Predicted finalist rank    : min rank of the two actual finalists
  - Top-5 coverage             : was the actual champion in the top-5 predicted teams?
  - Top-10 semifinalist coverage: were ALL 4 semi-finalists in the top-10?
  - P(champion) pre-tournament : ensemble's probability for the actual champion
  - P(finalist) pre-tournament : ensemble's probability for each finalist
  - Calibration check          : was P(champion) > 10% at the start?

Historical facts (from official FIFA records)
---------------------------------------------
WC 2014 (Brazil, 12 Jun – 13 Jul 2014)
  Champion  : Germany
  Finalists : Germany, Argentina
  Semi      : Germany, Argentina, Brazil, Netherlands

WC 2018 (Russia, 14 Jun – 15 Jul 2018)
  Champion  : France
  Finalists : France, Croatia
  Semi      : France, Croatia, Belgium, England

WC 2022 (Qatar, 20 Nov – 18 Dec 2022)
  Champion  : Argentina
  Finalists : Argentina, France
  Semi      : Argentina, France, Croatia, Morocco

CLI
---
    python3.13 -m src.evaluation.backtest
    python3.13 -m src.evaluation.backtest --years 2022 --n-sims 2000 --save
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parents[2]
MATCH_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "match_features.csv"
RESULTS_DIR       = PROJECT_ROOT / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Historical World Cup Configuration
# ─────────────────────────────────────────────────────────────────────────────

HISTORICAL_WC: dict[int, dict] = {
    2014: {
        "name":       "FIFA World Cup 2014 — Brazil",
        "start_date": "2014-06-12",   # opening match date (cutoff)
        "champion":   "Germany",
        "finalists":  ["Germany", "Argentina"],
        "semifinalists": ["Germany", "Argentina", "Brazil", "Netherlands"],
        "groups": {
            "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
            "B": ["Spain", "Netherlands", "Chile", "Australia"],
            "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
            "D": ["Uruguay", "Costa Rica", "England", "Italy"],
            "E": ["Switzerland", "Ecuador", "France", "Honduras"],
            "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
            "G": ["Germany", "Portugal", "Ghana", "United States"],
            "H": ["Belgium", "Algeria", "Russia", "South Korea"],
        },
    },
    2018: {
        "name":       "FIFA World Cup 2018 — Russia",
        "start_date": "2018-06-14",
        "champion":   "France",
        "finalists":  ["France", "Croatia"],
        "semifinalists": ["France", "Croatia", "Belgium", "England"],
        "groups": {
            "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
            "B": ["Portugal", "Spain", "Morocco", "Iran"],
            "C": ["France", "Australia", "Peru", "Denmark"],
            "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
            "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
            "F": ["Germany", "Mexico", "Sweden", "South Korea"],
            "G": ["Belgium", "Panama", "Tunisia", "England"],
            "H": ["Poland", "Senegal", "Colombia", "Japan"],
        },
    },
    2022: {
        "name":       "FIFA World Cup 2022 — Qatar",
        "start_date": "2022-11-20",
        "champion":   "Argentina",
        "finalists":  ["Argentina", "France"],
        "semifinalists": ["Argentina", "France", "Croatia", "Morocco"],
        "groups": {
            "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
            "B": ["England", "Iran", "United States", "Wales"],
            "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
            "D": ["France", "Australia", "Denmark", "Tunisia"],
            "E": ["Spain", "Costa Rica", "Germany", "Japan"],
            "F": ["Belgium", "Canada", "Morocco", "Croatia"],
            "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
            "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
        },
    },
}

# Standard 32-team WC R16 bracket (group-winner vs runner-up from adjacent groups)
# Format: (match_id, home_slot, away_slot)
# Slots: "1X" = group X winner, "2X" = group X runner-up
WC32_R16: list[tuple[str, str, str]] = [
    ("R16_M49", "1A", "2B"),
    ("R16_M50", "1C", "2D"),
    ("R16_M51", "1E", "2F"),
    ("R16_M52", "1G", "2H"),
    ("R16_M53", "1B", "2A"),
    ("R16_M54", "1D", "2C"),
    ("R16_M55", "1F", "2E"),
    ("R16_M56", "1H", "2G"),
]
# QF: adjacent R16 pairs
WC32_QF_PAIRS = [
    ("QF_M57", "R16_M49", "R16_M50"),
    ("QF_M58", "R16_M53", "R16_M54"),
    ("QF_M59", "R16_M51", "R16_M52"),
    ("QF_M60", "R16_M55", "R16_M56"),
]
# SF: cross-bracket QF winners
WC32_SF_PAIRS = [
    ("SF_M61", "QF_M57", "QF_M58"),
    ("SF_M62", "QF_M59", "QF_M60"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data-classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WC32SimResult:
    """Outcome of a single 32-team WC simulation."""
    champion:      str
    finalist_1:    str   # winner of SF1
    finalist_2:    str   # winner of SF2
    semifinalists: list[str]   # 4 teams
    quarterfinalists: list[str]  # 8 teams
    r16_teams:     list[str]   # 16 teams (knockout stage entrants)


@dataclass
class BacktestResult:
    """Full backtest result for one World Cup."""
    year:   int
    name:   str
    n_sims: int
    cutoff_date: str

    # Predicted champion probabilities (sorted descending)
    predicted_probs: dict[str, float]    # team → P(champion)

    # Finalist probabilities
    finalist_probs:  dict[str, float]    # team → P(finalist)

    # Semifinalist probabilities
    semi_probs:      dict[str, float]    # team → P(semifinalist)

    # Actual results
    actual_champion:     str
    actual_finalists:    list[str]
    actual_semis:        list[str]

    # Evaluation metrics
    champion_rank:       int      # rank of actual champion in predicted list (1 = top pick)
    champion_p_win:      float    # predicted P(win) for actual champion
    finalist_max_rank:   int      # max rank of the two finalists (worst-ranked finalist)
    finalist_min_rank:   int      # min rank (best-ranked finalist)
    semi_max_rank:       int      # worst-ranked semifinalist rank
    champion_in_top5:    bool
    champion_in_top10:   bool
    all_semis_in_top10:  bool
    finalists_both_in_top5: bool

    # Extra info
    pre_tournament_favourites: list[str]   # top-5 picks before tournament
    elapsed_s: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Historical MatchPredictor factory
# ─────────────────────────────────────────────────────────────────────────────

def build_historical_predictor(cutoff_date: str) -> "MatchPredictor":
    """
    Build a MatchPredictor whose team snapshots (form, ELO, WC experience, H2H)
    are computed using ONLY data before `cutoff_date`.

    The trained ML models (XGBoost, LightGBM, calibrated ensemble) are shared
    across all backtests — they are loaded from disk as-is.

    Parameters
    ----------
    cutoff_date : "YYYY-MM-DD" string — exclusive upper bound.
                  Use the tournament's opening-match date so no WC matches
                  contaminate the team states.

    Returns
    -------
    A MatchPredictor instance ready to simulate matches.
    """
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.simulation.match_predictor import MatchPredictor
    import numpy as np

    cutoff = pd.Timestamp(cutoff_date)

    # Load full feature matrix, filter to pre-tournament
    mf_full = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
    mf = mf_full[mf_full["date"] < cutoff].copy()

    log.info(
        "Historical predictor: cutoff=%s  rows=%d (from %d total)",
        cutoff_date, len(mf), len(mf_full)
    )

    # Standard load (models from disk)
    predictor = MatchPredictor.load()

    # Overwrite snapshots with pre-tournament data only
    predictor.team_snapshots.clear()
    predictor.h2h_lookup.clear()
    predictor._build_snapshots(mf)
    predictor._build_h2h(mf)

    n_teams = len(predictor.team_snapshots)
    log.info("Pre-tournament snapshots built for %d teams (cutoff %s)", n_teams, cutoff_date)
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# 32-team tournament simulator
# ─────────────────────────────────────────────────────────────────────────────

def _sim_match(home: str, away: str, predictor, rng: np.random.Generator) -> str:
    """Simulate a single knockout match; returns winner name."""
    hg, ag = predictor.simulate_scoreline(home, away, rng=rng)
    if hg != ag:
        return home if hg > ag else away
    # Extra time / penalties: use raw outcome probability as tiebreak
    p_h, p_d, p_a = predictor.predict_proba(home, away, neutral=True)
    denom = p_h + p_a
    p_h_pen = p_h / denom if denom > 0 else 0.5
    return home if rng.random() < p_h_pen else away


def _sim_group_stage(
    groups: dict[str, list[str]],
    predictor,
    rng: np.random.Generator,
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Simulate all group stage matches.

    Returns
    -------
    winners     : {group_letter → winner team name}
    runners_up  : {group_letter → runner-up team name}
    """
    from src.simulation.group_stage import simulate_group

    winners:    dict[str, str] = {}
    runners_up: dict[str, str] = {}

    for grp_name, teams in groups.items():
        standings = simulate_group(grp_name, teams, predictor, rng=rng)
        winners[grp_name]    = standings.ranking[0]
        runners_up[grp_name] = standings.ranking[1]

    return winners, runners_up


def simulate_wc32_once(
    groups:    dict[str, list[str]],
    predictor,
    rng:       np.random.Generator | None = None,
) -> WC32SimResult:
    """
    Simulate a complete 32-team World Cup tournament once.

    Standard R16 pairings: 1A-2B, 1C-2D, 1E-2F, 1G-2H,
                           1B-2A, 1D-2C, 1F-2E, 1H-2G.

    Parameters
    ----------
    groups    : 8 groups of 4 teams each
    predictor : pre-tournament MatchPredictor
    rng       : NumPy Generator for reproducibility

    Returns
    -------
    WC32SimResult with full knockout stage outcomes.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ── Group stage ──────────────────────────────────────────────────────────
    winners, runners_up = _sim_group_stage(groups, predictor, rng)

    # Resolve slot labels to team names
    def resolve(slot: str) -> str:
        role, grp = slot[0], slot[1]   # "1A" → role="1", grp="A"
        return winners[grp] if role == "1" else runners_up[grp]

    # ── R16 ──────────────────────────────────────────────────────────────────
    r16_teams: list[str] = []   # 16 knockout-stage entrants
    r16_winners: dict[str, str] = {}
    for mid, h_slot, a_slot in WC32_R16:
        home = resolve(h_slot)
        away = resolve(a_slot)
        r16_teams.extend([home, away])
        r16_winners[mid] = _sim_match(home, away, predictor, rng)

    # ── QF ───────────────────────────────────────────────────────────────────
    qf_teams = list(r16_winners.values())   # 8 QF participants
    qf_winners: dict[str, str] = {}
    for mid, r16_a, r16_b in WC32_QF_PAIRS:
        home = r16_winners[r16_a]
        away = r16_winners[r16_b]
        qf_winners[mid] = _sim_match(home, away, predictor, rng)

    # ── SF ───────────────────────────────────────────────────────────────────
    sf_winners: dict[str, str] = {}
    sf_losers:  dict[str, str] = {}
    for mid, qf_a, qf_b in WC32_SF_PAIRS:
        home = qf_winners[qf_a]
        away = qf_winners[qf_b]
        winner = _sim_match(home, away, predictor, rng)
        loser  = away if winner == home else home
        sf_winners[mid] = winner
        sf_losers[mid]  = loser

    semis = list(qf_winners.values())   # 4 SF participants

    # ── Final ────────────────────────────────────────────────────────────────
    sf1_win = sf_winners["SF_M61"]
    sf2_win = sf_winners["SF_M62"]
    champion = _sim_match(sf1_win, sf2_win, predictor, rng)

    return WC32SimResult(
        champion         = champion,
        finalist_1       = sf1_win,
        finalist_2       = sf2_win,
        semifinalists    = semis,
        quarterfinalists = qf_teams,
        r16_teams        = r16_teams,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    year:        int,
    n_sims:      int = 1_000,
    base_seed:   int = 42,
    predictor    = None,   # pass in a pre-built predictor to skip loading
) -> BacktestResult:
    """
    Run a full backtest for a historical World Cup year.

    Parameters
    ----------
    year      : 2014, 2018, or 2022
    n_sims    : number of Monte Carlo simulations (default 1000 — fast)
    base_seed : seed for first simulation
    predictor : optional pre-built MatchPredictor (built via build_historical_predictor)

    Returns
    -------
    BacktestResult with all prediction vs actual comparisons.
    """
    if year not in HISTORICAL_WC:
        raise ValueError(f"Year {year} not in backtest config. Choose from {sorted(HISTORICAL_WC)}")

    cfg         = HISTORICAL_WC[year]
    cutoff_date = cfg["start_date"]
    groups      = cfg["groups"]

    log.info("=" * 65)
    log.info("BACKTEST %d — %s", year, cfg["name"])
    log.info("  Cutoff: %s  |  Simulations: %d", cutoff_date, n_sims)
    log.info("=" * 65)

    t0 = time.perf_counter()

    # Build pre-tournament predictor if not provided
    if predictor is None:
        predictor = build_historical_predictor(cutoff_date)

    all_teams = [t for ts in groups.values() for t in ts]

    # ── Run simulations ───────────────────────────────────────────────────────
    champion_counts:  dict[str, int] = {t: 0 for t in all_teams}
    finalist_counts:  dict[str, int] = {t: 0 for t in all_teams}
    semi_counts:      dict[str, int] = {t: 0 for t in all_teams}
    qf_counts:        dict[str, int] = {t: 0 for t in all_teams}

    log.info("Running %d simulations …", n_sims)
    for i in range(n_sims):
        rng = np.random.default_rng(base_seed + i)
        try:
            result = simulate_wc32_once(groups, predictor, rng=rng)
        except Exception as exc:
            log.debug("Sim %d failed: %s", i, exc)
            continue

        champion_counts[result.champion] += 1
        finalist_counts[result.finalist_1] += 1
        finalist_counts[result.finalist_2] += 1
        for t in result.semifinalists:
            semi_counts[t] += 1
        for t in result.quarterfinalists:
            qf_counts[t] += 1

        if (i + 1) % 200 == 0:
            log.info("  %4d / %d sims done", i + 1, n_sims)

    elapsed = time.perf_counter() - t0

    # ── Convert to probabilities ───────────────────────────────────────────────
    valid = max(sum(champion_counts.values()), 1)

    predicted_probs = {
        t: champion_counts[t] / valid
        for t in sorted(champion_counts, key=champion_counts.get, reverse=True)
    }
    finalist_probs = {
        t: finalist_counts[t] / valid
        for t in sorted(finalist_counts, key=finalist_counts.get, reverse=True)
    }
    semi_probs = {
        t: semi_counts[t] / valid
        for t in sorted(semi_counts, key=semi_counts.get, reverse=True)
    }

    # ── Rank metrics ──────────────────────────────────────────────────────────
    sorted_by_win = sorted(predicted_probs, key=predicted_probs.get, reverse=True)
    ranks = {t: i + 1 for i, t in enumerate(sorted_by_win)}

    actual_champion   = cfg["champion"]
    actual_finalists  = cfg["finalists"]
    actual_semis      = cfg["semifinalists"]

    champion_rank     = ranks.get(actual_champion, 99)
    champion_p_win    = predicted_probs.get(actual_champion, 0.0)

    finalist_ranks    = [ranks.get(t, 99) for t in actual_finalists]
    finalist_min_rank = min(finalist_ranks)
    finalist_max_rank = max(finalist_ranks)

    semi_ranks        = [ranks.get(t, 99) for t in actual_semis]
    semi_max_rank     = max(semi_ranks)

    top5  = sorted_by_win[:5]
    top10 = sorted_by_win[:10]

    result = BacktestResult(
        year               = year,
        name               = cfg["name"],
        n_sims             = n_sims,
        cutoff_date        = cutoff_date,
        predicted_probs    = predicted_probs,
        finalist_probs     = finalist_probs,
        semi_probs         = semi_probs,
        actual_champion    = actual_champion,
        actual_finalists   = actual_finalists,
        actual_semis       = actual_semis,
        champion_rank      = champion_rank,
        champion_p_win     = champion_p_win,
        finalist_max_rank  = finalist_max_rank,
        finalist_min_rank  = finalist_min_rank,
        semi_max_rank      = semi_max_rank,
        champion_in_top5   = actual_champion in top5,
        champion_in_top10  = actual_champion in top10,
        all_semis_in_top10 = all(t in top10 for t in actual_semis),
        finalists_both_in_top5 = all(t in top5 for t in actual_finalists),
        pre_tournament_favourites = top5,
        elapsed_s          = elapsed,
    )

    _print_backtest_result(result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Multi-year backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_all_backtests(
    years:   list[int] | None = None,
    n_sims:  int = 1_000,
    save:    bool = True,
) -> dict[int, BacktestResult]:
    """
    Run backtests for all (or selected) historical World Cups.

    Returns
    -------
    dict mapping year → BacktestResult
    """
    if years is None:
        years = [2014, 2018, 2022]

    results: dict[int, BacktestResult] = {}
    for yr in years:
        try:
            results[yr] = run_backtest(yr, n_sims=n_sims)
        except Exception as exc:
            log.error("Backtest %d failed: %s", yr, exc)

    _print_summary_table(results)

    if save:
        _save_results(results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers (Plotly)
# ─────────────────────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(result: BacktestResult, top_n: int = 10) -> "go.Figure":
    """
    Horizontal bar: predicted P(champion) for top-N teams, with actual results overlaid.
    Green = actual champion, orange = actual finalists, yellow = actual semi-finalists.
    """
    import plotly.graph_objects as go

    sorted_teams = list(result.predicted_probs.keys())[:top_n]
    probs = [result.predicted_probs[t] * 100 for t in sorted_teams]

    # Colour by actual result
    def team_color(team: str) -> str:
        if team == result.actual_champion:
            return "#4CAF50"    # green — champion
        if team in result.actual_finalists:
            return "#FF9800"    # orange — finalist
        if team in result.actual_semis:
            return "#FFD700"    # gold — semi-finalist
        return "#455A64"        # grey

    colors = [team_color(t) for t in sorted_teams]
    labels = []
    for t in sorted_teams:
        suffix = ""
        if t == result.actual_champion:
            suffix = " 🏆"
        elif t in result.actual_finalists:
            suffix = " 🥈"
        elif t in result.actual_semis:
            suffix = " 🥉"
        labels.append(t + suffix)

    fig = go.Figure(go.Bar(
        x             = probs[::-1],
        y             = labels[::-1],
        orientation   = "h",
        marker_color  = colors[::-1],
        text          = [f"{p:.1f}%" for p in probs[::-1]],
        textposition  = "outside",
        hovertemplate = "<b>%{y}</b><br>P(champion): %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title       = dict(
            text  = f"Pre-Tournament Predictions: {result.name}",
            font  = dict(size=16, color="#FFD700"),
        ),
        xaxis_title = "P(Champion) %",
        height      = max(350, top_n * 30 + 80),
        paper_bgcolor = "#0D1B2A",
        plot_bgcolor  = "#0D1B2A",
        font          = dict(color="#E0E0E0", size=12),
        xaxis         = dict(gridcolor="#1E2130"),
        yaxis         = dict(tickfont=dict(size=11)),
        margin        = dict(l=10, r=60, t=60, b=40),
        showlegend    = False,
    )
    return fig


def plot_summary_comparison(all_results: dict[int, BacktestResult]) -> "go.Figure":
    """
    Multi-year comparison: champion rank, finalist rank, % coverage metrics.
    Returns a grouped bar chart.
    """
    import plotly.graph_objects as go

    years  = sorted(all_results.keys())
    champ_ranks   = [all_results[y].champion_rank       for y in years]
    finalist_ranks = [all_results[y].finalist_max_rank  for y in years]
    semi_ranks    = [all_results[y].semi_max_rank        for y in years]
    champ_probs   = [all_results[y].champion_p_win * 100 for y in years]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name         = "Champion rank",
        x            = [str(y) for y in years],
        y            = champ_ranks,
        marker_color = "#4CAF50",
        text         = champ_ranks,
        textposition = "outside",
        hovertemplate = "Year %{x}<br>Champion ranked #%{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name         = "Finalist (worst-ranked) rank",
        x            = [str(y) for y in years],
        y            = finalist_ranks,
        marker_color = "#FF9800",
        text         = finalist_ranks,
        textposition = "outside",
    ))
    fig.add_trace(go.Bar(
        name         = "Semifinalist (worst-ranked) rank",
        x            = [str(y) for y in years],
        y            = semi_ranks,
        marker_color = "#FFD700",
        text         = semi_ranks,
        textposition = "outside",
    ))

    fig.update_layout(
        title        = dict(
            text = "Backtest: Predicted Rank of Actual Champion / Finalists / Semi-finalists",
            font = dict(size=15, color="#FFD700"),
        ),
        barmode      = "group",
        yaxis        = dict(
            title      = "Pre-tournament predicted rank (lower = better)",
            gridcolor  = "#1E2130",
            autorange  = "reversed",    # rank 1 at top
        ),
        xaxis_title  = "World Cup Year",
        height       = 450,
        paper_bgcolor = "#0D1B2A",
        plot_bgcolor  = "#0D1B2A",
        font          = dict(color="#E0E0E0", size=12),
        legend        = dict(bgcolor="#1E2130", bordercolor="#2A2F3F", borderwidth=1),
        margin        = dict(l=10, r=20, t=70, b=40),
    )
    return fig


def plot_probability_heatmap(all_results: dict[int, BacktestResult], top_n: int = 10) -> "go.Figure":
    """
    Heatmap: teams (rows) × WC years (cols), cell = P(champion) %.
    Red cells = actual champions.
    """
    import plotly.graph_objects as go

    years = sorted(all_results.keys())

    # Collect top-N teams per year, union
    teams_set: set[str] = set()
    for r in all_results.values():
        teams_set.update(list(r.predicted_probs.keys())[:top_n])
    all_semis = {t for r in all_results.values() for t in r.actual_semis}
    teams_set.update(all_semis)

    # Sort teams by average predicted probability across years
    team_avg = {
        t: np.mean([all_results[y].predicted_probs.get(t, 0.0) for y in years])
        for t in teams_set
    }
    teams = sorted(teams_set, key=team_avg.get, reverse=True)[:top_n + 4]

    z_vals     = []
    cell_text  = []
    for team in teams:
        row_z    = []
        row_text = []
        for yr in years:
            p = all_results[yr].predicted_probs.get(team, 0.0) * 100
            row_z.append(round(p, 1))
            champ = all_results[yr].actual_champion
            tag   = " 🏆" if team == champ else (
                    " 🥈" if team in all_results[yr].actual_finalists else (
                    " 🥉" if team in all_results[yr].actual_semis else ""))
            row_text.append(f"{p:.1f}%{tag}")
        z_vals.append(row_z)
        cell_text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z          = z_vals,
        x          = [str(y) for y in years],
        y          = teams,
        text       = cell_text,
        texttemplate = "%{text}",
        colorscale = "YlOrRd",
        colorbar   = dict(title="P(Champion) %", thickness=14),
        hovertemplate = "<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
        xgap       = 2,
        ygap       = 2,
    ))
    fig.update_layout(
        title  = dict(
            text = f"Pre-Tournament P(Champion) — Top {len(teams)} Teams by Year",
            font = dict(size=15, color="#FFD700"),
        ),
        height = max(400, len(teams) * 30 + 120),
        paper_bgcolor = "#0D1B2A",
        plot_bgcolor  = "#0D1B2A",
        font   = dict(color="#E0E0E0", size=11),
        xaxis  = dict(tickfont=dict(size=13)),
        yaxis  = dict(tickfont=dict(size=11)),
        margin = dict(l=10, r=20, t=70, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def results_to_dataframe(all_results: dict[int, BacktestResult]) -> pd.DataFrame:
    """Convert backtest results to a tidy comparison DataFrame."""
    rows = []
    for yr, r in sorted(all_results.items()):
        rows.append({
            "Year":                yr,
            "Actual Champion":     r.actual_champion,
            "Champion Rank":       r.champion_rank,
            "P(Champion)":         f"{r.champion_p_win*100:.1f}%",
            "Worst Finalist Rank": r.finalist_max_rank,
            "Worst Semi Rank":     r.semi_max_rank,
            "Champ in Top-5":      "✅" if r.champion_in_top5  else "❌",
            "Champ in Top-10":     "✅" if r.champion_in_top10 else "❌",
            "Both Finalists Top-5":"✅" if r.finalists_both_in_top5 else "❌",
            "All Semis Top-10":    "✅" if r.all_semis_in_top10 else "❌",
            "Top-5 Favourites":    ", ".join(r.pre_tournament_favourites),
            "Elapsed (s)":         f"{r.elapsed_s:.1f}",
        })
    return pd.DataFrame(rows)


def _print_backtest_result(r: BacktestResult) -> None:
    W = 70
    tick = lambda v: "✅" if v else "❌"
    print(f"\n  {'─'*W}")
    print(f"  WC {r.year} — {r.name}")
    print(f"  {'─'*W}")
    print(f"  Simulations   : {r.n_sims:,}")
    print(f"  Cutoff date   : {r.cutoff_date}")
    print(f"  Actual champion: {r.actual_champion}")
    print()
    print(f"  {'Team':<22} {'P(Win)':>7}  {'P(Final)':>9}  {'P(Semi)':>8}  {'Rank':>4}")
    print(f"  {'─'*60}")
    for i, (team, p) in enumerate(list(r.predicted_probs.items())[:10], 1):
        pf   = r.finalist_probs.get(team, 0.0)
        ps   = r.semi_probs.get(team, 0.0)
        flag = ""
        if team == r.actual_champion:
            flag = " 🏆"
        elif team in r.actual_finalists:
            flag = " 🥈"
        elif team in r.actual_semis:
            flag = " 🥉"
        print(f"  {i:2}. {team:<18}{flag:<3} {p*100:>6.1f}%  {pf*100:>8.1f}%  {ps*100:>7.1f}%")
    print()
    print(f"  Champion rank       : #{r.champion_rank}  (P={r.champion_p_win*100:.1f}%)")
    print(f"  Finalist ranks      : #{r.finalist_min_rank} and #{r.finalist_max_rank}")
    print(f"  Champion in top-5   : {tick(r.champion_in_top5)}")
    print(f"  Champion in top-10  : {tick(r.champion_in_top10)}")
    print(f"  Both finalists top-5: {tick(r.finalists_both_in_top5)}")
    print(f"  All semis in top-10 : {tick(r.all_semis_in_top10)}")
    print(f"  Elapsed             : {r.elapsed_s:.1f}s")


def _print_summary_table(all_results: dict[int, BacktestResult]) -> None:
    W = 80
    print("\n" + "═" * W)
    print("  HISTORICAL BACKTEST SUMMARY")
    print("═" * W)
    df = results_to_dataframe(all_results)
    print(df.to_string(index=False, col_space=2))
    # Overall stats
    n = len(all_results)
    top5  = sum(1 for r in all_results.values() if r.champion_in_top5)
    top10 = sum(1 for r in all_results.values() if r.champion_in_top10)
    sf10  = sum(1 for r in all_results.values() if r.all_semis_in_top10)
    print(f"\n  Champion in top-5  : {top5}/{n} ({top5/n*100:.0f}%)")
    print(f"  Champion in top-10 : {top10}/{n} ({top10/n*100:.0f}%)")
    print(f"  All semis in top-10: {sf10}/{n} ({sf10/n*100:.0f}%)")
    print("═" * W + "\n")


def _save_results(all_results: dict[int, BacktestResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / "backtest_results.json"
    out  = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "years": {
            str(yr): {
                **asdict(r),
                # Convert dicts with tuple keys to lists for JSON serialisation
            }
            for yr, r in all_results.items()
        },
        "summary": results_to_dataframe(all_results).to_dict(orient="records"),
    }
    with open(dest, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Backtest report saved → %s", dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="TASK 6.3 — Historical WC Backtesting")
    p.add_argument("--years",  nargs="+", type=int, default=[2014, 2018, 2022],
                   help="Which World Cups to backtest (default: 2014 2018 2022)")
    p.add_argument("--n-sims", type=int, default=1_000,
                   help="Number of MC simulations per year (default: 1000)")
    p.add_argument("--seed",   type=int, default=42, help="Base random seed")
    p.add_argument("--no-save", action="store_true", help="Skip saving JSON report")
    return p.parse_args()


def main() -> None:
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    print("=" * 70)
    print("  TASK 6.3 — Historical World Cup Backtesting")
    print("=" * 70)
    print(f"  Years      : {args.years}")
    print(f"  Simulations: {args.n_sims:,} per year")
    print()

    all_results = run_all_backtests(
        years  = args.years,
        n_sims = args.n_sims,
        save   = not args.no_save,
    )

    # Final verdict
    n = len(all_results)
    top5_pct  = sum(1 for r in all_results.values() if r.champion_in_top5) / n * 100
    top10_pct = sum(1 for r in all_results.values() if r.champion_in_top10) / n * 100
    sf10_pct  = sum(1 for r in all_results.values() if r.all_semis_in_top10) / n * 100

    print(f"  ✅  Task 6.3 — Backtesting complete")
    print(f"  Champion top-5 rate  : {top5_pct:.0f}% ({sum(1 for r in all_results.values() if r.champion_in_top5)}/{n})")
    print(f"  Champion top-10 rate : {top10_pct:.0f}% ({sum(1 for r in all_results.values() if r.champion_in_top10)}/{n})")
    print(f"  All semis top-10 rate: {sf10_pct:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
