"""
knockout.py — FIFA World Cup 2026 Knockout Stage Simulator (Task 4.3).

Simulates the complete knockout bracket:
    Round of 32 (16 matches) → Round of 16 (8) → Quarter-finals (4)
    → Semi-finals (2) → Final (1)

For each knockout match:
    1.  Simulate a 90-minute scoreline using the Poisson model (λ_home, λ_away).
    2.  If draw → simulate 30 minutes of extra time at ET_GOAL_RATE_FACTOR × λ.
    3.  If still level → simulate a full penalty shootout via penalties.py
        (per-kick rates: 75% base, adjusted by historical data / specialist table).

Seeding constraint enforced via the bracket structure:
    Spain   (Group F) → LEFT half  — can only meet Argentina in the Final.
    Argentina (Group A) → RIGHT half — same.

Bracket progression:
    Adjacent R32 pairs (M49+M50, M51+M52, …, M63+M64) feed R16 slots 1–8.
    Adjacent R16 pairs (1+2, 3+4, …) feed QF, then SF, then Final.

Usage:
    from src.simulation.knockout import simulate_knockout, print_knockout_results
    result = simulate_knockout(bracket, predictor, rng=np.random.default_rng(42))
    print_knockout_results(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.simulation.match_predictor import MatchPredictor
from src.simulation.penalties import ShootoutResult, simulate_shootout
from src.simulation.third_place import BracketSlot, KnockoutBracket

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Extra-time goal-rate factor
# ─────────────────────────────────────────────────────────────────────────────
# 30 extra-time minutes at fatigued intensity.
# 30/90 × fatigue_factor(≈1.11) ≈ 0.37 of the 90-min rate.
ET_GOAL_RATE_FACTOR: float = 0.37

# ─────────────────────────────────────────────────────────────────────────────
# R32 match ordering for bracket progression
# ─────────────────────────────────────────────────────────────────────────────
_R32_ORDER = [f"R32_M{i}" for i in range(49, 65)]   # M49 … M64  (16 matches)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnockoutMatchResult:
    """Full record of one knockout match."""
    match_id:       str
    round_name:     str        # "R32" | "R16" | "QF" | "SF" | "Final"
    home:           str
    away:           str
    home_goals_90:  int
    away_goals_90:  int
    # Extra time
    went_to_et:     bool = False
    home_goals_et:  int  = 0
    away_goals_et:  int  = 0
    # Penalties
    went_to_pens:   bool = False
    pen_result:     Optional[ShootoutResult] = None
    winner:         str  = ""

    # ── Derived properties ────────────────────────────────────────────────────
    @property
    def home_goals_total(self) -> int:
        return self.home_goals_90 + self.home_goals_et

    @property
    def away_goals_total(self) -> int:
        return self.away_goals_90 + self.away_goals_et

    @property
    def loser(self) -> str:
        return self.away if self.winner == self.home else self.home

    def score_line(self) -> str:
        """Human-readable score, e.g. '2–1', '1–1 (AET)', '0–0 (4–3 pens)'."""
        h = self.home_goals_total
        a = self.away_goals_total
        if self.went_to_pens and self.pen_result is not None:
            pen_h = self.pen_result.goals_a if self.pen_result.team_a == self.home else self.pen_result.goals_b
            pen_a = self.pen_result.goals_b if self.pen_result.team_a == self.home else self.pen_result.goals_a
            return f"{h}–{a} ({pen_h}–{pen_a} pens)"
        if self.went_to_et:
            return f"{h}–{a} aet"
        return f"{h}–{a}"

    def method(self) -> str:
        """Short string describing how the match was decided."""
        if self.went_to_pens:
            return "Pens"
        if self.went_to_et:
            return "AET"
        return "FT"


@dataclass
class TournamentResult:
    """Full knockout-stage outcome."""
    r32_results:   list[KnockoutMatchResult] = field(default_factory=list)
    r16_results:   list[KnockoutMatchResult] = field(default_factory=list)
    qf_results:    list[KnockoutMatchResult] = field(default_factory=list)
    sf_results:    list[KnockoutMatchResult] = field(default_factory=list)
    final_result:  Optional[KnockoutMatchResult] = None
    champion:      str = ""
    runner_up:     str = ""
    sf_losers:     list[str] = field(default_factory=list)   # 3rd/4th-place candidates

    def all_results(self) -> list[KnockoutMatchResult]:
        all_r = (self.r32_results + self.r16_results +
                 self.qf_results + self.sf_results)
        if self.final_result:
            all_r.append(self.final_result)
        return all_r

    def champion_path(self) -> list[KnockoutMatchResult]:
        """Ordered list of matches won by the champion."""
        return [r for r in self.all_results() if r.winner == self.champion]


# ─────────────────────────────────────────────────────────────────────────────
# Single-match simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_knockout_match(
    match_id:   str,
    round_name: str,
    home:       str,
    away:       str,
    predictor:  MatchPredictor,
    rng:        np.random.Generator,
) -> KnockoutMatchResult:
    """
    Simulate one knockout match.  No draws are allowed as a final result.

    Pipeline
    --------
    1. Poisson scoreline for 90 minutes.
    2. If level: extra-time Poisson at ET_GOAL_RATE_FACTOR × λ.
    3. If still level: full kick-by-kick penalty shootout.
    """
    # ── 90 minutes ────────────────────────────────────────────────────────────
    hg, ag = predictor.simulate_scoreline(home, away, rng=rng)

    result = KnockoutMatchResult(
        match_id=match_id, round_name=round_name,
        home=home, away=away,
        home_goals_90=hg, away_goals_90=ag,
    )

    if hg != ag:
        result.winner = home if hg > ag else away
        return result

    # ── Extra time (30 min) ───────────────────────────────────────────────────
    result.went_to_et = True
    lam_h, lam_a = predictor.predict_lambdas(home, away)
    et_hg = int(rng.poisson(lam_h * ET_GOAL_RATE_FACTOR))
    et_ag = int(rng.poisson(lam_a * ET_GOAL_RATE_FACTOR))
    result.home_goals_et = et_hg
    result.away_goals_et = et_ag

    total_h = hg + et_hg
    total_a = ag + et_ag

    if total_h != total_a:
        result.winner = home if total_h > total_a else away
        return result

    # ── Penalty shootout ──────────────────────────────────────────────────────
    result.went_to_pens = True
    pen = simulate_shootout(home, away, rng)
    result.pen_result = pen
    result.winner     = pen.winner

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Round helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_round(
    pairs:      list[tuple[str, str]],
    round_name: str,
    id_prefix:  str,
    predictor:  MatchPredictor,
    rng:        np.random.Generator,
) -> list[KnockoutMatchResult]:
    """Simulate all matches in one round; return results in order."""
    results = []
    for idx, (home, away) in enumerate(pairs, start=1):
        mid = f"{id_prefix}{idx}"
        res = _simulate_knockout_match(mid, round_name, home, away, predictor, rng)
        log.debug("%s %s %s %s  → %s [%s]",
                  round_name, mid, home, away, res.winner, res.method())
        results.append(res)
    return results


def _winners(results: list[KnockoutMatchResult]) -> list[str]:
    return [r.winner for r in results]


def _adjacent_pairs(teams: list[str]) -> list[tuple[str, str]]:
    """Pair sequential items: (0,1), (2,3), (4,5), …"""
    return [(teams[i], teams[i + 1]) for i in range(0, len(teams), 2)]


# ─────────────────────────────────────────────────────────────────────────────
# Full tournament simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_knockout(
    bracket:   KnockoutBracket,
    predictor: MatchPredictor,
    rng:       Optional[np.random.Generator] = None,
) -> TournamentResult:
    """
    Simulate the entire knockout stage from R32 through the Final.

    Parameters
    ----------
    bracket   : KnockoutBracket from third_place.build_bracket()
    predictor : MatchPredictor wrapping the calibrated ensemble + Poisson model
    rng       : NumPy Generator for reproducibility (created if not supplied)

    Returns
    -------
    TournamentResult with all 31 match results and the champion.
    """
    if rng is None:
        rng = np.random.default_rng()

    tournament = TournamentResult()

    # ── Round of 32 (16 matches) ──────────────────────────────────────────────
    slot_map: dict[str, BracketSlot] = {m.match_id: m for m in bracket.r32_matches}

    r32_results: list[KnockoutMatchResult] = []
    for mid in _R32_ORDER:
        if mid not in slot_map:
            raise ValueError(
                f"Bracket is missing R32 match {mid}. "
                f"Available: {sorted(slot_map)}"
            )
        slot = slot_map[mid]
        res  = _simulate_knockout_match(mid, "R32", slot.home, slot.away,
                                        predictor, rng)
        r32_results.append(res)

    tournament.r32_results = r32_results

    # ── Round of 16 (8 matches) ───────────────────────────────────────────────
    # Adjacent R32 pairs in bracket order: M49+M50 → R16_1, M51+M52 → R16_2, …
    r32_sorted  = sorted(r32_results, key=lambda r: _R32_ORDER.index(r.match_id))
    r16_pairs   = _adjacent_pairs(_winners(r32_sorted))
    r16_results = _simulate_round(r16_pairs, "R16", "R16_M", predictor, rng)
    tournament.r16_results = r16_results

    # ── Quarter-finals (4 matches) ────────────────────────────────────────────
    qf_pairs   = _adjacent_pairs(_winners(r16_results))
    qf_results = _simulate_round(qf_pairs, "QF", "QF_M", predictor, rng)
    tournament.qf_results = qf_results

    # ── Semi-finals (2 matches) ───────────────────────────────────────────────
    sf_pairs   = _adjacent_pairs(_winners(qf_results))
    sf_results = _simulate_round(sf_pairs, "SF", "SF_M", predictor, rng)
    tournament.sf_results = sf_results

    # Semi-final losers are 3rd/4th-place candidates
    tournament.sf_losers = [r.loser for r in sf_results]

    # ── Final ─────────────────────────────────────────────────────────────────
    final_home, final_away = _winners(sf_results)
    final_res = _simulate_knockout_match(
        "Final", "Final", final_home, final_away, predictor, rng
    )
    tournament.final_result = final_res
    tournament.champion  = final_res.winner
    tournament.runner_up = final_res.loser

    log.info("🏆 Champion: %s", tournament.champion)
    return tournament


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

_W = 72   # console width

def _section(title: str, char: str = "─") -> str:
    return f"\n{char * _W}\n  {title}\n{char * _W}"


def print_knockout_results(result: TournamentResult) -> None:
    """Print the full bracket results to stdout."""

    def _print_round(name: str, results: list[KnockoutMatchResult]) -> None:
        print(_section(name))
        print(f"  {'Match':<10} {'Home':<26} {'Score':^14} {'Away':<26} {'Dec':>5}")
        print(f"  {'─'*66}")
        for r in results:
            score = r.score_line()
            dec   = r.method()
            # Bold the winner column
            hw = f"► {r.home}" if r.winner == r.home else f"  {r.home}"
            aw = f"► {r.away}" if r.winner == r.away else f"  {r.away}"
            print(f"  {r.match_id:<10} {hw:<26} {score:^14} {aw:<26} {dec:>5}")

    print(_section("FIFA WORLD CUP 2026 — KNOCKOUT STAGE", "═"))
    _print_round("ROUND OF 32",    result.r32_results)
    _print_round("ROUND OF 16",    result.r16_results)
    _print_round("QUARTER-FINALS", result.qf_results)
    _print_round("SEMI-FINALS",    result.sf_results)

    fr = result.final_result
    if fr:
        print(_section("⚽  THE FINAL", "═"))
        print(f"\n  {'Home':<28} {'Score':^16} {'Away':<28}")
        print(f"  {'─'*70}")
        print(f"  {fr.home:<28} {fr.score_line():^16} {fr.away:<28}")
        print(f"\n  🥇  CHAMPION  : {result.champion}")
        print(f"  🥈  RUNNER-UP : {result.runner_up}")
        if result.sf_losers:
            print(f"  🥉  SF Losers : {' & '.join(result.sf_losers)}")
        print(f"\n  (AET = after extra time; pens = penalty shootout)")

    # Champion's path
    print(_section(f"{result.champion}'s Road to Glory"))
    for r in result.champion_path():
        opp = r.away if r.home == result.champion else r.home
        print(f"  {r.round_name:<8}  beat {opp:<26}  {r.score_line()} [{r.method()}]")
    print()


def print_penalty_details(result: TournamentResult) -> None:
    """Print kick-by-kick detail for every shootout in the tournament."""
    pen_matches = [r for r in result.all_results() if r.went_to_pens]
    if not pen_matches:
        print("No penalty shootouts in this simulation.")
        return

    print(_section("PENALTY SHOOTOUT DETAILS"))
    for r in pen_matches:
        pr = r.pen_result
        if pr is None:
            continue
        print(f"\n  {r.round_name} — {r.home} vs {r.away}")
        print(f"  After {r.home_goals_total}–{r.away_goals_total} aet")
        print(f"  {r.home:<26}  {pr.kick_display(r.home)}  ({pr.goals_a if pr.team_a==r.home else pr.goals_b})")
        print(f"  {r.away:<26}  {pr.kick_display(r.away)}  ({pr.goals_b if pr.team_a==r.home else pr.goals_a})")
        if pr.sudden_death:
            print(f"  → Sudden death ({pr.sd_rounds} rounds)")
        print(f"  → Winner: {pr.winner}  (kick rates: {pr.rate_a:.3f} vs {pr.rate_b:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def knockout_stats(result: TournamentResult) -> dict:
    """Return summary statistics dict for a single knockout simulation."""
    all_r = result.all_results()
    n_et  = sum(1 for r in all_r if r.went_to_et)
    n_pen = sum(1 for r in all_r if r.went_to_pens)
    goals = sum(r.home_goals_total + r.away_goals_total for r in all_r)
    return {
        "total_matches":       len(all_r),
        "decided_at_ft":       len(all_r) - n_et,
        "went_to_et":          n_et,
        "decided_by_penalties": n_pen,
        "total_goals":         goals,
        "avg_goals_per_match": round(goals / max(len(all_r), 1), 2),
        "champion":            result.champion,
        "runner_up":           result.runner_up,
        "sf_losers":           result.sf_losers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("=" * 60)
    log.info("TASK 4.3 — Knockout Stage Simulator")
    log.info("=" * 60)

    from src.simulation.group_stage import simulate_all_groups
    from src.simulation.third_place import build_bracket

    predictor     = MatchPredictor.load()
    rng           = np.random.default_rng(2026)

    log.info("Simulating group stage...")
    all_standings = simulate_all_groups(predictor, rng=rng)

    log.info("Building 32-team bracket...")
    bracket, best_8, _ = build_bracket(all_standings, rng=rng)

    # Verify seeding constraint
    r32_map = {m.match_id: m for m in bracket.r32_matches}
    left_half_teams  = {t for mid in _R32_ORDER[:8]
                        for t in (r32_map[mid].home, r32_map[mid].away)}
    right_half_teams = {t for mid in _R32_ORDER[8:]
                        for t in (r32_map[mid].home, r32_map[mid].away)}

    # Find Spain and Argentina in the bracket
    group_f_teams = {"Spain", "Cape Verde", "Saudi Arabia", "Uruguay"}  # Group F
    group_a_teams = {"Algeria", "Argentina", "Austria", "Jordan"}        # Group A

    spain_team  = left_half_teams  & group_f_teams
    arg_team    = right_half_teams & group_a_teams
    cross_check = (left_half_teams & group_a_teams) | (right_half_teams & group_f_teams)

    log.info("Seeding check — Spain (Group F) in LEFT half:  %s", spain_team or "(eliminated)")
    log.info("Seeding check — Argentina (Group A) in RIGHT half: %s", arg_team or "(eliminated)")
    if cross_check:
        log.warning("Seeding violation detected: %s", cross_check)

    log.info("Simulating knockout rounds (seed=2026)...")
    result = simulate_knockout(bracket, predictor, rng=rng)

    # Print full bracket
    print_knockout_results(result)

    # Print penalty details
    print_penalty_details(result)

    # Summary stats
    stats = knockout_stats(result)
    print("  ─" * 36)
    print("  SUMMARY STATISTICS")
    print("  ─" * 36)
    print(f"  Total matches          : {stats['total_matches']}")
    print(f"  Decided at 90 min (FT) : {stats['decided_at_ft']}")
    print(f"  Decided in Extra Time  : {stats['went_to_et'] - stats['decided_by_penalties']}")
    print(f"  Decided by Penalties   : {stats['decided_by_penalties']}")
    print(f"  Total goals (incl. ET) : {stats['total_goals']}")
    print(f"  Avg goals / match      : {stats['avg_goals_per_match']}")
    print()
    print("  ✅  Task 4.3 — Knockout Stage Simulator complete.")


if __name__ == "__main__":
    main()
