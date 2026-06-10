from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────────────
# Rate model constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_CONVERSION_RATE  = 0.75   # per-kick base rate (tournament average)
MIN_KICK_RATE         = 0.60   # floor — even weakest team converts 3/5 on average
MAX_KICK_RATE         = 0.90   # ceiling — even strongest team misses ~1/10
WIN_TO_KICK_SLOPE     = 0.15   # maps shootout win% deviation to per-kick adjustment
MIN_APPEARANCES       = 3      # need at least this many shootouts for data-driven rate

# ─────────────────────────────────────────────────────────────────────────────
# Domain-knowledge overrides (used for teams with sparse historical data)
# Based on academic research and tournament records.
# ─────────────────────────────────────────────────────────────────────────────
SPECIALIST_ADJUSTMENTS: dict[str, float] = {
    # Strong penalty takers (historically)
    "Germany":           +0.05,   # 75% win rate (WC 1982, 1990, 2006…)
    "Iraq":              +0.04,   # 73% historical win rate
    "Saudi Arabia":      +0.04,   # 78% historical win rate
    "Panama":            +0.04,   # 71% win rate
    "Argentina":         +0.03,   # 65% win rate — Messi era improvements
    "Bosnia and Herzegovina": +0.03,
    "Czech Republic":    +0.03,   # 75% win rate in limited data
    "Croatia":           +0.03,   # 64% win rate, consistent performers
    "Australia":         +0.02,
    "DR Congo":          +0.02,   # 63% historical win rate
    "Portugal":          +0.02,   # 63% win rate
    "Sweden":            +0.02,   # 67% win rate
    "Brazil":            +0.01,   # 56% slight edge, but history of high-pressure misses
    "Colombia":          +0.01,   # 58% win rate
    "South Korea":       +0.01,   # 60% win rate
    "United States":     +0.01,   # 60% win rate

    # Average / neutral
    "Spain":              0.00,   # 50% win rate
    "Mexico":             0.00,   # 50% win rate
    "Ecuador":            0.00,   # 50% win rate
    "Senegal":            0.00,   # 52% win rate
    "Uruguay":            0.00,   # 53% win rate
    "Tunisia":            0.00,   # 54% win rate
    "Qatar":              0.00,   # 56% win rate
    "Algeria":           -0.01,   # 47% win rate
    "France":            -0.01,   # 45% win rate — inconsistent
    "Iran":              -0.01,   # 43% win rate
    "Paraguay":          -0.01,   # 46% win rate

    # Weaker takers (historically)
    "Morocco":           -0.02,   # 43% win rate
    "Japan":             -0.02,   # 42% win rate
    "Egypt":             -0.02,   # 58% win rate (large sample, mild -)
    "Ghana":             -0.02,   # 38% win rate
    "Cape Verde":        -0.02,   # 40% win rate
    "Uzbekistan":        -0.02,   # 40% win rate
    "Canada":            -0.03,   # 38% win rate
    "Switzerland":       -0.04,   # 25% win rate — famous pressure collapses
    "New Zealand":       -0.04,   # 25% win rate
    "England":           -0.04,   # 33% win rate — THIRTY years of heartbreak
    "Netherlands":       -0.06,   # 20% win rate — historically weakest at pens

    # Teams with very sparse data — small adjustments
    "Turkey":            +0.01,
    "Jordan":            -0.03,   # 0% in 3 shootouts
    "Norway":             0.00,   # no data
    "Scotland":          +0.01,   # 100% but only n=2
    "Belgium":            0.00,   # 100% but only n=2 (limited data)
    "Haiti":              0.00,   # n=2
    "Curaçao":           -0.03,   # 0% in 2 shootouts
    "Austria":           -0.02,   # 0% in 2 shootouts
    "South Africa":      -0.01,   # 46% win rate (large sample)
    "Ivory Coast":       +0.01,   # 56% win rate
}

# ─────────────────────────────────────────────────────────────────────────────
# Per-kick conversion rate computation
# ─────────────────────────────────────────────────────────────────────────────

def _load_shootout_data(
    csv_path: Path | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Load historical shootout data.  Returns {team: (appearances, wins)}.
    """
    path = csv_path or (PROJECT_ROOT / "shootouts.csv")
    if not path.exists():
        log.warning("shootouts.csv not found at %s — using specialist priors only", path)
        return {}

    df = pd.read_csv(path)
    wins: dict[str, int] = df["winner"].value_counts().to_dict()
    played: dict[str, int] = {}
    for _, row in df.iterrows():
        for team in (row["home_team"], row["away_team"]):
            played[team] = played.get(team, 0) + 1

    return {team: (n, wins.get(team, 0)) for team, n in played.items()}


def _build_kick_rates(
    shootout_data: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """
    Build per-kick conversion rate for every team that has historical data.

    Formula: kick_rate = clip(BASE + (win_rate − 0.5) × slope, MIN, MAX)
    """
    rates: dict[str, float] = {}
    for team, (n, w) in shootout_data.items():
        if n >= MIN_APPEARANCES:
            win_rate = w / n
            adj = (win_rate - 0.50) * WIN_TO_KICK_SLOPE
            rates[team] = float(np.clip(BASE_CONVERSION_RATE + adj,
                                        MIN_KICK_RATE, MAX_KICK_RATE))
    return rates


def _build_fallback_rates() -> dict[str, float]:
    """Build rates for all teams using specialist adjustments."""
    return {
        team: float(np.clip(BASE_CONVERSION_RATE + adj,
                            MIN_KICK_RATE, MAX_KICK_RATE))
        for team, adj in SPECIALIST_ADJUSTMENTS.items()
    }


# Compute once at import time
_SHOOTOUT_DATA: dict[str, tuple[int, int]] = _load_shootout_data()
_DATA_KICK_RATES: dict[str, float] = _build_kick_rates(_SHOOTOUT_DATA)
_FALLBACK_KICK_RATES: dict[str, float] = _build_fallback_rates()


def get_kick_rate(team: str) -> float:
    """
    Return the per-kick conversion probability for *team*.

    Priority:
        1. Historical data-driven rate (≥ 3 shootouts)
        2. Specialist domain-knowledge adjustment
        3. BASE_CONVERSION_RATE (0.75)
    """
    if team in _DATA_KICK_RATES:
        return _DATA_KICK_RATES[team]
    if team in _FALLBACK_KICK_RATES:
        return _FALLBACK_KICK_RATES[team]
    return BASE_CONVERSION_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Shootout result data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShootoutResult:
    """Full record of a penalty shootout."""
    team_a:         str
    team_b:         str
    winner:         str
    loser:          str
    goals_a:        int
    goals_b:        int
    kicks_a:        list[bool]    # True = scored
    kicks_b:        list[bool]
    rate_a:         float
    rate_b:         float
    sudden_death:   bool = False  # True if went beyond 5 kicks each
    sd_rounds:      int  = 0      # number of sudden-death rounds played

    def score_summary(self) -> str:
        score = f"{self.goals_a}–{self.goals_b}"
        suffix = " (SD)" if self.sudden_death else ""
        return f"{self.team_a} {score} {self.team_b}{suffix}"

    def kick_display(self, team: str) -> str:
        """Show kick sequence as e.g. '✓ ✓ ✗ ✓ ✓'."""
        kicks = self.kicks_a if team == self.team_a else self.kicks_b
        return "  ".join("✓" if k else "✗" for k in kicks)


# ─────────────────────────────────────────────────────────────────────────────
# Early-elimination helper
# ─────────────────────────────────────────────────────────────────────────────

def _is_decided(
    goals_a: int, kicks_a: int,
    goals_b: int, kicks_b: int,
    max_kicks: int,
) -> tuple[bool, str]:
    """
    Return (decided, 'A'|'B'|'') after each kick in the regulation 5-round phase.

    A team is eliminated if their maximum possible score is less than the
    opponent's current score (opponent cannot be caught).

    Both teams must have taken the same number of kicks OR B has taken one
    fewer (i.e. A just kicked).
    """
    remaining_a = max_kicks - kicks_a
    remaining_b = max_kicks - kicks_b
    max_a = goals_a + remaining_a
    max_b = goals_b + remaining_b

    # B is eliminated — cannot reach A's score
    if goals_a > max_b:
        return True, "A"
    # A is eliminated — cannot reach B's score
    if goals_b > max_a:
        return True, "B"
    # Both have taken all kicks
    if kicks_a == max_kicks and kicks_b == max_kicks:
        if goals_a != goals_b:
            return True, "A" if goals_a > goals_b else "B"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main shootout simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_shootout(
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    rate_a: float | None = None,
    rate_b: float | None = None,
    max_regulation_kicks: int = 5,
) -> ShootoutResult:
    """
    Simulate a FIFA-rules penalty shootout between team_a and team_b.

    Parameters
    ----------
    team_a, team_b : team names
    rng            : NumPy random generator (for reproducibility)
    rate_a, rate_b : optional per-kick rate overrides (default: looked up)
    max_regulation_kicks : kicks per team in regulation (default 5)

    Returns
    -------
    ShootoutResult with full kick record.

    FIFA Rules Implemented
    ----------------------
    - Team A takes kick 1, then Team B takes kick 1, alternating
    - After every kick, check if the outcome is already decided
    - If level after 5 kicks each → sudden death (one kick each per round)
    - In sudden death: if A scores and B misses → A wins;
      if both score or both miss → next round
    """
    ra = rate_a if rate_a is not None else get_kick_rate(team_a)
    rb = rate_b if rate_b is not None else get_kick_rate(team_b)

    kicks_a_list: list[bool] = []
    kicks_b_list: list[bool] = []
    goals_a = 0
    goals_b = 0
    n_kicks_a = 0
    n_kicks_b = 0

    # ── Regulation: up to max_regulation_kicks each ───────────────────────────
    for _ in range(max_regulation_kicks):
        # Team A kicks
        scored_a = bool(rng.random() < ra)
        kicks_a_list.append(scored_a)
        goals_a  += int(scored_a)
        n_kicks_a += 1

        decided, leader = _is_decided(goals_a, n_kicks_a,
                                      goals_b, n_kicks_b,
                                      max_regulation_kicks)
        if decided:
            winner = team_a if leader == "A" else team_b
            loser  = team_b if leader == "A" else team_a
            return ShootoutResult(
                team_a=team_a, team_b=team_b,
                winner=winner, loser=loser,
                goals_a=goals_a, goals_b=goals_b,
                kicks_a=kicks_a_list, kicks_b=kicks_b_list,
                rate_a=ra, rate_b=rb,
            )

        # Team B kicks
        scored_b = bool(rng.random() < rb)
        kicks_b_list.append(scored_b)
        goals_b  += int(scored_b)
        n_kicks_b += 1

        decided, leader = _is_decided(goals_a, n_kicks_a,
                                      goals_b, n_kicks_b,
                                      max_regulation_kicks)
        if decided:
            winner = team_a if leader == "A" else team_b
            loser  = team_b if leader == "A" else team_a
            return ShootoutResult(
                team_a=team_a, team_b=team_b,
                winner=winner, loser=loser,
                goals_a=goals_a, goals_b=goals_b,
                kicks_a=kicks_a_list, kicks_b=kicks_b_list,
                rate_a=ra, rate_b=rb,
            )

    # ── Sudden death ──────────────────────────────────────────────────────────
    sd_rounds = 0
    while True:
        sd_rounds += 1
        scored_a = bool(rng.random() < ra)
        scored_b = bool(rng.random() < rb)
        kicks_a_list.append(scored_a)
        kicks_b_list.append(scored_b)
        goals_a += int(scored_a)
        goals_b += int(scored_b)

        if scored_a and not scored_b:
            return ShootoutResult(
                team_a=team_a, team_b=team_b,
                winner=team_a, loser=team_b,
                goals_a=goals_a, goals_b=goals_b,
                kicks_a=kicks_a_list, kicks_b=kicks_b_list,
                rate_a=ra, rate_b=rb,
                sudden_death=True, sd_rounds=sd_rounds,
            )
        if scored_b and not scored_a:
            return ShootoutResult(
                team_a=team_a, team_b=team_b,
                winner=team_b, loser=team_a,
                goals_a=goals_a, goals_b=goals_b,
                kicks_a=kicks_a_list, kicks_b=kicks_b_list,
                rate_a=ra, rate_b=rb,
                sudden_death=True, sd_rounds=sd_rounds,
            )
        # Both scored or both missed → next sudden-death round
        # Guard against infinite loop (theoretical, extremely unlikely)
        if sd_rounds > 50:
            log.warning("Penalty shootout exceeded 50 sudden-death rounds; "
                        "breaking tie by higher rate.")
            winner = team_a if ra >= rb else team_b
            loser  = team_b if ra >= rb else team_a
            return ShootoutResult(
                team_a=team_a, team_b=team_b,
                winner=winner, loser=loser,
                goals_a=goals_a, goals_b=goals_b,
                kicks_a=kicks_a_list, kicks_b=kicks_b_list,
                rate_a=ra, rate_b=rb,
                sudden_death=True, sd_rounds=sd_rounds,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Probability helper (for analysis / Monte Carlo reporting)
# ─────────────────────────────────────────────────────────────────────────────

def penalty_win_probability(
    team_a: str,
    team_b: str,
    n_simulations: int = 50_000,
    seed: int = 0,
) -> float:
    """
    Estimate the probability that team_a wins a shootout vs team_b
    via Monte Carlo simulation.  Uses the module-level kick rates.
    """
    rng = np.random.default_rng(seed)
    ra = get_kick_rate(team_a)
    rb = get_kick_rate(team_b)
    wins = sum(
        1 for _ in range(n_simulations)
        if simulate_shootout(team_a, team_b, rng, rate_a=ra, rate_b=rb).winner == team_a
    )
    return wins / n_simulations


# ─────────────────────────────────────────────────────────────────────────────
# CLI — quick self-test
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO,
                         format="%(asctime)s | %(levelname)-8s | %(message)s")

    rng = np.random.default_rng(2026)

    print("=" * 60)
    print("Penalty Shootout Simulator — self-test")
    print("=" * 60)

    # Show per-kick rates for key WC 2026 teams
    wc_teams = [
        "Germany", "Argentina", "Croatia", "Brazil", "Spain",
        "Portugal", "France", "England", "Netherlands", "Switzerland",
    ]
    print(f"\n{'Team':<20} {'Kick Rate':>10}  {'Source'}")
    print("-" * 50)
    for team in wc_teams:
        rate = get_kick_rate(team)
        src = "historical" if team in _DATA_KICK_RATES else "specialist"
        print(f"  {team:<18} {rate:>10.3f}  {src}")

    print("\n" + "=" * 60)
    print("Example shootout: Germany vs England")
    print("=" * 60)
    res = simulate_shootout("Germany", "England", rng)
    print(f"  Winner : {res.winner}")
    print(f"  Score  : {res.goals_a}–{res.goals_b}")
    print(f"  Germany : {res.kick_display('Germany')}")
    print(f"  England : {res.kick_display('England')}")
    if res.sudden_death:
        print(f"  Sudden Death rounds: {res.sd_rounds}")

    print("\nSimulated win probabilities (50 000 trials each):")
    matchups = [
        ("Germany",     "Netherlands"),
        ("Argentina",   "England"),
        ("Spain",       "France"),
        ("Brazil",      "Croatia"),
    ]
    for ta, tb in matchups:
        p = penalty_win_probability(ta, tb, n_simulations=50_000)
        ra = get_kick_rate(ta)
        rb = get_kick_rate(tb)
        print(f"  {ta:<18} vs {tb:<18} → {ta} wins {p*100:.1f}%  "
              f"(rates {ra:.3f} vs {rb:.3f})")


if __name__ == "__main__":
    main()
