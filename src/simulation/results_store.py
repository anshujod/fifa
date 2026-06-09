"""
results_store.py — Results Storage & Reproducibility (Task 4.5).

Provides:
    save_results()             — persist MC results to data/processed/mc_results.json
    load_results()             — reload from JSON → MCResults dataclass
    get_confidence_interval()  — Wilson CI for any (team, stage, alpha)
    run_variance_analysis()    — stability analysis at 1k / 5k / 10k sims
    print_variance_report()    — formatted stability table

JSON schema  (data/processed/mc_results.json)
─────────────────────────────────────────────
{
  "format_version": "2.0",
  "generated_at":   "...",
  "metadata":       { n_simulations, base_seed, elapsed_seconds, … },
  "stages":         ["group_qualify", …, "winner"],
  "teams": {
    "Spain": {
      "group":       "F",
      "rank":        1,
      "probabilities": { "winner": 0.177, … },
      "confidence_intervals": {
        "0.90": { "winner": [0.171, 0.183], … },
        "0.95": { … },
        "0.99": { … }
      }
    }, …
  },
  "variance_analysis": {
    "analytical":  { … },   ← no extra simulations needed
    "empirical":   { … }    ← 3 reps × N=1000, 2 reps × N=5000
  }
}

Reproducibility
───────────────
Set np.random.seed(42) for legacy code; use np.random.default_rng(42) for
modern NumPy.  All simulation seeds derive deterministically from base_seed.

Usage
─────
    python -m src.simulation.results_store           # full run + analysis
    python -m src.simulation.results_store --quick   # reuse existing MC results
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

log = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DEFAULT_PATH  = PROJECT_ROOT / "data" / "processed" / "mc_results.json"
MC_CSV_PATH   = PROJECT_ROOT / "results" / "monte_carlo_probabilities.csv"
MC_STATS_PATH = PROJECT_ROOT / "results" / "monte_carlo_stats.json"

# Canonical stage keys (short names used internally + in JSON)
STAGES: list[str] = [
    "group_qualify",
    "round_of_32",
    "round_of_16",
    "quarter_final",
    "semi_final",
    "final",
    "winner",
]

# CSV column → short stage key
_CSV_TO_STAGE: dict[str, str] = {
    "p_group_qualify":  "group_qualify",
    "p_round_of_32":    "round_of_32",
    "p_round_of_16":    "round_of_16",
    "p_quarter_final":  "quarter_final",
    "p_semi_final":     "semi_final",
    "p_final":          "final",
    "p_winner":         "winner",
}
_STAGE_TO_CSV: dict[str, str] = {v: k for k, v in _CSV_TO_STAGE.items()}

# Human-readable stage labels
STAGE_LABELS: dict[str, str] = {
    "group_qualify":  "Group Stage",
    "round_of_32":    "Round of 32",
    "round_of_16":    "Round of 16",
    "quarter_final":  "Quarter-Final",
    "semi_final":     "Semi-Final",
    "final":          "Final",
    "winner":         "Champion",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TeamResult:
    """Probability estimates + CIs for one team."""
    team:         str
    group:        str
    rank:         int
    probabilities: dict[str, float]                       # stage → p
    cis:           dict[str, dict[str, tuple[float, float]]]  # alpha → stage → (lo, hi)
    n_simulations: int


@dataclass
class MCResults:
    """Complete Monte Carlo simulation output with all metadata."""
    teams:         dict[str, TeamResult]     # team_name → TeamResult
    metadata:      dict                       # run parameters + timing
    variance_analysis: Optional[dict] = None  # set after running analysis
    generated_at:  str = ""

    # ── Convenience accessors ─────────────────────────────────────────────────

    def probability(self, team: str, stage: str) -> float:
        """Return P(team reaches stage)."""
        _validate_stage(stage)
        return self.teams[team].probabilities[stage]

    def top_n(self, stage: str = "winner", n: int = 10) -> list[tuple[str, float]]:
        """Return top-N teams by probability at given stage."""
        _validate_stage(stage)
        ranked = sorted(
            self.teams.items(),
            key=lambda kv: kv[1].probabilities[stage],
            reverse=True,
        )
        return [(name, t.probabilities[stage]) for name, t in ranked[:n]]


# ─────────────────────────────────────────────────────────────────────────────
# Confidence interval utilities
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(p: float, n: int, alpha: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Parameters
    ----------
    p     : point estimate (0 – 1)
    n     : number of trials
    alpha : confidence level (default 0.95)

    Returns
    -------
    (lower, upper) clipped to [0, 1]
    """
    if n <= 0:
        return (0.0, 1.0)
    z = scipy_stats.norm.ppf(1 - (1 - alpha) / 2)
    denom   = 1 + z ** 2 / n
    centre  = (p + z ** 2 / (2 * n)) / denom
    half    = z * (p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5 / denom
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid: {STAGES}"
        )


def get_confidence_interval(
    team:    str,
    stage:   str,
    alpha:   float = 0.95,
    results: Optional[MCResults] = None,
    path:    Path | str | None = None,
) -> tuple[float, float, float]:
    """
    Return the (lower, estimate, upper) confidence interval for *team* at *stage*.

    Parameters
    ----------
    team    : team name, e.g. "Spain"
    stage   : one of STAGES — "group_qualify", "round_of_32", "round_of_16",
              "quarter_final", "semi_final", "final", "winner"
    alpha   : confidence level (default 0.95)
    results : pre-loaded MCResults; if None, loads from *path* or DEFAULT_PATH
    path    : JSON file path (overrides DEFAULT_PATH)

    Returns
    -------
    (lower, estimate, upper) as floats in [0, 1]

    Example
    -------
    >>> lo, est, hi = get_confidence_interval("Spain", "winner")
    >>> print(f"Spain P(champion) = {est:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    Spain P(champion) = 0.177  95% CI [0.170, 0.185]
    """
    _validate_stage(stage)

    if results is None:
        results = load_results(path)

    if team not in results.teams:
        raise KeyError(
            f"Team '{team}' not found. Available: {sorted(results.teams)}"
        )

    tr  = results.teams[team]
    est = tr.probabilities[stage]
    n   = tr.n_simulations

    alpha_key = f"{alpha:.2f}"
    if alpha_key in tr.cis and stage in tr.cis[alpha_key]:
        lo, hi = tr.cis[alpha_key][stage]
    else:
        lo, hi = _wilson_ci(est, n, alpha)

    return lo, est, hi


# ─────────────────────────────────────────────────────────────────────────────
# Build MCResults from a probabilities DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _build_results(
    df:       pd.DataFrame,
    metadata: dict,
    alphas:   list[float] = (0.90, 0.95, 0.99),
) -> MCResults:
    """
    Construct an MCResults object from the aggregated probabilities DataFrame.

    The DataFrame must have columns: team, group, p_group_qualify, …, p_winner
    (output of monte_carlo.run_monte_carlo or the saved CSV).
    """
    n = int(metadata.get("n_simulations", df["n_simulations"].iloc[0]))

    teams: dict[str, TeamResult] = {}
    for rank, row in df.iterrows():
        name  = row["team"]
        probs = {s: float(row[_STAGE_TO_CSV[s]]) for s in STAGES}

        # Pre-compute CIs for standard alpha levels
        cis: dict[str, dict[str, tuple[float, float]]] = {}
        for alpha in alphas:
            alpha_key = f"{alpha:.2f}"
            cis[alpha_key] = {
                s: _wilson_ci(probs[s], n, alpha) for s in STAGES
            }

        teams[name] = TeamResult(
            team=name,
            group=str(row["group"]),
            rank=int(rank),
            probabilities=probs,
            cis=cis,
            n_simulations=n,
        )

    return MCResults(
        teams=teams,
        metadata=metadata,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variance analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analytical_variance(
    results: MCResults,
    n_sizes: list[int] = (1_000, 5_000, 10_000),
) -> dict:
    """
    Compute theoretical standard errors at different sample sizes.

    SE = sqrt(p * (1 - p) / n)   [normal approximation]

    No extra simulation runs required — derived purely from the stored
    probability estimates.
    """
    out: dict = {
        "methodology":
            "Theoretical SE = √(p·(1−p) / n)  [normal approximation]",
        "n_sizes": list(n_sizes),
        "stages_analysed": STAGES,
        "by_team": {},
    }

    for team, tr in results.teams.items():
        out["by_team"][team] = {}
        for stage in STAGES:
            p = tr.probabilities[stage]
            stage_info: dict[str, dict] = {}
            for n in n_sizes:
                se   = float(np.sqrt(p * (1 - p) / n)) if n > 0 else 0.0
                cv   = float(se / p) if p > 1e-6 else 0.0
                lo, hi = _wilson_ci(p, n, 0.95)
                stage_info[str(n)] = {
                    "se":           round(se, 5),
                    "cv":           round(cv, 4),       # coefficient of variation
                    "ci95_half":    round((hi - lo) / 2, 5),
                    "stable":       cv < 0.05,          # CV < 5% = "stable"
                }
            out["by_team"][team][stage] = {
                "estimate": p,
                "by_n":     stage_info,
                # At which N does SE first become "stable" (CV < 5%)?
                "converges_at": next(
                    (n for n in sorted(n_sizes)
                     if stage_info[str(n)]["stable"]),
                    None,
                ),
            }
    return out


def _empirical_variance(
    n_sizes: list[int] = (1_000, 5_000),
    n_reps:  int = 3,
    n_workers: int | None = None,
    base_seed: int = 42,
) -> dict:
    """
    Run n_reps independent simulations at each N and measure empirical std.

    Uses the same multiprocessing infrastructure as run_monte_carlo.
    """
    from src.simulation.monte_carlo import (
        ALL_WC_TEAMS, run_monte_carlo,
    )

    out: dict = {
        "methodology":
            f"{n_reps} independent runs at each N; "
            f"std measured across runs",
        "n_sizes":      list(n_sizes),
        "n_reps":       n_reps,
        "n_workers":    n_workers or mp.cpu_count(),
        "by_team":      {},
    }

    # Collect rep estimates: {n: {team: [p1, p2, …]}}
    rep_data: dict[int, dict[str, list[float]]] = {}

    for n in n_sizes:
        rep_data[n] = {team: [] for team in ALL_WC_TEAMS}
        log.info("  Empirical variance: N=%d, %d reps…", n, n_reps)
        for rep in range(n_reps):
            # Each rep uses a deterministic, non-overlapping seed block
            seed = base_seed + rep * 1_000_000 + n * 100
            df, _ = run_monte_carlo(
                n_simulations=n,
                base_seed=seed,
                n_workers=n_workers,
                verbose=False,
                save=False,
            )
            p_col = df.set_index("team")["p_winner"].to_dict()
            for team in ALL_WC_TEAMS:
                rep_data[n][team].append(float(p_col.get(team, 0.0)))

    # Compute statistics per team
    for team in ALL_WC_TEAMS:
        team_info: dict[str, dict] = {}
        for n in n_sizes:
            vals = rep_data[n][team]
            arr  = np.array(vals)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            cv   = float(std / mean) if mean > 1e-6 else 0.0
            team_info[str(n)] = {
                "rep_estimates": [round(v, 5) for v in vals],
                "mean":  round(mean, 5),
                "std":   round(std,  5),
                "cv":    round(cv,   4),
                "stable": cv < 0.05,
            }
        out["by_team"][team] = team_info

    return out


def run_variance_analysis(
    results:   MCResults,
    n_sizes:   list[int] = (1_000, 5_000, 10_000),
    n_reps:    int = 3,
    n_workers: int | None = None,
    seed:      int = 42,
) -> dict:
    """
    Run full variance analysis (analytical + empirical) and attach to results.

    Parameters
    ----------
    results   : MCResults from build_results or load_results
    n_sizes   : sample sizes to evaluate (default 1k / 5k / 10k)
    n_reps    : independent replications at each N for empirical analysis
    n_workers : parallel workers (None = CPU count)
    seed      : base RNG seed

    Returns
    -------
    variance dict (also stored in results.variance_analysis)
    """
    log.info("Running analytical variance analysis…")
    analytical = _analytical_variance(results, n_sizes)

    # Only run empirical for sizes < the main simulation's N
    main_n = int(results.metadata.get("n_simulations", 10_000))
    emp_sizes = [n for n in n_sizes if n < main_n]

    empirical = None
    if emp_sizes:
        log.info("Running empirical variance analysis for N=%s…", emp_sizes)
        empirical = _empirical_variance(
            n_sizes=emp_sizes,
            n_reps=n_reps,
            n_workers=n_workers,
            base_seed=seed,
        )

    variance = {
        "analytical": analytical,
        "empirical":  empirical,
    }
    results.variance_analysis = variance
    return variance


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation
# ─────────────────────────────────────────────────────────────────────────────

def _to_json_safe(obj):
    """Recursively convert numpy scalars / tuples to JSON-serialisable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def save_results(
    results: MCResults,
    path:    Path | str | None = None,
) -> Path:
    """
    Serialise MCResults to JSON.

    Saves to data/processed/mc_results.json by default.
    All per-team probabilities, CIs, metadata, and variance analysis
    (if run) are included.
    """
    dest = Path(path) if path else DEFAULT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    doc: dict = {
        "format_version": "2.0",
        "generated_at":   results.generated_at or datetime.now(timezone.utc).isoformat(),
        "metadata":       _to_json_safe(results.metadata),
        "stages":         STAGES,
        "stage_labels":   STAGE_LABELS,
        "teams":          {},
    }

    for name, tr in results.teams.items():
        doc["teams"][name] = {
            "group":       tr.group,
            "rank":        tr.rank,
            "probabilities": {s: round(p, 6) for s, p in tr.probabilities.items()},
            "confidence_intervals": {
                alpha_key: {
                    s: list(bounds)
                    for s, bounds in stage_cis.items()
                }
                for alpha_key, stage_cis in tr.cis.items()
            },
            "n_simulations": tr.n_simulations,
        }

    if results.variance_analysis:
        doc["variance_analysis"] = _to_json_safe(results.variance_analysis)

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    log.info("MC results saved → %s  (%.1f KB)", dest,
             dest.stat().st_size / 1024)
    return dest


def load_results(path: Path | str | None = None) -> MCResults:
    """
    Load MCResults from JSON (data/processed/mc_results.json by default).

    Returns an MCResults with all teams, probabilities, CIs, and
    variance analysis (if present in the file).
    """
    src = Path(path) if path else DEFAULT_PATH
    if not src.exists():
        raise FileNotFoundError(
            f"mc_results.json not found at {src}. "
            "Run results_store.main() to generate it."
        )

    with open(src, encoding="utf-8") as f:
        doc = json.load(f)

    metadata = doc.get("metadata", {})
    n = int(metadata.get("n_simulations", 10_000))

    teams: dict[str, TeamResult] = {}
    for name, td in doc.get("teams", {}).items():
        cis_raw = td.get("confidence_intervals", {})
        cis: dict[str, dict[str, tuple[float, float]]] = {
            alpha_key: {
                stage: tuple(bounds)  # type: ignore[assignment]
                for stage, bounds in stage_dict.items()
            }
            for alpha_key, stage_dict in cis_raw.items()
        }
        teams[name] = TeamResult(
            team=name,
            group=td["group"],
            rank=td["rank"],
            probabilities={s: td["probabilities"][s] for s in STAGES},
            cis=cis,
            n_simulations=td.get("n_simulations", n),
        )

    results = MCResults(
        teams=teams,
        metadata=metadata,
        variance_analysis=doc.get("variance_analysis"),
        generated_at=doc.get("generated_at", ""),
    )
    log.info("MC results loaded ← %s  (%d teams, N=%d)",
             src, len(teams), n)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_ci_table(results: MCResults, teams: list[str] | None = None) -> None:
    """Print a confidence-interval table for the given teams."""
    if teams is None:
        teams = [name for name, _ in results.top_n("winner", n=10)]

    n = results.metadata.get("n_simulations", "?")
    print(f"\n{'═'*90}")
    print(f"  Confidence Intervals  (N = {n:,} simulations)")
    print(f"{'═'*90}")
    header = (
        f"  {'Team':<22} {'P(champion)':>12}  "
        f"{'90% CI':^18}  {'95% CI':^18}  {'99% CI':^18}"
    )
    print(header)
    print(f"  {'─'*84}")
    for name in teams:
        if name not in results.teams:
            continue
        tr = results.teams[name]
        p  = tr.probabilities["winner"]
        ci90 = tr.cis.get("0.90", {}).get("winner", _wilson_ci(p, tr.n_simulations, 0.90))
        ci95 = tr.cis.get("0.95", {}).get("winner", _wilson_ci(p, tr.n_simulations, 0.95))
        ci99 = tr.cis.get("0.99", {}).get("winner", _wilson_ci(p, tr.n_simulations, 0.99))
        print(
            f"  {name:<22} {p*100:>10.2f}%  "
            f"[{ci90[0]*100:5.2f}%, {ci90[1]*100:5.2f}%]  "
            f"[{ci95[0]*100:5.2f}%, {ci95[1]*100:5.2f}%]  "
            f"[{ci99[0]*100:5.2f}%, {ci99[1]*100:5.2f}%]"
        )
    print(f"{'═'*90}\n")


def print_variance_report(
    results: MCResults,
    top_n:   int = 10,
) -> None:
    """Print a formatted variance / stability report."""
    va = results.variance_analysis
    if va is None:
        print("No variance analysis available. Run run_variance_analysis() first.")
        return

    analytical  = va.get("analytical", {})
    empirical   = va.get("empirical")
    n_sizes     = analytical.get("n_sizes", [1000, 5000, 10000])
    by_team_ana = analytical.get("by_team", {})

    W = 90
    print(f"\n{'═'*W}")
    print("  VARIANCE ANALYSIS — Stability of Probabilities vs Sample Size")
    print(f"{'═'*W}")
    print(f"  {'Analytical SE = √(p·(1−p)/n)'}")
    print(f"  Stage: Champion (p_winner)   |   Stable = CV < 5%")
    print()
    print(f"  {'Team':<22}", end="")
    for n in n_sizes:
        label = f"N={n:,}"
        print(f"  {label:^18}", end="")
    print()
    print(f"  {'─'*22}", end="")
    for _ in n_sizes:
        print(f"  {'─'*18}", end="")
    print()

    # Sort by p_winner descending
    top_teams = [name for name, _ in results.top_n("winner", n=top_n)]
    for name in top_teams:
        if name not in by_team_ana:
            continue
        stage_data = by_team_ana[name].get("winner", {})
        p = stage_data.get("estimate", 0)
        print(f"  {name:<22}", end="")
        for n in n_sizes:
            info = stage_data.get("by_n", {}).get(str(n), {})
            se   = info.get("se", 0)
            cv   = info.get("cv", 0)
            ok   = "✓" if info.get("stable") else "✗"
            print(f"  SE={se*100:.2f}% CV={cv*100:.0f}% {ok:1}  ", end="")
        print()
    print(f"  {'─'*W}")
    print(f"  ✓ = CV < 5%  (probability estimate is stable at that N)")
    print()

    # Empirical section
    if empirical:
        emp_sizes  = empirical.get("n_sizes", [])
        n_reps     = empirical.get("n_reps", 0)
        by_team_e  = empirical.get("by_team", {})
        print(f"  Empirical std  ({n_reps} independent runs per N)")
        print(f"  Stage: Champion (p_winner)")
        print()
        print(f"  {'Team':<22}", end="")
        for n in emp_sizes:
            label = f"N={n:,} (empirical)"
            print(f"  {label:^22}", end="")
        print()
        print(f"  {'─'*22}", end="")
        for _ in emp_sizes:
            print(f"  {'─'*22}", end="")
        print()
        for name in top_teams:
            if name not in by_team_e:
                continue
            print(f"  {name:<22}", end="")
            for n in emp_sizes:
                info = by_team_e[name].get(str(n), {})
                mean = info.get("mean", 0)
                std  = info.get("std",  0)
                cv   = info.get("cv",   0)
                ok   = "✓" if info.get("stable") else "✗"
                print(f"  μ={mean*100:.1f}% σ={std*100:.2f}% CV={cv*100:.0f}% {ok}", end="")
            print()
        print(f"  {'─'*W}")
        print()

    # Convergence summary
    print(f"  Convergence summary  (first N where CV < 5% for p_winner)")
    print(f"  {'─'*60}")
    for name in top_teams:
        if name not in by_team_ana:
            continue
        stage_d = by_team_ana[name].get("winner", {})
        conv_at = stage_d.get("converges_at")
        p       = stage_d.get("estimate", 0)
        label   = f"N={conv_at:,}" if conv_at else "Not stable even at max N"
        print(f"  {name:<22}  p={p*100:.1f}%  converges at {label}")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 4.5 — MC Results Storage & Reproducibility"
    )
    p.add_argument("--quick",     action="store_true",
                   help="Reuse existing MC CSV (skip re-running 10k sims)")
    p.add_argument("--no-empirical", action="store_true",
                   help="Skip empirical variance runs (analytical only)")
    p.add_argument("--n",          type=int, default=10_000,
                   help="Simulations for main run (default 10 000)")
    p.add_argument("--n-reps",     type=int, default=3,
                   help="Reps for empirical variance (default 3)")
    p.add_argument("--workers",    type=int, default=None)
    p.add_argument("--seed",       type=int, default=42,
                   help="Base RNG seed (default 42)")
    p.add_argument("--output",     type=str, default=None,
                   help=f"JSON output path (default {DEFAULT_PATH})")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    # Legacy NumPy global seed (for any code using np.random directly)
    np.random.seed(42)
    # Modern: use np.random.default_rng(42) for actual simulation seeding

    args = _parse_args()

    log.info("=" * 60)
    log.info("TASK 4.5 — Results Storage & Reproducibility")
    log.info("=" * 60)

    # ── Step 1: Get (or re-run) the main Monte Carlo results ──────────────────
    if args.quick and MC_CSV_PATH.exists() and MC_STATS_PATH.exists():
        log.info("--quick: loading existing MC results from %s", MC_CSV_PATH)
        df = pd.read_csv(MC_CSV_PATH)
        with open(MC_STATS_PATH) as f:
            stats = json.load(f)
        # Update seed in metadata to match what we're documenting
        stats["base_seed"] = args.seed
    else:
        log.info("Running %d-simulation Monte Carlo (seed=%d)…", args.n, args.seed)
        from src.simulation.monte_carlo import run_monte_carlo
        df, stats = run_monte_carlo(
            n_simulations=args.n,
            base_seed=args.seed,
            n_workers=args.workers,
            verbose=True,
            save=True,
        )

    stats["base_seed"] = args.seed   # ensure seed is documented

    # ── Step 2: Build MCResults ───────────────────────────────────────────────
    log.info("Building MCResults object…")
    results = _build_results(df, stats, alphas=[0.90, 0.95, 0.99])

    # ── Step 3: Variance analysis ─────────────────────────────────────────────
    emp_sizes = [] if args.no_empirical else [1_000, 5_000]

    if not args.no_empirical:
        log.info("Running variance analysis (analytical + empirical)…")
    else:
        log.info("Running analytical variance analysis only…")

    run_variance_analysis(
        results,
        n_sizes   = [1_000, 5_000, 10_000],
        n_reps    = args.n_reps,
        n_workers = args.workers,
        seed      = args.seed,
    ) if not args.no_empirical else _attach_analytical_only(results)

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    dest = save_results(results, path=args.output)
    log.info("Saved → %s", dest)

    # ── Step 5: Display ───────────────────────────────────────────────────────
    top_10 = [name for name, _ in results.top_n("winner", 10)]
    print_ci_table(results, teams=top_10)
    print_variance_report(results, top_n=10)

    # Demonstrate get_confidence_interval()
    print("  get_confidence_interval() demonstration:")
    print("  ─" * 36)
    demo_cases = [
        ("Spain",       "winner",       0.95),
        ("Spain",       "quarter_final",0.95),
        ("Argentina",   "winner",       0.95),
        ("England",     "winner",       0.99),
        ("Netherlands", "winner",       0.90),
    ]
    for team, stage, alpha in demo_cases:
        lo, est, hi = get_confidence_interval(team, stage, alpha, results)
        label = STAGE_LABELS[stage]
        print(f"  {team:<18} P({label:<15}) = "
              f"{est*100:5.1f}%   "
              f"{int(alpha*100)}% CI [{lo*100:.2f}%, {hi*100:.2f}%]")
    print()
    print(f"  ✅  Task 4.5 — Results Storage & Reproducibility complete.")
    print(f"  📁  mc_results.json → {dest}")


def _attach_analytical_only(results: MCResults) -> None:
    """Attach only the analytical variance (no extra sims)."""
    analytical = _analytical_variance(results, [1_000, 5_000, 10_000])
    results.variance_analysis = {"analytical": analytical, "empirical": None}


if __name__ == "__main__":
    main()
