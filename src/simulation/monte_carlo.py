from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.simulation.group_stage import WC2026_GROUPS, simulate_all_groups
from src.simulation.knockout import simulate_knockout
from src.simulation.match_predictor import MatchPredictor
from src.simulation.third_place import build_bracket

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results"

# Flat list of all 48 WC 2026 teams in group order
ALL_WC_TEAMS: list[str] = [
    team for teams in WC2026_GROUPS.values() for team in teams
]
# Team → group letter
TEAM_GROUP: dict[str, str] = {
    team: grp
    for grp, teams in WC2026_GROUPS.items()
    for team in teams
}

# ─────────────────────────────────────────────────────────────────────────────
# Worker-process globals
# ─────────────────────────────────────────────────────────────────────────────

_worker_predictor: Optional[MatchPredictor] = None


def _init_worker(log_level: int = logging.WARNING) -> None:
    """
    Called once per worker process.
    Loads the MatchPredictor (all models) into process memory.
    Sets fast_mode=True for ~8× faster scoreline sampling.
    """
    global _worker_predictor
    logging.basicConfig(level=log_level,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")
    predictor = MatchPredictor.load()
    predictor.fast_mode = True    # use direct Poisson sampling for speed
    _worker_predictor = predictor


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation
# ─────────────────────────────────────────────────────────────────────────────

def _run_one(seed: int) -> dict | None:
    """
    Run one complete tournament simulation.

    Returns a minimal dict with frozensets of team names at each advancement
    stage.  Returns None if the simulation fails (logged as warning).

    The result dict keys:
        qualified   — teams that advance from the group stage (32 teams)
        r32_win     — teams that win their R32 match          (16 teams)
        r16_win     — teams that win their R16 match          ( 8 teams)
        qf_win      — teams that win their QF match           ( 4 teams)
        sf_win      — teams that win their SF match           ( 2 teams)
        champion    — the tournament winner                   ( 1 team )
    """
    global _worker_predictor
    rng = np.random.default_rng(seed)

    try:
        # ── Group stage ───────────────────────────────────────────────────────
        standings = simulate_all_groups(_worker_predictor, rng=rng)
        bracket, best_8, _ = build_bracket(standings, rng=rng)

        # Teams that qualify from the group stage (top 2 + best 8 thirds)
        qualified: set[str] = set()
        for gs in standings.values():
            qualified.add(gs.ranking[0])
            qualified.add(gs.ranking[1])
        for t in best_8:
            qualified.add(t.team)

        # ── Knockout stage ────────────────────────────────────────────────────
        result = simulate_knockout(bracket, _worker_predictor, rng=rng)

        return {
            "qualified": frozenset(qualified),
            "r32_win":   frozenset(r.winner for r in result.r32_results),
            "r16_win":   frozenset(r.winner for r in result.r16_results),
            "qf_win":    frozenset(r.winner for r in result.qf_results),
            "sf_win":    frozenset(r.winner for r in result.sf_results),
            "champion":  result.champion,
        }

    except Exception as exc:
        log.warning("Simulation seed=%d failed: %s", seed, exc, exc_info=False)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _ci_95(p: float, n: int) -> float:
    """Half-width of the 95% Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0
    z = 1.96
    # Normal approximation: ±z*sqrt(p(1-p)/n), clipped to [0, 1]
    return z * (p * (1 - p) / n) ** 0.5


def _aggregate(
    runs: list[dict | None],
    all_teams: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Aggregate successful simulation runs into per-team probability estimates.

    Returns (DataFrame, n_successful).
    """
    # Count successes
    valid  = [r for r in runs if r is not None]
    n      = len(valid)
    if n == 0:
        raise RuntimeError("All simulations failed — no results to aggregate.")

    # Per-team counters
    counts: dict[str, dict[str, int]] = {
        team: {k: 0 for k in
               ("qualified", "r32_win", "r16_win", "qf_win", "sf_win", "champion")}
        for team in all_teams
    }

    for run in valid:
        for team in all_teams:
            c = counts[team]
            if team in run["qualified"]:  c["qualified"] += 1
            if team in run["r32_win"]:    c["r32_win"]   += 1
            if team in run["r16_win"]:    c["r16_win"]   += 1
            if team in run["qf_win"]:     c["qf_win"]    += 1
            if team in run["sf_win"]:     c["sf_win"]    += 1
            if team == run["champion"]:   c["champion"]  += 1

    rows = []
    for team in all_teams:
        c  = counts[team]
        pq = c["qualified"] / n
        pr = c["r32_win"]   / n
        pq2 = c["r16_win"]  / n
        psf = c["qf_win"]   / n
        pfi = c["sf_win"]   / n
        pw  = c["champion"] / n

        rows.append({
            "team":             team,
            "group":            TEAM_GROUP.get(team, "?"),
            "p_group_qualify":  round(pq,  4),
            "p_round_of_32":    round(pr,  4),   # probability of winning R32 match
            "p_round_of_16":    round(pr,  4),   # = p_round_of_32 (reaching R16 = winning R32)
            "p_quarter_final":  round(pq2, 4),
            "p_semi_final":     round(psf, 4),
            "p_final":          round(pfi, 4),
            "p_winner":         round(pw,  4),
            # 95 % Wilson confidence intervals (half-width)
            "ci95_group_qualify": round(_ci_95(pq,  n), 4),
            "ci95_round_of_32":   round(_ci_95(pr,  n), 4),
            "ci95_quarter_final": round(_ci_95(pq2, n), 4),
            "ci95_semi_final":    round(_ci_95(psf, n), 4),
            "ci95_final":         round(_ci_95(pfi, n), 4),
            "ci95_winner":        round(_ci_95(pw,  n), 4),
            "n_simulations":      n,
        })

    df = (
        pd.DataFrame(rows)
          .sort_values("p_winner", ascending=False)
          .reset_index(drop=True)
    )
    df.index += 1   # 1-based rank
    return df, n


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    n_simulations: int = 10_000,
    n_workers:     int | None = None,
    base_seed:     int = 2026,
    chunk_size:    int = 50,
    verbose:       bool = True,
    save:          bool = True,
    output_dir:    Path | None = None,
    return_raw:    bool = False,
) -> tuple[pd.DataFrame, dict] | tuple[pd.DataFrame, dict, list]:
    """
    Run N parallel tournament simulations and return aggregated probabilities.

    Parameters
    ----------
    n_simulations : int   (default 10 000)
    n_workers     : int   workers for Pool; None = os.cpu_count()
    base_seed     : int   seeds = base_seed … base_seed + n_simulations − 1
    chunk_size    : int   tasks per IPC batch (default 50)
    verbose       : bool  print progress and final table
    save          : bool  write CSV + JSON to results/
    output_dir    : Path  override output directory
    return_raw    : bool  if True, also return the list of raw per-run dicts
                          (used by results_store for variance analysis)

    Returns
    -------
    (df, stats)          when return_raw=False (default)
    (df, stats, raw)     when return_raw=True
        df    : DataFrame sorted by p_winner (descending)
        stats : dict with n_simulations, n_workers, elapsed_s, sim_per_s, …
        raw   : list[dict | None] — one dict per simulation run
    """
    n_workers = n_workers or mp.cpu_count()
    n_workers = max(1, min(n_workers, n_simulations))
    seeds     = list(range(base_seed, base_seed + n_simulations))

    if verbose:
        log.info("=" * 60)
        log.info("Monte Carlo — %d simulations on %d workers", n_simulations, n_workers)
        log.info("=" * 60)

    t_start = time.perf_counter()

    # ── Run simulations ───────────────────────────────────────────────────────
    if n_workers == 1:
        # Sequential path — useful for debugging / notebooks
        if verbose:
            log.info("Sequential mode (n_workers=1)")
        _init_worker(logging.WARNING)
        all_runs: list[dict | None] = []
        for i, seed in enumerate(seeds, 1):
            all_runs.append(_run_one(seed))
            if verbose and i % max(1, n_simulations // 10) == 0:
                pct = 100 * i / n_simulations
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed
                log.info("  %5d / %d  (%.0f%%)  %.1f sim/s",
                         i, n_simulations, pct, rate)
    else:
        # Parallel path using multiprocessing.Pool
        ctx = mp.get_context("spawn")    # consistent on macOS + Linux
        all_runs = []
        completed = 0

        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(logging.WARNING,),
        ) as pool:
            for result in pool.imap_unordered(
                _run_one, seeds, chunksize=chunk_size
            ):
                all_runs.append(result)
                completed += 1
                if verbose and completed % max(1, n_simulations // 10) == 0:
                    elapsed = time.perf_counter() - t_start
                    rate    = completed / elapsed
                    eta     = (n_simulations - completed) / rate
                    log.info("  %5d / %d  (%.0f%%)  %.1f sim/s  ETA %.0fs",
                             completed, n_simulations,
                             100 * completed / n_simulations, rate, eta)

    elapsed   = time.perf_counter() - t_start
    n_ok      = sum(1 for r in all_runs if r is not None)
    n_failed  = n_simulations - n_ok
    sim_per_s = n_ok / elapsed if elapsed > 0 else 0.0

    if verbose:
        log.info("Completed %d/%d in %.1fs  (%.1f sim/s)%s",
                 n_ok, n_simulations, elapsed, sim_per_s,
                 f"  [{n_failed} failed]" if n_failed else "")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    df, n_used = _aggregate(all_runs, ALL_WC_TEAMS)

    stats: dict = {
        "n_simulations":    n_simulations,
        "n_successful":     n_ok,
        "n_failed":         n_failed,
        "n_workers":        n_workers,
        "base_seed":        base_seed,
        "elapsed_seconds":  round(elapsed, 2),
        "sim_per_second":   round(sim_per_s, 1),
        "fast_mode":        True,
        "champion_top5": df.head(5)[["team", "p_winner"]].to_dict(orient="records"),
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    if save:
        out_dir = Path(output_dir) if output_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path  = out_dir / "monte_carlo_probabilities.csv"
        json_path = out_dir / "monte_carlo_stats.json"
        df.to_csv(csv_path, index_label="rank")
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        if verbose:
            log.info("Saved probabilities → %s", csv_path)
            log.info("Saved stats         → %s", json_path)

    if return_raw:
        return df, stats, all_runs
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

_PCT = lambda p: f"{p * 100:5.1f}%"

def print_results_table(df: pd.DataFrame, top_n: int = 48) -> None:
    """Print a formatted probability table to stdout."""
    W = 112
    sep = "─" * W
    print(f"\n{'═' * W}")
    print("  FIFA WORLD CUP 2026 — MONTE CARLO WIN PROBABILITIES")
    print(f"  ({df['n_simulations'].iloc[0]:,} simulations)")
    print(f"{'═' * W}")
    header = (
        f"  {'#':>3}  {'Team':<26} {'Grp':>3}  "
        f"{'Qualify':>8}  {'R32 Win':>8}  {'QF':>8}  "
        f"{'SF':>8}  {'Final':>8}  {'Champion':>9}  {'CI±':>7}"
    )
    print(header)
    print(f"  {sep}")

    # Colour zones
    MEDAL   = "🥇🥈🥉"
    TROPHY  = "🏆"

    for rank, row in df.head(top_n).iterrows():
        medal = TROPHY if rank == 1 else (MEDAL[rank - 2] if rank <= 4 else "  ")
        print(
            f"  {rank:>3}  {row['team']:<26} {row['group']:>3}  "
            f"{_PCT(row['p_group_qualify']):>8}  "
            f"{_PCT(row['p_round_of_32']):>8}  "
            f"{_PCT(row['p_quarter_final']):>8}  "
            f"{_PCT(row['p_semi_final']):>8}  "
            f"{_PCT(row['p_final']):>8}  "
            f"{_PCT(row['p_winner']):>9}  "
            f"±{row['ci95_winner']*100:4.2f}%  {medal}"
        )

    print(f"  {sep}")
    print()


def print_group_summary(df: pd.DataFrame) -> None:
    """Print per-group qualification probabilities."""
    print(f"\n{'─' * 80}")
    print("  Group-stage qualification probabilities")
    print(f"{'─' * 80}")
    print(f"  {'Grp':>3}  {'Team':<28}  {'Qualify':>8}  {'R32 Win':>8}  {'QF':>8}")
    print(f"  {'─' * 70}")
    for grp in sorted(WC2026_GROUPS.keys()):
        grp_df = df[df["group"] == grp].sort_values("p_group_qualify", ascending=False)
        for _, row in grp_df.iterrows():
            print(f"  {row['group']:>3}  {row['team']:<28}  "
                  f"{_PCT(row['p_group_qualify']):>8}  "
                  f"{_PCT(row['p_round_of_32']):>8}  "
                  f"{_PCT(row['p_quarter_final']):>8}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Monte Carlo World Cup 2026 Simulator"
    )
    p.add_argument("--n",       type=int, default=10_000, help="Number of simulations")
    p.add_argument("--workers", type=int, default=None,   help="Worker processes (default: CPU count)")
    p.add_argument("--seed",    type=int, default=2026,   help="Base RNG seed")
    p.add_argument("--chunk",   type=int, default=50,     help="IPC chunk size")
    p.add_argument("--no-save", action="store_true",      help="Skip saving results")
    p.add_argument("--top",     type=int, default=16,     help="Teams to show in table")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    log.info("=" * 60)
    log.info("Monte Carlo Tournament Runner")
    log.info("=" * 60)

    df, stats = run_monte_carlo(
        n_simulations = args.n,
        n_workers     = args.workers,
        base_seed     = args.seed,
        chunk_size    = args.chunk,
        verbose       = True,
        save          = not args.no_save,
    )

    # Print tables
    print_results_table(df, top_n=args.top)
    print_group_summary(df)

    # Summary stats
    print(f"  Run statistics:")
    print(f"    Simulations  : {stats['n_simulations']:,}")
    print(f"    Successful   : {stats['n_successful']:,}")
    print(f"    Workers      : {stats['n_workers']}")
    print(f"    Elapsed      : {stats['elapsed_seconds']:.1f}s")
    print(f"    Speed        : {stats['sim_per_second']:.1f} sim/s")
    print()
    print("Monte Carlo complete.")


if __name__ == "__main__":
    main()
