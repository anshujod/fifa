"""
data_loader.py — Cached data-loading helpers for the Streamlit dashboard.

All heavy I/O goes through @st.cache_data / @st.cache_resource so Streamlit
only loads each resource once per TTL window (default 3600 s = 1 h).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Flag emoji lookup (ISO 3166-1 alpha-2 based)
# ─────────────────────────────────────────────────────────────────────────────
_FLAG_MAP: dict[str, str] = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Canada": "🇨🇦", "Cape Verde": "🇨🇻", "Colombia": "🇨🇴", "Croatia": "🇭🇷",
    "Curaçao": "🇨🇼", "Czech Republic": "🇨🇿", "DR Congo": "🇨🇩",
    "Ecuador": "🇪🇨", "Egypt": "🇪🇬", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷",
    "Germany": "🇩🇪", "Ghana": "🇬🇭", "Haiti": "🇭🇹", "Iran": "🇮🇷",
    "Iraq": "🇮🇶", "Ivory Coast": "🇨🇮", "Japan": "🇯🇵", "Jordan": "🇯🇴",
    "Mexico": "🇲🇽", "Morocco": "🇲🇦", "Netherlands": "🇳🇱",
    "New Zealand": "🇳🇿", "Norway": "🇳🇴", "Panama": "🇵🇦",
    "Paraguay": "🇵🇾", "Portugal": "🇵🇹", "Qatar": "🇶🇦",
    "Saudi Arabia": "🇸🇦", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Senegal": "🇸🇳",
    "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Spain": "🇪🇸",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Tunisia": "🇹🇳", "Turkey": "🇹🇷",
    "United States": "🇺🇸", "Uruguay": "🇺🇾", "Uzbekistan": "🇺🇿",
}

# ── medal tiers (for Home page colouring) ────────────────────────────────────
TIER_COLOURS: dict[str, str] = {
    "gold":   "#FFD700",
    "silver": "#C0C0C0",
    "bronze": "#CD7F32",
    "blue":   "#4A90D9",
    "grey":   "#9E9E9E",
}


def flag(team: str) -> str:
    return _FLAG_MAP.get(team, "🌍")


def flag_team(team: str) -> str:
    """Return '🇪🇸 Spain'."""
    return f"{flag(team)} {team}"


# ─────────────────────────────────────────────────────────────────────────────
# Static data loaders (TTL=3600s)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_mc_probabilities() -> pd.DataFrame:
    """Monte Carlo probability table (48 rows × 17 cols)."""
    path = ROOT / "results" / "monte_carlo_probabilities.csv"
    df = pd.read_csv(path)
    df["flag_team"] = df["team"].map(flag_team)
    return df


@st.cache_data(ttl=3600)
def load_mc_results_json() -> dict:
    """Full mc_results.json (includes CIs and variance analysis)."""
    path = ROOT / "data" / "processed" / "mc_results.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600)
def load_elo_ratings() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "elo_ratings.csv"
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_squad_profiles() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "squad_profiles.csv"
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_team_snapshots() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "team_snapshots.csv"
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_squad_json(team: str) -> list[dict]:
    """Load raw squad JSON for one team (list of player dicts)."""
    slug = team.lower().replace(" ", "_").replace("ç", "ç")
    path = ROOT / "data" / "raw" / "squads_2026" / f"{slug}.json"
    # Try a few slugs for tricky names
    for candidate in [
        slug,
        slug.replace("ç", "c"),
        team.lower().replace(" ", "_"),
    ]:
        p = ROOT / "data" / "raw" / "squads_2026" / f"{candidate}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return []


@st.cache_data(ttl=3600)
def load_wc_groups() -> dict[str, list[str]]:
    from src.simulation.group_stage import WC2026_GROUPS
    return WC2026_GROUPS


@st.cache_data(ttl=3600)
def load_all_teams() -> list[str]:
    from src.simulation.monte_carlo import ALL_WC_TEAMS
    return sorted(ALL_WC_TEAMS)


@st.cache_data(ttl=3600)
def load_historical_results() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "match_results.csv"
    return pd.read_csv(path, parse_dates=["date"], low_memory=False)


# ─────────────────────────────────────────────────────────────────────────────
# Heavy model loader (resource cache — never expired)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_predictor():
    """Load MatchPredictor once per process (not per user session)."""
    from src.simulation.match_predictor import MatchPredictor
    mp = MatchPredictor.load()
    mp.fast_mode = True
    return mp


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helpers (uncached — called with user-chosen seeds)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_group_live(group_id: str, seed: int = 42) -> "GroupStandings":  # type: ignore[name-defined]
    from src.simulation.group_stage import simulate_group
    groups = load_wc_groups()
    predictor = load_predictor()
    rng = np.random.default_rng(seed)
    return simulate_group(group_id, groups[group_id], predictor, rng)


@st.cache_data(ttl=3600)
def load_feature_importance(n_features: int = 20) -> "pd.DataFrame | None":
    """
    Compute SHAP-based feature importance for the XGBoost outcome model.
    Returns DataFrame with columns [feature, importance, category, source].
    Falls back to native XGBoost gain-importance if SHAP is unavailable.
    Returns None on total failure.
    """
    import warnings
    warnings.filterwarnings("ignore")

    _CATS = {
        "elo":              "ELO/Ranking",
        "rank":             "ELO/Ranking",
        "expected_goals":   "Expected Goals",
        "h2h":              "Head-to-Head",
        "conf_":            "Confederation",
        "confederation":    "Confederation",
        "form":             "Form/Momentum",
        "win_pct":          "Form/Momentum",
        "decay":            "Form/Momentum",
        "goals_scored":     "Goals",
        "goals_conceded":   "Goals",
        "neutral":          "Match Context",
        "match_importance": "Match Context",
        "wc_experience":    "Match Context",
        "days_since":       "Match Context",
    }

    def _categorize(feat: str) -> str:
        fl = feat.lower()
        for kw, cat in _CATS.items():
            if kw in fl:
                return cat
        return "Other"

    try:
        import joblib as _jl
        import numpy as _np
        data  = _jl.load(ROOT / "models" / "saved" / "xgboost_outcome.joblib")
        model = data["model"]

        from src.features.encoder import FeaturePipeline
        fp = FeaturePipeline()
        X, _, _, _ = fp.prepare_training_data(start_year=2010, competitive_only=True)
        feature_names = list(X.columns)

        try:
            import shap as _shap
            X_sample  = X.tail(500)
            explainer = _shap.TreeExplainer(model)
            sv        = explainer.shap_values(X_sample.values)   # (N, F, 3)
            mean_abs  = _np.mean(_np.abs(sv), axis=(0, 2))       # (F,)
            source    = "SHAP"
        except Exception:
            mean_abs = model.feature_importances_
            source   = "Native"

        df = (
            pd.DataFrame({"feature": feature_names, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .head(n_features)
        )
        df["category"] = df["feature"].apply(_categorize)
        df["source"]   = source
        return df.reset_index(drop=True)

    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_h2h_stats(team1: str, team2: str) -> dict:
    """
    Head-to-head statistics between two teams from the full historical dataset.

    Returns
    -------
    dict with keys: total, team1_wins, draws, team2_wins,
                    team1_goals, team2_goals, team1_win_pct, matches
    """
    hist = load_historical_results()
    mask = (
        ((hist["home_team"] == team1) & (hist["away_team"] == team2)) |
        ((hist["home_team"] == team2) & (hist["away_team"] == team1))
    )
    h2h = hist[mask].dropna(subset=["home_score", "away_score"]).copy()

    t1_wins = draws = t2_wins = 0
    t1_goals = t2_goals = 0
    matches: list[dict] = []

    for _, r in h2h.iterrows():
        hg = int(r["home_score"])
        ag = int(r["away_score"])
        is_t1_home = (r["home_team"] == team1)
        g1, g2 = (hg, ag) if is_t1_home else (ag, hg)
        t1_goals += g1
        t2_goals += g2
        if g1 > g2:    t1_wins += 1
        elif g1 == g2: draws   += 1
        else:          t2_wins += 1
        matches.append({
            "date":       str(r.get("date", "")),
            "tournament": str(r.get("tournament", "")),
            "home":       r["home_team"],
            "away":       r["away_team"],
            "score":      f"{hg} – {ag}",
        })

    total = t1_wins + draws + t2_wins
    return {
        "total":         total,
        "team1_wins":    t1_wins,
        "draws":         draws,
        "team2_wins":    t2_wins,
        "team1_goals":   t1_goals,
        "team2_goals":   t2_goals,
        "team1_win_pct": t1_wins / total if total > 0 else 0.5,
        "matches":       sorted(matches, key=lambda x: x["date"], reverse=True)[:20],
    }


@st.cache_data(ttl=3600, show_spinner=False)
def predict_h2h(
    home: str,
    away: str,
    neutral: bool = True,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Return H2H prediction dict:
      p_home, p_draw, p_away, lam_home, lam_away,
      scoreline_counts, most_likely_score
    """
    predictor = load_predictor()
    p_home, p_draw, p_away = predictor.predict_proba(home, away, neutral=neutral)
    lam_h, lam_a = predictor.predict_lambdas(home, away, neutral=neutral)

    rng = np.random.default_rng(seed)
    scores: dict[tuple[int, int], int] = {}
    for _ in range(n_sims):
        h, a = predictor.simulate_scoreline(home, away, neutral=neutral, rng=rng)
        scores[(h, a)] = scores.get((h, a), 0) + 1

    most_likely = max(scores, key=scores.__getitem__)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "lam_home": lam_h,
        "lam_away": lam_a,
        "scoreline_counts": scores,
        "most_likely_score": most_likely,
        "n_sims": n_sims,
    }
