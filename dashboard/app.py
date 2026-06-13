"""
app.py — FIFA World Cup 2026 Prediction Dashboard
===================================================
Multi-page Streamlit app.

Run:
    cd /path/to/fifa
    streamlit run dashboard/app.py

Pages
-----
    Home              — Overview, KPI cards, championship race, groups
    Match Predictor   — H2H win/draw/loss + scoreline distribution
    Group Standings   — Live group simulation + standings table
    Bracket Simulator — MC probability bracket + single tournament sim
    Team Profiles     — Squad, stats, WC history, confidence intervals
    Team Comparison   — Side-by-side team analytics
    Model Insights    — Calibration, variance, feature importance
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 — Prediction Engine",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "**FIFA World Cup 2026 Prediction System**\n\n"
            "Ensemble model (Poisson · XGBoost · LightGBM · Elo)  |  "
            "10,000 Monte Carlo simulations  |  Wilson score confidence intervals"
        ),
    },
)

from dashboard.utils import theme

theme.inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    "Home",
    "Match Predictor",
    "Group Standings",
    "Bracket Simulator",
    "Team Profiles",
    "Team Comparison",
    "Model Insights",
]

with st.sidebar:
    theme.sidebar_logo()

    page = st.radio(
        "Navigate",
        options=PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Model status + quick stats
    try:
        from dashboard.utils.data_loader import load_mc_probabilities, load_mc_results_json

        mc = load_mc_probabilities()
        top3 = mc.nlargest(3, "p_winner")[["team", "p_winner"]]
        st.markdown(
            "<div style='font-size:11px;font-weight:600;letter-spacing:.09em;"
            "text-transform:uppercase;color:#94A3B8;margin:16px 0 6px 4px'>"
            "Title favourites</div>",
            unsafe_allow_html=True,
        )
        theme.sidebar_favourites([
            {"pos": i + 1, "team": row["team"], "pct": f"{row['p_winner']*100:.1f}%"}
            for i, (_, row) in enumerate(top3.iterrows())
        ])
    except Exception:
        pass

    st.markdown("---")
    st.caption("Simulation-driven tournament intelligence")

# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────
from dashboard.pages import (
    home,
    match_predictor,
    group_standings,
    bracket_simulator,
    team_profiles,
    team_comparison,
    model_insights,
)

ROUTES = {
    "Home":              home.render,
    "Match Predictor":   match_predictor.render,
    "Group Standings":   group_standings.render,
    "Bracket Simulator": bracket_simulator.render,
    "Team Profiles":     team_profiles.render,
    "Team Comparison":   team_comparison.render,
    "Model Insights":    model_insights.render,
}

ROUTES.get(page, home.render)()
