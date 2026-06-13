"""
match_predictor.py — Pick any 2 teams → win/draw/loss % + expected scoreline.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import (
    load_all_teams, load_predictor, predict_h2h,
    load_elo_ratings, load_team_snapshots, flag, flag_team, flag_url,
)
from dashboard.utils.charts import scoreline_heatmap
from dashboard.utils import theme


def _form_letters(wins: int, draws: int, losses: int) -> str:
    """Last-5 form as compact coloured letters."""
    letters = ["W"] * wins + ["D"] * draws + ["L"] * losses
    colours = {"W": theme.SUCCESS, "D": theme.TEXT_3, "L": theme.ERROR}
    spans = "".join(
        f"<span style='color:{colours[lt]};font-weight:700;margin-right:9px'>{lt}</span>"
        for lt in letters[:5]
    )
    return f"<div style='font-size:15px;letter-spacing:.04em'>{spans}</div>"


def render() -> None:
    theme.page_header(
        eyebrow="Head-to-Head",
        title="Match Predictor",
        subtitle=(
            "Pick any two World Cup 2026 teams for win probabilities, expected goals, "
            "and the full simulated scoreline distribution."
        ),
    )

    all_teams = load_all_teams()
    elo_df    = load_elo_ratings()
    snap_df   = load_team_snapshots()

    # ── Team selection ────────────────────────────────────────────────────────
    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home = st.selectbox("Home team", all_teams,
                            index=all_teams.index("Spain") if "Spain" in all_teams else 0,
                            format_func=flag_team,
                            key="mp_home")
    with col_vs:
        st.markdown("<div class='mp-vs'>VS</div>", unsafe_allow_html=True)
    with col2:
        away_default = "Germany" if "Germany" in all_teams else all_teams[1]
        away = st.selectbox("Away team", all_teams,
                            index=all_teams.index(away_default),
                            format_func=flag_team,
                            key="mp_away")

    opt1, opt2 = st.columns([1, 2])
    with opt1:
        neutral = st.checkbox("Neutral venue", value=True, key="mp_neutral")
    with opt2:
        n_sims = st.select_slider(
            "Simulation samples",
            options=[1_000, 5_000, 10_000, 50_000],
            value=10_000, key="mp_nsims",
        )

    run = st.button("Run simulation", type="primary",
                    use_container_width=True, key="mp_run")

    if home == away:
        st.warning("Please select two different teams.")
        return

    # First visit renders with defaults; afterwards results refresh on demand.
    if run or "mp_result" not in st.session_state:
        try:
            with st.spinner(f"Simulating {n_sims:,} {home} – {away} matches…"):
                result = predict_h2h(home, away, neutral=neutral, n_sims=n_sims)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return
        st.session_state["mp_result"] = result
        st.session_state["mp_params"] = (home, away, neutral, n_sims)

    result = st.session_state["mp_result"]
    home, away, neutral, n_sims = st.session_state["mp_params"]

    p_home = result["p_home"]
    p_draw = result["p_draw"]
    p_away = result["p_away"]
    lam_h  = result["lam_home"]
    lam_a  = result["lam_away"]
    ml_h, ml_a = result["most_likely_score"]

    # ── Matchup banner ────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    elo_h = elo_df[elo_df["team"] == home]["elo_rating"].values
    elo_a = elo_df[elo_df["team"] == away]["elo_rating"].values
    venue = "Neutral venue" if neutral else "Home advantage"

    theme.versus_header(
        left={"flag_url": flag_url(home, 80), "name": home,
              "sub": f"Elo {int(elo_h[0])}" if len(elo_h) else ""},
        right={"flag_url": flag_url(away, 80), "name": away,
               "sub": f"Elo {int(elo_a[0])}" if len(elo_a) else ""},
        venue=venue,
    )

    # ── Outcome probabilities ─────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    theme.outcome_bar(p_home, p_draw, p_away, home, away)

    fav = home if p_home >= p_away else away
    theme.kpi_row([
        {"label": f"{home} win", "value": f"{p_home*100:.1f}%",
         "delta": "Favourite" if fav == home and p_home > p_draw else "",
         "delta_class": "accent"},
        {"label": "Draw", "value": f"{p_draw*100:.1f}%"},
        {"label": f"{away} win", "value": f"{p_away*100:.1f}%",
         "delta": "Favourite" if fav == away and p_away > p_draw else "",
         "delta_class": "accent"},
        {"label": "Most likely score",
         "value": f"{ml_h} – {ml_a}",
         "delta": f"{result['scoreline_counts'][(ml_h, ml_a)]/n_sims*100:.1f}% of simulations"},
    ])

    # ── Expected goals ────────────────────────────────────────────────────────
    theme.kpi_row([
        {"label": f"Expected goals — {home}", "value": f"{lam_h:.2f}"},
        {"label": f"Expected goals — {away}", "value": f"{lam_a:.2f}"},
    ])

    # ── Scoreline distribution ────────────────────────────────────────────────
    theme.section(
        "Scoreline Distribution",
        f"Poisson model · probability of each scoreline across {n_sims:,} simulated matches.",
    )
    ch1, ch2 = st.columns([6, 5], gap="large")
    with ch1:
        fig_heat = scoreline_heatmap(result["scoreline_counts"], home, away, n_sims)
        st.plotly_chart(fig_heat, use_container_width=True)
    with ch2:
        top_scores = sorted(
            result["scoreline_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        score_data = [
            {
                "Scoreline":   f"{home} {h} – {a} {away}",
                "Probability": f"{cnt/n_sims*100:.2f}%",
            }
            for (h, a), cnt in top_scores
        ]
        st.markdown(
            f"<div style='font-size:11px;font-weight:700;letter-spacing:.12em;"
            f"text-transform:uppercase;color:{theme.TEXT_3};margin:14px 0 10px'>"
            "Ten most likely scorelines</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(score_data),
            use_container_width=True,
            hide_index=True,
            height=390,
        )

    # ── Recent form comparison ────────────────────────────────────────────────
    theme.section("Recent Form", "Results and scoring rates over the last 10 matches.")
    snap_h = snap_df[snap_df["team"] == home]
    snap_a = snap_df[snap_df["team"] == away]

    if not snap_h.empty and not snap_a.empty:
        sh = snap_h.iloc[0]
        sa = snap_a.iloc[0]
        form_cols = st.columns(2, gap="large")
        for col, team_name, s in ((form_cols[0], home, sh), (form_cols[1], away, sa)):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{flag(team_name)} {team_name}**")
                    st.markdown(_form_letters(
                        int(s.get("wins_last_5",   0)),
                        int(s.get("draws_last_5",  0)),
                        int(s.get("losses_last_5", 0)),
                    ), unsafe_allow_html=True)
                    st.caption("Last 5 results")
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Goals/game", f"{s.get('goals_scored_avg_10', 0):.1f}")
                    fc2.metric("Conceded/game", f"{s.get('goals_conceded_avg_10', 0):.1f}")
                    fc3.metric("Clean sheets", f"{int(s.get('clean_sheets_last_10', 0))}/10")

    st.caption(
        f"Probabilities from ensemble model · {n_sims:,} scoreline simulations · "
        "Venue adjustment applied when not neutral"
    )
