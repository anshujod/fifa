"""
match_predictor.py — ⚔️ Pick any 2 teams → win/draw/loss % + expected scoreline.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import (
    load_all_teams, load_predictor, predict_h2h,
    load_elo_ratings, load_team_snapshots, flag, flag_team,
)
from dashboard.utils.charts import outcome_donut, scoreline_heatmap


def _form_badge(wins: int, draws: int, losses: int) -> str:
    """Return coloured badge string for last-5 form."""
    letters = ["W"] * wins + ["D"] * draws + ["L"] * losses
    coloured = []
    for lt in letters[:5]:
        if lt == "W":
            coloured.append(f":green[**W**]")
        elif lt == "D":
            coloured.append(f":orange[**D**]")
        else:
            coloured.append(f":red[**L**]")
    return "  ".join(coloured)


def render() -> None:
    st.title("⚔️ Match Predictor")
    st.caption("Select any two World Cup 2026 teams and get win probabilities, "
               "expected goals, and a simulated scoreline distribution.")

    all_teams = load_all_teams()
    elo_df    = load_elo_ratings()
    snap_df   = load_team_snapshots()

    # ── Team selection ────────────────────────────────────────────────────────
    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home = st.selectbox("🏠 Home Team", all_teams,
                            index=all_teams.index("Spain") if "Spain" in all_teams else 0,
                            key="mp_home")
    with col_vs:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### VS", unsafe_allow_html=False)
    with col2:
        away_default = "Germany" if "Germany" in all_teams else all_teams[1]
        away = st.selectbox("✈️ Away Team", all_teams,
                            index=all_teams.index(away_default),
                            key="mp_away")

    neutral = st.checkbox("Neutral venue", value=True, key="mp_neutral")
    n_sims  = st.select_slider(
        "Simulation samples for scoreline distribution",
        options=[1_000, 5_000, 10_000, 50_000],
        value=10_000, key="mp_nsims",
    )

    if home == away:
        st.warning("Please select two different teams.")
        return

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        with st.spinner("Running prediction…"):
            result = predict_h2h(home, away, neutral=neutral, n_sims=n_sims)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    p_home = result["p_home"]
    p_draw = result["p_draw"]
    p_away = result["p_away"]
    lam_h  = result["lam_home"]
    lam_a  = result["lam_away"]
    ml_h, ml_a = result["most_likely_score"]

    # ── Header banner ─────────────────────────────────────────────────────────
    st.markdown("---")
    bh, bc, ba = st.columns([2, 1, 2])
    with bh:
        st.markdown(f"## {flag(home)} {home}")
        elo_h = elo_df[elo_df["team"] == home]["elo_rating"].values
        if len(elo_h):
            st.markdown(f"**ELO:** {int(elo_h[0])}")
    with bc:
        st.markdown("<h2 style='text-align:center'>vs</h2>", unsafe_allow_html=True)
        venue = "Neutral" if neutral else "Home/Away"
        st.caption(f"_{venue} venue_")
    with ba:
        st.markdown(f"## {flag(away)} {away}")
        elo_a = elo_df[elo_df["team"] == away]["elo_rating"].values
        if len(elo_a):
            st.markdown(f"**ELO:** {int(elo_a[0])}")

    # ── Outcome probability metrics ───────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        delta = "Favourite" if p_home > p_away and p_home > p_draw else ""
        st.metric(f"🔵 {home} Win", f"{p_home*100:.1f}%", delta)
    with m2:
        st.metric("⬜ Draw", f"{p_draw*100:.1f}%")
    with m3:
        delta = "Favourite" if p_away > p_home and p_away > p_draw else ""
        st.metric(f"🔴 {away} Win", f"{p_away*100:.1f}%", delta)

    # ── Expected goals ────────────────────────────────────────────────────────
    eg1, eg2 = st.columns(2)
    with eg1:
        st.metric("⚽ Expected Goals", f"{lam_h:.2f}", f"{home}")
    with eg2:
        st.metric("⚽ Expected Goals", f"{lam_a:.2f}", f"{away}")

    # ── Most likely scoreline ─────────────────────────────────────────────────
    st.info(
        f"**Most likely scoreline:** {flag(home)} {home} **{ml_h} – {ml_a}** {away} {flag(away)}  "
        f"(probability: {result['scoreline_counts'][(ml_h,ml_a)]/n_sims*100:.1f}%)"
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_donut = outcome_donut(p_home, p_draw, p_away, home, away)
        st.plotly_chart(fig_donut, use_container_width=True)
    with ch2:
        fig_heat = scoreline_heatmap(result["scoreline_counts"], home, away, n_sims)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top 10 scorelines ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Top 10 Most Likely Scorelines")
    top_scores = sorted(
        result["scoreline_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    score_data = [
        {
            "Scoreline": f"{h} – {a}",
            "Label": f"{home} {h} – {a} {away}",
            "Count": cnt,
            "Probability": f"{cnt/n_sims*100:.2f}%",
        }
        for (h, a), cnt in top_scores
    ]
    st.dataframe(
        pd.DataFrame(score_data)[["Label", "Probability"]],
        use_container_width=True,
        hide_index=True,
    )

    # ── Recent form comparison ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Recent Form Comparison (Last 10 Matches)")
    snap_h = snap_df[snap_df["team"] == home]
    snap_a = snap_df[snap_df["team"] == away]

    if not snap_h.empty and not snap_a.empty:
        sh = snap_h.iloc[0]
        sa = snap_a.iloc[0]
        form_cols = st.columns(2)
        with form_cols[0]:
            with st.container(border=True):
                st.markdown(f"**{flag(home)} {home}**")
                st.markdown(_form_badge(
                    int(sh.get("wins_last_5",  0)),
                    int(sh.get("draws_last_5", 0)),
                    int(sh.get("losses_last_5",0)),
                ))
                st.caption("Last 5 results")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Goals/game", f"{sh.get('goals_scored_avg_10', 0):.1f}")
                fc2.metric("Conceded/game", f"{sh.get('goals_conceded_avg_10', 0):.1f}")
                fc3.metric("Clean sheets", f"{int(sh.get('clean_sheets_last_10', 0))}/10")
        with form_cols[1]:
            with st.container(border=True):
                st.markdown(f"**{flag(away)} {away}**")
                st.markdown(_form_badge(
                    int(sa.get("wins_last_5",  0)),
                    int(sa.get("draws_last_5", 0)),
                    int(sa.get("losses_last_5",0)),
                ))
                st.caption("Last 5 results")
                fc1, fc2, fc3 = st.columns(3)
                fc1.metric("Goals/game", f"{sa.get('goals_scored_avg_10', 0):.1f}")
                fc2.metric("Conceded/game", f"{sa.get('goals_conceded_avg_10', 0):.1f}")
                fc3.metric("Clean sheets", f"{int(sa.get('clean_sheets_last_10', 0))}/10")

    st.caption(
        f"Probabilities from ensemble model · {n_sims:,} scoreline simulations  |  "
        "Venue adjustment applied when not neutral"
    )
