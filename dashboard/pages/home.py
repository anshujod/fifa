"""
home.py — Home page: overview header, KPI cards, championship race, groups.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from dashboard.utils.data_loader import (
    load_mc_probabilities, load_elo_ratings,
    load_mc_results_json, flag,
)
from dashboard.utils.charts import tournament_funnel
from dashboard.utils import theme

# Stage column → short label
_STAGE_SHORT = {
    "p_group_qualify":  "Group",
    "p_round_of_32":    "R32",
    "p_round_of_16":    "R16",
    "p_quarter_final":  "QF",
    "p_semi_final":     "SF",
    "p_final":          "Final",
    "p_winner":         "Champion",
}


def render() -> None:
    df   = load_mc_probabilities()
    elo  = load_elo_ratings()
    meta = load_mc_results_json().get("metadata", {})

    n       = int(meta.get("n_simulations", 10_000))
    n_fail  = int(meta.get("n_failed", 0))
    elapsed = float(meta.get("elapsed_seconds", 0) or 0)
    sp      = n / elapsed if elapsed > 0 else 0

    # ── Header ───────────────────────────────────────────────────────────────
    theme.page_header(
        eyebrow="FIFA World Cup 2026 · USA · Canada · Mexico",
        title="Tournament Prediction Engine",
        subtitle=(
            "Championship probabilities from 10,000 full-tournament Monte Carlo "
            "simulations, powered by an ensemble of Poisson, XGBoost, LightGBM "
            "and Elo models."
        ),
        hero=True,
    )

    # ── KPI cards ────────────────────────────────────────────────────────────
    top      = df.iloc[0]
    top_elo  = elo.nlargest(1, "elo_rating").iloc[0]
    runner   = df.iloc[1]
    lead_gap = (top["p_winner"] - runner["p_winner"]) * 100

    theme.kpi_row([
        {
            "label": "Top favourite",
            "value": f"{flag(top['team'])} {top['team']}",
            "delta": f"{top['p_winner']*100:.1f}% champion · +{lead_gap:.1f} pts on {runner['team']}",
            "delta_class": "accent",
        },
        {
            "label": "Simulations run",
            "value": f"{n:,}",
            "delta": f"{sp:.0f} tournaments per second",
        },
        {
            "label": "Highest Elo",
            "value": f"{flag(top_elo['team'])} {int(top_elo['elo_rating'])}",
            "delta": f"{top_elo['team']} leads the rating pool",
        },
        {
            "label": "Model confidence",
            "value": f"{(n - n_fail)/n*100:.1f}%",
            "delta": f"{n - n_fail:,} clean runs · {n_fail} failed",
            "delta_class": "up",
        },
    ])

    # ── Championship race ────────────────────────────────────────────────────
    theme.section(
        "Championship Race",
        "Probability of winning the tournament — ranked by Monte Carlo estimate.",
    )

    col_left, col_right = st.columns([5, 6], gap="large")

    with col_left:
        n_teams = st.slider("Contenders shown", 5, 20, 10, key="home_n_teams")
        top_n = df.nlargest(n_teams, "p_winner").reset_index(drop=True)
        elo_map = dict(zip(elo["team"], elo["elo_rating"]))
        theme.ranking_cards([
            {
                "pos":   i + 1,
                "flag":  flag(row["team"]),
                "team":  row["team"],
                "prob":  float(row["p_winner"]),
                "elo":   int(elo_map.get(row["team"], 0)),
                "group": row["group"],
            }
            for i, row in top_n.iterrows()
        ])

    with col_right:
        st.markdown(
            "<div style='height:14px'></div><div style='font-size:11px;font-weight:600;"
            "letter-spacing:.09em;text-transform:uppercase;color:#94A3B8;"
            "margin-bottom:10px'>Full probability table — all 48 teams</div>",
            unsafe_allow_html=True,
        )
        stage_cols = list(_STAGE_SHORT.keys())
        display = df[["flag_team", "group"] + stage_cols].copy()
        display.columns = ["Team", "Grp"] + [_STAGE_SHORT[c] for c in stage_cols]

        st.dataframe(
            display,
            use_container_width=True,
            height=560,
            hide_index=True,
            column_config={
                "Team": st.column_config.TextColumn("Team", width="medium"),
                "Grp":  st.column_config.TextColumn("Grp", width="small"),
                **{
                    _STAGE_SHORT[c]: st.column_config.NumberColumn(
                        _STAGE_SHORT[c], format="percent", width="small",
                    )
                    for c in stage_cols[:-1]
                },
                "Champion": st.column_config.ProgressColumn(
                    "Champion", format="percent", min_value=0.0,
                    max_value=float(df["p_winner"].max()),
                ),
            },
        )

    # ── Tournament trajectories ──────────────────────────────────────────────
    theme.section(
        "Tournament Trajectories",
        "How each side's survival probability decays from the group stage to the final.",
    )
    all_teams = sorted(df["team"].tolist())
    defaults = df.nlargest(5, "p_winner")["team"].tolist()
    selected = st.multiselect(
        "Teams to compare",
        options=all_teams,
        default=defaults,
        key="home_compare_teams",
        label_visibility="collapsed",
    )
    if selected:
        fig2 = tournament_funnel(df, selected)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Groups at a glance ───────────────────────────────────────────────────
    theme.section(
        "Groups at a Glance",
        "All 12 groups with each team's championship probability.",
    )
    from src.simulation.group_stage import WC2026_GROUPS

    group_data: dict[str, list[dict]] = {}
    for gid in sorted(WC2026_GROUPS.keys()):
        teams_in_g = WC2026_GROUPS[gid]
        group_df = (
            df[df["team"].isin(teams_in_g)]
            .sort_values("p_winner", ascending=False)
        )
        group_data[gid] = [
            {"flag": flag(r["team"]), "team": r["team"], "prob": float(r["p_winner"])}
            for _, r in group_df.iterrows()
        ]
    theme.group_cards(group_data)

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin-top:32px;padding-top:14px;border-top:1px solid "
        f"rgba(148,163,184,.10);font-size:12px;color:#94A3B8'>"
        f"Ensemble of Poisson goal model · XGBoost · LightGBM · Elo &nbsp;·&nbsp; "
        f"N = {n:,} simulations &nbsp;·&nbsp; Seed = {meta.get('base_seed', 42)}</div>",
        unsafe_allow_html=True,
    )
