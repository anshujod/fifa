"""
bracket_simulator.py — Visual knockout bracket with MC probabilities on each path.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from dashboard.utils.data_loader import (
    load_mc_probabilities, load_wc_groups, load_predictor,
    flag, flag_team,
)
from dashboard.utils import theme


# ── MC stage columns ──────────────────────────────────────────────────────────
STAGE_COLS = {
    "R32":      "p_round_of_32",
    "R16":      "p_round_of_16",
    "QF":       "p_quarter_final",
    "SF":       "p_semi_final",
    "Final":    "p_final",
    "Champion": "p_winner",
}


def _prob_colour(p: float) -> str:
    """Blue intensity scaled to MC probability."""
    if   p >= 0.15: return "#93C5FD"
    elif p >= 0.08: return "#60A5FA"
    elif p >= 0.04: return "#3B82F6"
    elif p >= 0.02: return "#2563EB"
    else:           return "#334155"


def _make_bracket_figure(mc_df: pd.DataFrame, groups: dict) -> go.Figure:
    """
    Build a simplified visual bracket using Plotly scatter + lines.

    Layout: left half (Groups A–F, M49–M56) + right half (Groups G–L, M57–M64)
    Each column = one round: R32 | R16 | QF | SF | Final
    """
    # Top 2 from each group as R32 seeds (use MC rankings as proxy)
    group_order = sorted(groups.keys())

    def top_n_from_group(gid: str, n: int = 2) -> list[str]:
        teams = groups[gid]
        ranked = (
            mc_df[mc_df["team"].isin(teams)]
            .sort_values("p_winner", ascending=False)["team"]
            .tolist()
        )
        return ranked[:n]

    # Build R32 matchups (16 matches, 32 slots)
    # Left half: groups A–F; Right half: groups G–L
    left_groups  = group_order[:6]   # A B C D E F
    right_groups = group_order[6:]   # G H I J K L

    left_r32: list[tuple[str, str]] = []
    right_r32: list[tuple[str, str]] = []

    # Pair 1st of group[i] vs 2nd of group[i+3] (across halves)
    for i in range(3):
        lg = left_groups[i]
        rg = left_groups[i + 3]
        l1, l2 = top_n_from_group(lg)
        r1, r2 = top_n_from_group(rg)
        left_r32.append((l1, r2))   # 1st A vs 2nd D
        left_r32.append((r1, l2))   # 1st D vs 2nd A

    for i in range(3):
        lg = right_groups[i]
        rg = right_groups[i + 3]
        l1, l2 = top_n_from_group(lg)
        r1, r2 = top_n_from_group(rg)
        right_r32.append((l1, r2))
        right_r32.append((r1, l2))

    all_r32 = left_r32 + right_r32   # 12 matches (uses only group-stage qualifiers)
    # pad to 16 if needed (some groups provide 3rd-place slots — simplified here)
    n_rounds = 5
    n_matches_r32 = len(all_r32)

    fig = go.Figure()

    # Column x-positions
    round_x = {0: 0.05, 1: 0.22, 2: 0.38, 3: 0.55, 4: 0.72, 5: 0.88}
    col_labels = ["R32", "R16", "QF", "SF", "Final", "Champion"]

    # Draw column headers
    for col, label in enumerate(col_labels):
        fig.add_annotation(
            x=round_x[col], y=1.03, xref="paper", yref="paper",
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=12, color="#94A3B8"),
            xanchor="center",
        )

    # --- simplified bracket: show top MC teams per round ---
    rounds_teams = {}
    stage_col_map = [
        "p_round_of_32", "p_round_of_16", "p_quarter_final",
        "p_semi_final", "p_final", "p_winner",
    ]
    for col_idx, stage in enumerate(stage_col_map):
        n_spots = [32, 16, 8, 4, 2, 1][col_idx]
        top = mc_df.nlargest(n_spots, stage)
        rounds_teams[col_idx] = top["team"].tolist()

    # Draw team cards for each round
    for col_idx, teams in rounds_teams.items():
        n = len(teams)
        for row_i, team in enumerate(teams):
            y = 1 - (row_i + 0.5) / n
            stage = stage_col_map[col_idx]
            p = float(mc_df[mc_df["team"] == team][stage].values[0])
            colour = _prob_colour(p)

            # Team card (rectangle annotation)
            x = round_x[col_idx]
            label = f"{flag(team)} {team}  {p*100:.0f}%"
            if n > 8:
                label = f"{flag(team)} {team[:3].upper()}  {p*100:.0f}%"

            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=colour,
                    symbol="square",
                    line=dict(color="rgba(248,250,252,.25)", width=1),
                ),
                text=[label],
                textposition="middle right" if col_idx < 3 else "middle left",
                textfont=dict(size=9 if n > 8 else 11, color="#F8FAFC"),
                hovertemplate=f"<b>{team}</b><br>P({col_labels[col_idx]}): {p*100:.1f}%<extra></extra>",
                showlegend=False,
            ))

    # Highlight champion
    champ = mc_df.nlargest(1, "p_winner").iloc[0]
    fig.add_annotation(
        x=round_x[5], y=0.5,
        text=f"{flag(champ['team'])} <b>{champ['team']}</b>",
        showarrow=False,
        font=dict(size=15, color="#60A5FA"),
        xanchor="center",
        yshift=20,
    )

    fig.update_layout(
        title=dict(text="WC 2026 Bracket — MC Top Teams by Round", font_size=15),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        height=700,
        xaxis=dict(visible=False, range=[-0.05, 1.1]),
        yaxis=dict(visible=False, range=[-0.02, 1.1]),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def _run_single_tournament(seed: int) -> dict:
    """Simulate one full tournament and return round-by-round winners."""
    from src.simulation.group_stage import simulate_all_groups
    from src.simulation.third_place import build_bracket
    from src.simulation.knockout import simulate_knockout

    predictor = load_predictor()
    groups    = load_wc_groups()
    rng       = np.random.default_rng(seed)

    # Group stage  (signature: predictor, groups, rng)
    group_results = simulate_all_groups(predictor, groups, rng)

    # Build R32 bracket  (returns tuple: bracket, best8_thirds, eliminated)
    bracket, _best8, _elim = build_bracket(group_results, rng)

    # Knockout
    tournament = simulate_knockout(bracket, predictor, rng)

    return {
        "r32_winners":  [r.winner for r in tournament.r32_results],
        "r16_winners":  [r.winner for r in tournament.r16_results],
        "qf_winners":   [r.winner for r in tournament.qf_results],
        "sf_winners":   [r.winner for r in tournament.sf_results],
        "finalist_a":   tournament.final_result.home,
        "finalist_b":   tournament.final_result.away,
        "champion":     tournament.champion,
        "runner_up":    tournament.runner_up,
    }


def render() -> None:
    theme.page_header(
        title="Bracket Simulator",
        subtitle=(
            "Visualise the knockout bracket using Monte Carlo probabilities, "
            "or run a full single-seed tournament simulation."
        ),
    )

    mc_df  = load_mc_probabilities()
    groups = load_wc_groups()

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab_mc, tab_sim = st.tabs(["Probability Bracket", "Single Tournament"])

    # ────────────── Tab 1: MC bracket ────────────────────────────────────────
    with tab_mc:
        theme.section(
            "Most Likely Bracket Paths",
            "Each column shows the teams most likely to reach that round, "
            "shaded by championship probability.",
        )

        fig = _make_bracket_figure(mc_df, groups)
        st.plotly_chart(fig, use_container_width=True)

        # ── Stage probability table ───────────────────────────────────────────
        theme.section("Stage Probability Breakdown")
        stage_cols = list(STAGE_COLS.values())
        stage_labels = list(STAGE_COLS.keys())

        disp = mc_df[["flag_team", "group"] + stage_cols].copy()
        disp.columns = ["Team", "Grp"] + stage_labels
        for lbl in stage_labels:
            disp[lbl] = disp[lbl].map(lambda x: f"{x*100:.1f}%")

        group_filter = st.multiselect(
            "Filter by group:", sorted(groups.keys()), default=[],
            key="bs_group_filter"
        )
        if group_filter:
            disp = disp[disp["Grp"].isin(group_filter)]

        st.dataframe(disp, use_container_width=True, hide_index=True, height=500)

    # ────────────── Tab 2: Single simulation ─────────────────────────────────
    with tab_sim:
        theme.section(
            "Single Full-Tournament Simulation",
            "One complete tournament run — group stage through the final.",
        )

        col_seed, col_btn = st.columns([2, 1])
        with col_seed:
            sim_seed = st.number_input(
                "Simulation seed", min_value=0, max_value=999_999,
                value=2026, step=1, key="bs_seed"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_sim = st.button("Run tournament", type="primary", key="bs_run",
                                use_container_width=True)

        if run_sim:
            with st.spinner("Simulating full tournament (R32 → Final)…"):
                try:
                    result = _run_single_tournament(int(sim_seed))
                    st.session_state["bs_sim_result"] = result
                    st.session_state["bs_sim_seed"]   = sim_seed
                except Exception as e:
                    st.error(f"Simulation error: {e}")
                    return

        sim = st.session_state.get("bs_sim_result")
        if sim is None:
            st.info("Run the tournament to simulate a full bracket.")
            return

        # ── Champion callout ──────────────────────────────────────────────────
        champ    = sim["champion"]
        runner   = sim["runner_up"]
        champ_p  = float(mc_df[mc_df["team"] == champ]["p_winner"].values[0]) * 100
        runner_p = float(mc_df[mc_df["team"] == runner]["p_winner"].values[0]) * 100

        st.markdown("---")
        theme.kpi_row([
            {"label": "World champion",
             "value": f"{flag(champ)} {champ}",
             "delta": f"Pre-tournament P(Champion) {champ_p:.1f}%",
             "delta_class": "accent"},
            {"label": "Runner-up",
             "value": f"{flag(runner)} {runner}",
             "delta": f"Pre-tournament P(Champion) {runner_p:.1f}%"},
            {"label": "Final",
             "value": f"{champ} d. {runner}",
             "delta": f"Seed {st.session_state.get('bs_sim_seed', sim_seed)}"},
        ])

        # ── Round-by-round bracket ────────────────────────────────────────────
        theme.section("Round-by-Round Results")
        round_map = [
            ("Round of 32 (R32)", sim.get("r32_winners",  [])),
            ("Round of 16 (R16)", sim.get("r16_winners",  [])),
            ("Quarter-Finals",    sim.get("qf_winners",   [])),
            ("Semi-Finals",       sim.get("sf_winners",   [])),
            ("Finalists",         [sim.get("finalist_a"), sim.get("finalist_b")]),
        ]
        for round_name, winners in round_map:
            if not winners:
                continue
            with st.expander(f"{round_name} — {len(winners)} teams", expanded=True):
                w_cols = st.columns(min(len(winners), 8))
                for i, team in enumerate(winners):
                    if team is None:
                        continue
                    p = float(mc_df[mc_df["team"] == team]["p_winner"].values[0]) * 100
                    with w_cols[i % len(w_cols)]:
                        with st.container(border=True):
                            st.markdown(f"**{flag(team)} {team}**")
                            st.caption(f"MC {p:.1f}%")

    st.markdown("---")
    st.caption(
        "Bracket built from WC 2026 group assignments · "
        "Knockout uses Poisson model + Elo confidence · "
        "Penalties: 75% base conversion rate"
    )
