"""
group_standings.py — 📊 Simulate a group stage live and see standings.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import (
    load_wc_groups, simulate_group_live, load_mc_probabilities, flag,
)
from dashboard.utils.charts import group_standings_bar, goals_chart, plot_group_standings


def _standing_emoji(rank: int) -> str:
    return ["🥇", "🥈", "🥉", "❌"][min(rank, 3)]


def render() -> None:
    st.title("📊 Group Stage Simulator")
    st.caption(
        "Pick a group, set a random seed, and simulate the full round-robin. "
        "Run multiple seeds to see how standings fluctuate."
    )

    groups = load_wc_groups()
    mc_df  = load_mc_probabilities()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col, seed_col, btn_col = st.columns([2, 2, 1])
    with ctrl_col:
        group_id = st.selectbox(
            "Select Group", sorted(groups.keys()), key="gs_group"
        )
    with seed_col:
        seed = st.number_input(
            "Random Seed", min_value=0, max_value=999_999,
            value=42, step=1, key="gs_seed"
        )
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("▶ Simulate", type="primary", key="gs_run")

    # ── MC prior probabilities for this group ─────────────────────────────────
    group_teams = groups[group_id]
    mc_group = mc_df[mc_df["team"].isin(group_teams)].copy()

    st.markdown("---")
    st.subheader(f"Group {group_id} — MC Prior Probabilities")
    prior_cols = st.columns(len(group_teams))
    for i, (_, row) in enumerate(
        mc_group.sort_values("p_winner", ascending=False).iterrows()
    ):
        with prior_cols[i]:
            with st.container(border=True):
                st.markdown(f"#### {flag(row['team'])} {row['team']}")
                st.metric("P(Champion)", f"{row['p_winner']*100:.1f}%")
                st.metric("P(Qualify)",  f"{row['p_group_qualify']*100:.1f}%")
                st.metric("P(R32 Win)",  f"{row['p_round_of_32']*100:.1f}%")

    # ── Run simulation ────────────────────────────────────────────────────────
    # Only auto-run on first ever visit; after that require button click
    first_visit = "gs_last_result" not in st.session_state
    if run or first_visit:
        with st.spinner(f"Simulating Group {group_id}…"):
            standings = simulate_group_live(group_id, seed=int(seed))
        st.session_state["gs_last_result"] = standings
        st.session_state["gs_last_group"]  = group_id
        st.session_state["gs_last_seed"]   = int(seed)

    standings = st.session_state.get("gs_last_result")
    # Re-run automatically if user changes the group without clicking the button
    if standings is not None and st.session_state.get("gs_last_group") != group_id:
        with st.spinner(f"Simulating Group {group_id}…"):
            standings = simulate_group_live(group_id, seed=int(seed))
        st.session_state["gs_last_result"] = standings
        st.session_state["gs_last_group"]  = group_id
        st.session_state["gs_last_seed"]   = int(seed)

    if standings is None:
        st.info("Click **▶ Simulate** to run the group stage.")
        return

    st.markdown("---")
    st.subheader(
        f"Group {group_id} Simulation Results  (seed = {st.session_state['gs_last_seed']})"
    )

    ranking = standings.ranking  # list[str] — 1st to 4th
    records = standings.records

    # ── Plotly standings table (colour-coded qualification zones) ─────────────
    fig_table = plot_group_standings(records, ranking, group_id)
    st.plotly_chart(fig_table, use_container_width=True)

    # ── Qualifier callout ─────────────────────────────────────────────────────
    q1, q2, q3 = st.columns(3)
    with q1:
        st.success(f"🥇 **Winner:** {flag(ranking[0])} {ranking[0]}")
    with q2:
        st.success(f"🥈 **Runner-up:** {flag(ranking[1])} {ranking[1]}")
    with q3:
        st.warning(f"🔁 **3rd place:** {flag(ranking[2])} {ranking[2]} (may qualify)")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_bar = group_standings_bar(records, group_id)
        st.plotly_chart(fig_bar, use_container_width=True)
    with ch2:
        fig_goals = goals_chart(records, group_id)
        st.plotly_chart(fig_goals, use_container_width=True)

    # ── Match results grid ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏟️ Match Results")
    match_rows = []
    for mr in standings.results:
        winner = (
            mr.home if mr.home_goals > mr.away_goals else
            mr.away if mr.away_goals > mr.home_goals else
            "Draw"
        )
        match_rows.append({
            "Home":        f"{flag(mr.home)} {mr.home}",
            "Score":       f"**{mr.home_goals} – {mr.away_goals}**",
            "Away":        f"{flag(mr.away)} {mr.away}",
            "Result":      ("🟢 Home Win" if winner == mr.home else
                           "🟡 Draw"     if winner == "Draw" else
                           "🔴 Away Win"),
        })
    st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)

    # ── Multi-seed stability ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔄 Stability Check — Multiple Seeds")
    st.caption(
        "Run the group 10 times to see how often each team finishes in each position."
    )
    if st.button("Run 10-seed stability check", key="gs_stability"):
        pos_counts: dict[str, list[int]] = {t: [0, 0, 0, 0] for t in group_teams}
        with st.spinner("Running 10 simulations…"):
            for s in range(10):
                st_ = simulate_group_live(group_id, seed=s * 137 + 7)
                for rank, team in enumerate(st_.ranking):
                    pos_counts[team][rank] += 1

        stab_rows = [
            {
                "Team": f"{flag(t)} {t}",
                "1st": f"{pos_counts[t][0]*10}%",
                "2nd": f"{pos_counts[t][1]*10}%",
                "3rd": f"{pos_counts[t][2]*10}%",
                "4th": f"{pos_counts[t][3]*10}%",
                "P(Top 2)": f"{(pos_counts[t][0]+pos_counts[t][1])*10}%",
                "_first": pos_counts[t][0],
            }
            for t in group_teams
        ]
        stab_df = pd.DataFrame(stab_rows)
        stab_df = stab_df.sort_values("_first", ascending=False).drop(columns="_first")
        st.dataframe(stab_df, use_container_width=True, hide_index=True)
