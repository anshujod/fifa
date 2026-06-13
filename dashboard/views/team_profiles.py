"""
team_profiles.py — Deep-dive on any team: squad, stats, historical WC performance.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import (
    load_all_teams, load_mc_probabilities, load_elo_ratings,
    load_squad_profiles, load_team_snapshots, load_squad_json,
    load_historical_results, flag, flag_team,
)
from dashboard.utils.charts import (
    squad_position_pie, form_radar, probability_stage_bar, elo_history_placeholder,
)
from dashboard.utils import theme

# Stage column → label
STAGE_LABELS = {
    "group_qualify":  "Group Stage",
    "round_of_32":    "Round of 32",
    "round_of_16":    "Round of 16",
    "quarter_final":  "Quarter-Final",
    "semi_final":     "Semi-Final",
    "final":          "Final",
    "winner":         "Champion",
}


def _format_market_value(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v >= 1e9:
        return f"€{v/1e9:.2f}B"
    if v >= 1e6:
        return f"€{v/1e6:.0f}M"
    return f"€{v/1e3:.0f}K"


def _wc_history(hist_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Extract World Cup matches for a team from historical results."""
    wc = hist_df[
        (hist_df.get("tournament", pd.Series(dtype=str)).str.contains(
            "FIFA World Cup", case=False, na=False
        )) &
        (
            (hist_df.get("home_team", pd.Series(dtype=str)) == team) |
            (hist_df.get("away_team", pd.Series(dtype=str)) == team)
        )
    ].copy()

    if wc.empty:
        return wc

    # Drop rows with NaN scores before computing result
    wc = wc.dropna(subset=["home_score", "away_score"]).copy()
    if wc.empty:
        return wc

    # Use the stored result column ('H'=home win, 'A'=away win, 'D'=draw)
    # and re-map to team's perspective
    def row_result(r):
        stored = r.get("result", "D")   # H / A / D
        is_home = (r.get("home_team") == team)
        if stored == "D":
            return "D"
        if is_home:
            return "W" if stored == "H" else "L"
        else:
            return "W" if stored == "A" else "L"

    wc["Result"] = wc.apply(row_result, axis=1)
    wc["Opponent"] = wc.apply(
        lambda r: r.get("away_team") if r.get("home_team") == team
                  else r.get("home_team"), axis=1
    )
    wc["Score"] = wc.apply(
        lambda r: f"{int(r['home_score'])} – {int(r['away_score'])}",
        axis=1,
    )
    cols = [c for c in ["date", "tournament", "Opponent", "Score", "Result"]
            if c in wc.columns]
    return wc[cols].sort_values("date", ascending=False).head(20)


def render() -> None:
    theme.page_header(
        title="Team Profiles",
        subtitle="Deep dive into any of the 48 World Cup 2026 teams.",
    )

    all_teams    = load_all_teams()
    mc_df        = load_mc_probabilities()
    elo_df       = load_elo_ratings()
    squad_prof   = load_squad_profiles()
    snap_df      = load_team_snapshots()

    # ── Team selector ─────────────────────────────────────────────────────────
    team = st.selectbox(
        "Select a team",
        all_teams,
        index=all_teams.index("Spain") if "Spain" in all_teams else 0,
        format_func=flag_team,
        key="tp_team",
    )

    # Load all data for this team
    mc_row   = mc_df[mc_df["team"] == team]
    elo_row  = elo_df[elo_df["team"] == team]
    sq_prof  = squad_prof[squad_prof["team"] == team]
    snap     = snap_df[snap_df["team"] == team]
    squad    = load_squad_json(team)

    mc_probs = {}
    if not mc_row.empty:
        row = mc_row.iloc[0]
        mc_probs = {
            "group_qualify":  float(row["p_group_qualify"]),
            "round_of_32":    float(row["p_round_of_32"]),
            "round_of_16":    float(row["p_round_of_16"]),
            "quarter_final":  float(row["p_quarter_final"]),
            "semi_final":     float(row["p_semi_final"]),
            "final":          float(row["p_final"]),
            "winner":         float(row["p_winner"]),
        }

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("---")
    meta_bits = []
    if not mc_row.empty:
        meta_bits.append(f"Group {mc_row.iloc[0]['group']}")
        meta_bits.append(f"P(Champion) <b style='color:{theme.TEXT}'>{mc_probs['winner']*100:.1f}%</b>")
    if not elo_row.empty:
        elo_val  = int(elo_row.iloc[0]["elo_rating"])
        elo_rank = int(elo_row.iloc[0]["rank"]) if "rank" in elo_row.columns else "–"
        meta_bits.append(f"Elo <b style='color:{theme.TEXT}'>{elo_val}</b> · Rank #{elo_rank}")
    meta_html = " &nbsp;·&nbsp; ".join(meta_bits)
    st.markdown(
        f"""
<div style='display:flex;align-items:center;gap:18px;margin:6px 0 10px'>
  <span style='font-size:44px;line-height:1'>{flag(team)}</span>
  <div>
    <div style='font-size:28px;font-weight:700;letter-spacing:-.02em;color:{theme.TEXT}'>{team}</div>
    <div style='font-size:13.5px;color:{theme.TEXT_3};margin-top:4px'>{meta_html}</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Top metrics ───────────────────────────────────────────────────────────
    st.markdown("---")
    if not sq_prof.empty:
        sp = sq_prof.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Squad Size",     int(sp.get("squad_size", 0)))
        c2.metric("Avg Age",        f"{sp.get('avg_age', 0):.1f}")
        c3.metric("Avg Caps",       f"{sp.get('avg_caps', 0):.0f}")
        c4.metric("Market Value",   _format_market_value(sp.get("squad_market_value_eur")))
        c5.metric("Experience",     f"{sp.get('experience_score', 0):.0f}")

    if not snap.empty and not snap.iloc[0].empty:
        sn = snap.iloc[0]
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Form Score",       f"{sn.get('form_score', 0):.2f}")
        d2.metric("Goals/game (L10)", f"{sn.get('goals_scored_avg_10', 0):.1f}")
        d3.metric("Conceded/game",    f"{sn.get('goals_conceded_avg_10', 0):.1f}")
        d4.metric("Win % (L10)",      f"{sn.get('win_pct_last_10', 0):.0f}%")
        d5.metric("Clean Sheets",     f"{int(sn.get('clean_sheets_last_10', 0))}/10")

    # ── Charts row ────────────────────────────────────────────────────────────
    st.markdown("---")
    charts_col1, charts_col2, charts_col3 = st.columns(3)

    with charts_col1:
        if mc_probs:
            fig_probs = probability_stage_bar(team, mc_probs)
            st.plotly_chart(fig_probs, use_container_width=True)
        else:
            st.info("No MC probability data.")

    with charts_col2:
        if squad:
            fig_pie = squad_position_pie(squad, team)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No squad data available.")

    with charts_col3:
        if not snap.empty:
            fig_radar = form_radar(snap.iloc[0], team)
            st.plotly_chart(fig_radar, use_container_width=True)
        elif not elo_row.empty:
            fig_elo = elo_history_placeholder(team, elo_row.iloc[0])
            st.plotly_chart(fig_elo, use_container_width=True)

    # ── Squad roster ──────────────────────────────────────────────────────────
    theme.section(f"{team} — 2026 World Cup Squad")

    if squad:
        # Group by position
        pos_order = ["GK", "DF", "MF", "FW"]
        by_pos: dict[str, list] = {p: [] for p in pos_order}
        for player in squad:
            pos = player.get("position", "?")
            if pos in by_pos:
                by_pos[pos].append(player)

        tabs = st.tabs(["All"] + pos_order)
        all_rows = []
        for player in squad:
            all_rows.append({
                "Pos":   player.get("position", "?"),
                "Name":  player.get("name", "?"),
                "Club":  player.get("club", "?"),
                "Age":   player.get("age", "?"),
                "Caps":  player.get("caps", "?"),
                "Goals": player.get("goals", "?"),
                "Value": player.get("market_value", "?"),
            })

        with tabs[0]:
            st.dataframe(
                pd.DataFrame(all_rows),
                use_container_width=True,
                hide_index=True,
            )

        for i, pos in enumerate(pos_order):
            with tabs[i + 1]:
                pos_rows = [
                    {
                        "Name":  p.get("name", "?"),
                        "Club":  p.get("club", "?"),
                        "Age":   p.get("age", "?"),
                        "Caps":  p.get("caps", "?"),
                        "Goals": p.get("goals", "?"),
                        "Value": p.get("market_value", "?"),
                    }
                    for p in by_pos[pos]
                ]
                if pos_rows:
                    st.dataframe(
                        pd.DataFrame(pos_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(f"No {pos} players found in squad data.")

        # Squad highlights
        theme.section("Squad Highlights")
        if not sq_prof.empty:
            sp = sq_prof.iloc[0]
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                star_val = _format_market_value(sp.get("star_player_value_eur"))
                st.metric("Star Player Value", star_val)
            with hc2:
                youth = sp.get("youth_ratio", 0)
                st.metric("Youth Ratio (U23)", f"{youth:.1f}%")
            with hc3:
                vet = sp.get("veteran_ratio", 0)
                st.metric("Veteran Ratio (30+)", f"{vet:.1f}%")

            # Most capped / valuable players
            if squad:
                most_capped = max(squad, key=lambda p: p.get("caps", 0))
                st.caption(
                    f"Most capped: **{most_capped['name']}** — "
                    f"{most_capped.get('caps', 0)} caps, {most_capped.get('goals', 0)} goals"
                )
    else:
        st.warning(f"No squad data found for {team}.")

    # ── Historical WC performance ─────────────────────────────────────────────
    theme.section(f"{team} — World Cup History")

    try:
        hist_df = load_historical_results()
        wc_hist = _wc_history(hist_df, team)
        if not wc_hist.empty:
            st.caption(f"Last {len(wc_hist)} World Cup matches found in dataset")
            st.dataframe(
                wc_hist.rename(columns={"date": "Date", "tournament": "Tournament"}),
                use_container_width=True,
                hide_index=True,
            )

            # Win/Draw/Loss summary
            wdl = wc_hist["Result"].value_counts()
            ws1, ws2, ws3 = st.columns(3)
            ws1.metric("World Cup Wins",   int(wdl.get("W", 0)))
            ws2.metric("World Cup Draws",  int(wdl.get("D", 0)))
            ws3.metric("World Cup Losses", int(wdl.get("L", 0)))
        else:
            st.info(f"No World Cup match records found for {team}.")
    except Exception as e:
        st.warning(f"Could not load historical data: {e}")

    # ── Confidence intervals ──────────────────────────────────────────────────
    if mc_probs:
        theme.section(
            "Confidence Intervals",
            "Wilson score intervals at 95% confidence for each tournament stage.",
        )
        try:
            from src.simulation.results_store import load_results, get_confidence_interval
            results = load_results()
            ci_rows = []
            for stage, label in STAGE_LABELS.items():
                try:
                    lo, est, hi = get_confidence_interval(team, stage, 0.95, results)
                    ci_rows.append({
                        "Stage":    label,
                        "P":        f"{est*100:.1f}%",
                        "95% CI":   f"[{lo*100:.1f}%, {hi*100:.1f}%]",
                        "Width":    f"±{(hi-lo)/2*100:.2f}%",
                    })
                except Exception:
                    pass
            if ci_rows:
                st.dataframe(
                    pd.DataFrame(ci_rows),
                    use_container_width=True,
                    hide_index=True,
                )
        except Exception as e:
            st.caption(f"CI data not available: {e}")
