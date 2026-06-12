"""
team_comparison.py — Side-by-side comparison of any two WC 2026 teams.

Sections
--------
1. Key Metrics table   — numeric stats side by side, winner highlighted
2. Radar Chart         — 8 dimensions normalised across all 48 teams
3. Tournament Probs    — overlapping funnel for both teams
4. Squad Viewer        — tabbed roster for each team, key-player callouts
5. Head-to-Head        — all-time H2H record + recent matches table
6. WC History Timeline — scatter plot of World Cup matches for each team
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.utils.data_loader import (
    load_all_teams, load_elo_ratings, load_mc_probabilities,
    load_squad_profiles, load_team_snapshots, load_squad_json,
    load_historical_results, get_h2h_stats, flag, flag_team,
)
from dashboard.utils.charts import (
    comparison_radar, tournament_funnel, wc_history_timeline,
)
from dashboard.utils import theme

# ── Stage labels ──────────────────────────────────────────────────────────────
STAGE_LABELS = {
    "group_qualify": "Group Stage", "round_of_32": "Round of 32",
    "round_of_16": "Round of 16",   "quarter_final": "Quarter-Final",
    "semi_final": "Semi-Final",     "final": "Final",
    "winner": "Champion",
}
STAGE_KEYS = list(STAGE_LABELS.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Helper: normalise a value into [0, 1] given population range
# ─────────────────────────────────────────────────────────────────────────────
def _norm(val: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.5
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _safe(series_or_none, col: str, default: float = 0.0) -> float:
    """Extract a scalar from a pandas Series gracefully."""
    if series_or_none is None:
        return default
    try:
        v = series_or_none[col]
        return float(v) if not pd.isna(v) else default
    except (KeyError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Build normalised radar values for one team
# ─────────────────────────────────────────────────────────────────────────────
def _radar_vals(
    snap:         "pd.Series | None",
    elo_row:      "pd.Series | None",
    sq_row:       "pd.Series | None",
    h2h_win_pct:  float,
    ranges:       dict,
) -> list[float]:
    """
    Return 8 normalised [0,1] values for the radar dimensions:
    Attack, Defence, Form, ELO, FIFA Rank, Squad Value, Experience, H2H
    """
    attack     = _norm(_safe(snap,   "goals_scored_avg_10"),    ranges["min_gs"], ranges["max_gs"])
    defence    = _norm(
        ranges["max_gc"] - _safe(snap, "goals_conceded_avg_10", ranges["max_gc"]),
        0, ranges["max_gc"] - ranges["min_gc"],
    )
    form       = _norm(_safe(snap,   "win_pct_last_10"),         0,                100)
    elo        = _norm(_safe(elo_row, "elo_rating",    1500),    ranges["min_elo"], ranges["max_elo"])
    # FIFA rank: lower rank # = better → invert
    rank_val   = _safe(elo_row, "rank", ranges["max_rank"])
    fifa_rank  = _norm(ranges["max_rank"] - rank_val, 0, ranges["max_rank"] - ranges["min_rank"])
    # Squad value: log-scale
    mv_raw     = _safe(sq_row,  "squad_market_value_eur", 0)
    squad_val  = _norm(np.log1p(mv_raw), 0, ranges["max_log_mv"])
    experience = _norm(_safe(sq_row,  "experience_score"), 0, ranges["max_exp"])
    h2h        = float(np.clip(h2h_win_pct, 0, 1))

    return [attack, defence, form, elo, fifa_rank, squad_val, experience, h2h]


# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute population ranges for normalisation
# ─────────────────────────────────────────────────────────────────────────────
def _population_ranges(
    snaps: pd.DataFrame,
    elos:  pd.DataFrame,
    squads: pd.DataFrame,
) -> dict:
    def _rng(df, col, default_lo=0, default_hi=1):
        if df.empty or col not in df.columns:
            return default_lo, default_hi
        s = df[col].dropna()
        return float(s.min()), float(s.max())

    min_gs, max_gs = _rng(snaps, "goals_scored_avg_10",    0, 4)
    min_gc, max_gc = _rng(snaps, "goals_conceded_avg_10",  0, 4)
    min_elo, max_elo = _rng(elos, "elo_rating",           1200, 2200)
    min_rank, max_rank = _rng(elos, "rank",                  1, 50)
    _, max_exp = _rng(squads, "experience_score", 0, 30)
    mv_max = squads["squad_market_value_eur"].dropna().max() if not squads.empty else 1
    max_log_mv = float(np.log1p(mv_max)) if mv_max > 0 else 1.0

    return dict(
        min_gs=min_gs, max_gs=max_gs,
        min_gc=min_gc, max_gc=max_gc,
        min_elo=min_elo, max_elo=max_elo,
        min_rank=min_rank, max_rank=max_rank,
        max_exp=max_exp,
        max_log_mv=max_log_mv,
    )


# ─────────────────────────────────────────────────────────────────────────────
# World Cup history helper  (reused from team_profiles logic)
# ─────────────────────────────────────────────────────────────────────────────
def _wc_history(hist_df: pd.DataFrame, team: str) -> pd.DataFrame:
    wc = hist_df[
        (hist_df["tournament"].str.contains("FIFA World Cup", case=False, na=False)) &
        ((hist_df["home_team"] == team) | (hist_df["away_team"] == team))
    ].copy()
    if wc.empty:
        return wc
    wc = wc.dropna(subset=["home_score", "away_score"]).copy()
    if wc.empty:
        return wc

    def _result(r):
        stored, is_home = r.get("result", "D"), (r.get("home_team") == team)
        if stored == "D":
            return "D"
        return ("W" if stored == "H" else "L") if is_home else ("W" if stored == "A" else "L")

    wc["Result"]   = wc.apply(_result, axis=1)
    wc["Opponent"] = wc.apply(
        lambda r: r.get("away_team") if r.get("home_team") == team else r.get("home_team"),
        axis=1,
    )
    wc["Score"] = wc.apply(
        lambda r: f"{int(r['home_score'])} – {int(r['away_score'])}", axis=1,
    )
    cols = [c for c in ["date", "tournament", "Opponent", "Score", "Result"] if c in wc.columns]
    return wc[cols].sort_values("date", ascending=False).head(25)


# ─────────────────────────────────────────────────────────────────────────────
# Market-value formatter
# ─────────────────────────────────────────────────────────────────────────────
def _mv(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if v >= 1e9:
        return f"€{v/1e9:.2f}B"
    if v >= 1e6:
        return f"€{v/1e6:.0f}M"
    return f"€{v/1e3:.0f}K"


# ─────────────────────────────────────────────────────────────────────────────
# Squad roster component (shared for both teams)
# ─────────────────────────────────────────────────────────────────────────────
def _squad_block(team: str) -> None:
    """Render squad roster with position tabs and key-player callout."""
    squad = load_squad_json(team)
    if not squad:
        st.warning(f"No squad data for {team}.")
        return

    pos_order = ["GK", "DF", "MF", "FW"]
    by_pos: dict[str, list] = {p: [] for p in pos_order}
    for p in squad:
        pos = p.get("position", "?")
        if pos in by_pos:
            by_pos[pos].append(p)

    all_rows = [
        {
            "Pos":   p.get("position", "?"),
            "Name":  p.get("name", "?"),
            "Club":  p.get("club", "?"),
            "Age":   p.get("age", "?"),
            "Caps":  p.get("caps", "?"),
            "Goals": p.get("goals", "?"),
        }
        for p in squad
    ]

    tabs = st.tabs(["All"] + pos_order)
    with tabs[0]:
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)
    for i, pos in enumerate(pos_order):
        with tabs[i + 1]:
            pos_rows = [
                {"Name": p.get("name","?"), "Club": p.get("club","?"),
                 "Age": p.get("age","?"), "Caps": p.get("caps","?"),
                 "Goals": p.get("goals","?")}
                for p in by_pos[pos]
            ]
            if pos_rows:
                st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)
            else:
                st.caption(f"No {pos} data.")

    # Star player callout
    try:
        star = max(squad, key=lambda p: p.get("caps", 0))
        st.caption(
            f"Most capped: **{star['name']}** — "
            f"{star.get('caps',0)} caps, {star.get('goals',0)} goals"
        )
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    theme.page_header(
        title="Team Comparison",
        subtitle=(
            "Compare any two World Cup 2026 teams across stats, squad, "
            "head-to-head record, and historical World Cup performance."
        ),
    )

    # ── Load shared data ──────────────────────────────────────────────────────
    all_teams  = load_all_teams()
    mc_df      = load_mc_probabilities()
    elo_df     = load_elo_ratings()
    snap_df    = load_team_snapshots()
    squad_prof = load_squad_profiles()

    # ── Team selectors ────────────────────────────────────────────────────────
    col1, vs_col, col2 = st.columns([5, 1, 5])
    with col1:
        team1 = st.selectbox(
            "Team 1", all_teams,
            index=all_teams.index("Spain") if "Spain" in all_teams else 0,
            format_func=flag_team, key="tc_team1",
        )
    with vs_col:
        st.markdown(
            "<div style='text-align:center;margin-top:38px;font-size:13px;"
            "font-weight:600;letter-spacing:.08em;color:#94A3B8'>VS</div>",
            unsafe_allow_html=True,
        )
    with col2:
        default2 = "Argentina" if "Argentina" in all_teams else all_teams[1]
        team2 = st.selectbox(
            "Team 2", all_teams,
            index=all_teams.index(default2),
            format_func=flag_team, key="tc_team2",
        )

    if team1 == team2:
        st.warning("Please select two **different** teams.")
        return

    # ── Extract per-team data ─────────────────────────────────────────────────
    snap1  = snap_df[snap_df["team"] == team1].iloc[0]  if not snap_df[snap_df["team"] == team1].empty  else None
    snap2  = snap_df[snap_df["team"] == team2].iloc[0]  if not snap_df[snap_df["team"] == team2].empty  else None
    elo1   = elo_df[elo_df["team"] == team1].iloc[0]    if not elo_df[elo_df["team"] == team1].empty    else None
    elo2   = elo_df[elo_df["team"] == team2].iloc[0]    if not elo_df[elo_df["team"] == team2].empty    else None
    sq1    = squad_prof[squad_prof["team"] == team1].iloc[0] if not squad_prof[squad_prof["team"] == team1].empty else None
    sq2    = squad_prof[squad_prof["team"] == team2].iloc[0] if not squad_prof[squad_prof["team"] == team2].empty else None
    mc1    = mc_df[mc_df["team"] == team1].iloc[0]      if not mc_df[mc_df["team"] == team1].empty      else None
    mc2    = mc_df[mc_df["team"] == team2].iloc[0]      if not mc_df[mc_df["team"] == team2].empty      else None

    # ── H2H stats ─────────────────────────────────────────────────────────────
    h2h = get_h2h_stats(team1, team2)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: Key Metrics
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section("Key Metrics", "Side-by-side numbers with the stronger value highlighted.")

    metric_rows = []
    def _add(label, v1, v2, fmt="{}", higher_is_better=True):
        try:
            n1 = float(str(v1).replace("%","").replace("N/A","0").replace("€","").replace("B","e9").replace("M","e6").replace("K","e3"))
            n2 = float(str(v2).replace("%","").replace("N/A","0").replace("€","").replace("B","e9").replace("M","e6").replace("K","e3"))
            if higher_is_better:
                winner = team1 if n1 > n2 else (team2 if n2 > n1 else "—")
            else:
                winner = team1 if n1 < n2 else (team2 if n2 < n1 else "—")
        except Exception:
            winner = "—"
        metric_rows.append({"Metric": label, team1: str(v1), team2: str(v2), "Better": winner})

    # ELO
    _add("ELO Rating",
         int(_safe(elo1, "elo_rating", 0)) or "N/A",
         int(_safe(elo2, "elo_rating", 0)) or "N/A")
    # Rank
    _add("FIFA/ELO Rank",
         int(_safe(elo1, "rank", 0)) or "N/A",
         int(_safe(elo2, "rank", 0)) or "N/A",
         higher_is_better=False)
    # Form
    _add("Win % (Last 10)",
         f"{_safe(snap1, 'win_pct_last_10'):.0f}%",
         f"{_safe(snap2, 'win_pct_last_10'):.0f}%")
    _add("Goals Scored /game",
         f"{_safe(snap1, 'goals_scored_avg_10'):.2f}",
         f"{_safe(snap2, 'goals_scored_avg_10'):.2f}")
    _add("Goals Conceded /game",
         f"{_safe(snap1, 'goals_conceded_avg_10'):.2f}",
         f"{_safe(snap2, 'goals_conceded_avg_10'):.2f}",
         higher_is_better=False)
    _add("Squad Avg Age",
         f"{_safe(sq1, 'avg_age'):.1f}",
         f"{_safe(sq2, 'avg_age'):.1f}",
         higher_is_better=False)
    _add("Squad Avg Caps",
         f"{_safe(sq1, 'avg_caps'):.0f}",
         f"{_safe(sq2, 'avg_caps'):.0f}")
    _add("Squad Market Value",
         _mv(_safe(sq1, "squad_market_value_eur")),
         _mv(_safe(sq2, "squad_market_value_eur")))
    _add("Experience Score",
         f"{_safe(sq1, 'experience_score'):.0f}",
         f"{_safe(sq2, 'experience_score'):.0f}")
    if mc1 is not None and mc2 is not None:
        _add("P(Champion)",
             f"{float(mc1.get('p_winner',0))*100:.1f}%",
             f"{float(mc2.get('p_winner',0))*100:.1f}%")

    metrics_df = pd.DataFrame(metric_rows)

    # colour "Better" column
    def _colour_row(row):
        styles = [""] * len(row)
        if row["Better"] == team1:
            styles[1] = "background-color:rgba(16,185,129,.10); font-weight:600"
        elif row["Better"] == team2:
            styles[2] = "background-color:rgba(16,185,129,.10); font-weight:600"
        return styles

    st.dataframe(
        metrics_df.style.apply(_colour_row, axis=1),
        use_container_width=True, hide_index=True,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: Radar Chart
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(
        "Multi-Dimensional Comparison",
        "All 8 dimensions are normalised to 0–100 across all 48 World Cup teams. "
        "H2H is the historical win rate between these two specific teams.",
    )

    ranges = _population_ranges(snap_df, elo_df, squad_prof)
    h2h_t1 = h2h["team1_win_pct"]
    h2h_t2 = h2h["team2_wins"] / h2h["total"] if h2h["total"] > 0 else 0.5

    vals1 = _radar_vals(snap1, elo1, sq1, h2h_t1, ranges)
    vals2 = _radar_vals(snap2, elo2, sq2, h2h_t2, ranges)

    categories = [
        "Attack", "Defence", "Form",
        "Elo", "FIFA Rank",
        "Squad\nValue", "Experience", "H2H",
    ]
    fig_radar = comparison_radar(vals1, vals2, team1, team2, categories)
    st.plotly_chart(fig_radar, use_container_width=True)

    # Radar dimension explainer
    with st.expander("How each dimension is calculated"):
        st.markdown(
            """
| Dimension | Source | Notes |
|---|---|---|
| Attack | Goals scored avg (last 10 matches) | Normalised across 48 teams |
| Defence | Goals conceded avg (last 10) — inverted | Lower conceded = higher score |
| Form | Win % last 10 matches | 0–100% → 0–1 |
| Elo | Current Elo rating | Normalised to [1200–2200] range |
| FIFA Rank | Elo rank — inverted | Lower rank number = higher score |
| Squad Value | Log(squad market value €) | Normalised across 48 teams |
| Experience | Experience score (caps × WC apps) | Normalised across 48 teams |
| H2H | Historical win rate between these two teams | 0.5 if no head-to-head data |
"""
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Tournament Probability Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section("Tournament Probabilities")

    if mc1 is not None and mc2 is not None:
        fig_funnel = tournament_funnel(mc_df, [team1, team2])
        st.plotly_chart(fig_funnel, use_container_width=True)

        # Compact probability comparison table
        prob_rows = []
        for stage, label in STAGE_LABELS.items():
            p1_col = f"p_{stage}" if f"p_{stage}" in mc_df.columns else stage
            p2_col = p1_col
            p1 = float(mc1.get(p1_col, mc1.get(stage, 0)))
            p2 = float(mc2.get(p2_col, mc2.get(stage, 0)))
            prob_rows.append({
                "Stage": label,
                f"{flag(team1)} {team1}": f"{p1*100:.1f}%",
                f"{flag(team2)} {team2}": f"{p2*100:.1f}%",
                "Edge": team1 if p1 > p2 else (team2 if p2 > p1 else "Even"),
            })
        st.dataframe(pd.DataFrame(prob_rows), use_container_width=True, hide_index=True)
    else:
        st.info("MC probability data not available for one or both teams.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Squad Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section("Squad Comparison")
    sq_col1, sq_col2 = st.columns(2)
    with sq_col1:
        st.markdown(f"#### {flag(team1)} {team1}")
        _squad_block(team1)
    with sq_col2:
        st.markdown(f"#### {flag(team2)} {team2}")
        _squad_block(team2)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 5: Head-to-Head History
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(f"Head-to-Head — {team1} vs {team2}")

    total = h2h["total"]
    if total == 0:
        st.info(f"No recorded matches found between {team1} and {team2}.")
    else:
        # Summary metrics
        hm1, hd, hm2, hg = st.columns(4)
        hm1.metric(f"{flag(team1)} {team1} Wins", h2h["team1_wins"])
        hd.metric("Draws", h2h["draws"])
        hm2.metric(f"{flag(team2)} {team2} Wins", h2h["team2_wins"])
        hg.metric("Total Matches", total)

        gc1, gc2 = st.columns(2)
        gc1.metric(f"{team1} Goals", h2h["team1_goals"])
        gc2.metric(f"{team2} Goals", h2h["team2_goals"])

        # Dominance bar
        t1_pct = h2h["team1_wins"] / total
        t2_pct = h2h["team2_wins"] / total
        d_pct  = h2h["draws"] / total
        st.markdown(
            f"""
<div style="margin:14px 0 0; border-radius:99px; overflow:hidden; height:10px; display:flex;
            background:rgba(148,163,184,.12)">
  <div style="width:{t1_pct*100:.1f}%; background:#3B82F6; min-width:1px;"></div>
  <div style="width:{d_pct*100:.1f}%;  background:#475569; min-width:1px;"></div>
  <div style="width:{t2_pct*100:.1f}%; background:#94A3B8; min-width:1px;"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:12px; color:#94A3B8; margin:8px 0 14px">
  <span><b style='color:#F8FAFC'>{team1}</b> {t1_pct*100:.0f}% wins</span>
  <span>{d_pct*100:.0f}% draws</span>
  <span><b style='color:#F8FAFC'>{team2}</b> {t2_pct*100:.0f}% wins</span>
</div>
""",
            unsafe_allow_html=True,
        )

        # Recent matches
        st.markdown("**Recent meetings:**")
        match_rows = []
        for m in h2h["matches"][:10]:
            home_win = False
            away_win = False
            try:
                hg_val, ag_val = m["score"].split(" – ")
                home_win = int(hg_val) > int(ag_val)
                away_win = int(ag_val) > int(hg_val)
            except Exception:
                pass
            result_lbl = (f"{team1} win" if (home_win and m["home"] == team1) or
                                            (away_win and m["away"] == team1)
                          else f"{team2} win" if (home_win and m["home"] == team2) or
                                                 (away_win and m["away"] == team2)
                          else "Draw")
            match_rows.append({
                "Date":       m["date"][:10],
                "Home":       f"{flag(m['home'])} {m['home']}",
                "Score":      m["score"],
                "Away":       f"{flag(m['away'])} {m['away']}",
                "Tournament": m["tournament"],
                "Result":     result_lbl,
            })
        st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 6: WC History Timeline
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section("World Cup History Timeline")

    try:
        hist_df = load_historical_results()
        wc1 = _wc_history(hist_df, team1)
        wc2 = _wc_history(hist_df, team2)

        tab1, tab2 = st.tabs([f"{flag(team1)} {team1}", f"{flag(team2)} {team2}"])
        with tab1:
            if not wc1.empty:
                fig1 = wc_history_timeline(wc1, team1)
                st.plotly_chart(fig1, use_container_width=True)
                wdl1 = wc1["Result"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("WC Wins",   int(wdl1.get("W", 0)))
                c2.metric("WC Draws",  int(wdl1.get("D", 0)))
                c3.metric("WC Losses", int(wdl1.get("L", 0)))
            else:
                st.info(f"No World Cup records found for {team1}.")
        with tab2:
            if not wc2.empty:
                fig2 = wc_history_timeline(wc2, team2)
                st.plotly_chart(fig2, use_container_width=True)
                wdl2 = wc2["Result"].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric("WC Wins",   int(wdl2.get("W", 0)))
                c2.metric("WC Draws",  int(wdl2.get("D", 0)))
                c3.metric("WC Losses", int(wdl2.get("L", 0)))
            else:
                st.info(f"No World Cup records found for {team2}.")
    except Exception as exc:
        st.warning(f"Could not load WC history: {exc}")
