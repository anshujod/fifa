"""
charts.py — Reusable Plotly chart builders for the FIFA WC 2026 dashboard.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots


# ── colour palette — Terminal (cyan data accent · amber champion signal) ────────
ACCENT   = "#3AC9E0"        # signal cyan
ACCENT_2 = "#74E2F2"
GOLD     = "#E8A33D"        # amber — champion / leader (name kept for compatibility)
GOLD_2   = "#F4C36B"
SUCCESS  = "#3FB950"
WARNING  = "#D29922"
ERROR    = "#F85149"
NEUTRAL  = "#3A434F"
TEXT     = "#E6EDF3"
TEXT_DIM = "#7D8794"
BLUE     = ACCENT
BG       = "rgba(0,0,0,0)"  # transparent — lets the terminal canvas show through
CARD     = "#11151C"        # panel background

MONO     = "IBM Plex Mono, ui-monospace, monospace"

# cyan ramp for sequential / staged data (dark → light)
BLUE_RAMP = ["#0E3A44", "#125663", "#1C7A8C", "#2AA5BD", "#3AC9E0", "#74E2F2", "#A9EFF8"]

# field-cyan ramp with an amber cap reserved for the clear leader
PROB_RAMP = [
    [0.0, "#11151C"], [0.30, "#125663"], [0.58, "#1C8DA3"],
    [0.80, "#3AC9E0"], [0.92, "#74E2F2"], [0.965, "#E8A33D"], [1.0, "#F4C36B"],
]

# ── global template (Plex Mono, faint grid, cyan-led colorway) ───────────────
pio.templates["wc_premium"] = go.layout.Template(
    layout=dict(
        font=dict(family=MONO, color=TEXT, size=12),
        title=dict(font=dict(family=MONO, size=14, color=TEXT)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#161B23",
            bordercolor="rgba(255,255,255,.12)",
            font=dict(family=MONO, color=TEXT, size=12),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.06)", zerolinecolor="rgba(255,255,255,.12)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        colorway=["#3AC9E0", "#E8A33D", "#74E2F2", "#3FB950",
                  "#9BA7B4", "#A9EFF8", "#D29922", "#6B7480"],
        margin=dict(t=56, b=44, l=56, r=24),
    )
)
pio.templates.default = "plotly_dark+wc_premium"

STAGE_COLOURS = {
    "p_group_qualify":  BLUE_RAMP[0],
    "p_round_of_32":    BLUE_RAMP[1],
    "p_round_of_16":    BLUE_RAMP[2],
    "p_quarter_final":  BLUE_RAMP[3],
    "p_semi_final":     BLUE_RAMP[4],
    "p_final":          BLUE_RAMP[5],
    "p_winner":         BLUE_RAMP[6],
}

STAGE_LABELS = {
    "p_group_qualify":  "Group Stage",
    "p_round_of_32":    "Round of 32",
    "p_round_of_16":    "Round of 16",
    "p_quarter_final":  "Quarter-Final",
    "p_semi_final":     "Semi-Final",
    "p_final":          "Final",
    "p_winner":         "Champion",
}


# ─────────────────────────────────────────────────────────────────────────────
# Home page charts
# ─────────────────────────────────────────────────────────────────────────────

def top_favourites_bar(df: pd.DataFrame, n: int = 10) -> go.Figure:
    """Horizontal bar chart — top N teams by P(champion)."""
    top = df.nlargest(n, "p_winner").sort_values("p_winner")
    colours = [GOLD if i == len(top) - 1 else ACCENT
               for i in range(len(top))]

    fig = go.Figure(go.Bar(
        x=top["p_winner"] * 100,
        y=top["flag_team"],
        orientation="h",
        marker_color=colours,
        text=[f"{v*100:.1f}%" for v in top["p_winner"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>P(Champion): %{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Top {n} Championship Favourites", font_size=18),
        xaxis_title="P(Champion) %",
        yaxis_title=None,
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=420,
        margin=dict(l=160, r=80, t=50, b=40),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)", range=[0, top["p_winner"].max() * 120]),
    )
    return fig


def tournament_funnel(df: pd.DataFrame, teams: list[str]) -> go.Figure:
    """Multi-team probability funnel across tournament stages."""
    stage_cols = list(STAGE_LABELS.keys())
    labels = list(STAGE_LABELS.values())

    fig = go.Figure()
    palette = ["#3AC9E0", "#E8A33D", "#74E2F2", "#3FB950",
               "#9BA7B4", "#A9EFF8", "#D29922", "#6B7480"]
    for i, team in enumerate(teams):
        row = df[df["team"] == team]
        if row.empty:
            continue
        vals = [float(row[c].values[0]) * 100 for c in stage_cols]
        fig.add_trace(go.Scatter(
            x=labels, y=vals,
            mode="lines+markers",
            name=team,
            line=dict(width=2.5, color=palette[i % len(palette)]),
            marker=dict(size=7),
            hovertemplate=f"<b>{team}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Tournament Progression Probability",
        xaxis_title="Stage",
        yaxis_title="Probability (%)",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=380,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", range=[0, 105]),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)"),
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def probability_landscape(df: pd.DataFrame) -> go.Figure:
    """
    Treemap of all 48 nations, each tile sized and shaded by P(champion).
    A single broadcast-grade visual that shows the entire field at a glance.
    """
    d = df.sort_values("p_winner", ascending=False).copy()
    pct = d["p_winner"] * 100
    labels = [f"{t}<br><b>{p:.1f}%</b>" for t, p in zip(d["team"], pct)]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(d),
        values=d["p_winner"],
        marker=dict(
            colors=d["p_winner"],
            colorscale=PROB_RAMP,
            line=dict(width=2, color="#0A0C10"),
            cornerradius=4,
        ),
        text=d["team"],
        textinfo="label",
        textfont=dict(family=MONO, size=14, color=TEXT),
        hovertemplate="<b>%{text}</b><br>Championship probability: %{value:.2%}<extra></extra>",
        tiling=dict(pad=3),
        sort=True,
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color=TEXT,
        height=460,
        margin=dict(t=10, b=10, l=6, r=6),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Match predictor charts
# ─────────────────────────────────────────────────────────────────────────────

def outcome_donut(p_home: float, p_draw: float, p_away: float,
                  home: str, away: str) -> go.Figure:
    """Donut chart for match outcome probabilities."""
    labels = [f"{home} Win", "Draw", f"{away} Win"]
    values = [p_home * 100, p_draw * 100, p_away * 100]
    colours = [ACCENT, "#3A434F", "#6B7480"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker_colors=colours,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        textfont_size=13,
    ))
    fig.update_layout(
        title="Match Outcome Probabilities",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=340,
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def scoreline_heatmap(
    scores: dict[tuple[int, int], int],
    home: str,
    away: str,
    n_sims: int,
    max_goals: int = 5,
) -> go.Figure:
    """Heatmap of simulated scoreline probabilities."""
    import numpy as np
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for (h, a), cnt in scores.items():
        if h <= max_goals and a <= max_goals:
            grid[h, a] += cnt
    grid = grid / n_sims * 100  # convert to %

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=[str(i) for i in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        colorscale=[[0.0, "#0E1A20"], [0.5, "#1C7A8C"], [1.0, "#3AC9E0"]],
        text=[[f"{grid[i, j]:.1f}%" for j in range(max_goals + 1)]
              for i in range(max_goals + 1)],
        hovertemplate=f"<b>{home} %{{y}} – %{{x}} {away}</b><br>Prob: %{{text}}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Prob %", ticksuffix="%", tickfont_color="white",
                      title_font_color="white"),
    ))

    # Per-cell labels: the cyan scale runs dark (low) → bright cyan (high), so a
    # single text colour fails one end. Ink the labels by their own value — dark
    # ink on bright cells, light ink on dark cells — to clear WCAG contrast.
    cell_threshold = grid.max() * 0.55 if grid.max() else 1.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            fig.add_annotation(
                x=str(j), y=str(i), text=f"{grid[i, j]:.1f}%",
                showarrow=False,
                font=dict(
                    family=MONO,
                    color="#06222A" if grid[i, j] >= cell_threshold else TEXT,
                    size=12,
                ),
            )

    fig.update_layout(
        title=f"Scoreline Probability Matrix",
        xaxis_title=f"{away} Goals",
        yaxis_title=f"{home} Goals",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=380,
        margin=dict(t=50, b=60, l=70, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Group stage charts
# ─────────────────────────────────────────────────────────────────────────────

def group_standings_bar(records: dict, group_id: str) -> go.Figure:
    """Stacked bar chart of group standings (W/D/L)."""
    teams = list(records.keys())
    wins   = [records[t].wins   for t in teams]
    draws  = [records[t].draws  for t in teams]
    losses = [records[t].losses for t in teams]
    pts    = [records[t].wins * 3 + records[t].draws for t in teams]
    gf     = [records[t].goals_for     for t in teams]
    ga     = [records[t].goals_against for t in teams]
    gd     = [records[t].goals_for - records[t].goals_against for t in teams]

    # Sort by pts then GD
    order = sorted(range(len(teams)), key=lambda i: (pts[i], gd[i]), reverse=True)
    teams = [teams[i] for i in order]
    wins  = [wins[i]  for i in order]
    draws = [draws[i] for i in order]
    losses= [losses[i] for i in order]
    pts   = [pts[i]   for i in order]
    gf    = [gf[i]    for i in order]
    ga    = [ga[i]    for i in order]
    gd    = [gd[i]    for i in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Wins",   x=teams, y=wins,   marker_color=SUCCESS,
                         hovertemplate="<b>%{x}</b><br>Wins: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(name="Draws",  x=teams, y=draws,  marker_color=NEUTRAL,
                         hovertemplate="<b>%{x}</b><br>Draws: %{y}<extra></extra>"))
    fig.add_trace(go.Bar(name="Losses", x=teams, y=losses, marker_color=ERROR,
                         hovertemplate="<b>%{x}</b><br>Losses: %{y}<extra></extra>"))

    # Annotate total points
    for i, (team, pt) in enumerate(zip(teams, pts)):
        fig.add_annotation(x=team, y=3.2, text=f"<b>{pt} pts</b>",
                           showarrow=False, font=dict(color="white", size=12))

    fig.update_layout(
        barmode="stack",
        title=f"Group {group_id} Standings (W/D/L)",
        xaxis_title="Team",
        yaxis_title="Matches",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=360,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", range=[0, 4.2]),
        margin=dict(t=50, b=60, l=50, r=20),
    )
    return fig


def goals_chart(records: dict, group_id: str) -> go.Figure:
    """Goals For / Against bar chart."""
    teams  = list(records.keys())
    pts    = [records[t].wins * 3 + records[t].draws for t in teams]
    gd     = [records[t].goals_for - records[t].goals_against for t in teams]
    order  = sorted(range(len(teams)), key=lambda i: (pts[i], gd[i]), reverse=True)
    teams  = [teams[i] for i in order]
    gf     = [records[teams[i]].goals_for     for i in range(len(teams))]
    ga     = [records[teams[i]].goals_against for i in range(len(teams))]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Goals For",     x=teams, y=gf, marker_color=ACCENT))
    fig.add_trace(go.Bar(name="Goals Against", x=teams, y=ga, marker_color=NEUTRAL,
                         base=[-v for v in ga]))

    fig.update_layout(
        title=f"Group {group_id} — Goals For / Against",
        barmode="overlay",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=320,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", title="Goals"),
        margin=dict(t=50, b=60, l=50, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Team profile charts
# ─────────────────────────────────────────────────────────────────────────────

def squad_position_pie(squad: list[dict], team: str) -> go.Figure:
    """Pie chart of squad positions."""
    from collections import Counter
    counts = Counter(p.get("position", "?") for p in squad)
    pos_order = ["GK", "DF", "MF", "FW"]
    # Monochrome brand-blue ramp (deep → light) across the four position groups
    colours = {"GK": BLUE_RAMP[1], "DF": BLUE_RAMP[2], "MF": ACCENT, "FW": ACCENT_2}
    labels = [p for p in pos_order if p in counts]
    values = [counts[p] for p in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker_colors=[colours.get(p, "#64748B") for p in labels],
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>%{value} players<extra></extra>",
    ))
    fig.update_layout(
        title=f"{team} — Squad Composition",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=300,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def form_radar(snapshot: pd.Series, team: str) -> go.Figure:
    """Radar chart of recent form metrics."""
    import numpy as np

    metrics = {
        "Win Rate\n(last 10)": min(snapshot.get("win_pct_last_10", 0) / 100, 1),
        "Goals\nScored": min(snapshot.get("goals_scored_avg_10", 0) / 4, 1),
        "Goals\nConceeded": 1 - min(snapshot.get("goals_conceded_avg_10", 0) / 4, 1),
        "Clean\nSheets": min(snapshot.get("clean_sheets_last_10", 0) / 10, 1),
        "Form\nScore": min(snapshot.get("form_score", 0) / 3, 1),
    }
    cats = list(metrics.keys())
    vals = list(metrics.values())
    vals.append(vals[0])  # close the polygon
    cats.append(cats[0])

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(58, 201, 224, 0.16)",
        line=dict(color=ACCENT, width=2),
        name=team,
    ))
    fig.update_layout(
        title=f"{team} — Recent Form",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            tickfont_color="white", gridcolor="rgba(148,163,184,.10)"),
            angularaxis=dict(tickfont_color="white", gridcolor="rgba(148,163,184,.10)"),
            bgcolor=CARD,
        ),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=340,
        showlegend=False,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


def probability_stage_bar(team: str, probs: dict[str, float]) -> go.Figure:
    """Bar chart of a team's probability at each tournament stage."""
    stage_order = [
        "group_qualify", "round_of_32", "round_of_16",
        "quarter_final", "semi_final", "final", "winner",
    ]
    label_map = {
        "group_qualify": "Group", "round_of_32": "R32",
        "round_of_16": "R16",    "quarter_final": "QF",
        "semi_final": "SF",      "final": "Final",
        "winner": "Champion",
    }
    colour_map = {
        "group_qualify": BLUE_RAMP[0], "round_of_32": BLUE_RAMP[1],
        "round_of_16": BLUE_RAMP[2],   "quarter_final": BLUE_RAMP[3],
        "semi_final": BLUE_RAMP[4],    "final": BLUE_RAMP[5],
        "winner": BLUE_RAMP[6],
    }
    labels = [label_map[s] for s in stage_order if s in probs]
    vals   = [probs[s] * 100 for s in stage_order if s in probs]
    clrs   = [colour_map[s]  for s in stage_order if s in probs]

    fig = go.Figure(go.Bar(
        x=labels, y=vals,
        marker_color=clrs,
        text=[f"{v:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=f"{team} — Tournament Probabilities",
        yaxis_title="Probability (%)",
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font_color="white",
        height=340,
        yaxis=dict(range=[0, 115], gridcolor="rgba(148,163,184,.10)"),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)"),
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def elo_history_placeholder(team: str, elo_row: pd.Series) -> go.Figure:
    """Mini card showing current ELO rating as a gauge."""
    elo = int(elo_row.get("elo_rating", 1500))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=elo,
        title={"text": "Elo Rating", "font": {"size": 14, "color": TEXT_DIM}},
        number={"font": {"color": TEXT, "size": 28}},
        gauge={
            "axis": {"range": [1200, 2300], "tickcolor": TEXT_DIM,
                     "tickfont": {"color": TEXT_DIM}},
            "bar": {"color": ACCENT},
            "steps": [
                {"range": [1200, 1700], "color": "#1E293B"},
                {"range": [1700, 1900], "color": "#172033"},
                {"range": [1900, 2300], "color": "#111827"},
            ],
            "threshold": {"line": {"color": ACCENT_2, "width": 3}, "value": elo},
            "bgcolor": CARD,
        },
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color="white",
        height=260,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.2 — Public API chart functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_win_probabilities(
    team: str,
    probs: dict[str, float],
    ci_lo: dict[str, float] | None = None,
    ci_hi: dict[str, float] | None = None,
) -> go.Figure:
    """
    Horizontal stacked bar: P(reach each stage) with optional 95% CI error bars.

    Parameters
    ----------
    team   : team name for title
    probs  : {stage_key: probability}  e.g. {"winner": 0.178, "final": 0.32, ...}
    ci_lo  : {stage_key: lower_bound}  optional Wilson lower bounds
    ci_hi  : {stage_key: upper_bound}  optional Wilson upper bounds
    """
    stage_order = [
        "group_qualify", "round_of_32", "round_of_16",
        "quarter_final", "semi_final", "final", "winner",
    ]
    label_map = {
        "group_qualify": "Group Stage",  "round_of_32": "Round of 32",
        "round_of_16": "Round of 16",    "quarter_final": "Quarter-Final",
        "semi_final": "Semi-Final",      "final": "Final",
        "winner": "Champion",
    }
    colour_map = {
        "group_qualify": BLUE_RAMP[0], "round_of_32": BLUE_RAMP[1],
        "round_of_16": BLUE_RAMP[2],   "quarter_final": BLUE_RAMP[3],
        "semi_final": BLUE_RAMP[4],    "final": BLUE_RAMP[5],
        "winner": BLUE_RAMP[6],
    }
    stages  = [s for s in stage_order if s in probs]
    labels  = [label_map[s] for s in stages]
    vals    = [probs[s] * 100 for s in stages]
    colours = [colour_map[s] for s in stages]

    err_minus = err_plus = None
    if ci_lo and ci_hi:
        err_minus = [max(0, (probs[s] - ci_lo.get(s, probs[s])) * 100) for s in stages]
        err_plus  = [max(0, (ci_hi.get(s, probs[s]) - probs[s]) * 100) for s in stages]

    bar_kw: dict = dict(
        x=vals, y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"  {v:.1f}%" for v in vals],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
    )
    if err_minus:
        bar_kw["error_x"] = dict(
            type="data", symmetric=False,
            array=err_plus, arrayminus=err_minus,
            color="rgba(255,255,255,0.55)", thickness=1.5, width=5,
        )

    fig = go.Figure(go.Bar(**bar_kw))
    fig.update_layout(
        title=f"{team} — Tournament Win Probabilities",
        xaxis_title="Probability (%)",
        yaxis_title=None,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color="white",
        height=360,
        xaxis=dict(range=[0, 115], gridcolor="rgba(148,163,184,.10)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", autorange="reversed"),
        margin=dict(t=50, b=40, l=130, r=90),
    )
    return fig


# Public aliases so callers can use the named API from the task spec
def plot_match_prediction(
    p_home: float, p_draw: float, p_away: float,
    home: str, away: str,
) -> go.Figure:
    """Donut chart: W/D/L probabilities.  (Alias for outcome_donut.)"""
    return outcome_donut(p_home, p_draw, p_away, home, away)


def plot_expected_scoreline(
    scores: dict, home: str, away: str,
    n_sims: int, max_goals: int = 5,
) -> go.Figure:
    """Heatmap of goal-score probabilities 0–max_goals × 0–max_goals.
    (Alias for scoreline_heatmap.)"""
    return scoreline_heatmap(scores, home, away, n_sims, max_goals)


def plot_group_standings(
    records: dict,
    ranking: list[str],
    group_id: str,
) -> go.Figure:
    """
    Sortable Plotly Table with colour-coded qualification zones.

    Zone colours:
      1st & 2nd  — dark green  (qualify directly)
      3rd        — dark amber  (potential third-place qualifier)
      4th        — dark red    (eliminated)
    """
    n_rows = len(ranking)
    zone_fill = {
        0: "rgba(63,185,80,.10)", 1: "rgba(63,185,80,.10)",     # SUCCESS — qualify
        2: "rgba(210,153,34,.10)", 3: "rgba(255,255,255,.03)",  # WARNING — third / out
    }
    row_colors = [zone_fill.get(i, "rgba(0,0,0,0)") for i in range(n_rows)]

    headers = ["Pos", "Team", "Pld", "W", "D", "L", "GF", "GA", "GD", "Pts", "Status"]
    col_data: list[list] = [[] for _ in range(len(headers))]

    for i, team in enumerate(ranking):
        rec = records[team]
        pts = rec.wins * 3 + rec.draws
        gd  = rec.goals_for - rec.goals_against
        status = ("Qualified" if i < 2 else
                  "Third place" if i == 2 else "Eliminated")
        row = [str(i + 1), team, rec.played, rec.wins, rec.draws,
               rec.losses, rec.goals_for, rec.goals_against,
               f"{gd:+d}", pts, status]
        for c, v in enumerate(row):
            col_data[c].append(v)

    # fill_color shape: [n_cols][n_rows]
    fill = [[row_colors[r] for r in range(n_rows)] for _ in range(len(headers))]

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#161B23",
            font=dict(color=TEXT_DIM, size=12, family=MONO),
            align=["center", "left"] + ["center"] * 9,
            line_color="rgba(255,255,255,.10)",
            height=36,
        ),
        cells=dict(
            values=col_data,
            fill_color=fill,
            font=dict(color=TEXT, size=12.5, family=MONO),
            align=["center", "left"] + ["center"] * 9,
            line_color="rgba(255,255,255,.07)",
            height=34,
        ),
        columnwidth=[40, 180, 45, 40, 40, 40, 45, 45, 50, 45, 110],
    ))
    fig.update_layout(
        title=f"Group {group_id} — Standings",
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color=TEXT,
        height=270,
        margin=dict(t=50, b=30, l=10, r=10),
        annotations=[dict(
            x=0, y=-0.08, xref="paper", yref="paper",
            text="Top two qualify directly · third place may advance as best-ranked third",
            showarrow=False,
            font=dict(color=TEXT_DIM, size=11),
            xanchor="left",
        )],
    )
    return fig


def plot_feature_importance(imp_df: "pd.DataFrame") -> go.Figure:
    """
    Horizontal bar chart of SHAP (or native) feature importances,
    colour-coded by feature category.

    imp_df: DataFrame with columns [feature, importance, category, source]
    """
    if imp_df is None or imp_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Feature importance data not available",
            plot_bgcolor=BG, paper_bgcolor=BG, font_color="white",
        )
        return fig

    CAT_COL: dict[str, str] = {
        "ELO/Ranking":    ACCENT,      # signal cyan
        "Expected Goals": SUCCESS,     # green
        "Head-to-Head":   WARNING,     # amber
        "Confederation":  "#8B7FE8",   # one violet — separates the 9-family chart
        "Form/Momentum":  ACCENT_2,    # bright cyan
        "Goals":          "#52C7A6",   # teal-green (bridges cyan↔green)
        "Match Context":  "#64748B",
        "Other":          "#94A3B8",
        "Unknown":        "#94A3B8",
    }

    def fmt(f: str) -> str:
        f = (f.replace("home_", "H:").replace("away_", "A:")
              .replace("_avg_10", "/10").replace("_avg_5", "/5")
              .replace("_before", "").replace("_decay", " decay")
              .replace("_conf_", ":").replace("feat_", "")
              .replace("_", " ").title())
        return f

    df     = imp_df.sort_values("importance", ascending=True)
    source = df["source"].iloc[-1] if "source" in df.columns else "Importance"

    fig = go.Figure()
    for cat in df["category"].unique():
        mask = df["category"] == cat
        fig.add_trace(go.Bar(
            x=df.loc[mask, "importance"],
            y=[fmt(f) for f in df.loc[mask, "feature"]],
            orientation="h",
            name=cat,
            marker_color=CAT_COL.get(cat, "#9E9E9E"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.5f}<extra></extra>",
        ))

    n = len(df)
    fig.update_layout(
        title=dict(
            text=(f"Top {n} Feature Importances"
                  + (" — SHAP TreeExplainer" if source == "SHAP" else "")),
            font_size=16,
        ),
        xaxis_title="Mean |SHAP Value|" if source == "SHAP" else "Importance Score",
        yaxis_title=None,
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color="white",
        height=max(440, n * 24 + 100),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font_size=12,
            title=dict(text="Feature Category", font_color="white"),
        ),
        barmode="overlay",
        xaxis=dict(gridcolor="rgba(148,163,184,.10)"),
        margin=dict(l=200, r=80, t=70, b=50),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5.3 — Team Comparison charts
# ─────────────────────────────────────────────────────────────────────────────

def comparison_radar(
    vals1: list[float],
    vals2: list[float],
    team1: str,
    team2: str,
    categories: list[str],
) -> go.Figure:
    """
    Dual-trace polar radar for side-by-side team comparison.

    Parameters
    ----------
    vals1/vals2 : list of N floats in [0, 1], one per category
    categories  : list of N dimension labels (same order as vals)
    """
    v1   = list(vals1) + [vals1[0]]
    v2   = list(vals2) + [vals2[0]]
    cats = list(categories) + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v1, theta=cats, fill="toself",
        fillcolor="rgba(58,201,224,0.14)",
        line=dict(color=ACCENT, width=2),
        name=team1,
        hovertemplate=f"<b>{team1}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=v2, theta=cats, fill="toself",
        fillcolor="rgba(232,163,61,0.13)",
        line=dict(color=GOLD, width=2),
        name=team2,
        hovertemplate=f"<b>{team2}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{team1}  vs  {team2}", font_size=18, x=0.5, xanchor="center"),
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25", "50", "75", "100"],
                tickfont=dict(color="#9E9E9E", size=10),
                gridcolor="rgba(148,163,184,.10)",
                linecolor="rgba(148,163,184,.18)",
            ),
            angularaxis=dict(
                tickfont=dict(size=13, color="white"),
                gridcolor="rgba(148,163,184,.10)",
                linecolor="rgba(148,163,184,.18)",
                direction="clockwise",
                rotation=90,
            ),
            bgcolor=CARD,
        ),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color="white",
        height=520,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font_size=14,
            orientation="h",
            x=0.5, xanchor="center", y=-0.08,
        ),
        margin=dict(t=80, b=100, l=80, r=80),
    )
    return fig


def wc_history_timeline(wc_df: "pd.DataFrame", team: str) -> go.Figure:
    """
    Scatter timeline of a team's World Cup match history.
    X axis = date, Y axis = opponent, colour/symbol = W/D/L.
    """
    if wc_df is None or wc_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{team} — No World Cup history found",
            plot_bgcolor=BG, paper_bgcolor=BG,
            font_color="white", height=200,
        )
        return fig

    RESULT_COL = {"W": SUCCESS, "D": "#6B7480", "L": ERROR}
    RESULT_SYM = {"W": "circle",  "D": "diamond", "L": "x"}

    # Ensure date is parseable
    import pandas as pd
    dates = pd.to_datetime(wc_df["date"], errors="coerce")
    opponents = wc_df["Opponent"]
    scores    = wc_df["Score"]
    tourn     = (wc_df["tournament"].values
                 if "tournament" in wc_df.columns
                 else [""] * len(wc_df))

    fig = go.Figure()
    for res in ["W", "D", "L"]:
        mask = wc_df["Result"] == res
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=dates[mask],
            y=opponents[mask],
            mode="markers",
            name=res,
            marker=dict(
                size=11, color=RESULT_COL[res],
                symbol=RESULT_SYM[res],
                line=dict(color="rgba(248,250,252,.35)", width=1),
            ),
            text=scores[mask],
            customdata=[[t] for t in tourn[mask]],
            hovertemplate=(
                f"<b>{team}</b> vs <b>%{{y}}</b><br>"
                "Score: %{text}<br>"
                "Tournament: %{customdata[0]}<br>"
                "Date: %{x|%Y-%m-%d}<extra></extra>"
            ),
        ))

    n_opps = opponents.nunique()
    fig.update_layout(
        title=f"{team} — World Cup Match History",
        xaxis=dict(title="Year", type="date", gridcolor="rgba(148,163,184,.10)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", categoryorder="total ascending"),
        plot_bgcolor=BG, paper_bgcolor=BG,
        font_color="white",
        height=max(300, n_opps * 36 + 110),
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            title=dict(text="Result", font_color="white"),
            orientation="h", x=0.5, xanchor="center", y=-0.14,
        ),
        margin=dict(t=60, b=100, l=140, r=30),
    )
    return fig
