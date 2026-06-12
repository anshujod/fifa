"""
theme.py — Design system for the WC 2026 dashboard.

Single source of truth for the colour palette, global CSS, and the custom
HTML components (page header, KPI cards, ranking rows, group cards, team
probability cards) used across pages.

Design language: premium sports-intelligence platform — quiet surfaces,
typography-led hierarchy, a single blue accent used sparingly, and no
decorative iconography. Country flags are the only pictorial element.
"""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0B1020"
SURFACE   = "#172033"
SURFACE_2 = "#1E293B"
BORDER    = "rgba(148, 163, 184, 0.12)"

TEXT      = "#F8FAFC"
TEXT_2    = "#CBD5E1"
TEXT_3    = "#94A3B8"

ACCENT    = "#3B82F6"
ACCENT_2  = "#60A5FA"
SUCCESS   = "#10B981"
WARNING   = "#F59E0B"
ERROR     = "#EF4444"


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp { background: #0B1020; }
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }

h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; color: #F8FAFC; }
hr { border-color: rgba(148,163,184,.10) !important; margin: 1.6rem 0 !important; }

/* hide markdown header anchor links */
h1 a, h2 a, h3 a, [data-testid="stHeaderActionElements"] { display: none !important; }

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #283349; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #36435e; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0F172A;
    border-right: 1px solid rgba(148,163,184,.10);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* nav radio → quiet text items */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 8px;
    padding: 8px 12px;
    margin: 0;
    width: 100%;
    border: none;
    transition: background .15s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(148,163,184,.07);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(59,130,246,.12);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #F8FAFC !important; font-weight: 600;
}
/* hide the radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 13.5px; color: #CBD5E1; font-weight: 500;
}
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.10); }

/* ── Native metric cards ───────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 10px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,.25);
    transition: border-color .15s ease;
}
div[data-testid="stMetric"]:hover { border-color: rgba(148,163,184,.24); }
div[data-testid="stMetric"] label p {
    color: #94A3B8 !important; font-size: 11px;
    text-transform: uppercase; letter-spacing: .08em; font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-weight: 700; font-variant-numeric: tabular-nums; color: #F8FAFC;
}
div[data-testid="stMetricDelta"] { font-size: 12.5px; }

/* ── Bordered containers → cards ───────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #172033;
    border: 1px solid rgba(148,163,184,.12) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.25);
    transition: border-color .15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(148,163,184,.22) !important;
}

/* ── Dataframes / tables ───────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 10px;
    overflow: hidden;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 14px;
    border: 1px solid rgba(148,163,184,.20);
    background: #1E293B;
    color: #F8FAFC;
    transition: background .15s ease, border-color .15s ease;
}
.stButton > button:hover { background: #243246; border-color: rgba(148,163,184,.32); }
.stButton > button[kind="primary"] {
    background: #3B82F6;
    color: #FFFFFF; border: none;
    padding: 0.6rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #2563EB; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-size: 13.5px; font-weight: 600; color: #94A3B8; }
button[data-baseweb="tab"][aria-selected="true"] { color: #F8FAFC; }
div[data-baseweb="tab-highlight"] { background-color: #3B82F6; }
div[data-baseweb="tab-border"] { background-color: rgba(148,163,184,.12); }

/* ── Inputs ────────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: #172033 !important;
    border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #3B82F6; border-color: #3B82F6;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 10px;
    background: #131B2B;
}

/* ═══════════════════════════════ Custom components ═════════════════════ */

/* Page header */
.wc-header { margin-bottom: 28px; }
.wc-header .eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: #60A5FA; margin-bottom: 10px;
}
.wc-header h1 {
    font-size: 30px; font-weight: 700; margin: 0 0 8px;
    letter-spacing: -0.02em; line-height: 1.15; color: #F8FAFC;
}
.wc-header.hero h1 { font-size: 38px; }
.wc-header p { color: #94A3B8; font-size: 14.5px; margin: 0; max-width: 640px; line-height: 1.55; }

/* KPI grid */
.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;
    margin: 4px 0 10px;
}
.kpi-card {
    background: #172033;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 12px;
    padding: 20px 22px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,.25);
    transition: border-color .15s ease;
}
.kpi-card:hover { border-color: rgba(148,163,184,.24); }
.kpi-label {
    font-size: 11px; font-weight: 600; letter-spacing: .09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 10px;
}
.kpi-value {
    font-size: 26px; font-weight: 700; color: #F8FAFC;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}
.kpi-delta { font-size: 12.5px; font-weight: 500; margin-top: 6px; color: #94A3B8; }
.kpi-delta.up      { color: #10B981; }
.kpi-delta.accent  { color: #60A5FA; }

/* Section headers */
.wc-section { margin: 34px 0 16px; }
.wc-section h2 {
    font-size: 19px; font-weight: 700; margin: 0; letter-spacing: -0.01em; color: #F8FAFC;
}
.wc-section p { color: #94A3B8; font-size: 13.5px; margin: 4px 0 0; }

/* Ranking rows */
.rank-list { display: flex; flex-direction: column; gap: 8px; }
.rank-card {
    display: flex; align-items: center; gap: 14px;
    background: #172033;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 10px;
    padding: 12px 16px;
    transition: border-color .15s ease, background .15s ease;
}
.rank-card:hover { border-color: rgba(148,163,184,.24); background: #1A2438; }
.rank-pos {
    min-width: 24px; text-align: center;
    font-weight: 600; font-size: 13px; color: #94A3B8;
    font-variant-numeric: tabular-nums;
}
.rank-flag { font-size: 21px; line-height: 1; }
.rank-body { flex: 1; min-width: 0; }
.rank-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.rank-team { font-weight: 600; font-size: 14px; color: #F8FAFC; }
.rank-prob { font-weight: 700; font-size: 14.5px; color: #F8FAFC; font-variant-numeric: tabular-nums; }
.rank-bar {
    height: 4px; border-radius: 99px; background: rgba(148,163,184,.12); overflow: hidden;
}
.rank-fill { height: 100%; border-radius: 99px; background: #3B82F6; }
.rank-meta { font-size: 11.5px; color: #94A3B8; margin-top: 6px; font-variant-numeric: tabular-nums; }

/* Group cards */
.group-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; }
@media (max-width: 1100px) { .group-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card {
    background: #172033;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 10px;
    padding: 16px 18px;
    transition: border-color .15s ease;
}
.group-card:hover { border-color: rgba(148,163,184,.24); }
.group-title {
    font-size: 11px; font-weight: 600; letter-spacing: .10em; text-transform: uppercase;
    color: #94A3B8; margin-bottom: 10px;
}
.group-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.group-row .t { flex: 1; font-size: 13px; color: #CBD5E1; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.group-row .p { font-size: 12px; font-weight: 600; color: #94A3B8; font-variant-numeric: tabular-nums; }
.group-row .bar { width: 52px; height: 3px; border-radius: 99px; background: rgba(148,163,184,.12); }
.group-row .bar > div { height: 100%; border-radius: 99px; background: #3B82F6; }

/* Team probability cards */
.team-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 1100px) { .team-grid { grid-template-columns: repeat(2, 1fr); } }
.team-card {
    background: #172033;
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 12px;
    padding: 20px 22px;
    transition: border-color .15s ease;
}
.team-card:hover { border-color: rgba(148,163,184,.24); }
.team-card .head { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.team-card .head .flag { font-size: 22px; line-height: 1; }
.team-card .head .name { font-size: 14.5px; font-weight: 600; color: #F8FAFC; }
.team-card .main-label {
    font-size: 11px; font-weight: 600; letter-spacing: .09em;
    text-transform: uppercase; color: #94A3B8;
}
.team-card .main-value {
    font-size: 30px; font-weight: 700; color: #F8FAFC;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
    margin: 2px 0 14px;
}
.team-card .sub { display: flex; justify-content: space-between; align-items: baseline;
    padding: 6px 0; border-top: 1px solid rgba(148,163,184,.08); }
.team-card .sub .l { font-size: 12px; color: #94A3B8; }
.team-card .sub .v { font-size: 12.5px; font-weight: 600; color: #CBD5E1;
    font-variant-numeric: tabular-nums; }

/* Outcome probability bar */
.outcome-wrap { margin: 8px 0 4px; }
.outcome-bar {
    display: flex; height: 10px; border-radius: 99px; overflow: hidden;
    background: rgba(148,163,184,.12);
}
.outcome-bar .h { background: #3B82F6; }
.outcome-bar .d { background: #475569; }
.outcome-bar .a { background: #94A3B8; }
.outcome-legend {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #94A3B8; margin-top: 8px;
    font-variant-numeric: tabular-nums;
}
.outcome-legend b { color: #F8FAFC; font-weight: 600; }

/* Sidebar widgets */
.sb-logo { display: flex; align-items: center; gap: 11px; padding: 2px 4px 16px; }
.sb-logo .mark {
    width: 34px; height: 34px; border-radius: 8px;
    background: #1E293B; border: 1px solid rgba(148,163,184,.16);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: #60A5FA; letter-spacing: .02em;
}
.sb-logo .name { font-size: 14px; font-weight: 600; color: #F8FAFC; line-height: 1.2; }
.sb-logo .sub { font-size: 10.5px; color: #94A3B8; letter-spacing: .07em; text-transform: uppercase; }

.sb-status { display: flex; align-items: center; gap: 8px; padding: 6px 4px 2px; }
.sb-status .dot { width: 7px; height: 7px; border-radius: 50%; background: #10B981; }
.sb-status .txt { font-size: 12px; font-weight: 500; color: #CBD5E1; }
.sb-status .txt span { color: #94A3B8; font-weight: 400; }

.sb-fav { display: flex; align-items: center; gap: 10px; padding: 4px 4px; }
.sb-fav .pos { width: 14px; font-size: 12px; font-weight: 600; color: #94A3B8;
    font-variant-numeric: tabular-nums; }
.sb-fav .team { flex: 1; font-size: 13px; font-weight: 500; color: #CBD5E1; }
.sb-fav .pct { font-size: 12px; font-weight: 600; color: #F8FAFC; font-variant-numeric: tabular-nums; }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once per page render, right after set_page_config."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", eyebrow: str = "", hero: bool = False) -> None:
    """Standard page header: eyebrow label, title, muted subtitle."""
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    cls = "wc-header hero" if hero else "wc-header"
    st.markdown(
        f'<div class="{cls}">{eyebrow_html}<h1>{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    """cards: [{label, value, delta?, delta_class? ('up'|'accent')}]"""
    items = "".join(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{c['label']}</div>
  <div class="kpi-value">{c['value']}</div>
  <div class="kpi-delta {c.get('delta_class','')}">{c.get('delta','')}</div>
</div>"""
        for c in cards
    )
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="wc-section"><h2>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def ranking_cards(rows: list[dict]) -> None:
    """rows: [{pos, flag, team, prob (0-1), elo, group}] — prob bars scaled to leader."""
    max_p = max((r["prob"] for r in rows), default=1.0) or 1.0
    items = []
    for r in rows:
        width = r["prob"] / max_p * 100
        items.append(f"""
<div class="rank-card">
  <div class="rank-pos">{r['pos']}</div>
  <div class="rank-flag">{r['flag']}</div>
  <div class="rank-body">
    <div class="rank-top">
      <span class="rank-team">{r['team']}</span>
      <span class="rank-prob">{r['prob'] * 100:.1f}%</span>
    </div>
    <div class="rank-bar"><div class="rank-fill" style="width:{width:.1f}%"></div></div>
    <div class="rank-meta">Elo {r['elo']} &nbsp;·&nbsp; Group {r['group']}</div>
  </div>
</div>""")
    st.markdown(f'<div class="rank-list">{"".join(items)}</div>', unsafe_allow_html=True)


def group_cards(groups: dict[str, list[dict]]) -> None:
    """groups: {gid: [{flag, team, prob (0-1)}]} — bars scaled per group."""
    cards = []
    for gid, teams in groups.items():
        max_p = max((t["prob"] for t in teams), default=1.0) or 1.0
        rows = "".join(
            f"""
<div class="group-row">
  <span>{t['flag']}</span><span class="t">{t['team']}</span>
  <span class="bar"><div style="width:{t['prob'] / max_p * 100:.0f}%"></div></span>
  <span class="p">{t['prob'] * 100:.1f}%</span>
</div>"""
            for t in teams
        )
        cards.append(
            f'<div class="group-card"><div class="group-title">Group {gid}</div>{rows}</div>'
        )
    st.markdown(f'<div class="group-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def team_prob_cards(cards: list[dict]) -> None:
    """cards: [{flag, team, main_label, main_value, metrics: [(label, value), ...]}]"""
    items = []
    for c in cards:
        subs = "".join(
            f'<div class="sub"><span class="l">{label}</span><span class="v">{value}</span></div>'
            for label, value in c.get("metrics", [])
        )
        items.append(f"""
<div class="team-card">
  <div class="head"><span class="flag">{c['flag']}</span><span class="name">{c['team']}</span></div>
  <div class="main-label">{c['main_label']}</div>
  <div class="main-value">{c['main_value']}</div>
  {subs}
</div>""")
    st.markdown(f'<div class="team-grid">{"".join(items)}</div>', unsafe_allow_html=True)


def outcome_bar(p_home: float, p_draw: float, p_away: float, home: str, away: str) -> None:
    """Horizontal segmented probability bar for match outcome (win / draw / win)."""
    st.markdown(
        f"""
<div class="outcome-wrap">
  <div class="outcome-bar">
    <div class="h" style="width:{p_home*100:.1f}%"></div>
    <div class="d" style="width:{p_draw*100:.1f}%"></div>
    <div class="a" style="width:{p_away*100:.1f}%"></div>
  </div>
  <div class="outcome-legend">
    <span><b>{home}</b> {p_home*100:.1f}%</span>
    <span>Draw {p_draw*100:.1f}%</span>
    <span><b>{away}</b> {p_away*100:.1f}%</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def sidebar_logo() -> None:
    st.markdown(
        """
<div class="sb-logo">
  <div class="mark">26</div>
  <div>
    <div class="name">World Cup 2026</div>
    <div class="sub">Prediction Engine</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_status(n_sims: int) -> None:
    st.markdown(
        f"""
<div class="sb-status">
  <div class="dot"></div>
  <div class="txt">Model online <span>· {n_sims:,} simulations</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_favourites(rows: list[dict]) -> None:
    """rows: [{pos, team, pct}]"""
    items = "".join(
        f"""
<div class="sb-fav">
  <span class="pos">{r['pos']}</span>
  <span class="team">{r['team']}</span>
  <span class="pct">{r['pct']}</span>
</div>"""
        for r in rows
    )
    st.markdown(items, unsafe_allow_html=True)
