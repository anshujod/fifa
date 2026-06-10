"""
theme.py — Design system for the WC 2026 dashboard.

Single source of truth for the colour palette, global CSS, and the custom
HTML components (hero, KPI cards, ranking cards, group cards) used across
pages. Inspired by Stripe / Vercel / Linear dashboards.
"""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0B1020"
SURFACE   = "#111827"
CARD      = "#171F2E"
CARD_HI   = "#1C2638"
BORDER    = "rgba(148, 163, 184, 0.14)"
GOLD      = "#F4C430"
BLUE      = "#3B82F6"
GREEN     = "#22C55E"
RED       = "#EF4444"
TEXT      = "#F9FAFB"
TEXT_DIM  = "#94A3B8"

MEDALS = {0: GOLD, 1: "#C0C8D4", 2: "#CD7F32"}


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background:
        radial-gradient(1100px 500px at 80% -10%, rgba(59,130,246,.07), transparent 60%),
        radial-gradient(900px 500px at 0% 0%, rgba(244,196,48,.05), transparent 55%),
        #0B1020;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }

h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em; color: #F9FAFB; }

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #283349; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #36435e; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1322 0%, #0A0F1C 100%);
    border-right: 1px solid rgba(148,163,184,.10);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* nav radio → Linear-style items */
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 3px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 10px;
    padding: 9px 12px;
    margin: 0;
    width: 100%;
    border: 1px solid transparent;
    transition: background .15s ease, border .15s ease, box-shadow .2s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.045);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(244,196,48,.14), rgba(244,196,48,.03));
    border: 1px solid rgba(244,196,48,.32);
    box-shadow: 0 0 16px rgba(244,196,48,.12), inset 0 0 12px rgba(244,196,48,.04);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #F4C430 !important; font-weight: 600;
}
/* hide the radio circle */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px; color: #C8D2E0;
}
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.10); }

/* ── Native metric cards (used on inner pages) ─────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #1A2336 0%, #151D2D 100%);
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25);
    transition: transform .18s ease, border-color .18s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    border-color: rgba(244,196,48,.35);
}
div[data-testid="stMetric"] label p { color: #94A3B8 !important; font-size: 12px;
    text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
div[data-testid="stMetricValue"] { font-weight: 800; }

/* ── Bordered containers → cards ───────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #151D2D;
    border: 1px solid rgba(148,163,184,.13) !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 10px 30px rgba(0,0,0,.20);
    transition: border-color .18s ease, transform .18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(148,163,184,.28) !important;
}

/* ── Dataframes / tables ───────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 6px 22px rgba(0,0,0,.22);
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid rgba(148,163,184,.22);
    transition: all .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 18px rgba(0,0,0,.35); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #F4C430, #E0A800);
    color: #0B1020; border: none;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
div[data-baseweb="tab-highlight"] { background-color: #F4C430; }

/* ── Inputs ────────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stNumberInput input {
    background-color: #171F2E !important;
    border-radius: 10px !important;
}

/* ═══════════════════════════════ Custom components ═════════════════════ */

/* Hero */
.wc-hero {
    position: relative;
    border-radius: 20px;
    padding: 38px 42px 34px;
    margin-bottom: 22px;
    background:
        radial-gradient(1100px 420px at 15% -20%, rgba(59,130,246,.20), transparent 60%),
        radial-gradient(800px 360px at 90% -10%, rgba(244,196,48,.16), transparent 55%),
        linear-gradient(180deg, #131B2C 0%, #0E1524 100%);
    border: 1px solid rgba(148,163,184,.16);
    box-shadow: 0 20px 60px rgba(0,0,0,.35);
    overflow: hidden;
}
.wc-hero::after {
    content: "";
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(148,163,184,.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(148,163,184,.05) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(600px 300px at 30% 0%, black, transparent);
    pointer-events: none;
}
.wc-hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 700; letter-spacing: .14em;
    color: #F4C430; text-transform: uppercase;
    background: rgba(244,196,48,.10);
    border: 1px solid rgba(244,196,48,.30);
    border-radius: 99px; padding: 6px 14px; margin-bottom: 16px;
}
.wc-hero h1 {
    font-size: 40px; font-weight: 900; margin: 0 0 8px;
    letter-spacing: -0.03em; line-height: 1.05; color: #F9FAFB;
}
.wc-hero h1 .gold {
    background: linear-gradient(90deg, #F4C430, #FFE08A);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.wc-hero p { color: #94A3B8; font-size: 15px; margin: 0; max-width: 640px; }

/* KPI grid */
.kpi-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px;
    margin: 4px 0 10px;
}
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi-card {
    background: linear-gradient(180deg, #1A2336 0%, #141C2B 100%);
    border: 1px solid rgba(148,163,184,.14);
    border-radius: 18px;
    padding: 18px 20px 16px;
    box-shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 28px rgba(0,0,0,.22);
    transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    animation: cardIn .45s ease both;
}
.kpi-card:nth-child(2) { animation-delay: .06s; }
.kpi-card:nth-child(3) { animation-delay: .12s; }
.kpi-card:nth-child(4) { animation-delay: .18s; }
.kpi-card:hover {
    transform: translateY(-3px);
    border-color: rgba(244,196,48,.38);
    box-shadow: 0 14px 40px rgba(0,0,0,.35), 0 0 24px rgba(244,196,48,.06);
}
.kpi-head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.kpi-icon {
    width: 30px; height: 30px; border-radius: 9px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 15px; background: rgba(59,130,246,.13);
    border: 1px solid rgba(59,130,246,.25);
}
.kpi-icon.gold { background: rgba(244,196,48,.12); border-color: rgba(244,196,48,.30); }
.kpi-icon.green { background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.30); }
.kpi-label {
    font-size: 11px; font-weight: 700; letter-spacing: .09em;
    text-transform: uppercase; color: #94A3B8;
}
.kpi-value { font-size: 26px; font-weight: 800; color: #F9FAFB; letter-spacing: -0.02em; }
.kpi-delta { font-size: 12.5px; font-weight: 600; margin-top: 5px; color: #94A3B8; }
.kpi-delta.up   { color: #22C55E; }
.kpi-delta.gold { color: #F4C430; }

/* Section headers */
.wc-section { margin: 26px 0 14px; }
.wc-section h2 {
    font-size: 22px; font-weight: 800; margin: 0; letter-spacing: -0.02em;
    display: flex; align-items: center; gap: 10px;
}
.wc-section .accent {
    width: 4px; height: 20px; border-radius: 4px; display: inline-block;
    background: linear-gradient(180deg, #F4C430, #E0A800);
}
.wc-section p { color: #94A3B8; font-size: 13.5px; margin: 5px 0 0 14px; }

/* Ranking cards */
.rank-list { display: flex; flex-direction: column; gap: 9px; }
.rank-card {
    display: flex; align-items: center; gap: 14px;
    background: linear-gradient(180deg, #192236 0%, #141C2B 100%);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 14px;
    padding: 12px 16px;
    transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    animation: cardIn .4s ease both;
}
.rank-card:hover {
    transform: translateX(3px);
    border-color: rgba(244,196,48,.35);
    box-shadow: 0 8px 26px rgba(0,0,0,.30);
}
.rank-pos {
    min-width: 30px; height: 30px; border-radius: 9px;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 13px;
    background: rgba(148,163,184,.10); color: #94A3B8;
    border: 1px solid rgba(148,163,184,.16);
}
.rank-pos.m0 { background: rgba(244,196,48,.15); color: #F4C430; border-color: rgba(244,196,48,.45);
    box-shadow: 0 0 14px rgba(244,196,48,.18); }
.rank-pos.m1 { background: rgba(192,200,212,.12); color: #C0C8D4; border-color: rgba(192,200,212,.40); }
.rank-pos.m2 { background: rgba(205,127,50,.13); color: #CD7F32; border-color: rgba(205,127,50,.45); }
.rank-flag { font-size: 24px; line-height: 1; }
.rank-body { flex: 1; min-width: 0; }
.rank-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.rank-team { font-weight: 700; font-size: 14.5px; color: #F9FAFB; }
.rank-prob { font-weight: 800; font-size: 15px; color: #F4C430; font-variant-numeric: tabular-nums; }
.rank-bar {
    height: 6px; border-radius: 99px; background: rgba(148,163,184,.12); overflow: hidden;
}
.rank-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #3B82F6, #F4C430);
    animation: growBar 1s cubic-bezier(.22,.8,.36,1) both;
}
.rank-meta { font-size: 11.5px; color: #94A3B8; margin-top: 6px; font-variant-numeric: tabular-nums; }

/* Group cards */
.group-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; }
@media (max-width: 1100px) { .group-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card {
    background: linear-gradient(180deg, #182135 0%, #131B2A 100%);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 14px;
    padding: 14px 16px;
    transition: transform .16s ease, border-color .16s ease;
}
.group-card:hover { transform: translateY(-2px); border-color: rgba(59,130,246,.40); }
.group-title {
    font-size: 12px; font-weight: 800; letter-spacing: .10em; text-transform: uppercase;
    color: #3B82F6; margin-bottom: 10px;
    display: flex; align-items: center; gap: 7px;
}
.group-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; }
.group-row .t { flex: 1; font-size: 13px; color: #E3E9F2; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.group-row .p { font-size: 12px; font-weight: 700; color: #94A3B8; font-variant-numeric: tabular-nums; }
.group-row .bar { width: 52px; height: 4px; border-radius: 99px; background: rgba(148,163,184,.12); }
.group-row .bar > div { height: 100%; border-radius: 99px; background: #F4C430;
    animation: growBar .9s ease both; }

/* Sidebar widgets */
.sb-logo { display: flex; align-items: center; gap: 11px; padding: 2px 4px 14px; }
.sb-logo .mark {
    width: 38px; height: 38px; border-radius: 11px;
    background: linear-gradient(135deg, #F4C430, #E08700);
    display: flex; align-items: center; justify-content: center; font-size: 19px;
    box-shadow: 0 4px 14px rgba(244,196,48,.30);
}
.sb-logo .name { font-size: 15px; font-weight: 800; color: #F9FAFB; line-height: 1.15; }
.sb-logo .sub { font-size: 10.5px; color: #94A3B8; letter-spacing: .06em; text-transform: uppercase; }

.sb-status {
    display: flex; align-items: center; gap: 9px;
    background: rgba(34,197,94,.07);
    border: 1px solid rgba(34,197,94,.22);
    border-radius: 11px; padding: 9px 12px; margin: 4px 0 2px;
}
.sb-status .dot {
    width: 8px; height: 8px; border-radius: 50%; background: #22C55E;
    box-shadow: 0 0 0 0 rgba(34,197,94,.55);
    animation: pulse 2s infinite;
}
.sb-status .txt { font-size: 12px; font-weight: 600; color: #C8D2E0; }
.sb-status .txt span { color: #94A3B8; font-weight: 500; }

.sb-fav { display: flex; align-items: center; gap: 8px; padding: 5px 4px; }
.sb-fav .medal { font-size: 13px; width: 18px; }
.sb-fav .team { flex: 1; font-size: 13px; font-weight: 600; color: #E3E9F2; }
.sb-fav .pct { font-size: 12px; font-weight: 700; color: #F4C430; font-variant-numeric: tabular-nums; }

/* Animations */
@keyframes cardIn { from { opacity: 0; transform: translateY(8px); } }
@keyframes growBar { from { width: 0; } }
@keyframes pulse { 70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); } }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet. Call once per page render, right after set_page_config."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Components
# ─────────────────────────────────────────────────────────────────────────────

def hero(title_plain: str, title_gold: str, subtitle: str, badge: str) -> None:
    st.markdown(
        f"""
<div class="wc-hero">
  <div class="wc-hero-badge">⚽ {badge}</div>
  <h1>{title_plain} <span class="gold">{title_gold}</span></h1>
  <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    """cards: [{icon, tint('gold'|'blue'|'green'), label, value, delta, delta_class}]"""
    items = "".join(
        f"""
<div class="kpi-card">
  <div class="kpi-head">
    <span class="kpi-icon {c.get('tint','')}">{c['icon']}</span>
    <span class="kpi-label">{c['label']}</span>
  </div>
  <div class="kpi-value">{c['value']}</div>
  <div class="kpi-delta {c.get('delta_class','')}">{c.get('delta','')}</div>
</div>"""
        for c in cards
    )
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="wc-section"><h2><span class="accent"></span>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def ranking_cards(rows: list[dict]) -> None:
    """rows: [{pos, flag, team, prob (0-1), elo, group}] — prob bars scaled to leader."""
    max_p = max((r["prob"] for r in rows), default=1.0) or 1.0
    items = []
    for i, r in enumerate(rows):
        medal = f"m{i}" if i < 3 else ""
        width = r["prob"] / max_p * 100
        delay = f"animation-delay:{i * 0.05:.2f}s"
        items.append(f"""
<div class="rank-card" style="{delay}">
  <div class="rank-pos {medal}">{r['pos']}</div>
  <div class="rank-flag">{r['flag']}</div>
  <div class="rank-body">
    <div class="rank-top">
      <span class="rank-team">{r['team']}</span>
      <span class="rank-prob">{r['prob'] * 100:.1f}%</span>
    </div>
    <div class="rank-bar"><div class="rank-fill" style="width:{width:.1f}%;{delay}"></div></div>
    <div class="rank-meta">ELO {r['elo']} &nbsp;·&nbsp; Group {r['group']}</div>
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


def sidebar_logo() -> None:
    st.markdown(
        """
<div class="sb-logo">
  <div class="mark">🏆</div>
  <div>
    <div class="name">WC 2026</div>
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
  <div class="txt">Ensemble model online <span>· N={n_sims:,} sims</span></div>
</div>
""",
        unsafe_allow_html=True,
    )


def sidebar_favourites(rows: list[dict]) -> None:
    """rows: [{medal, team, pct}]"""
    items = "".join(
        f"""
<div class="sb-fav">
  <span class="medal">{r['medal']}</span>
  <span class="team">{r['team']}</span>
  <span class="pct">{r['pct']}</span>
</div>"""
        for r in rows
    )
    st.markdown(items, unsafe_allow_html=True)
