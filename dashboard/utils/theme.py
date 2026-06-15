"""
theme.py — Design system for the WC 2026 Forecasting Terminal.

Single source of truth for the colour palette, global CSS, motion language and
the custom HTML components used across pages.

Design language — "The Forecasting Terminal"
--------------------------------------------
A Bloomberg/trading-terminal for football. Near-black graph-paper canvas, monospace
data (IBM Plex Mono) so every figure aligns into columns, IBM Plex Sans for prose.
ONE cyan accent carries all data and interaction; ONE amber signal marks exactly the
single champion/leader per view. Flat surfaces, a single hairline border, sharp 4–6px
radii, near-square bars. No gradients, no glow, no gold-gamer chrome. Motion is fast
and minimal — things appear, they don't perform. See DESIGN.md for the full system.
"""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette — Terminal (near-black · cyan data accent · amber champion signal)
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#0A0C10"        # near-black terminal canvas
BG_2      = "#0D1016"
SURFACE   = "#11151C"        # flat panel
SURFACE_2 = "#161B23"        # raised panel / hover
BORDER    = "rgba(255, 255, 255, 0.08)"   # single hairline weight

TEXT      = "#E6EDF3"        # cool off-white ink
TEXT_2    = "#9BA7B4"
TEXT_3    = "#7D8794"        # muted (≥4.5:1 on canvas)

# GOLD* kept for code compatibility — repurposed as the AMBER champion signal.
GOLD      = "#E8A33D"        # amber — the single champion / leader mark
GOLD_2    = "#F4C36B"
GOLD_DEEP = "#A06A1E"
ACCENT    = "#3AC9E0"        # signal cyan — the one data accent
ACCENT_2  = "#74E2F2"
SUCCESS   = "#3FB950"        # terminal green
WARNING   = "#D29922"        # amber
ERROR     = "#F85149"        # terminal red


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ═══ THE FORECASTING TERMINAL ═══ near-black · mono · cyan · amber-champion */

/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

/* Force Plex Sans on every body-text surface; headings & figures take Plex Mono. */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] a,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
button[data-baseweb="tab"], [data-baseweb="tab"] *,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
.stSelectbox, .stSelectbox * {
    font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
}
/* Data surfaces take the mono face — figures align into columns. */
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p,
[data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
.stNumberInput, .stNumberInput input, .stSlider, .stSlider * {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}
.stApp {
    background-color: #0A0C10;
    background-image:
        linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px),
        radial-gradient(900px 600px at 90% -6%, rgba(58,201,224,.05), transparent 60%);
    background-size: 32px 32px, 32px 32px, 100% 100%;
    background-attachment: fixed;
}
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 2.0rem; padding-bottom: 5rem; max-width: 1200px; }

h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-weight: 600 !important; letter-spacing: -0.01em; color: #E6EDF3;
}
hr { border-color: rgba(255,255,255,.07) !important; margin: 1.6rem 0 !important; }
h1 a, h2 a, h3 a, [data-testid="stHeaderActionElements"] { display: none !important; }

/* ── Motion language — fast & minimal ──────────────────────────────────── */
@keyframes revealUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes growBar { from { transform: scaleX(0); } }
.reveal { animation: revealUp .42s cubic-bezier(.2,.6,.2,1) both; }
@media (prefers-reduced-motion: reduce) {
    .reveal, .cb-fill, .wb-fill { animation: none !important; }
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #232A33; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #313A45; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(8, 10, 14, 0.97);
    border-right: 1px solid rgba(255,255,255,.07);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 6px;
    padding: 9px 12px;
    margin: 0; width: 100%;
    border: 1px solid transparent;
    transition: background .15s ease, border-color .15s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,.04);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(58,201,224,.12);
    border-color: rgba(58,201,224,.32);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #E6EDF3 !important; font-weight: 600;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label svg { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] { display: block !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 13.5px; color: #9BA7B4; font-weight: 500;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.07); }

/* Navigation IS the sidebar — on desktop/tablet keep it permanently open so the
   menu can never be lost. Below 769px (phones) this is RELAXED: the sidebar
   collapses to an overlay and the expand control returns, so content gets the
   full width. */
@media (min-width: 769px) {
    section[data-testid="stSidebar"] {
        position: relative !important;
        min-width: 300px !important;
        width: 300px !important;
        transform: none !important;
        visibility: visible !important;
        margin-left: 0 !important;
        flex-shrink: 0 !important;
    }
    [data-testid="stSidebarContent"] { display: flex !important; }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }
    [data-testid="stMain"] { position: relative !important; inset: auto !important; }
}
/* On phones the nav can't hide behind a collapse control — stack the whole
   sidebar full-width ON TOP of the content (in normal flow), so navigation is
   always visible and the page gets full width below it. The page links lay out
   as compact wrapping chips so all 7 fit in a couple of rows. */
/* Mobile top-nav: hidden on desktop (the sidebar is the nav there). */
.mobile-nav { display: none; }

/* On phones the Streamlit sidebar is hidden entirely (its positioning traps
   scroll); navigation is the in-flow .mobile-nav row of links instead, which
   scrolls with the page and drives the ?page= query param. */
@media (max-width: 768px) {
    section[data-testid="stSidebar"] { display: none !important; }
    [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    [data-testid="stMain"] { position: relative !important; inset: auto !important; width: 100% !important; }
    .mobile-nav {
        display: flex; flex-wrap: wrap; gap: 6px;
        margin: 0 0 20px; padding-bottom: 14px;
        border-bottom: 1px solid rgba(255,255,255,.08);
    }
    .mobile-nav .brand {
        flex-basis: 100%; font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 13px; font-weight: 600; color: #E6EDF3; margin-bottom: 4px;
    }
    .mobile-nav .brand b { color: #3AC9E0; }
    .mobile-nav a.mnav-link {
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-size: 12.5px; font-weight: 500; text-decoration: none; white-space: nowrap;
        color: #9BA7B4; background: #161B23; border: 1px solid rgba(255,255,255,.10);
        border-radius: 6px; padding: 7px 12px;
    }
    .mobile-nav a.mnav-link.active {
        color: #E6EDF3; background: rgba(58,201,224,.12); border-color: rgba(58,201,224,.42);
    }
}

/* ── Native metric cards ───────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px;
    padding: 18px 20px;
    transition: border-color .15s ease;
}
div[data-testid="stMetric"]:hover { border-color: rgba(58,201,224,.40); }
div[data-testid="stMetric"] label p {
    color: #7D8794 !important; font-size: 11px;
    text-transform: uppercase; letter-spacing: .16em; font-weight: 500;
}
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-weight: 600; font-variant-numeric: tabular-nums; color: #E6EDF3;
}
div[data-testid="stMetricDelta"] { font-size: 12.5px; }

/* ── Bordered containers → flat panels ─────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08) !important;
    border-radius: 6px !important;
    transition: border-color .15s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(58,201,224,.32) !important;
}

/* ── Dataframes / tables ───────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px;
    overflow: hidden;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 6px;
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    font-weight: 600; font-size: 13.5px; letter-spacing: .01em;
    border: 1px solid rgba(255,255,255,.12);
    background: #161B23;
    color: #E6EDF3;
    transition: background .12s ease, border-color .12s ease;
}
.stButton > button:hover {
    background: #1B222C; border-color: rgba(58,201,224,.45);
}
.stButton > button[kind="primary"] {
    background: #3AC9E0;
    color: #06222A; border: none; font-weight: 600;
    padding: 0.55rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #74E2F2; }

/* ── Tabs ──────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-size: 13px; font-weight: 600; color: #7D8794;
    letter-spacing: .02em; }
button[data-baseweb="tab"][aria-selected="true"] { color: #E6EDF3; }
div[data-baseweb="tab-highlight"] { background-color: #3AC9E0; }
div[data-baseweb="tab-border"] { background-color: rgba(255,255,255,.08); }

/* ── Inputs ────────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
    background-color: #11151C !important;
    border-radius: 6px !important;
    border-color: rgba(255,255,255,.10) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #3AC9E0; border-color: #3AC9E0;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div > div {
    background: #3AC9E0;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px;
    background: #11151C;
}

/* ── Plotly chart frame ────────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 6px;
    padding: 10px 12px 12px;
    background: #0D1016;
}

/* Sidebar caption */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #7D8794 !important;
}

/* ═══════════════════════════════ Custom components ═════════════════════ */

/* Page header */
.wc-header { margin-bottom: 26px; }
.wc-header .eyebrow {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: .16em;
    text-transform: uppercase; color: #3AC9E0; margin-bottom: 12px;
}
.wc-header .eyebrow::before { content: "\\25B8  "; opacity: .85; }
.wc-header h1 {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 30px; font-weight: 600; margin: 0 0 10px;
    letter-spacing: -0.01em; line-height: 1.12; color: #E6EDF3;
}
.wc-header.hero h1 { font-size: 40px; }
.wc-header p { color: #9BA7B4; font-size: 15px; margin: 0; max-width: 620px; line-height: 1.6; }

/* KPI grid */
.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;
    margin: 4px 0 10px;
}
.kpi-card {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px;
    padding: 18px 20px 16px;
    transition: border-color .15s ease;
}
.kpi-card:hover { border-color: rgba(58,201,224,.38); }
.kpi-label {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: .14em;
    text-transform: uppercase; color: #7D8794; margin-bottom: 12px;
}
.kpi-value {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 26px; font-weight: 600; color: #E6EDF3;
    letter-spacing: 0; font-variant-numeric: tabular-nums;
    display: flex; align-items: center; gap: 10px;
}
.kpi-value img { width: 30px; height: 20px; border-radius: 2px; object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,.5); }
.kpi-delta { font-size: 12px; font-weight: 500; margin-top: 8px; color: #7D8794;
    font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.kpi-delta.up      { color: #3FB950; }
.kpi-delta.accent  { color: #3AC9E0; }
.kpi-delta.gold    { color: #E8A33D; }

/* Section headers */
.wc-section { margin: 46px 0 18px; }
.wc-section .kicker {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: .16em; text-transform: uppercase;
    color: #3AC9E0; margin-bottom: 8px;
}
.wc-section .kicker::before { content: "\\25B8  "; opacity: .85; }
.wc-section h2 {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 21px; font-weight: 600; margin: 0; letter-spacing: 0; color: #E6EDF3;
}
.wc-section p { color: #9BA7B4; font-size: 14px; margin: 6px 0 0; max-width: 620px; line-height: 1.55; }

/* ── Contender board (image flags) ─────────────────────────────────────── */
.cb { display: flex; flex-direction: column; gap: 6px; }
.cb-row {
    display: grid;
    grid-template-columns: 30px 34px 1fr 116px;
    align-items: center; gap: 16px;
    background: #11151C;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 6px;
    padding: 13px 18px;
    transition: border-color .15s ease, background .15s ease;
}
.cb-row:hover { border-color: rgba(58,201,224,.35); background: #141A22; }
.cb-row.lead {
    border-color: rgba(232,163,61,.50);
    background: rgba(232,163,61,.07);
}
.cb-pos {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 14px; font-weight: 600; color: #7D8794; text-align: center;
    font-variant-numeric: tabular-nums;
}
.cb-row.lead .cb-pos { color: #E8A33D; }
.cb-flag { width: 34px; height: 23px; border-radius: 2px; object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,.5); }
.cb-mid { min-width: 0; }
.cb-name { font-size: 15px; font-weight: 600; color: #E6EDF3; margin-bottom: 8px;
    display: flex; align-items: baseline; gap: 10px; }
.cb-name .meta { font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11.5px; font-weight: 400; color: #7D8794; font-variant-numeric: tabular-nums; }
.cb-track { height: 6px; border-radius: 2px; background: rgba(255,255,255,.06); overflow: hidden; }
.cb-fill { height: 100%; border-radius: 2px;
    background: #3AC9E0;
    transform-origin: left center;
    animation: growBar .9s cubic-bezier(.2,.7,.2,1) both; }
.cb-row.lead .cb-fill { background: #E8A33D; }
.cb-right { text-align: right; }
.cb-pct {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 20px; font-weight: 600; color: #E6EDF3; font-variant-numeric: tabular-nums;
    letter-spacing: 0;
}
.cb-row.lead .cb-pct { color: #E8A33D; }
.cb-delta { font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11.5px; font-weight: 500; margin-top: 2px; font-variant-numeric: tabular-nums; }
.cb-delta.up { color: #3FB950; }
.cb-delta.down { color: #F85149; }
.cb-delta.flat { color: #7D8794; }

/* ── Methodology / engine cards ────────────────────────────────────────── */
.eng-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.eng-card {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px;
    padding: 20px 20px;
    transition: border-color .15s ease;
}
.eng-card:hover { border-color: rgba(58,201,224,.35); }
.eng-card .num {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 32px; font-weight: 600; color: #E6EDF3;
    letter-spacing: 0; font-variant-numeric: tabular-nums; line-height: 1;
}
.eng-card .num .u { font-size: 16px; color: #7D8794; font-weight: 500; margin-left: 4px; }
.eng-card .lab { font-size: 13px; font-weight: 600; color: #9BA7B4; margin-top: 12px; }
.eng-card .desc { font-size: 12.5px; color: #7D8794; margin-top: 6px; line-height: 1.5; }

/* ── Pipeline chips (ensemble models) ──────────────────────────────────── */
.pipe { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 4px; }
.pipe .chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: #161B23; border: 1px solid rgba(255,255,255,.10);
    border-radius: 4px; padding: 7px 14px;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12.5px; font-weight: 500; color: #C3CDD8;
}
.pipe .chip .dot { width: 6px; height: 6px; border-radius: 1px; background: #3AC9E0; }
.pipe .arrow { color: #3A434F; font-size: 14px; }

/* ── Weight / contribution board ───────────────────────────────────────── */
.wb { display: flex; flex-direction: column; gap: 6px; }
.wb-row {
    display: grid; grid-template-columns: 1fr 92px; align-items: center; gap: 18px;
    background: #11151C;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 6px; padding: 15px 18px;
    transition: border-color .15s ease, background .15s ease;
}
.wb-row:hover { border-color: rgba(58,201,224,.35); background: #141A22; }
.wb-row.lead {
    border-color: rgba(232,163,61,.48);
    background: rgba(232,163,61,.06);
}
.wb-mid { min-width: 0; }
.wb-name { font-size: 14.5px; font-weight: 600; color: #E6EDF3; }
.wb-sub { font-size: 12px; color: #7D8794; margin-top: 4px; line-height: 1.45; }
.wb-track { height: 6px; border-radius: 2px; background: rgba(255,255,255,.06);
    overflow: hidden; margin-top: 11px; }
.wb-fill { height: 100%; border-radius: 2px;
    background: #3AC9E0;
    transform-origin: left center;
    animation: growBar .9s cubic-bezier(.2,.7,.2,1) both; }
.wb-row.lead .wb-fill { background: #E8A33D; }
.wb-pct { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 21px; font-weight: 600;
    color: #E6EDF3; text-align: right; font-variant-numeric: tabular-nums; letter-spacing: 0; }
.wb-row.lead .wb-pct { color: #E8A33D; }

/* ── Versus banner (matchup header) ────────────────────────────────────── */
.vs-wrap { container-type: inline-size; }
.vs {
    display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr);
    align-items: center; gap: 18px;
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px; padding: 22px 26px;
}
.vs-side { display: flex; align-items: center; gap: 15px; min-width: 0; }
.vs-side.right { flex-direction: row-reverse; }
.vs-side.right .vs-text { text-align: right; }
.vs-text { min-width: 0; }
.vs-side img { width: 56px; height: 38px; border-radius: 3px; object-fit: cover;
    box-shadow: 0 2px 8px rgba(0,0,0,.5); flex-shrink: 0; }
.vs-name { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 22px; font-weight: 600;
    color: #E6EDF3; letter-spacing: -0.01em; line-height: 1.1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vs-sub { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12px; color: #7D8794;
    margin-top: 4px; font-variant-numeric: tabular-nums;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vs-mid { text-align: center; }
.vs-mid .lab { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 15px; font-weight: 600;
    color: #3AC9E0; letter-spacing: .12em; }
.vs-mid .venue { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 10px; color: #7D8794;
    margin-top: 5px; text-transform: uppercase; letter-spacing: .12em; white-space: nowrap; }
@container (max-width: 540px) {
    .vs { padding: 16px 16px; gap: 10px; }
    .vs-side { gap: 10px; }
    .vs-side img { width: 38px; height: 26px; }
    .vs-name { font-size: 16px; }
    .vs-mid .lab { font-size: 13px; }
    .vs-mid .venue { font-size: 9px; letter-spacing: .08em; }
}

/* Match Predictor "VS" divider */
.mp-vs { font-family: 'IBM Plex Mono', ui-monospace, monospace;
    text-align: center; margin-top: 38px; font-size: 13px; font-weight: 600;
    letter-spacing: .14em; color: #3AC9E0; }
@media (max-width: 640px) { .mp-vs { margin-top: 4px; margin-bottom: 4px; } }

/* Ranking rows (legacy API) */
.rank-list { display: flex; flex-direction: column; gap: 6px; }
.rank-card {
    display: flex; align-items: center; gap: 14px;
    background: #11151C;
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 6px; padding: 12px 16px;
    transition: border-color .15s ease, background .15s ease;
}
.rank-card:hover { border-color: rgba(58,201,224,.35); background: #141A22; }
.rank-pos { font-family: 'IBM Plex Mono', ui-monospace, monospace; min-width: 24px; text-align: center;
    font-weight: 600; font-size: 13px; color: #7D8794; font-variant-numeric: tabular-nums; }
.rank-flag { font-size: 21px; line-height: 1; }
.rank-body { flex: 1; min-width: 0; }
.rank-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.rank-team { font-weight: 600; font-size: 14px; color: #E6EDF3; }
.rank-prob { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-weight: 600; font-size: 14px;
    color: #E6EDF3; font-variant-numeric: tabular-nums; }
.rank-bar { height: 4px; border-radius: 2px; background: rgba(255,255,255,.07); overflow: hidden; }
.rank-fill { height: 100%; border-radius: 2px; background: #3AC9E0; }
.rank-meta { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 11.5px; color: #7D8794;
    margin-top: 6px; font-variant-numeric: tabular-nums; }

/* Group cards */
.group-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1100px) { .group-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px; padding: 16px 18px;
    transition: border-color .15s ease;
}
.group-card:hover { border-color: rgba(58,201,224,.32); }
.group-title { font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    color: #7D8794; margin-bottom: 12px; }
.group-row { display: flex; align-items: center; gap: 9px; padding: 5px 0; }
.group-row img { width: 22px; height: 15px; border-radius: 2px; object-fit: cover;
    box-shadow: 0 1px 2px rgba(0,0,0,.5); }
.group-row .t { flex: 1; font-size: 13px; color: #C3CDD8; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.group-row .p { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12px; font-weight: 500;
    color: #7D8794; font-variant-numeric: tabular-nums; }
.group-row .bar { width: 50px; height: 3px; border-radius: 2px; background: rgba(255,255,255,.07); }
.group-row .bar > div { height: 100%; border-radius: 2px; background: #3AC9E0; }

/* Team probability cards */
.team-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 1100px) { .team-grid { grid-template-columns: repeat(2, 1fr); } }
.team-card {
    background: #11151C;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 6px; padding: 20px 22px;
    transition: border-color .15s ease;
}
.team-card:hover { border-color: rgba(58,201,224,.32); }
.team-card .head { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.team-card .head .flag { font-size: 22px; line-height: 1; }
.team-card .head .name { font-size: 14.5px; font-weight: 600; color: #E6EDF3; }
.team-card .main-label { font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 11px; font-weight: 500; letter-spacing: .14em;
    text-transform: uppercase; color: #7D8794; }
.team-card .main-value {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 30px; font-weight: 600; color: #E6EDF3;
    letter-spacing: 0; font-variant-numeric: tabular-nums; margin: 2px 0 14px; }
.team-card .sub { display: flex; justify-content: space-between; align-items: baseline;
    padding: 6px 0; border-top: 1px solid rgba(255,255,255,.06); }
.team-card .sub .l { font-size: 12px; color: #7D8794; }
.team-card .sub .v { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12.5px;
    font-weight: 500; color: #C3CDD8; font-variant-numeric: tabular-nums; }

/* Outcome probability bar */
.outcome-wrap { margin: 8px 0 4px; }
.outcome-bar { display: flex; height: 11px; border-radius: 2px; overflow: hidden;
    background: rgba(255,255,255,.07); }
.outcome-bar .h { background: #3AC9E0; }
.outcome-bar .d { background: #3A434F; }
.outcome-bar .a { background: #6B7480; }
.outcome-legend { display: flex; justify-content: space-between; font-size: 12px;
    color: #7D8794; margin-top: 8px; font-variant-numeric: tabular-nums;
    font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.outcome-legend b { color: #E6EDF3; font-weight: 600; }

/* Sidebar widgets */
.sb-logo { display: flex; align-items: center; gap: 12px; padding: 2px 4px 18px; }
.sb-logo .mark {
    width: 38px; height: 38px; border-radius: 6px;
    background: rgba(58,201,224,.10); border: 1px solid #3AC9E0;
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 13px; font-weight: 600; color: #3AC9E0; letter-spacing: .01em;
}
.sb-logo .name { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 14px;
    font-weight: 600; color: #E6EDF3; line-height: 1.2; }
.sb-logo .sub { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 10px;
    color: #7D8794; letter-spacing: .14em; text-transform: uppercase; }

.sb-status { display: flex; align-items: center; gap: 8px; padding: 6px 4px 2px; }
.sb-status .dot { width: 7px; height: 7px; border-radius: 50%; background: #3FB950;
    box-shadow: 0 0 0 0 rgba(63,185,80,.5); animation: pulse 2.2s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(63,185,80,.45); }
    70%  { box-shadow: 0 0 0 7px rgba(63,185,80,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
}
.sb-status .txt { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12px;
    font-weight: 500; color: #C3CDD8; }
.sb-status .txt span { color: #7D8794; font-weight: 400; }

.sb-fav { display: flex; align-items: center; gap: 10px; padding: 5px 4px; }
.sb-fav img { width: 22px; height: 15px; border-radius: 2px; object-fit: cover;
    box-shadow: 0 1px 2px rgba(0,0,0,.5); }
.sb-fav .pos { font-family: 'IBM Plex Mono', ui-monospace, monospace; width: 12px; font-size: 12px;
    font-weight: 600; color: #7D8794; font-variant-numeric: tabular-nums; }
.sb-fav .team { flex: 1; font-size: 13px; font-weight: 500; color: #C3CDD8; }
.sb-fav .pct { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: 12.5px;
    font-weight: 600; color: #3AC9E0; font-variant-numeric: tabular-nums; }

/* ── Mobile layout ─────────────────────────────────────────────────────── */
@media (max-width: 640px) {
    .block-container { padding-left: .7rem !important; padding-right: .7rem !important;
        padding-top: 1.1rem; }
    .wc-header h1 { font-size: 25px; }
    .wc-header.hero h1 { font-size: 29px; }
    .wc-header p { font-size: 14px; }
    .wc-section { margin: 34px 0 14px; }
    .wc-section h2 { font-size: 19px; }
    /* Multi-column card grids collapse to one column */
    .group-grid, .team-grid, .eng-grid, .kpi-grid { grid-template-columns: 1fr !important; }
    /* Contender / weight rows tighten so nothing overflows the viewport */
    .cb-row { grid-template-columns: 20px 26px 1fr 64px; gap: 10px; padding: 11px 12px; }
    .cb-name { font-size: 13.5px; }
    .cb-pct { font-size: 17px; }
    .cb-flag { width: 26px; height: 18px; }
    .wb-row { grid-template-columns: 1fr 66px; gap: 12px; padding: 13px 14px; }
    .wb-pct { font-size: 18px; }
    /* KPI / metric values shrink slightly to avoid wrapping */
    .kpi-value, .team-card .main-value { font-size: 24px; }
}
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
    cls = "wc-header hero reveal" if hero else "wc-header reveal"
    st.markdown(
        f'<div class="{cls}">{eyebrow_html}<h1>{title}</h1>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]) -> None:
    """cards: [{label, value, delta?, delta_class? ('up'|'accent'|'gold')}]"""
    items = "".join(
        f"""
<div class="kpi-card reveal" style="animation-delay:{i*70}ms">
  <div class="kpi-label">{c['label']}</div>
  <div class="kpi-value">{c['value']}</div>
  <div class="kpi-delta {c.get('delta_class','')}">{c.get('delta','')}</div>
</div>"""
        for i, c in enumerate(cards)
    )
    st.markdown(f'<div class="kpi-grid">{items}</div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "", kicker: str = "") -> None:
    kick = f'<div class="kicker">{kicker}</div>' if kicker else ""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="wc-section reveal">{kick}<h2>{title}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def contender_board(rows: list[dict]) -> None:
    """
    Broadcast-style contender ranking with rectangular flags.

    rows: [{pos, flag_url, team, prob (0-1), meta (str), delta (str|None), lead (bool)}]
    Bars are scaled to the leader.
    """
    max_p = max((r["prob"] for r in rows), default=1.0) or 1.0
    items = []
    for i, r in enumerate(rows):
        width = r["prob"] / max_p * 100
        delta = r.get("delta")
        if delta is None:
            delta_html = ""
        else:
            cls = "up" if delta.startswith("+") else "down" if delta.startswith("−") or delta.startswith("-") else "flat"
            delta_html = f'<div class="cb-delta {cls}">{delta}</div>'
        lead_cls = " lead" if r.get("lead") else ""
        items.append(f"""
<div class="cb-row{lead_cls} reveal" style="animation-delay:{i*55}ms">
  <div class="cb-pos">{r['pos']}</div>
  <img class="cb-flag" src="{r['flag_url']}" alt="" loading="lazy">
  <div class="cb-mid">
    <div class="cb-name">{r['team']} <span class="meta">{r.get('meta','')}</span></div>
    <div class="cb-track"><div class="cb-fill" style="width:{width:.1f}%"></div></div>
  </div>
  <div class="cb-right">
    <div class="cb-pct">{r['prob']*100:.1f}%</div>
    {delta_html}
  </div>
</div>""")
    st.markdown(f'<div class="cb">{"".join(items)}</div>', unsafe_allow_html=True)


def engine_cards(cards: list[dict]) -> None:
    """cards: [{num, unit?, label, desc}]"""
    items = "".join(
        f"""
<div class="eng-card reveal" style="animation-delay:{i*70}ms">
  <div class="num">{c['num']}{f'<span class="u">{c["unit"]}</span>' if c.get('unit') else ''}</div>
  <div class="lab">{c['label']}</div>
  <div class="desc">{c.get('desc','')}</div>
</div>"""
        for i, c in enumerate(cards)
    )
    st.markdown(f'<div class="eng-grid">{items}</div>', unsafe_allow_html=True)


def model_pipeline(models: list[str]) -> None:
    """Render the ensemble as a row of connected chips."""
    parts = []
    for i, m in enumerate(models):
        if i > 0:
            parts.append('<span class="arrow">+</span>')
        parts.append(f'<span class="chip"><span class="dot"></span>{m}</span>')
    st.markdown(f'<div class="pipe">{"".join(parts)}</div>', unsafe_allow_html=True)


def versus_header(left: dict, right: dict, mid_label: str = "VS", venue: str = "") -> None:
    """
    Broadcast matchup banner. left / right: {flag_url, name, sub}.
    """
    venue_html = f'<div class="venue">{venue}</div>' if venue else ""
    st.markdown(
        f"""
<div class="vs-wrap reveal">
 <div class="vs">
  <div class="vs-side">
    <img src="{left['flag_url']}" alt="" loading="lazy">
    <div class="vs-text"><div class="vs-name">{left['name']}</div>
      <div class="vs-sub">{left.get('sub','')}</div></div>
  </div>
  <div class="vs-mid"><div class="lab">{mid_label}</div>{venue_html}</div>
  <div class="vs-side right">
    <img src="{right['flag_url']}" alt="" loading="lazy">
    <div class="vs-text"><div class="vs-name">{right['name']}</div>
      <div class="vs-sub">{right.get('sub','')}</div></div>
  </div>
 </div>
</div>""",
        unsafe_allow_html=True,
    )


def weight_board(rows: list[dict]) -> None:
    """
    Contribution / weight ranking without flags — for model blend weights or
    feature drivers. Bars scale to the leader; the lead row earns gold.

    rows: [{name, sub (str), pct (0-1), lead (bool)}]
    """
    max_p = max((r["pct"] for r in rows), default=1.0) or 1.0
    items = []
    for i, r in enumerate(rows):
        w = r["pct"] / max_p * 100
        lead = " lead" if r.get("lead") else ""
        sub = f'<div class="wb-sub">{r["sub"]}</div>' if r.get("sub") else ""
        items.append(f"""
<div class="wb-row{lead} reveal" style="animation-delay:{i*55}ms">
  <div class="wb-mid">
    <div class="wb-name">{r['name']}</div>
    {sub}
    <div class="wb-track"><div class="wb-fill" style="width:{w:.1f}%"></div></div>
  </div>
  <div class="wb-pct">{r['pct']*100:.1f}%</div>
</div>""")
    st.markdown(f'<div class="wb">{"".join(items)}</div>', unsafe_allow_html=True)


# ── Legacy components (used by other pages) ──────────────────────────────────

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
    """groups: {gid: [{flag_url?|flag, team, prob (0-1)}]} — bars scaled per group."""
    cards = []
    for gid, teams in groups.items():
        max_p = max((t["prob"] for t in teams), default=1.0) or 1.0
        rows = ""
        for t in teams:
            if t.get("flag_url"):
                flag_html = f'<img src="{t["flag_url"]}" alt="" loading="lazy">'
            else:
                flag_html = f'<span>{t.get("flag","")}</span>'
            rows += f"""
<div class="group-row">
  {flag_html}<span class="t">{t['team']}</span>
  <span class="bar"><div style="width:{t['prob'] / max_p * 100:.0f}%"></div></span>
  <span class="p">{t['prob'] * 100:.1f}%</span>
</div>"""
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
    <div class="sub">Prediction Lab</div>
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
    """rows: [{pos, team, pct, flag_url?}]"""
    items = ""
    for r in rows:
        flag_html = f'<img src="{r["flag_url"]}" alt="">' if r.get("flag_url") else ""
        items += f"""
<div class="sb-fav">
  <span class="pos">{r['pos']}</span>
  {flag_html}
  <span class="team">{r['team']}</span>
  <span class="pct">{r['pct']}</span>
</div>"""
    st.markdown(items, unsafe_allow_html=True)
