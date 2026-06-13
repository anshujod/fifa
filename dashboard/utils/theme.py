"""
theme.py — Design system for the WC 2026 prediction platform.

Single source of truth for the colour palette, global CSS, motion language and
the custom HTML components used across pages.

Design language
---------------
A premium World Cup intelligence platform — cinematic dark canvas, broadcast-grade
typography, a restrained FIFA-gold accent reserved for the champion and an electric
blue for live data. No decorative iconography; rectangular country flags are the
only pictorial element. Motion is quiet and confident (Apple / Linear / Stripe),
never flashy.
"""

from __future__ import annotations

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#070B14"        # deep midnight canvas
BG_2      = "#0B1020"
SURFACE   = "#10182B"        # rich charcoal surface
SURFACE_2 = "#16203A"
BORDER    = "rgba(148, 163, 184, 0.10)"

TEXT      = "#F4F7FB"
TEXT_2    = "#AEB9CC"
TEXT_3    = "#8B98AF"

GOLD      = "#E9C46A"        # FIFA-inspired gold — the champion accent
GOLD_2    = "#F2D98C"
GOLD_DEEP = "#C9A24B"
ACCENT    = "#4C8DFF"        # electric blue — live data
ACCENT_2  = "#7DB0FF"
SUCCESS   = "#34D399"
WARNING   = "#F59E0B"
ERROR     = "#F87171"


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────── */
html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

/* Several Streamlit components ship their own default face (Source Sans). Force
   Inter on every body-text surface so type is consistent across all pages;
   headings and numerals keep Space Grotesk via the rules below. */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] a,
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
button[data-baseweb="tab"], [data-baseweb="tab"] *,
div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p,
[data-testid="stDataFrame"], [data-testid="stDataFrame"] *,
section[data-testid="stSidebar"] div[role="radiogroup"] label p,
.stSelectbox, .stSelectbox *, .stNumberInput, .stSlider, .stSlider * {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background:
        radial-gradient(820px 560px at 8% 4%, rgba(76,141,255,.08), transparent 60%),
        radial-gradient(1200px 800px at 50% 120%, rgba(76,141,255,.04), transparent 72%),
        #070B14;
    background-attachment: fixed;
}
#MainMenu, footer { visibility: hidden; height: 0; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

.block-container { padding-top: 2.0rem; padding-bottom: 5rem; max-width: 1200px; }

h1, h2, h3 {
    font-family: 'Space Grotesk', 'Inter', sans-serif !important;
    font-weight: 600 !important; letter-spacing: -0.02em; color: #F4F7FB;
}
hr { border-color: rgba(148,163,184,.08) !important; margin: 1.6rem 0 !important; }

/* hide markdown header anchor links */
h1 a, h2 a, h3 a, [data-testid="stHeaderActionElements"] { display: none !important; }

/* ── Motion language ───────────────────────────────────────────────────── */
@keyframes revealUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes growBar { from { transform: scaleX(0); } }
@keyframes shimmer {
    0%   { background-position: -160% 0; }
    100% { background-position: 260% 0; }
}
.reveal { animation: revealUp .7s cubic-bezier(.22,.7,.2,1) both; }
@media (prefers-reduced-motion: reduce) {
    .reveal { animation: none; }
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #25304a; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #34425f; }

/* ── Sidebar ───────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 22, 0.97);
    border-right: 1px solid rgba(148,163,184,.08);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    border-radius: 9px;
    padding: 9px 12px;
    margin: 0; width: 100%;
    border: 1px solid transparent;
    transition: background .18s ease, border-color .18s ease;
    cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(148,163,184,.06);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(233,196,106,.10);
    border-color: rgba(233,196,106,.22);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #F4F7FB !important; font-weight: 600;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label svg { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] { display: none; }
section[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] { display: block !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 13.5px; color: #AEB9CC; font-weight: 500;
}
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.08); }

/* Navigation IS the sidebar — keep it permanently open so the menu can never
   be lost. Force it visible at full width (overriding Streamlit's collapsed
   state) and remove both the collapse chevron and the reopen control. */
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
/* Collapsed state sets stMain to position:absolute (escaping under the forced
   sidebar). Pin it back into normal flow so it stays in its own column. */
[data-testid="stMain"] { position: relative !important; inset: auto !important; }

/* ── Native metric cards ───────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(22,32,58,.55), rgba(16,24,43,.55));
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color .2s ease, transform .2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(233,196,106,.30); transform: translateY(-2px);
}
div[data-testid="stMetric"] label p {
    color: #8B98AF !important; font-size: 11px;
    text-transform: uppercase; letter-spacing: .10em; font-weight: 600;
}
div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600; font-variant-numeric: tabular-nums; color: #F4F7FB;
}
div[data-testid="stMetricDelta"] { font-size: 12.5px; }

/* ── Bordered containers → cards ───────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, rgba(22,32,58,.40), rgba(16,24,43,.40));
    border: 1px solid rgba(148,163,184,.10) !important;
    border-radius: 16px !important;
    transition: border-color .2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(148,163,184,.20) !important;
}

/* ── Dataframes / tables ───────────────────────────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 14px;
    overflow: hidden;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px;
    font-weight: 600; font-size: 14px;
    border: 1px solid rgba(148,163,184,.18);
    background: rgba(22,32,58,.6);
    color: #F4F7FB;
    transition: background .18s ease, border-color .18s ease, transform .12s ease;
}
.stButton > button:hover {
    background: rgba(30,42,70,.8); border-color: rgba(233,196,106,.35);
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E9C46A, #C9A24B);
    color: #1A1407; border: none; font-weight: 700;
    padding: 0.62rem 1.5rem;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #F2D98C, #E9C46A);
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] { font-size: 13.5px; font-weight: 600; color: #8B98AF; }
button[data-baseweb="tab"][aria-selected="true"] { color: #F4F7FB; }
div[data-baseweb="tab-highlight"] { background-color: #E9C46A; }
div[data-baseweb="tab-border"] { background-color: rgba(148,163,184,.10); }

/* ── Inputs ────────────────────────────────────────────────────────────── */
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
    background-color: rgba(16,24,43,.8) !important;
    border-radius: 10px !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #E9C46A; border-color: #E9C46A;
}
.stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] ~ div > div {
    background: #E9C46A;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 12px;
    background: rgba(16,24,43,.5);
}

/* ── Plotly chart frame ────────────────────────────────────────────────── */
div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(148,163,184,.08);
    border-radius: 16px;
    padding: 10px 12px 12px;
    background: linear-gradient(180deg, rgba(22,32,58,.28), rgba(16,24,43,.28));
}

/* Sidebar caption — explicit colour so it clears WCAG over the surface */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #8B98AF !important;
}

/* ═══════════════════════════════ Custom components ═════════════════════ */

/* Page header */
.wc-header { margin-bottom: 26px; }
.wc-header .eyebrow {
    font-size: 11px; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #E9C46A; margin-bottom: 12px;
}
.wc-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px; font-weight: 600; margin: 0 0 10px;
    letter-spacing: -0.03em; line-height: 1.12; color: #F4F7FB;
}
.wc-header.hero h1 { font-size: 44px; }
.wc-header p { color: #AEB9CC; font-size: 15px; margin: 0; max-width: 600px; line-height: 1.6; }

/* KPI grid */
.kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;
    margin: 4px 0 10px;
}
.kpi-card {
    background: linear-gradient(180deg, rgba(22,32,58,.55), rgba(16,24,43,.55));
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 16px;
    padding: 20px 22px 18px;
    transition: border-color .2s ease, transform .2s ease;
}
.kpi-card:hover { border-color: rgba(233,196,106,.28); transform: translateY(-2px); }
.kpi-label {
    font-size: 11px; font-weight: 600; letter-spacing: .11em;
    text-transform: uppercase; color: #8B98AF; margin-bottom: 12px;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px; font-weight: 600; color: #F4F7FB;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
    display: flex; align-items: center; gap: 10px;
}
.kpi-value img { width: 30px; height: 20px; border-radius: 3px; object-fit: cover;
    box-shadow: 0 1px 4px rgba(0,0,0,.4); }
.kpi-delta { font-size: 12.5px; font-weight: 500; margin-top: 8px; color: #8B98AF; }
.kpi-delta.up      { color: #34D399; }
.kpi-delta.accent  { color: #7DB0FF; }
.kpi-delta.gold    { color: #E9C46A; }

/* Section headers */
.wc-section { margin: 46px 0 18px; }
.wc-section .kicker {
    font-size: 11px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
    color: #E9C46A; margin-bottom: 8px;
}
.wc-section h2 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px; font-weight: 600; margin: 0; letter-spacing: -0.02em; color: #F4F7FB;
}
.wc-section p { color: #8B97AD; font-size: 14px; margin: 6px 0 0; max-width: 560px; line-height: 1.55; }

/* ── Broadcast contender board (image flags) ───────────────────────────── */
.cb { display: flex; flex-direction: column; gap: 8px; }
.cb-row {
    display: grid;
    grid-template-columns: 30px 34px 1fr 116px;
    align-items: center; gap: 16px;
    background: linear-gradient(180deg, rgba(22,32,58,.5), rgba(16,24,43,.5));
    border: 1px solid rgba(148,163,184,.09);
    border-radius: 14px;
    padding: 14px 18px;
    transition: border-color .2s ease, transform .18s ease, background .2s ease;
}
.cb-row:hover {
    border-color: rgba(148,163,184,.22); transform: translateX(3px);
}
.cb-row.lead {
    border-color: rgba(233,196,106,.45);
    background: linear-gradient(180deg, rgba(233,196,106,.10), rgba(16,24,43,.55));
}
.cb-pos {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: #8B98AF; text-align: center;
    font-variant-numeric: tabular-nums;
}
.cb-row.lead .cb-pos { color: #E9C46A; }
.cb-flag { width: 34px; height: 23px; border-radius: 4px; object-fit: cover;
    box-shadow: 0 2px 6px rgba(0,0,0,.45); }
.cb-mid { min-width: 0; }
.cb-name { font-size: 15px; font-weight: 600; color: #F4F7FB; margin-bottom: 8px;
    display: flex; align-items: baseline; gap: 10px; }
.cb-name .meta { font-size: 11.5px; font-weight: 500; color: #8B98AF;
    font-variant-numeric: tabular-nums; }
.cb-track { height: 6px; border-radius: 99px; background: rgba(148,163,184,.10); overflow: hidden; }
.cb-fill { height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #4C8DFF, #7DB0FF);
    transform-origin: left center;
    animation: growBar 1.1s cubic-bezier(.2,.7,.2,1) both; }
.cb-row.lead .cb-fill { background: linear-gradient(90deg, #C9A24B, #F2D98C); }
.cb-right { text-align: right; }
.cb-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 21px; font-weight: 600; color: #F4F7FB; font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
}
.cb-row.lead .cb-pct { color: #E9C46A; }
.cb-delta { font-size: 11.5px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }
.cb-delta.up { color: #34D399; }
.cb-delta.down { color: #F87171; }
.cb-delta.flat { color: #8B98AF; }

/* ── Methodology / engine cards ────────────────────────────────────────── */
.eng-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.eng-card {
    background: linear-gradient(180deg, rgba(22,32,58,.5), rgba(16,24,43,.5));
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 16px;
    padding: 22px 22px;
    transition: border-color .2s ease, transform .2s ease;
}
.eng-card:hover { border-color: rgba(76,141,255,.30); transform: translateY(-2px); }
.eng-card .num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 34px; font-weight: 600; color: #F4F7FB;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1;
}
.eng-card .num .u { font-size: 16px; color: #8B98AF; font-weight: 500; margin-left: 4px; }
.eng-card .lab { font-size: 13px; font-weight: 600; color: #AEB9CC; margin-top: 12px; }
.eng-card .desc { font-size: 12.5px; color: #8B98AF; margin-top: 6px; line-height: 1.5; }

/* ── Pipeline chips (ensemble models) ──────────────────────────────────── */
.pipe { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 4px; }
.pipe .chip {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(22,32,58,.6); border: 1px solid rgba(148,163,184,.14);
    border-radius: 99px; padding: 8px 16px;
    font-size: 13px; font-weight: 600; color: #D7DEEB;
}
.pipe .chip .dot { width: 7px; height: 7px; border-radius: 50%; background: #4C8DFF; }
.pipe .arrow { color: #3C4763; font-size: 14px; }

/* ── Weight / contribution board (no flags — model blend, drivers) ──────── */
.wb { display: flex; flex-direction: column; gap: 8px; }
.wb-row {
    display: grid; grid-template-columns: 1fr 92px; align-items: center; gap: 18px;
    background: linear-gradient(180deg, rgba(22,32,58,.5), rgba(16,24,43,.5));
    border: 1px solid rgba(148,163,184,.09);
    border-radius: 14px; padding: 15px 18px;
    transition: border-color .2s ease, transform .18s ease;
}
.wb-row:hover { border-color: rgba(148,163,184,.22); transform: translateX(3px); }
.wb-row.lead {
    border-color: rgba(233,196,106,.42);
    background: linear-gradient(180deg, rgba(233,196,106,.08), rgba(16,24,43,.55));
}
.wb-mid { min-width: 0; }
.wb-name { font-size: 14.5px; font-weight: 600; color: #F4F7FB; }
.wb-sub { font-size: 12px; color: #8B98AF; margin-top: 4px; line-height: 1.45; }
.wb-track { height: 6px; border-radius: 99px; background: rgba(148,163,184,.10);
    overflow: hidden; margin-top: 11px; }
.wb-fill { height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #4C8DFF, #7DB0FF);
    transform-origin: left center;
    animation: growBar 1.1s cubic-bezier(.2,.7,.2,1) both; }
.wb-row.lead .wb-fill { background: linear-gradient(90deg, #C9A24B, #F2D98C); }
.wb-pct { font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 600;
    color: #F4F7FB; text-align: right; font-variant-numeric: tabular-nums; letter-spacing: -0.01em; }
.wb-row.lead .wb-pct { color: #E9C46A; }

/* ── Versus banner (broadcast matchup header) ──────────────────────────── */
.vs-wrap { container-type: inline-size; }
.vs {
    display: grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr);
    align-items: center; gap: 18px;
    background: linear-gradient(180deg, rgba(22,32,58,.5), rgba(16,24,43,.5));
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 18px; padding: 22px 26px;
}
.vs-side { display: flex; align-items: center; gap: 15px; min-width: 0; }
.vs-side.right { flex-direction: row-reverse; }
.vs-side.right .vs-text { text-align: right; }
.vs-text { min-width: 0; }
.vs-side img { width: 56px; height: 38px; border-radius: 6px; object-fit: cover;
    box-shadow: 0 4px 12px rgba(0,0,0,.5); flex-shrink: 0; }
.vs-name { font-family: 'Space Grotesk', sans-serif; font-size: 23px; font-weight: 600;
    color: #F4F7FB; letter-spacing: -0.02em; line-height: 1.1;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vs-sub { font-size: 12px; color: #8B98AF; margin-top: 4px; font-variant-numeric: tabular-nums;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.vs-mid { text-align: center; }
.vs-mid .lab { font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 600;
    color: #8B98AF; letter-spacing: .10em; }
.vs-mid .venue { font-size: 10px; color: #8B98AF; margin-top: 5px;
    text-transform: uppercase; letter-spacing: .12em; white-space: nowrap; }
@container (max-width: 540px) {
    .vs { padding: 16px 16px; gap: 10px; }
    .vs-side { gap: 10px; }
    .vs-side img { width: 38px; height: 26px; }
    .vs-name { font-size: 16px; }
    .vs-mid .lab { font-size: 13px; }
    .vs-mid .venue { font-size: 9px; letter-spacing: .08em; }
}

/* Match Predictor "VS" divider — aligns with the selectbox inputs on desktop,
   tightens to the stacked teams on mobile (Streamlit stacks columns < 640px). */
.mp-vs { text-align: center; margin-top: 38px; font-size: 13px; font-weight: 700;
    letter-spacing: .10em; color: #8B98AF; }
@media (max-width: 640px) { .mp-vs { margin-top: 4px; margin-bottom: 4px; } }

/* Ranking rows (legacy API — kept for other pages) */
.rank-list { display: flex; flex-direction: column; gap: 8px; }
.rank-card {
    display: flex; align-items: center; gap: 14px;
    background: rgba(22,32,58,.5);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 12px; padding: 12px 16px;
    transition: border-color .18s ease, background .18s ease;
}
.rank-card:hover { border-color: rgba(148,163,184,.22); background: rgba(26,36,60,.7); }
.rank-pos { min-width: 24px; text-align: center; font-weight: 600; font-size: 13px;
    color: #8B98AF; font-variant-numeric: tabular-nums; }
.rank-flag { font-size: 21px; line-height: 1; }
.rank-body { flex: 1; min-width: 0; }
.rank-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.rank-team { font-weight: 600; font-size: 14px; color: #F4F7FB; }
.rank-prob { font-weight: 700; font-size: 14.5px; color: #F4F7FB; font-variant-numeric: tabular-nums; }
.rank-bar { height: 4px; border-radius: 99px; background: rgba(148,163,184,.12); overflow: hidden; }
.rank-fill { height: 100%; border-radius: 99px; background: #4C8DFF; }
.rank-meta { font-size: 11.5px; color: #8B98AF; margin-top: 6px; font-variant-numeric: tabular-nums; }

/* Group cards */
.group-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 13px; }
@media (max-width: 1100px) { .group-grid { grid-template-columns: repeat(2, 1fr); } }
.group-card {
    background: rgba(22,32,58,.45);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 14px; padding: 16px 18px;
    transition: border-color .2s ease;
}
.group-card:hover { border-color: rgba(148,163,184,.22); }
.group-title { font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: #8B98AF; margin-bottom: 12px; }
.group-row { display: flex; align-items: center; gap: 9px; padding: 5px 0; }
.group-row img { width: 22px; height: 15px; border-radius: 3px; object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,.4); }
.group-row .t { flex: 1; font-size: 13px; color: #C2CBDA; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; }
.group-row .p { font-size: 12px; font-weight: 600; color: #8B98AF; font-variant-numeric: tabular-nums; }
.group-row .bar { width: 50px; height: 3px; border-radius: 99px; background: rgba(148,163,184,.12); }
.group-row .bar > div { height: 100%; border-radius: 99px; background: #4C8DFF; }

/* Team probability cards */
.team-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
@media (max-width: 1100px) { .team-grid { grid-template-columns: repeat(2, 1fr); } }
.team-card {
    background: rgba(22,32,58,.5);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 16px; padding: 20px 22px;
    transition: border-color .2s ease;
}
.team-card:hover { border-color: rgba(148,163,184,.22); }
.team-card .head { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
.team-card .head .flag { font-size: 22px; line-height: 1; }
.team-card .head .name { font-size: 14.5px; font-weight: 600; color: #F4F7FB; }
.team-card .main-label { font-size: 11px; font-weight: 600; letter-spacing: .10em;
    text-transform: uppercase; color: #8B98AF; }
.team-card .main-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px; font-weight: 600; color: #F4F7FB;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin: 2px 0 14px; }
.team-card .sub { display: flex; justify-content: space-between; align-items: baseline;
    padding: 6px 0; border-top: 1px solid rgba(148,163,184,.08); }
.team-card .sub .l { font-size: 12px; color: #8B98AF; }
.team-card .sub .v { font-size: 12.5px; font-weight: 600; color: #C2CBDA;
    font-variant-numeric: tabular-nums; }

/* Outcome probability bar */
.outcome-wrap { margin: 8px 0 4px; }
.outcome-bar { display: flex; height: 11px; border-radius: 99px; overflow: hidden;
    background: rgba(148,163,184,.12); }
.outcome-bar .h { background: #4C8DFF; }
.outcome-bar .d { background: #475569; }
.outcome-bar .a { background: #94A3B8; }
.outcome-legend { display: flex; justify-content: space-between; font-size: 12px;
    color: #8B98AF; margin-top: 8px; font-variant-numeric: tabular-nums; }
.outcome-legend b { color: #F4F7FB; font-weight: 600; }

/* Sidebar widgets */
.sb-logo { display: flex; align-items: center; gap: 12px; padding: 2px 4px 18px; }
.sb-logo .mark {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #E9C46A, #C9A24B);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px; font-weight: 700; color: #1A1407; letter-spacing: .01em;
}
.sb-logo .name { font-family: 'Space Grotesk', sans-serif; font-size: 14.5px;
    font-weight: 600; color: #F4F7FB; line-height: 1.2; }
.sb-logo .sub { font-size: 10px; color: #8B98AF; letter-spacing: .12em; text-transform: uppercase; }

.sb-status { display: flex; align-items: center; gap: 8px; padding: 6px 4px 2px; }
.sb-status .dot { width: 7px; height: 7px; border-radius: 50%; background: #34D399;
    box-shadow: 0 0 0 0 rgba(52,211,153,.5); animation: pulse 2.2s infinite; }
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(52,211,153,.45); }
    70%  { box-shadow: 0 0 0 7px rgba(52,211,153,0); }
    100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}
.sb-status .txt { font-size: 12px; font-weight: 500; color: #C2CBDA; }
.sb-status .txt span { color: #8B98AF; font-weight: 400; }

.sb-fav { display: flex; align-items: center; gap: 10px; padding: 5px 4px; }
.sb-fav img { width: 22px; height: 15px; border-radius: 3px; object-fit: cover;
    box-shadow: 0 1px 3px rgba(0,0,0,.4); }
.sb-fav .pos { width: 12px; font-size: 12px; font-weight: 600; color: #8B98AF;
    font-variant-numeric: tabular-nums; }
.sb-fav .team { flex: 1; font-size: 13px; font-weight: 500; color: #C2CBDA; }
.sb-fav .pct { font-size: 12.5px; font-weight: 600; color: #E9C46A;
    font-family: 'Space Grotesk', sans-serif; font-variant-numeric: tabular-nums; }
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
