"""
home.py — Cinematic landing experience for the WC 2026 Prediction Lab.

A scroll-driven narrative:
    Hero            — projected champion + headline win probability (count-up)
    Top Contenders  — broadcast ranking board with rectangular flags
    Probability     — interactive landscape of all 48 nations
    Trajectories    — survival probability from group stage to the final
    Simulation      — the engine and ensemble behind the forecast
    Full table      — every team, every stage
    Groups          — all 12 groups at a glance
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from dashboard.utils.data_loader import (
    load_mc_probabilities, load_elo_ratings,
    load_mc_results_json, flag_url,
)
from dashboard.utils.charts import tournament_funnel, probability_landscape
from dashboard.utils import theme

_STAGE_SHORT = {
    "p_group_qualify":  "Group",
    "p_round_of_32":    "R32",
    "p_round_of_16":    "R16",
    "p_quarter_final":  "QF",
    "p_semi_final":     "SF",
    "p_final":          "Final",
    "p_winner":         "Champion",
}


# ─────────────────────────────────────────────────────────────────────────────
# Hero — self-contained HTML/CSS/JS component (count-up animation, flags)
# ─────────────────────────────────────────────────────────────────────────────

_HERO_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body { height:100%; }
  body {
    font-family:'IBM Plex Sans',-apple-system,sans-serif;
    background-color:#0A0C10;
    background-image:
      linear-gradient(rgba(255,255,255,.022) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,.022) 1px, transparent 1px),
      radial-gradient(760px 520px at 92% -8%, rgba(58,201,224,.07), transparent 60%);
    background-size:32px 32px, 32px 32px, 100% 100%;
    color:#E6EDF3; overflow:hidden;
  }
  .mono { font-family:'IBM Plex Mono', ui-monospace, monospace; }
  .wrap { position:relative; padding:10px 4px; min-height:100%;
    display:flex; flex-direction:column; justify-content:center; }
  .topbar { display:flex; justify-content:space-between; align-items:center; margin-bottom:22px;
    font-family:'IBM Plex Mono', ui-monospace, monospace; }
  .live { display:flex; align-items:center; gap:9px; font-size:11px; font-weight:500;
    letter-spacing:.16em; text-transform:uppercase; color:#9BA7B4; }
  .live .dot { width:8px; height:8px; border-radius:50%; background:#3FB950;
    box-shadow:0 0 0 0 rgba(63,185,80,.5); animation:pulse 2.2s infinite; }
  @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(63,185,80,.45)} 70%{box-shadow:0 0 0 8px rgba(63,185,80,0)} 100%{box-shadow:0 0 0 0 rgba(63,185,80,0)} }
  .host { font-size:11px; font-weight:500; letter-spacing:.14em; text-transform:uppercase; color:#6A7480; }

  .main { display:grid; grid-template-columns:1.06fr .94fr; gap:38px; align-items:center; }
  @media (max-width:680px){
    .main{ grid-template-columns:1fr; gap:18px; }
    .sub{ margin-bottom:18px; }
    .sp-num{ font-size:52px; }
  }

  .eyebrow { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:12px; font-weight:500;
    letter-spacing:.18em; text-transform:uppercase; color:#3AC9E0; margin-bottom:16px;
    opacity:0; animation:rise .5s .05s forwards; }
  h1 { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:clamp(30px,3.4vw,42px);
    font-weight:600; line-height:1.08; letter-spacing:-0.01em; margin-bottom:20px;
    opacity:0; animation:rise .55s .12s forwards; }
  h1 .accent { color:#3AC9E0; }
  .sub { font-size:15.5px; line-height:1.62; color:#9BA7B4; max-width:520px; margin-bottom:26px;
    opacity:0; animation:rise .55s .2s forwards; }
  .chips { display:flex; flex-wrap:wrap; gap:9px; opacity:0; animation:rise .55s .28s forwards; }
  .chip { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:12px; font-weight:500;
    color:#9BA7B4; background:#11151C; border:1px solid rgba(255,255,255,.10);
    border-radius:4px; padding:7px 13px; }
  .chip b { color:#E6EDF3; font-weight:600; }

  .spotlight {
    position:relative; border:1px solid rgba(255,255,255,.09); border-radius:6px;
    background:#11151C; padding:26px 28px 24px; opacity:0; animation:rise .6s .34s forwards;
  }
  .spotlight::before { content:""; position:absolute; left:0; top:0; right:0; height:1px;
    background:linear-gradient(90deg, #3AC9E0, transparent 70%); }
  .sp-label { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:11px; font-weight:500;
    letter-spacing:.16em; text-transform:uppercase; color:#3AC9E0; margin-bottom:18px; }
  .sp-label::before { content:"\\25B8  "; }
  .sp-team { display:flex; align-items:center; gap:16px; margin-bottom:22px; }
  .sp-team img { width:60px; height:40px; border-radius:3px; object-fit:cover;
    box-shadow:0 2px 10px rgba(0,0,0,.5); }
  .sp-team .nm { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:26px; font-weight:600;
    letter-spacing:-0.01em; color:#E6EDF3; }
  .sp-team .rk { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:12px; color:#7D8794;
    margin-top:4px; letter-spacing:.02em; }
  .sp-prob { display:flex; align-items:flex-end; gap:14px; margin-bottom:6px; }
  .sp-num { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:68px; font-weight:600;
    line-height:.92; letter-spacing:-0.02em; color:#F4C36B; font-variant-numeric:tabular-nums; }
  .sp-cap { font-family:'IBM Plex Mono', ui-monospace, monospace; font-size:11px; font-weight:500;
    letter-spacing:.14em; text-transform:uppercase; color:#7D8794; padding-bottom:12px; line-height:1.5; }
  .sp-gap { font-size:13px; color:#9BA7B4; padding-top:16px; margin-top:16px;
    border-top:1px solid rgba(255,255,255,.08); line-height:1.55; }
  .sp-gap b { color:#E6EDF3; font-weight:600; font-family:'IBM Plex Mono', ui-monospace, monospace; }

  @keyframes rise { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @media (prefers-reduced-motion: reduce){ *{animation:none!important;opacity:1!important} }
</style></head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="live"><span class="dot"></span> Forecast Engine // Live</div>
      <div class="host">USA · CAN · MEX</div>
    </div>
    <div class="main">
      <div class="intro">
        <div class="eyebrow">FIFA World Cup 2026</div>
        <h1>Who lifts the<br><span class="accent">2026 World Cup?</span></h1>
        <div class="sub">A four-model ensemble — Elo, Poisson, XGBoost, LightGBM — blended, calibrated, and run through 10,000 full-tournament simulations.</div>
        <div class="chips">
          <span class="chip"><b id="c-sims">10,000</b> SIMS</span>
          <span class="chip"><b>4</b> MODELS</span>
          <span class="chip"><b>48</b> NATIONS</span>
        </div>
      </div>
      <div class="spotlight">
        <div class="sp-label">Projected Champion</div>
        <div class="sp-team">
          <img id="sp-flag" src="data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==" alt="">
          <div>
            <div class="nm" id="sp-name">—</div>
            <div class="rk" id="sp-rank">—</div>
          </div>
        </div>
        <div class="sp-prob">
          <div class="sp-num" id="sp-num">0.0%</div>
          <div class="sp-cap">Title<br>probability</div>
        </div>
        <div class="sp-gap" id="sp-gap"></div>
      </div>
    </div>
  </div>
<script>
  const DATA = __DATA__;
  // populate champion
  document.getElementById('sp-flag').src = DATA.champ.flag;
  document.getElementById('sp-name').textContent = DATA.champ.team;
  document.getElementById('sp-rank').textContent = 'Group ' + DATA.champ.group + '  ·  Elo ' + DATA.champ.elo;
  document.getElementById('sp-gap').innerHTML =
    'Leads <b>' + DATA.runner.team + '</b> by <b>' + DATA.gap.toFixed(1) + ' points</b> across ' + DATA.nsims + ' simulated tournaments.';
  document.getElementById('c-sims').textContent = DATA.nsims;
  // count-up
  function animate(el, end, dur, dec, suf){
    const t0=performance.now();
    function tick(now){ let p=Math.min((now-t0)/dur,1); p=1-Math.pow(1-p,3);
      el.textContent=(end*p).toFixed(dec)+suf; if(p<1) requestAnimationFrame(tick); }
    requestAnimationFrame(tick);
  }
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const numEl = document.getElementById('sp-num');
  if (reduce) { numEl.textContent = DATA.champ.prob.toFixed(1)+'%'; }
  else { setTimeout(()=>animate(numEl, DATA.champ.prob, 1700, 1, '%'), 450); }
</script>
</body></html>
"""


def _render_hero(df, elo, n_sims: int) -> None:
    elo_map = dict(zip(elo["team"], elo["elo_rating"]))
    top = df.iloc[0]
    runner = df.iloc[1]
    gap = (top["p_winner"] - runner["p_winner"]) * 100

    contenders = []
    for _, r in df.iloc[1:6].iterrows():
        contenders.append({
            "team": r["team"],
            "prob": float(r["p_winner"]) * 100,
            "flag": flag_url(r["team"], 80),
        })

    data = {
        "champ": {
            "team": top["team"],
            "prob": float(top["p_winner"]) * 100,
            "group": top["group"],
            "elo": int(elo_map.get(top["team"], 0)),
            "flag": flag_url(top["team"], 160),
        },
        "runner": {"team": runner["team"]},
        "gap": gap,
        "nsims": f"{n_sims:,}",
        "contenders": contenders,
    }
    html = _HERO_TEMPLATE.replace("__DATA__", json.dumps(data))
    components.html(html, height=410, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# Page
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    df   = load_mc_probabilities()
    elo  = load_elo_ratings()
    meta = load_mc_results_json().get("metadata", {})

    n       = int(meta.get("n_simulations", 10_000))
    n_fail  = int(meta.get("n_failed", 0))
    elapsed = float(meta.get("elapsed_seconds", 0) or 0)
    sp      = n / elapsed if elapsed > 0 else 0

    elo_map = dict(zip(elo["team"], elo["elo_rating"]))

    # ── Hero ───────────────────────────────────────────────────────────────────
    _render_hero(df, elo, n)

    # ── Top Contenders (broadcast board) ───────────────────────────────────────
    theme.section(
        "The Contenders",
        "Title probability for every leading nation, with each side's edge over an "
        "average World Cup entrant (a 2.1% baseline share of the field).",
        kicker="Championship Race",
    )

    n_show = st.slider("Contenders shown", 6, 24, 12, key="home_n_teams",
                       label_visibility="collapsed")
    baseline = 1 / len(df)
    top_n = df.nlargest(n_show, "p_winner").reset_index(drop=True)
    rows = []
    for i, r in top_n.iterrows():
        edge = (r["p_winner"] - baseline) * 100
        rows.append({
            "pos": i + 1,
            "flag_url": flag_url(r["team"], 80),
            "team": r["team"],
            "prob": float(r["p_winner"]),
            "meta": f"Group {r['group']}  ·  Elo {int(elo_map.get(r['team'], 0))}",
            "delta": f"+{edge:.1f} pts vs field",
            "lead": i == 0,
        })
    theme.contender_board(rows)

    # ── Probability Landscape ──────────────────────────────────────────────────
    theme.section(
        "Probability Landscape",
        "Every one of the 48 qualified nations, sized and shaded by championship "
        "probability. Hover any tile for its exact forecast.",
        kicker="The Full Field",
    )
    st.plotly_chart(probability_landscape(df), use_container_width=True,
                    config={"displayModeBar": False})

    # ── Tournament Trajectories ────────────────────────────────────────────────
    theme.section(
        "Survival Curves",
        "How each side's probability of still being alive decays from the group "
        "stage through to the final.",
        kicker="Tournament Trajectories",
    )
    all_teams = sorted(df["team"].tolist())
    defaults = df.nlargest(5, "p_winner")["team"].tolist()
    selected = st.multiselect(
        "Teams to compare", options=all_teams, default=defaults,
        key="home_compare_teams", label_visibility="collapsed",
    )
    if selected:
        st.plotly_chart(tournament_funnel(df, selected), use_container_width=True,
                        config={"displayModeBar": False})
    else:
        st.caption("Pick one or more teams above to plot their survival curves.")

    # ── Simulation Engine ──────────────────────────────────────────────────────
    theme.section(
        "Inside the Engine",
        "Each forecast is the aggregate of thousands of independent tournaments. "
        "Every match is resolved by an ensemble, every knockout tie carried through "
        "extra time and penalties, and the bracket replayed end to end.",
        kicker="Simulation Methodology",
    )
    theme.engine_cards([
        {"num": f"{n:,}", "label": "Tournaments simulated",
         "desc": "Independent end-to-end runs of the full 104-match bracket."},
        {"num": "4", "label": "Models in the ensemble",
         "desc": "Poisson goals, XGBoost, LightGBM and Elo, blended per match."},
        {"num": f"{(n - n_fail) / n * 100:.1f}", "unit": "%", "label": "Clean completion",
         "desc": f"{n - n_fail:,} of {n:,} simulations resolved without error."},
        {"num": f"{sp:.0f}", "unit": "/s", "label": "Tournaments per second",
         "desc": f"Parallelised across {meta.get('n_workers', 8)} workers, seed {meta.get('base_seed', 42)}."},
    ])
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    theme.model_pipeline([
        "Elo ratings", "Poisson goal model", "XGBoost", "LightGBM",
        "Monte Carlo bracket",
    ])
    st.markdown(
        f"<div style='margin-top:14px;font-size:13.5px;color:{theme.TEXT_2}'>"
        f"See the calibration curves, out-of-sample accuracy and feature importance on "
        f"<a href='?page=Model+Insights' target='_self' "
        f"style='color:{theme.ACCENT};text-decoration:none;font-weight:600'>Model Insights →</a></div>",
        unsafe_allow_html=True,
    )

    # ── Drill-downs ────────────────────────────────────────────────────────────
    # The hero, contenders and landscape already carry the headline. These two
    # views answer narrower follow-up questions ("the per-stage breakdown", "the
    # draw"), so they live behind progressive disclosure to keep the scroll tight.
    theme.section(
        "Go Deeper",
        "The full per-stage matrix and the group draw, for when you want the detail "
        "behind the headline forecast.",
        kicker="Full Forecast",
    )

    with st.expander("Every team, every stage — the complete probability matrix", expanded=False):
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

    with st.expander("Groups at a glance — all 12 groups", expanded=False):
        try:
            from src.simulation.group_stage import WC2026_GROUPS
            group_data: dict[str, list[dict]] = {}
            for gid in sorted(WC2026_GROUPS.keys()):
                teams_in_g = WC2026_GROUPS[gid]
                group_df = (
                    df[df["team"].isin(teams_in_g)].sort_values("p_winner", ascending=False)
                )
                group_data[gid] = [
                    {"flag_url": flag_url(r["team"], 40), "team": r["team"],
                     "prob": float(r["p_winner"])}
                    for _, r in group_df.iterrows()
                ]
            theme.group_cards(group_data)
        except Exception:
            st.caption("Group draw data is unavailable right now.")

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin-top:44px;padding-top:18px;border-top:1px solid "
        f"{theme.BORDER};font-size:12px;color:{theme.TEXT_3}'>"
        f"Ensemble of Poisson goal model · XGBoost · LightGBM · Elo &nbsp;·&nbsp; "
        f"N = {n:,} simulations &nbsp;·&nbsp; Seed = {meta.get('base_seed', 42)} &nbsp;·&nbsp; ",
        unsafe_allow_html=True,
    )
