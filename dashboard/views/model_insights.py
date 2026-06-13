"""
model_insights.py — How the model thinks: ensemble architecture, evaluation,
explainability (SHAP) and Monte Carlo convergence.

The "show the work" surface — its job is to convince a technical reviewer the
forecast is rigorous. Every architecture claim here matches the production
prediction path in src/simulation/match_predictor.py.

Sections
--------
1. The Ensemble Blend   — the four-model weighted average + calibration (real weights)
2. Out-of-Sample        — genuine 2023-2024 evaluation metrics
3. What Drives It        — SHAP feature importance + deep-dive plots
4. Simulation           — Monte Carlo statistics + convergence
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import load_feature_importance, load_mc_results_json
from dashboard.utils.charts import plot_feature_importance
from dashboard.utils import theme

_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble — matches MatchPredictor.predict_proba (weighted blend + isotonic cal).
# Weights are the log-loss-optimised values stored in ensemble_calibrated.joblib.
# ─────────────────────────────────────────────────────────────────────────────
ENSEMBLE = [
    {
        "name": "XGBoost",
        "weight": 0.4249,
        "sub": "Gradient-boosted trees over 37 engineered features — Elo, form, "
               "expected goals, head-to-head, confederation, World-Cup experience → P(H / D / A).",
    },
    {
        "name": "Elo ratings",
        "weight": 0.4063,
        "sub": "A running team-strength rating (K = 32 World Cup · 20 competitive · 10 friendly) "
               "mapped to outcome probabilities through the rating difference.",
    },
    {
        "name": "Poisson goal model",
        "weight": 0.1218,
        "sub": "Attack / defence strengths with home advantage produce expected goals (λ), "
               "converted to a scoreline distribution. Also drives knockout scoreline simulation.",
    },
    {
        "name": "LightGBM",
        "weight": 0.0470,
        "sub": "A second gradient-boosting learner on the same 37-feature set, adding "
               "decorrelated signal to the blend.",
    },
]

# Feature category explanations (for the SHAP breakdown).
CAT_DESC = {
    "ELO/Ranking":    "ELO ratings, rank, and confederation-level ELO averages.",
    "Expected Goals": "Pre-match expected goal estimates (derived from a Dixon-Coles model).",
    "Head-to-Head":   "Historical H2H win rate, goal difference, and match count.",
    "Confederation":  "One-hot encoded confederation membership (UEFA, CONMEBOL, …).",
    "Form/Momentum":  "Rolling win rates, form decay weights, and a form composite score.",
    "Goals":          "Rolling goals scored / conceded averages (5- and 10-match windows).",
    "Match Context":  "Venue (neutral / home), match importance, World-Cup-experience gap.",
}


def render() -> None:
    theme.page_header(
        eyebrow="Model Intelligence",
        title="How the Model Thinks",
        subtitle=(
            "The forecast is an ensemble of four models, blended and calibrated, then run "
            "through 10,000 tournament simulations. Here is the architecture, the "
            "out-of-sample evidence that it works, and what drives an individual prediction."
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # 1 · The Ensemble Blend
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(
        "The Ensemble Blend",
        "Each model produces its own home / draw / away probabilities. They are combined as a "
        "weighted average — the weights tuned on 2023 holdout matches to minimise log-loss — "
        "then mapped to calibrated probabilities by an isotonic calibrator.",
        kicker="Architecture",
    )

    lead_w = max(m["weight"] for m in ENSEMBLE)
    theme.weight_board([
        {"name": m["name"], "sub": m["sub"], "pct": m["weight"], "lead": m["weight"] == lead_w}
        for m in ENSEMBLE
    ])

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    theme.model_pipeline([
        "37 features", "XGBoost", "Elo", "Poisson", "LightGBM",
        "Weighted blend", "Isotonic calibration", "Monte Carlo",
    ])

    with st.expander("The prediction pipeline, step by step"):
        st.markdown(
            """
1. **Feature pipeline** builds 37 normalised features for the match.
2. **Four models** each emit `[P(H), P(D), P(A)]` — XGBoost and LightGBM from the feature set,
   Poisson from its expected-goals (λ) scoreline, Elo from the rating difference.
3. **Weighted blend.** The four probability vectors are averaged with log-loss-optimised
   weights — **XGBoost 0.425 · Elo 0.406 · Poisson 0.122 · LightGBM 0.047**.
4. **Isotonic calibration** maps the blended probabilities onto observed frequencies, so a
   stated 30% really happens about 30% of the time.
5. **Scoreline simulation.** Independently, the Poisson model samples
   `Poisson(λ_home) × Poisson(λ_away)` to resolve knockout ties and penalties.
6. **Monte Carlo.** 10,000 full-tournament simulations aggregate these match-level
   probabilities into per-stage and championship odds.
"""
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # 2 · Out-of-Sample Performance
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(
        "Out-of-Sample Performance",
        "Measured on 1,464 competitive matches from 2023-2024 (UEFA Euros, Copa América, "
        "Nations League, AFCON and more) — none of which the models were trained on.",
        kicker="Evaluation",
    )

    eval_path = _ROOT / "results" / "evaluation_report.json"
    if eval_path.exists():
        with open(eval_path) as _f:
            eval_report = json.load(_f)

        oos = eval_report.get("oos_results", [])
        ens_oos = next((r for r in oos if r["model_name"] == "ensemble"), None)
        if ens_oos:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{ens_oos['accuracy']*100:.1f}%")
            c2.metric("Log Loss", f"{ens_oos['log_loss']:.4f}",
                      delta="Beats 0.85 target" if ens_oos["beats_log_loss_target"] else "Misses target",
                      delta_color="normal" if ens_oos["beats_log_loss_target"] else "inverse")
            c3.metric("RPS", f"{ens_oos['rps']:.4f}",
                      delta="Beats 0.19 target" if ens_oos["beats_rps_target"] else "Misses target",
                      delta_color="normal" if ens_oos["beats_rps_target"] else "inverse")
            c4.metric("RPS Skill Score", f"{ens_oos['rps_skill_score']:+.4f}")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        tab_oos, tab_insample = st.tabs([
            "Out-of-sample (2023-2024)",
            "In-sample reference (WC 2014-2022)",
        ])

        def _metric_table(rows: list[dict]) -> pd.DataFrame:
            return pd.DataFrame([{
                "Model":      r["model_name"].title(),
                "Accuracy":   f"{r['accuracy']*100:.1f}%",
                "Log Loss":   f"{r['log_loss']:.4f}",
                "Brier":      f"{r['brier_score']:.4f}",
                "RPS":        f"{r['rps']:.4f}",
                "RPSS":       f"{r['rps_skill_score']:+.4f}",
                "LL < 0.85":  "Pass" if r["beats_log_loss_target"] else "Miss",
                "RPS < 0.19": "Pass" if r["beats_rps_target"] else "Miss",
            } for r in rows])

        with tab_oos:
            if oos:
                st.dataframe(_metric_table(oos), use_container_width=True, hide_index=True)
                st.caption(
                    "Trained on data ≤ 2022, evaluated on 2023-2024 — genuine predictive skill. "
                    "RPSS > 0 means the model beats a naive base-rate forecast."
                )
            else:
                st.warning("OOS results not found in report. Re-run evaluation to generate them.")

        with tab_insample:
            st.warning(
                "World Cup 2014 / 2018 / 2022 matches (192 total) were in the training set. "
                "Tree-model numbers here are in-sample artifacts and do not reflect real "
                "predictive power — shown only for reference."
            )
            in_sample = eval_report.get("results", [])
            if in_sample:
                st.dataframe(_metric_table(in_sample), use_container_width=True, hide_index=True)

                per_t = eval_report.get("per_tournament", {})
                if per_t:
                    with st.expander("Per-World-Cup breakdown (ensemble)"):
                        rows = []
                        for yr, res_list in sorted(per_t.items()):
                            ens = next((r for r in res_list if r["model_name"] == "ensemble"), None)
                            if ens:
                                rows.append({
                                    "WC Year":  yr,
                                    "Matches":  ens["n_matches"],
                                    "Accuracy": f"{ens['accuracy']*100:.1f}%",
                                    "Log Loss": f"{ens['log_loss']:.4f}",
                                    "RPS":      f"{ens['rps']:.4f}",
                                })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Evaluation report not found. Run `python3 -m src.evaluation.metrics` to generate it.")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3 · What Drives a Prediction (SHAP)
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(
        "What Drives a Prediction",
        "Mean absolute SHAP values across 500 recent competitive matches and all three outcome "
        "classes, coloured by feature family. SHAP attributes each prediction back to the "
        "features that moved it.",
        kicker="Explainability",
    )

    n_feat = st.slider("Features shown", 10, 37, 20, key="mi_nfeat")
    with st.spinner("Computing SHAP values (cached after first run)…"):
        imp_df = load_feature_importance(n_features=n_feat)

    if imp_df is None:
        st.error(
            "Could not compute feature importances. "
            "Ensure the XGBoost model is trained (`python -m src.models.xgboost_model`)."
        )
    else:
        source = imp_df["source"].iloc[0] if "source" in imp_df.columns else "?"
        st.caption(
            "Source: SHAP TreeExplainer values"
            if source == "SHAP" else "Source: native XGBoost gain importance"
        )
        st.plotly_chart(plot_feature_importance(imp_df), use_container_width=True,
                        config={"displayModeBar": False})

        with st.expander("Feature-family breakdown"):
            cat_counts = imp_df.groupby("category")["importance"].agg(["sum", "count"]).reset_index()
            cat_counts.columns = ["Family", "Total importance", "# Features"]
            cat_counts = cat_counts.sort_values("Total importance", ascending=False)
            cat_counts["Total importance"] = cat_counts["Total importance"].map("{:.5f}".format)
            st.dataframe(cat_counts, use_container_width=True, hide_index=True)
            st.markdown("---")
            for cat, desc in CAT_DESC.items():
                if cat in imp_df["category"].values:
                    st.markdown(f"**{cat}** — {desc}")

    # ── SHAP deep-dive ─────────────────────────────────────────────────────────
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("SHAP deep-dive — beeswarm, single-match waterfall, dependency plots"):
        @st.cache_resource(show_spinner="Loading SHAP analyser …")
        def _get_shap_analyser(n_samples: int):
            import sys
            if str(_ROOT) not in sys.path:
                sys.path.insert(0, str(_ROOT))
            from src.evaluation.shap_analysis import SHAPAnalyser
            sa = SHAPAnalyser(n_samples=n_samples)
            sa.compute_shap_values()
            return sa

        shap_n = st.slider(
            "SHAP sample size (higher = more accurate, slower first load)",
            100, 800, 300, 50, key="mi_shap_n",
        )
        shap_class = st.radio(
            "Outcome class",
            options=["H", "D", "A"],
            format_func=lambda c: {"H": "Home Win", "D": "Draw", "A": "Away Win"}[c],
            horizontal=True, key="mi_shap_class",
        )

        try:
            sa = _get_shap_analyser(shap_n)
            t1, t2, t3 = st.tabs(["Beeswarm", "Waterfall (single match)", "Dependency"])

            with t1:
                st.caption(
                    "Each dot is one match. X = SHAP value (positive pushes toward the selected "
                    "outcome); colour = feature value (low → high)."
                )
                st.plotly_chart(sa.plot_beeswarm(top_n=20, outcome_class=shap_class, max_points=300),
                                use_container_width=True, config={"displayModeBar": False})

            with t2:
                st.caption(
                    "How each feature shifts the predicted probability from the model's baseline "
                    "for a specific matchup."
                )
                from dashboard.utils.data_loader import load_wc_groups
                grps = load_wc_groups()
                all_teams = sorted({t for teams in grps.values() for t in teams})
                wf_c1, wf_c2 = st.columns(2)
                with wf_c1:
                    wf_home = st.selectbox("Team A", all_teams,
                                           index=all_teams.index("Spain") if "Spain" in all_teams else 0,
                                           key="mi_wf_home")
                with wf_c2:
                    wf_away = st.selectbox("Team B", all_teams,
                                           index=all_teams.index("Argentina") if "Argentina" in all_teams else 1,
                                           key="mi_wf_away")
                wf_neutral = st.checkbox("Neutral venue", value=True, key="mi_wf_neutral")
                if wf_home == wf_away:
                    st.warning("Please select two different teams.")
                else:
                    with st.spinner(f"Computing waterfall for {wf_home} vs {wf_away} …"):
                        try:
                            st.plotly_chart(
                                sa.plot_waterfall(wf_home, wf_away, neutral=wf_neutral,
                                                  outcome_class=shap_class, top_n=15),
                                use_container_width=True, config={"displayModeBar": False})
                        except Exception as e:
                            st.error(f"Could not compute waterfall: {e}")

            with t3:
                st.caption("Top features: feature value (x) vs SHAP value (y); colour = a correlated feature.")
                n_dep = st.slider("Dependency subplots", 3, 6, 5, key="mi_dep_n")
                st.plotly_chart(sa.plot_dependency_grid(top_n=n_dep, outcome_class=shap_class),
                                use_container_width=True, config={"displayModeBar": False})

        except ModuleNotFoundError:
            st.caption("SHAP is not installed — deep-dive plots unavailable. Install with `pip install shap`.")
        except Exception as shap_exc:
            st.error(f"SHAP plots could not be generated: {shap_exc}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4 · Simulation Convergence
    # ═══════════════════════════════════════════════════════════════════════════
    theme.section(
        "Simulation Convergence",
        "How stable the championship estimates are as the number of simulated tournaments grows — "
        "a forecast is only as trustworthy as its Monte Carlo convergence.",
        kicker="Monte Carlo",
    )

    try:
        mc_json = load_mc_results_json()
        meta    = mc_json.get("metadata", {})
        va      = mc_json.get("variance_analysis", {})

        theme.engine_cards([
            {"num": f"{meta.get('n_simulations', 0):,}", "label": "Simulations",
             "desc": "Independent full-tournament runs."},
            {"num": str(meta.get("n_teams", 48)), "label": "Teams", "desc": "Every qualified nation."},
            {"num": str(meta.get("n_stages", 7)), "label": "Stages",
             "desc": "Group through to the final."},
            {"num": str(mc_json.get("generated_at", "—"))[:10], "label": "Generated",
             "desc": f"Seed {meta.get('base_seed', 42)} · {meta.get('n_workers', 8)} workers."},
        ])

        analytical = ((va or {}).get("analytical") or {}).get("by_team", {})
        if analytical:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:11px;font-weight:700;letter-spacing:.12em;"
                f"text-transform:uppercase;color:{theme.TEXT_3};margin-bottom:8px'>"
                "Championship estimate convergence (CV &lt; 5%)</div>",
                unsafe_allow_html=True,
            )
            conv_rows = []
            for team_name, v in analytical.items():
                if not isinstance(v, dict):
                    continue
                winner_data = v.get("winner", {})
                p = winner_data.get("estimate", None)
                if p is None:
                    continue
                conv_at = "—"
                for n_str in ["1000", "5000", "10000"]:
                    cv = winner_data.get("by_n", {}).get(n_str, {}).get("cv", 1.0)
                    if cv < 0.05:
                        conv_at = f"N = {int(n_str):,}"
                        break
                conv_rows.append({
                    "Team": team_name,
                    "P(Champion)": f"{p*100:.1f}%",
                    "Stabilises at": conv_at,
                })
            if conv_rows:
                conv_df = (pd.DataFrame(conv_rows)
                           .sort_values("P(Champion)", ascending=False)
                           .head(15))
                st.dataframe(conv_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.caption(f"Could not load simulation statistics: {exc}")
