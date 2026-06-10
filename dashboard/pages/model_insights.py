"""
model_insights.py — 🔬 Model internals: SHAP explainability, evaluation, architecture.

Sections
--------
1. SHAP Feature Importance   — mean |SHAP| bar, colour-coded by category
2. SHAP Beeswarm             — distribution of SHAP values per feature per match
3. SHAP Waterfall            — single-match prediction explanation
4. SHAP Dependency Plots     — feature value vs SHAP value (top-5 features)
5. Model Evaluation          — OOS & in-sample metric tables (Task 6.1)
6. Ensemble Architecture     — component descriptions
7. Monte Carlo Simulation Statistics
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from dashboard.utils.data_loader import load_feature_importance, load_mc_results_json
from dashboard.utils.charts import plot_feature_importance, BG, CARD, GOLD

_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble component descriptions
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":    "XGBoost Classifier",
        "icon":    "🌳",
        "role":    "Outcome probabilities (H / D / A)",
        "features": "37 features: ELO, form, xG, H2H, confederation, WC experience",
        "weight":  "Ensemble member (stacked via logistic regression)",
    },
    {
        "name":    "LightGBM Classifier",
        "icon":    "⚡",
        "role":    "Outcome probabilities (H / D / A)",
        "features": "Same 37-feature set as XGBoost",
        "weight":  "Ensemble member (stacked via logistic regression)",
    },
    {
        "name":    "Logistic Regression (meta-learner)",
        "icon":    "📐",
        "role":    "Combines XGBoost + LightGBM predictions",
        "features": "Stacked soft probabilities from both models",
        "weight":  "Final outcome probability output",
    },
    {
        "name":    "Poisson Goal Model",
        "icon":    "⚽",
        "role":    "Expected goals (λ_home, λ_away) → scoreline distribution",
        "features": "Team attack / defence strengths, home advantage, ELO adjustment",
        "weight":  "Used for simulate_scoreline() — independent of classifier",
    },
    {
        "name":    "ELO Rating System",
        "icon":    "📊",
        "role":    "Running strength metric updated after every match",
        "features": "K-factor = 32 (WC) / 20 (competitive) / 10 (friendly); draws treated correctly",
        "weight":  "Input feature for all models + used directly in Poisson model",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Feature category explanations
# ─────────────────────────────────────────────────────────────────────────────
CAT_DESC = {
    "ELO/Ranking":    "ELO ratings, rank, and confederation-level ELO averages.",
    "Expected Goals": "Pre-match expected goal estimates (derived from Dixon-Coles model).",
    "Head-to-Head":   "Historical H2H win rate, goal difference, and match count.",
    "Confederation":  "One-hot encoded confederation membership (UEFA, CONMEBOL, etc.).",
    "Form/Momentum":  "Rolling win rates, form decay weights, and form composite score.",
    "Goals":          "Rolling goals scored/conceded averages (5- and 10-match windows).",
    "Match Context":  "Venue (neutral/home), match importance weight, WC-experience gap.",
}


def render() -> None:
    st.title("🔬 Model Insights")
    st.caption(
        "Understand what drives the predictions: SHAP feature importances, "
        "the ensemble architecture, and how the models were evaluated."
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 1: SHAP Feature Importance
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📌 Feature Importance — SHAP TreeExplainer")
    st.caption(
        "Mean absolute SHAP values averaged over 500 recent competitive matches "
        "and all three outcome classes (Home / Draw / Away). "
        "Colours indicate feature category."
    )

    n_feat = st.slider("Number of features to display", 10, 37, 20, key="mi_nfeat")

    with st.spinner("Computing SHAP values (cached after first run)…"):
        imp_df = load_feature_importance(n_features=n_feat)

    if imp_df is None:
        st.error(
            "Could not compute feature importances. "
            "Ensure the XGBoost model is trained (`python -m src.models.xgboost_model`)."
        )
    else:
        source = imp_df["source"].iloc[0] if "source" in imp_df.columns else "?"
        if source == "SHAP":
            st.success("✅ Using SHAP TreeExplainer values")
        else:
            st.warning("⚠️ SHAP unavailable — showing native XGBoost gain importance instead")

        fig_imp = plot_feature_importance(imp_df)
        st.plotly_chart(fig_imp, use_container_width=True)

        # Category breakdown
        with st.expander("📂 Category breakdown"):
            cat_counts = imp_df.groupby("category")["importance"].agg(["sum", "count"]).reset_index()
            cat_counts.columns = ["Category", "Total Importance", "# Features"]
            cat_counts = cat_counts.sort_values("Total Importance", ascending=False)
            cat_counts["Total Importance"] = cat_counts["Total Importance"].map("{:.5f}".format)
            st.dataframe(cat_counts, use_container_width=True, hide_index=True)

            st.markdown("---")
            for cat, desc in CAT_DESC.items():
                if cat in imp_df["category"].values:
                    st.markdown(f"**{cat}** — {desc}")

        # Raw table
        with st.expander("🗃️ Raw feature importance table"):
            display = imp_df.copy()
            display["importance"] = display["importance"].map("{:.6f}".format)
            st.dataframe(display, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 2: SHAP Deep-Dive (Beeswarm / Waterfall / Dependency)
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🐝 SHAP Deep-Dive — Task 6.2")
    st.caption(
        "Interactive SHAP plots powered by `shap.TreeExplainer` on the XGBoost model. "
        "All plots are computed on 500 competitive matches (OOS 2023-2024 + WC holdout). "
        "Use the tabs below to explore different plot types."
    )

    shap_tab1, shap_tab2, shap_tab3 = st.tabs([
        "🐝 Beeswarm",
        "💧 Waterfall (Single Match)",
        "📈 Dependency Plots",
    ])

    @st.cache_resource(show_spinner="Loading SHAP analyser …")
    def _get_shap_analyser(n_samples: int):
        import sys
        if str(_ROOT) not in sys.path:
            sys.path.insert(0, str(_ROOT))
        from src.evaluation.shap_analysis import SHAPAnalyser
        sa = SHAPAnalyser(n_samples=n_samples)
        sa.compute_shap_values()     # pre-compute & cache on the object
        return sa

    shap_n = st.slider(
        "SHAP sample size (higher = more accurate, slower first load)",
        min_value=100, max_value=800, value=300, step=50,
        key="mi_shap_n",
    )
    shap_class = st.radio(
        "Outcome class for beeswarm / dependency / waterfall",
        options=["H", "D", "A"],
        format_func=lambda c: {"H": "🟢 Home Win", "D": "🟡 Draw", "A": "🔴 Away Win"}[c],
        horizontal=True,
        key="mi_shap_class",
    )

    try:
        sa = _get_shap_analyser(shap_n)

        with shap_tab1:
            st.caption(
                "Each dot = one match.  "
                "X-axis = SHAP value (positive → pushes prediction toward selected outcome).  "
                "Colour = feature value (blue=low → red=high)."
            )
            fig_bee = sa.plot_beeswarm(top_n=20, outcome_class=shap_class, max_points=300)
            st.plotly_chart(fig_bee, use_container_width=True)

        with shap_tab2:
            st.caption(
                "Select two teams to see how each feature individually shifts "
                "the predicted probability from the model's average baseline."
            )
            from dashboard.utils.data_loader import load_wc_groups
            grps = load_wc_groups()
            all_teams = sorted({t for teams in grps.values() for t in teams})

            wf_c1, wf_c2 = st.columns(2)
            with wf_c1:
                wf_home = st.selectbox("Home / Team A", all_teams,
                                       index=all_teams.index("Spain") if "Spain" in all_teams else 0,
                                       key="mi_wf_home")
            with wf_c2:
                wf_away = st.selectbox("Away / Team B", all_teams,
                                       index=all_teams.index("Argentina") if "Argentina" in all_teams else 1,
                                       key="mi_wf_away")

            wf_neutral = st.checkbox("Neutral venue (WC final style)", value=True, key="mi_wf_neutral")

            if wf_home == wf_away:
                st.warning("Please select two different teams.")
            else:
                with st.spinner(f"Computing waterfall for {wf_home} vs {wf_away} …"):
                    try:
                        fig_wf = sa.plot_waterfall(
                            wf_home, wf_away,
                            neutral=wf_neutral,
                            outcome_class=shap_class,
                            top_n=15,
                        )
                        st.plotly_chart(fig_wf, use_container_width=True)
                        class_lbl = {"H": "Home Win", "D": "Draw", "A": "Away Win"}[shap_class]
                        st.caption(
                            f"Prediction: P({class_lbl}) shifted from the model baseline (avg across all matches) "
                            "by the contributions of each individual feature. "
                            "Green bars push P up; red bars push P down."
                        )
                    except Exception as e:
                        st.error(f"Could not compute waterfall: {e}")

        with shap_tab3:
            st.caption(
                "For each of the top-5 features: feature value (x) vs SHAP value (y).  "
                "Colour = a correlated 'interaction' feature. "
                "A positive SHAP value means the feature increases P(selected outcome)."
            )
            n_dep = st.slider("Number of dependency subplots", 3, 6, 5, key="mi_dep_n")
            fig_dep = sa.plot_dependency_grid(top_n=n_dep, outcome_class=shap_class)
            st.plotly_chart(fig_dep, use_container_width=True)

    except Exception as shap_exc:
        st.error(f"SHAP plots could not be generated: {shap_exc}")
        st.caption("Ensure SHAP is installed: `pip install shap`")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Ensemble Architecture
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏗️ Ensemble Architecture")

    for m in MODELS:
        with st.container(border=True):
            c1, c2 = st.columns([1, 6])
            with c1:
                st.markdown(f"<h2 style='text-align:center'>{m['icon']}</h2>",
                            unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{m['name']}**")
                st.caption(f"Role: {m['role']}")
                st.caption(f"Features: {m['features']}")
                st.caption(f"Stacking: {m['weight']}")

    st.markdown(
        """
**Prediction pipeline:**
1. `FeaturePipeline` → 37 normalised features per match
2. XGBoost + LightGBM each produce `[P(H), P(D), P(A)]`
3. Logistic regression meta-learner stacks both outputs → final outcome probabilities
4. Platt scaling (isotonic regression) calibrates to true probabilities
5. Poisson model independently estimates λ_home / λ_away from attack–defence ratings
6. `simulate_scoreline(home, away)` samples from `Poisson(λ_home) × Poisson(λ_away)`
"""
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Model Evaluation Results
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📋 Model Evaluation — Task 6.1")

    eval_path = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "results" / "evaluation_report.json"
    )
    if eval_path.exists():
        import json as _json
        with open(eval_path) as _f:
            eval_report = _json.load(_f)

        # ── OOS tab first (the real numbers), then in-sample for reference ──
        tab_oos, tab_insample = st.tabs([
            "✅ True Out-of-Sample (2023-2024)",
            "⚠️ In-Sample / Reference (WC 2014-2022)",
        ])

        with tab_oos:
            st.info(
                "**1,464 competitive matches** from 2023-2024 "
                "(UEFA Euros, Copa América, Nations League, AFCON, …). "
                "Models were trained on data ≤ 2022 — **these results are genuine**."
            )
            oos = eval_report.get("oos_results", [])
            if oos:
                oos_df = pd.DataFrame([{
                    "Model":       r["model_name"],
                    "Accuracy":    f"{r['accuracy']*100:.1f}%",
                    "Log Loss":    f"{r['log_loss']:.4f}",
                    "Brier":       f"{r['brier_score']:.4f}",
                    "RPS":         f"{r['rps']:.4f}",
                    "RPSS":        f"{r['rps_skill_score']:+.4f}",
                    "LL < 0.85":   "✅" if r["beats_log_loss_target"] else "❌",
                    "RPS < 0.19":  "✅" if r["beats_rps_target"] else "❌",
                } for r in oos])
                st.dataframe(oos_df, use_container_width=True, hide_index=True)

                # Key metrics for ensemble
                ens_oos = next((r for r in oos if r["model_name"] == "ensemble"), None)
                if ens_oos:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Ensemble Accuracy",  f"{ens_oos['accuracy']*100:.1f}%")
                    c2.metric("Ensemble Log Loss",  f"{ens_oos['log_loss']:.4f}",
                              delta="< 0.85 ✅" if ens_oos["beats_log_loss_target"] else "≥ 0.85 ❌",
                              delta_color="normal")
                    c3.metric("Ensemble RPS",       f"{ens_oos['rps']:.4f}",
                              delta="< 0.19 ✅" if ens_oos["beats_rps_target"] else "≥ 0.19 ❌",
                              delta_color="normal")
                    c4.metric("Ensemble RPSS",      f"{ens_oos['rps_skill_score']:+.4f}")
            else:
                st.warning("OOS results not found in report. Re-run evaluation to generate them.")

        with tab_insample:
            st.warning(
                "⚠️ **WC 2014 / 2018 / 2022 matches (192 total) were in the training set.** "
                "LightGBM 76.6% accuracy and similar tree-model numbers are in-sample artifacts "
                "and do **not** reflect real predictive power."
            )
            in_sample = eval_report.get("results", [])
            if in_sample:
                is_df = pd.DataFrame([{
                    "Model":       r["model_name"],
                    "Accuracy":    f"{r['accuracy']*100:.1f}%",
                    "Log Loss":    f"{r['log_loss']:.4f}",
                    "Brier":       f"{r['brier_score']:.4f}",
                    "RPS":         f"{r['rps']:.4f}",
                    "RPSS":        f"{r['rps_skill_score']:+.4f}",
                    "LL < 0.85":   "✅" if r["beats_log_loss_target"] else "❌",
                    "RPS < 0.19":  "✅" if r["beats_rps_target"] else "❌",
                } for r in in_sample])
                st.dataframe(is_df, use_container_width=True, hide_index=True)

                # Per-tournament breakdown
                per_t = eval_report.get("per_tournament", {})
                if per_t:
                    with st.expander("Per-World-Cup breakdown (ensemble)"):
                        rows = []
                        for yr, res_list in sorted(per_t.items()):
                            ens = next((r for r in res_list if r["model_name"] == "ensemble"), None)
                            if ens:
                                rows.append({
                                    "WC Year":    yr,
                                    "Matches":    ens["n_matches"],
                                    "Accuracy":   f"{ens['accuracy']*100:.1f}%",
                                    "Log Loss":   f"{ens['log_loss']:.4f}",
                                    "RPS":        f"{ens['rps']:.4f}",
                                })
                        if rows:
                            st.dataframe(pd.DataFrame(rows),
                                         use_container_width=True, hide_index=True)
    else:
        st.info(
            "Evaluation report not found. Run: "
            "`python3.13 -m src.evaluation.metrics` to generate it."
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Simulation Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📉 Monte Carlo Simulation Statistics")

    try:
        mc_json = load_mc_results_json()
        meta    = mc_json.get("metadata", {})
        va      = mc_json.get("variance_analysis", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulations", f"{meta.get('n_simulations', '?'):,}")
        c2.metric("Teams",       meta.get("n_teams", "?"))
        c3.metric("Stages",      meta.get("n_stages", "?"))
        c4.metric("Generated",   str(mc_json.get("generated_at", "?"))[:10])

        # Convergence table
        analytical = ((va or {}).get("analytical") or {}).get("by_team", {})
        if analytical:
            st.markdown("**Analytical variance (SE / CV) at key sample sizes:**")
            conv_rows = []
            for team_name, v in analytical.items():
                if not isinstance(v, dict):
                    continue
                winner_data = v.get("winner", {})
                p = winner_data.get("estimate", None)
                if p is None:
                    continue
                conv_at = "N/A"
                for n_str in ["1000", "5000", "10000"]:
                    cv = winner_data.get("by_n", {}).get(n_str, {}).get("cv", 1.0)
                    if cv < 0.05:
                        conv_at = f"N={int(n_str):,}"
                        break
                conv_rows.append({
                    "Team":        team_name,
                    "P(Champion)": f"{p*100:.1f}%",
                    "Converges at": conv_at,
                })
            if conv_rows:
                st.dataframe(
                    pd.DataFrame(conv_rows).head(15),
                    use_container_width=True, hide_index=True,
                )
    except Exception as exc:
        st.caption(f"Could not load simulation statistics: {exc}")
