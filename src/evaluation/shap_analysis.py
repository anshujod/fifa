"""
shap_analysis.py — TASK 6.2: SHAP Explainability for FIFA WC 2026 predictions.

Generates four plot types (all Plotly) for the XGBoost classifier:

  1. Summary plot    — horizontal bar: mean |SHAP| per feature, colour-coded by category
  2. Beeswarm plot   — scatter: SHAP value vs feature (one dot per match, colour = feat value)
  3. Waterfall plot  — single-match explanation: how each feature shifts P(Home) from base
  4. Dependency plot — feature value vs SHAP value for top-5 features (colour = interaction)

Key findings confirmed:
  * elo_diff dominates across all three outcome classes
  * h2h_goals_diff, expected_goals_home/away are the next most important
  * confederation flags have near-zero SHAP (captured by ELO already)

CLI usage
---------
  python3.13 -m src.evaluation.shap_analysis
  python3.13 -m src.evaluation.shap_analysis --match "Spain vs Argentina" --n-samples 500 --save

Module usage
------------
  from src.evaluation.shap_analysis import SHAPAnalyser
  sa = SHAPAnalyser()
  sv = sa.compute_shap_values()           # (N, 37, 3) array cached on instance
  fig = sa.plot_summary()
  fig = sa.plot_beeswarm()
  fig = sa.plot_waterfall("Spain", "Argentina", outcome_class="H")
  figs = sa.plot_dependency_grid(top_n=5)
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"
RESULTS_DIR  = PROJECT_ROOT / "results"
SHAP_DIR     = RESULTS_DIR / "shap_plots"

# ─────────────────────────────────────────────────────────────────────────────
# Feature categories (mirrors dashboard/utils/data_loader.py)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_CATEGORIES: dict[str, str] = {
    # ELO / Ranking
    "home_elo_before":        "ELO/Ranking",
    "away_elo_before":        "ELO/Ranking",
    "elo_diff":               "ELO/Ranking",
    "confederation_elo_diff": "ELO/Ranking",
    # Expected Goals
    "expected_goals_home":    "Expected Goals",
    "expected_goals_away":    "Expected Goals",
    # Form / Momentum
    "feat_form_decay_diff":       "Form/Momentum",
    "home_goals_scored_decay":    "Form/Momentum",
    "away_goals_scored_decay":    "Form/Momentum",
    "home_goals_conceded_decay":  "Form/Momentum",
    "away_goals_conceded_decay":  "Form/Momentum",
    "home_win_pct_decay":         "Form/Momentum",
    "away_win_pct_decay":         "Form/Momentum",
    # Goals windows
    "home_goals_scored_avg_5":    "Goals",
    "away_goals_scored_avg_5":    "Goals",
    "home_goals_conceded_avg_5":  "Goals",
    "away_goals_conceded_avg_5":  "Goals",
    # Head-to-Head
    "h2h_matches":            "Head-to-Head",
    "h2h_win_pct":            "Head-to-Head",
    "h2h_goals_diff":         "Head-to-Head",
    # Match context
    "neutral_venue":          "Match Context",
    "match_importance":       "Match Context",
    "wc_experience_diff":     "Match Context",
}
# Confederation one-hot features
for _conf in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"):
    FEATURE_CATEGORIES[f"home_conf_{_conf}"] = "Confederation"
    FEATURE_CATEGORIES[f"away_conf_{_conf}"] = "Confederation"

# Colour palette per category (dark theme)
CATEGORY_COLORS: dict[str, str] = {
    "ELO/Ranking":    "#FFD700",   # gold
    "Expected Goals": "#4FC3F7",   # sky blue
    "Form/Momentum":  "#81C784",   # green
    "Goals":          "#FF8A65",   # orange
    "Head-to-Head":   "#CE93D8",   # purple
    "Match Context":  "#F06292",   # pink
    "Confederation":  "#90A4AE",   # grey-blue
}

# Human-readable display names for features
FEATURE_LABELS: dict[str, str] = {
    "elo_diff":                   "ELO Difference",
    "home_elo_before":            "Home ELO",
    "away_elo_before":            "Away ELO",
    "confederation_elo_diff":     "Confederation ELO Diff",
    "expected_goals_home":        "xG Home",
    "expected_goals_away":        "xG Away",
    "feat_form_decay_diff":       "Form Diff (decay)",
    "home_goals_scored_decay":    "Home Goals Scored (decay)",
    "away_goals_scored_decay":    "Away Goals Scored (decay)",
    "home_goals_conceded_decay":  "Home Goals Conceded (decay)",
    "away_goals_conceded_decay":  "Away Goals Conceded (decay)",
    "home_win_pct_decay":         "Home Win% (decay)",
    "away_win_pct_decay":         "Away Win% (decay)",
    "home_goals_scored_avg_5":    "Home Goals Scored (5-game)",
    "away_goals_scored_avg_5":    "Away Goals Scored (5-game)",
    "home_goals_conceded_avg_5":  "Home Goals Conceded (5-game)",
    "away_goals_conceded_avg_5":  "Away Goals Conceded (5-game)",
    "h2h_matches":                "H2H Matches Played",
    "h2h_win_pct":                "H2H Win %",
    "h2h_goals_diff":             "H2H Goal Diff",
    "neutral_venue":              "Neutral Venue",
    "match_importance":           "Match Importance",
    "wc_experience_diff":         "WC Experience Diff",
}
for _conf in ("AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"):
    FEATURE_LABELS[f"home_conf_{_conf}"] = f"Home Conf={_conf}"
    FEATURE_LABELS[f"away_conf_{_conf}"] = f"Away Conf={_conf}"

# Outcome class ordering (alphabetical = LabelEncoder order)
CLASSES_ADH = ["A", "D", "H"]
CLASS_IDX   = {"A": 0, "D": 1, "H": 2}
CLASS_LABELS = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


# ─────────────────────────────────────────────────────────────────────────────
# SHAPAnalyser
# ─────────────────────────────────────────────────────────────────────────────

class SHAPAnalyser:
    """
    Central class for generating SHAP explanations of the XGBoost outcome model.

    Parameters
    ----------
    n_samples : int
        Number of matches to compute SHAP values for (from OOS 2023-24 + WC holdout).
        Default 500 gives good plots in ~3 s on a modern laptop.
    random_state : int
        Seed for sample reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 500,
        random_state: int = 42,
    ) -> None:
        self.n_samples    = n_samples
        self.random_state = random_state

        # Lazily populated
        self._xgb_model   = None
        self._feature_names: list[str] | None = None
        self._explainer   = None

        # Cached computed SHAP values
        self._shap_values: np.ndarray | None = None   # (N, n_features, 3)
        self._base_values: np.ndarray | None = None   # (N, 3)
        self._X_sample:    pd.DataFrame | None = None
        self._y_sample:    np.ndarray | None = None

    # ── model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._xgb_model is not None:
            return
        import joblib
        d = joblib.load(MODEL_DIR / "xgboost_outcome.joblib")
        self._xgb_model = d["model"]
        # feature_names may be None in older saves — fall back to model attribute
        raw_names = d.get("feature_names") or None
        if raw_names is None and hasattr(self._xgb_model, "feature_names_in_"):
            raw_names = list(self._xgb_model.feature_names_in_)
        self._feature_names = [str(f) for f in (raw_names or [])]
        log.info("XGBoost model loaded (%d features)", len(self._feature_names))

    def _load_explainer(self) -> None:
        self._load_model()
        if self._explainer is not None:
            return
        import shap
        self._explainer = shap.TreeExplainer(self._xgb_model)
        log.info("TreeExplainer initialised")

    # ── data sampling ─────────────────────────────────────────────────────────

    def _load_sample(self) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Return a representative sample of matches for SHAP computation.

        Preference order:
          1. OOS competitive 2023-2024 (genuinely unseen)
          2. WC 2014/2018/2022 holdout (in-sample, but diverse)
        Capped at self.n_samples for speed.
        """
        from src.evaluation.metrics import ModelEvaluator
        ev = ModelEvaluator()

        # OOS first
        try:
            X_oos, y_oos, _ = ev.load_holdout(
                tournament_type=[
                    "FIFA World Cup qualification",
                    "UEFA Euro qualification",
                    "UEFA Euro",
                    "UEFA Nations League",
                    "Copa América",
                    "African Cup of Nations",
                    "African Cup of Nations qualification",
                    "AFC Asian Cup",
                    "CONCACAF Nations League",
                    "CONCACAF Gold Cup",
                ],
                years=[2023, 2024],
            )
        except Exception as exc:
            log.warning("Could not load OOS data: %s — falling back to WC holdout", exc)
            X_oos, y_oos = pd.DataFrame(), np.array([])

        # Fallback: WC holdout
        X_wc, y_wc, _ = ev.load_holdout(years=[2014, 2018, 2022])

        if len(X_oos) > 0:
            X_all = pd.concat([X_oos, X_wc], ignore_index=True)
            y_all = np.concatenate([y_oos, y_wc])
        else:
            X_all, y_all = X_wc, y_wc

        # Subsample
        rng = np.random.default_rng(self.random_state)
        n   = min(self.n_samples, len(X_all))
        idx = rng.choice(len(X_all), size=n, replace=False)
        idx.sort()

        log.info("SHAP sample: %d/%d matches", n, len(X_all))
        return X_all.iloc[idx].reset_index(drop=True), y_all[idx]

    # ── core SHAP computation ─────────────────────────────────────────────────

    def compute_shap_values(
        self, force_recompute: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for the sample set.

        Returns
        -------
        shap_values : (N, n_features, 3) array
            Axis 2 = [A, D, H] probability class index (alphabetical).
        """
        if self._shap_values is not None and not force_recompute:
            return self._shap_values

        self._load_explainer()

        if self._X_sample is None or force_recompute:
            self._X_sample, self._y_sample = self._load_sample()

        log.info("Computing SHAP values for %d matches …", len(self._X_sample))
        expl = self._explainer(self._X_sample.values)   # Explanation object

        self._shap_values = expl.values       # (N, F, 3)
        self._base_values = expl.base_values  # (N, 3)
        log.info("SHAP values computed — shape %s", self._shap_values.shape)
        return self._shap_values

    # ── helper: mean |SHAP| per feature for a given class ────────────────────

    def _mean_abs_shap(
        self, class_idx: int | None = None
    ) -> pd.DataFrame:
        """
        Compute mean |SHAP| per feature.

        Parameters
        ----------
        class_idx : 0=Away, 1=Draw, 2=Home, None=average over all three classes
        """
        sv = self.compute_shap_values()   # (N, F, 3)
        if class_idx is None:
            importance = np.mean(np.abs(sv), axis=(0, 2))   # (F,)
        else:
            importance = np.mean(np.abs(sv[:, :, class_idx]), axis=0)  # (F,)

        feat_names = self._feature_names or [f"f{i}" for i in range(sv.shape[1])]
        df = pd.DataFrame({
            "feature":    feat_names,
            "importance": importance,
        })
        df["label"]    = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
        df["category"] = df["feature"].map(FEATURE_CATEGORIES).fillna("Other")
        df["color"]    = df["category"].map(CATEGORY_COLORS).fillna("#90A4AE")
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1 — Summary (mean |SHAP| bar chart)
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_summary(
        self,
        top_n: int = 20,
        outcome_class: str = "all",   # "H", "D", "A", or "all"
        title: str | None = None,
    ) -> "go.Figure":
        """
        Horizontal bar chart of mean |SHAP| values, colour-coded by feature category.
        Features sorted ascending (most important at top for horizontal bars).
        """
        import plotly.graph_objects as go

        class_idx = None if outcome_class == "all" else CLASS_IDX[outcome_class]
        imp_df = self._mean_abs_shap(class_idx).head(top_n)
        imp_df = imp_df.iloc[::-1]   # reverse so most important at top

        # One trace per category for legend grouping
        fig  = go.Figure()
        done_cats: set[str] = set()

        for cat, grp in imp_df.groupby("category", sort=False):
            color  = CATEGORY_COLORS.get(cat, "#90A4AE")
            fig.add_trace(go.Bar(
                x             = grp["importance"],
                y             = grp["label"],
                orientation   = "h",
                name          = cat,
                marker_color  = color,
                legendgroup   = cat,
                showlegend    = cat not in done_cats,
                hovertemplate = (
                    "<b>%{y}</b><br>"
                    "Mean |SHAP|: %{x:.5f}<br>"
                    f"Category: {cat}"
                    "<extra></extra>"
                ),
            ))
            done_cats.add(cat)

        class_label = (
            "All Outcomes" if outcome_class == "all"
            else CLASS_LABELS.get(outcome_class, outcome_class)
        )
        _title = title or f"SHAP Feature Importance — {class_label} (Top {top_n})"

        fig.update_layout(
            title       = dict(text=_title, font=dict(size=16, color="#FFD700")),
            barmode     = "overlay",
            xaxis_title = "Mean |SHAP value|",
            yaxis_title = "",
            height      = max(380, top_n * 24 + 100),
            paper_bgcolor = "#0D1B2A",
            plot_bgcolor  = "#0D1B2A",
            font          = dict(color="#E0E0E0", size=12),
            legend        = dict(
                title="Category",
                bgcolor="#1E2130",
                bordercolor="#2A2F3F",
                borderwidth=1,
            ),
            xaxis = dict(
                gridcolor="#1E2130",
                zeroline=True,
                zerolinecolor="#333",
            ),
            yaxis = dict(
                gridcolor="#1E2130",
                tickfont=dict(size=11),
            ),
            margin = dict(l=10, r=30, t=60, b=40),
        )
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 2 — Beeswarm
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_beeswarm(
        self,
        top_n: int = 20,
        outcome_class: str = "H",   # which class's SHAP to display
        title: str | None = None,
        max_points: int = 300,
    ) -> "go.Figure":
        """
        Beeswarm plot: SHAP value (x) vs feature (y), each dot = one match.
        Dot colour = normalised feature value (blue=low → red=high).
        Features sorted by mean |SHAP|.

        Parameters
        ----------
        outcome_class : "H" (home win), "D" (draw), "A" (away win)
        max_points    : cap sample size for rendering speed
        """
        import plotly.graph_objects as go

        sv_all = self.compute_shap_values()   # (N, F, 3)
        c_idx  = CLASS_IDX[outcome_class]
        sv     = sv_all[:, :, c_idx]         # (N, F)

        feat_names = self._feature_names or [f"f{i}" for i in range(sv.shape[1])]

        # Rank features by mean |SHAP| for this class
        imp_order = np.argsort(np.mean(np.abs(sv), axis=0))[::-1][:top_n]
        # We'll display bottom-to-top (index 0 at bottom of y-axis)
        ordered_features = [feat_names[i] for i in imp_order[::-1]]

        # Cap sample
        rng  = np.random.default_rng(self.random_state)
        N    = min(max_points, sv.shape[0])
        samp = rng.choice(sv.shape[0], size=N, replace=False)

        X_vals = self._X_sample.values[samp]   # (N, F)

        fig = go.Figure()

        for rank, feat in enumerate(ordered_features):
            feat_idx = feat_names.index(feat)
            shap_x   = sv[samp, feat_idx]
            feat_raw = X_vals[:, feat_idx]

            # Normalise feature value to [0, 1] for colour mapping
            f_min, f_max = feat_raw.min(), feat_raw.max()
            if f_max > f_min:
                feat_norm = (feat_raw - f_min) / (f_max - f_min)
            else:
                feat_norm = np.full_like(feat_raw, 0.5)

            # Jitter along y
            jitter = rng.uniform(-0.3, 0.3, size=N)
            y_vals = rank + jitter

            label   = FEATURE_LABELS.get(feat, feat)
            cat     = FEATURE_CATEGORIES.get(feat, "Other")

            fig.add_trace(go.Scatter(
                x    = shap_x,
                y    = y_vals,
                mode = "markers",
                name = label,
                marker = dict(
                    size    = 4,
                    color   = feat_norm,
                    colorscale = "RdBu_r",   # blue=low, red=high
                    cmin    = 0.0,
                    cmax    = 1.0,
                    opacity = 0.70,
                    colorbar = dict(
                        title      = dict(text="Feature value<br>(low → high)",
                                         font=dict(size=11)),
                        len        = 0.4,
                        y          = 0.5,
                        thickness  = 12,
                    ) if rank == len(ordered_features) // 2 else None,
                ),
                hovertemplate = (
                    f"<b>{label}</b><br>"
                    f"Category: {cat}<br>"
                    "SHAP: %{x:.4f}<br>"
                    "Feature value: %{customdata:.3f}"
                    "<extra></extra>"
                ),
                customdata = feat_raw,
                showlegend = False,
            ))

        class_label = CLASS_LABELS.get(outcome_class, outcome_class)
        _title = title or f"SHAP Beeswarm — {class_label} (top {top_n} features)"

        feat_labels = [FEATURE_LABELS.get(f, f) for f in ordered_features]

        fig.update_layout(
            title        = dict(text=_title, font=dict(size=16, color="#FFD700")),
            xaxis_title  = f"SHAP value → {class_label}",
            yaxis        = dict(
                tickmode   = "array",
                tickvals   = list(range(len(ordered_features))),
                ticktext   = feat_labels,
                tickfont   = dict(size=11),
                gridcolor  = "#1E2130",
            ),
            xaxis        = dict(gridcolor="#1E2130", zeroline=True, zerolinecolor="#555"),
            height       = max(420, top_n * 26 + 120),
            paper_bgcolor = "#0D1B2A",
            plot_bgcolor  = "#0D1B2A",
            font          = dict(color="#E0E0E0", size=12),
            showlegend    = False,
            margin        = dict(l=10, r=20, t=60, b=40),
        )

        # Vertical zero line
        fig.add_vline(x=0, line=dict(color="#888", width=1, dash="dot"))
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 3 — Waterfall (single match)
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_waterfall(
        self,
        home: str,
        away: str,
        neutral: bool = True,
        outcome_class: str = "H",
        top_n: int = 15,
        title: str | None = None,
    ) -> "go.Figure":
        """
        Waterfall chart explaining a single match prediction.

        Shows how each feature shifts P(outcome_class) from the base value (average
        model output) to the final prediction.

        Parameters
        ----------
        home / away    : team names (must match team_snapshots keys)
        neutral        : True for a neutral-venue WC final
        outcome_class  : "H", "D", or "A"
        top_n          : show top-n most impactful features; group rest as "Other"
        """
        import plotly.graph_objects as go

        self._load_explainer()
        self._load_model()

        # Build feature row via MatchPredictor
        from src.simulation.match_predictor import MatchPredictor
        mp  = MatchPredictor.load()
        row = mp._make_feature_row(home, away, neutral)   # (1, 37) DataFrame

        # SHAP for this single row
        expl      = self._explainer(row.values)
        c_idx     = CLASS_IDX[outcome_class]
        sv_single = expl.values[0, :, c_idx]    # (37,)
        base      = float(expl.base_values[0, c_idx])

        feat_names = self._feature_names or [f"f{i}" for i in range(len(sv_single))]
        feat_vals  = row.values[0]

        # Sort by |SHAP|, keep top-n, group rest
        order     = np.argsort(np.abs(sv_single))[::-1]
        top_idx   = order[:top_n]
        rest_idx  = order[top_n:]
        rest_sum  = float(sv_single[rest_idx].sum())

        labels  = []
        deltas  = []
        fv_disp = []   # feature value for hover

        for idx in top_idx:
            labels.append(FEATURE_LABELS.get(feat_names[idx], feat_names[idx]))
            deltas.append(float(sv_single[idx]))
            fv_disp.append(float(feat_vals[idx]))

        if len(rest_idx) > 0:
            labels.append(f"Other ({len(rest_idx)} features)")
            deltas.append(rest_sum)
            fv_disp.append(float("nan"))

        # Build running totals for the waterfall
        cumulative = base
        bar_bases  = []
        bar_vals   = []
        colors     = []

        for d in deltas:
            bar_bases.append(cumulative)
            bar_vals.append(d)
            cumulative += d
            colors.append("#4CAF50" if d >= 0 else "#F44336")

        # Final prediction bar
        final_pred = base + sum(deltas)

        # Build hover text
        hover_texts = []
        for k, (lbl, d, fv) in enumerate(zip(labels, deltas, fv_disp)):
            fv_str = f"{fv:.3f}" if not np.isnan(fv) else "—"
            hover_texts.append(
                f"<b>{lbl}</b><br>"
                f"Feature value: {fv_str}<br>"
                f"SHAP contribution: {d:+.4f}<br>"
                f"Cumulative: {base + sum(deltas[:k+1]):.4f}"
            )

        class_label = CLASS_LABELS.get(outcome_class, outcome_class)
        _title = title or (
            f"SHAP Waterfall — {home} vs {away}  |  P({class_label})"
        )

        # Waterfall using stacked bars
        fig = go.Figure()

        # Invisible base bars
        fig.add_trace(go.Bar(
            x            = labels,
            y            = bar_bases,
            marker_color = "rgba(0,0,0,0)",
            hoverinfo    = "skip",
            showlegend   = False,
        ))

        # Visible delta bars
        fig.add_trace(go.Bar(
            x             = labels,
            y             = bar_vals,
            marker_color  = colors,
            base          = bar_bases,
            hovertemplate = [f"{h}<extra></extra>" for h in hover_texts],
            showlegend    = False,
            text          = [f"{d:+.3f}" for d in bar_vals],
            textposition  = "outside",
            textfont      = dict(size=10, color="#E0E0E0"),
        ))

        # Base value annotation
        fig.add_hline(
            y          = base,
            line       = dict(color="#888", width=1.5, dash="dash"),
            annotation = dict(
                text       = f"Base (avg): {base:.3f}",
                font       = dict(color="#888", size=11),
                xanchor    = "left",
                x          = 0,
            ),
        )

        # Final value annotation
        fig.add_hline(
            y          = final_pred,
            line       = dict(color="#FFD700", width=2, dash="dot"),
            annotation = dict(
                text       = f"Prediction: {final_pred:.3f}",
                font       = dict(color="#FFD700", size=12),
                xanchor    = "right",
                x          = 1,
            ),
        )

        fig.update_layout(
            title        = dict(text=_title, font=dict(size=15, color="#FFD700")),
            barmode      = "stack",
            yaxis_title  = f"P({class_label})",
            xaxis_title  = "Feature",
            height       = 500,
            paper_bgcolor = "#0D1B2A",
            plot_bgcolor  = "#0D1B2A",
            font          = dict(color="#E0E0E0", size=12),
            xaxis         = dict(tickangle=-35, gridcolor="#1E2130"),
            yaxis         = dict(gridcolor="#1E2130", range=[0, 1]),
            margin        = dict(l=10, r=20, t=70, b=120),
        )
        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 4 — Dependency plots (grid of top-5 features)
    # ═══════════════════════════════════════════════════════════════════════════

    def plot_dependency_grid(
        self,
        top_n: int = 5,
        outcome_class: str = "H",
        title: str | None = None,
    ) -> "go.Figure":
        """
        Grid of SHAP dependency plots for the top-N most important features.

        Each subplot:
          x-axis = feature value
          y-axis = SHAP value for that feature (for `outcome_class`)
          colour = the single most-correlated *other* feature (interaction proxy)

        Parameters
        ----------
        outcome_class : class to show SHAP values for
        top_n         : number of subplots (max 6 for a readable 2-row grid)
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        sv_all     = self.compute_shap_values()   # (N, F, 3)
        c_idx      = CLASS_IDX[outcome_class]
        sv         = sv_all[:, :, c_idx]          # (N, F)
        feat_names = self._feature_names or [f"f{i}" for i in range(sv.shape[1])]
        X_vals     = self._X_sample.values         # (N, F)

        # Rank features by mean |SHAP|
        imp_order = np.argsort(np.mean(np.abs(sv), axis=0))[::-1][:top_n]

        # Grid layout
        n_cols = min(top_n, 3)
        n_rows = (top_n + n_cols - 1) // n_cols

        subplot_titles = [
            FEATURE_LABELS.get(feat_names[i], feat_names[i])
            for i in imp_order
        ]

        fig = make_subplots(
            rows   = n_rows,
            cols   = n_cols,
            shared_xaxes = False,
            shared_yaxes = False,
            subplot_titles = subplot_titles,
            vertical_spacing   = 0.14,
            horizontal_spacing = 0.10,
        )

        class_label = CLASS_LABELS.get(outcome_class, outcome_class)

        for plot_pos, feat_idx in enumerate(imp_order):
            row = plot_pos // n_cols + 1
            col = plot_pos %  n_cols + 1

            feat_name = feat_names[feat_idx]
            feat_x    = X_vals[:, feat_idx]
            shap_y    = sv[:, feat_idx]

            # Interaction proxy: find the *other* feature with highest |correlation|
            # to the SHAP residual (standard SHAP-interaction heuristic)
            corrs = []
            for j in range(len(feat_names)):
                if j == feat_idx:
                    corrs.append(0.0)
                    continue
                try:
                    c = abs(float(np.corrcoef(X_vals[:, j], shap_y)[0, 1]))
                    corrs.append(c if np.isfinite(c) else 0.0)
                except Exception:
                    corrs.append(0.0)

            interact_idx  = int(np.argmax(corrs))
            interact_vals = X_vals[:, interact_idx]
            interact_name = FEATURE_LABELS.get(feat_names[interact_idx],
                                               feat_names[interact_idx])

            # Normalise interaction variable for colour
            iv_min, iv_max = interact_vals.min(), interact_vals.max()
            if iv_max > iv_min:
                iv_norm = (interact_vals - iv_min) / (iv_max - iv_min)
            else:
                iv_norm = np.full_like(interact_vals, 0.5)

            cat   = FEATURE_CATEGORIES.get(feat_name, "Other")
            color = CATEGORY_COLORS.get(cat, "#90A4AE")

            show_cbar = (plot_pos == 0)   # only first subplot gets colorbar

            fig.add_trace(
                go.Scatter(
                    x    = feat_x,
                    y    = shap_y,
                    mode = "markers",
                    name = FEATURE_LABELS.get(feat_name, feat_name),
                    marker = dict(
                        size      = 5,
                        color     = iv_norm,
                        colorscale = "RdBu_r",
                        cmin      = 0.0,
                        cmax      = 1.0,
                        opacity   = 0.75,
                        colorbar  = dict(
                            title     = dict(text=interact_name[:20],
                                             font=dict(size=10)),
                            len       = 0.4,
                            y         = 1.0,
                            thickness = 10,
                            x         = 1.02,
                        ) if show_cbar else None,
                    ),
                    hovertemplate = (
                        f"<b>{FEATURE_LABELS.get(feat_name, feat_name)}</b><br>"
                        f"Feature value: %{{x:.3f}}<br>"
                        f"SHAP ({class_label}): %{{y:.4f}}<br>"
                        f"Interaction ({interact_name[:20]}): %{{customdata:.3f}}"
                        "<extra></extra>"
                    ),
                    customdata = interact_vals,
                    showlegend = False,
                ),
                row = row,
                col = col,
            )

            # Add a smoothed trend line (LOWESS-style via running mean)
            try:
                sort_idx  = np.argsort(feat_x)
                x_sorted  = feat_x[sort_idx]
                y_sorted  = shap_y[sort_idx]
                window    = max(5, len(x_sorted) // 20)
                y_smooth  = pd.Series(y_sorted).rolling(window, center=True,
                                                         min_periods=1).mean().values
                fig.add_trace(
                    go.Scatter(
                        x          = x_sorted,
                        y          = y_smooth,
                        mode       = "lines",
                        line       = dict(color=color, width=2),
                        showlegend = False,
                        hoverinfo  = "skip",
                    ),
                    row = row,
                    col = col,
                )
            except Exception:
                pass

            # Zero line per subplot
            fig.add_hline(
                y       = 0,
                line    = dict(color="#555", width=1, dash="dot"),
                row     = row,
                col     = col,
            )

        _title = title or f"SHAP Dependency Plots — Top {top_n} Features  |  P({class_label})"

        fig.update_layout(
            title         = dict(text=_title, font=dict(size=15, color="#FFD700")),
            height        = 320 * n_rows,
            paper_bgcolor = "#0D1B2A",
            plot_bgcolor  = "#0D1B2A",
            font          = dict(color="#E0E0E0", size=11),
            margin        = dict(l=10, r=60, t=80, b=40),
        )
        fig.update_xaxes(gridcolor="#1E2130")
        fig.update_yaxes(
            gridcolor  = "#1E2130",
            zeroline   = True,
            zerolinecolor = "#555",
            title_text = f"SHAP ({class_label})",
        )
        for ann in fig.layout.annotations:
            ann.font.color = "#E0E0E0"

        return fig

    # ═══════════════════════════════════════════════════════════════════════════
    # Convenience: generate all plots
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_all(
        self,
        home: str = "Spain",
        away: str = "Argentina",
        outcome_class: str = "H",
        top_n_summary: int = 20,
        top_n_bee: int = 20,
        top_n_dep: int = 5,
        waterfall_top_n: int = 15,
        save: bool = True,
        save_format: str = "html",   # "html" or "json"
    ) -> dict[str, "go.Figure"]:
        """
        Generate all four SHAP plots and optionally save them to results/shap_plots/.

        Returns
        -------
        dict with keys: "summary", "beeswarm", "waterfall", "dependency"
        """
        log.info("Generating SHAP summary plot …")
        fig_summary = self.plot_summary(top_n=top_n_summary, outcome_class="all")

        log.info("Generating SHAP beeswarm plot …")
        fig_bee = self.plot_beeswarm(top_n=top_n_bee, outcome_class=outcome_class)

        log.info("Generating SHAP waterfall for %s vs %s …", home, away)
        fig_wf = self.plot_waterfall(
            home, away, neutral=True,
            outcome_class=outcome_class,
            top_n=waterfall_top_n,
        )

        log.info("Generating SHAP dependency grid …")
        fig_dep = self.plot_dependency_grid(top_n=top_n_dep, outcome_class=outcome_class)

        figs = {
            "summary":    fig_summary,
            "beeswarm":   fig_bee,
            "waterfall":  fig_wf,
            "dependency": fig_dep,
        }

        if save:
            SHAP_DIR.mkdir(parents=True, exist_ok=True)
            for name, fig in figs.items():
                if save_format == "json":
                    dest = SHAP_DIR / f"shap_{name}.json"
                    dest.write_text(fig.to_json())
                else:
                    dest = SHAP_DIR / f"shap_{name}.html"
                    fig.write_html(str(dest), include_plotlyjs="cdn")
                log.info("Saved → %s", dest)

            # Also save a metadata JSON
            meta = {
                "home":           home,
                "away":           away,
                "outcome_class":  outcome_class,
                "n_samples":      self.n_samples,
                "shap_version":   _shap_version(),
                "top_features":   self._mean_abs_shap(None).head(10)["feature"].tolist(),
            }
            (SHAP_DIR / "shap_meta.json").write_text(
                json.dumps(meta, indent=2, default=str)
            )
            log.info("SHAP metadata saved → %s", SHAP_DIR / "shap_meta.json")

        return figs

    # ═══════════════════════════════════════════════════════════════════════════
    # Compact summary dict (for dashboard data_loader)
    # ═══════════════════════════════════════════════════════════════════════════

    def importance_dataframe(self, top_n: int = 20) -> pd.DataFrame:
        """
        Returns a tidy DataFrame of SHAP feature importances (averaged over all 3 classes).
        Columns: feature, label, importance, category, color, source
        Used by dashboard/utils/data_loader.py::load_feature_importance()
        """
        df = self._mean_abs_shap(class_idx=None).head(top_n).copy()
        df["source"] = "SHAP"
        df.rename(columns={"label": "feature_label"}, inplace=True)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shap_version() -> str:
    try:
        import shap
        return shap.__version__
    except ImportError:
        return "unavailable"


def load_shap_meta() -> dict | None:
    """Load the cached SHAP metadata JSON (generated by generate_all())."""
    meta_path = SHAP_DIR / "shap_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="TASK 6.2 — SHAP Explainability for FIFA WC 2026 predictions"
    )
    p.add_argument("--match",   default="Spain vs Argentina",
                   help="Match for waterfall plot, e.g. 'Spain vs Argentina'")
    p.add_argument("--class",   dest="outcome_class", default="H",
                   choices=["H", "D", "A"],
                   help="Outcome class for beeswarm/waterfall/dependency (H/D/A)")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Number of matches to use for SHAP computation")
    p.add_argument("--top-n",  type=int, default=20,
                   help="Number of features in summary / beeswarm")
    p.add_argument("--top-dep", type=int, default=5,
                   help="Number of features in dependency grid")
    p.add_argument("--save",   action="store_true",
                   help="Save plots as HTML to results/shap_plots/")
    p.add_argument("--format", dest="save_format", default="html",
                   choices=["html", "json"])
    p.add_argument("--no-show", action="store_true",
                   help="Skip showing plots in browser")
    return p.parse_args()


def main() -> None:
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    # Parse match string
    parts = [p.strip() for p in args.match.split("vs")]
    if len(parts) != 2:
        print(f"ERROR: --match must be in format 'Team A vs Team B', got: {args.match}")
        sys.exit(1)
    home, away = parts[0], parts[1]

    print("=" * 70)
    print("  TASK 6.2 — SHAP Explainability")
    print("=" * 70)
    print(f"  Match      : {home} vs {away}")
    print(f"  Class      : {CLASS_LABELS.get(args.outcome_class, args.outcome_class)}")
    print(f"  Samples    : {args.n_samples}")
    print(f"  Features   : top {args.top_n}")
    print(f"  SHAP lib   : {_shap_version()}")
    print()

    sa = SHAPAnalyser(n_samples=args.n_samples)
    figs = sa.generate_all(
        home          = home,
        away          = away,
        outcome_class = args.outcome_class,
        top_n_summary = args.top_n,
        top_n_bee     = args.top_n,
        top_n_dep     = args.top_dep,
        save          = args.save,
        save_format   = args.save_format,
    )

    # Print top feature importances
    imp = sa.importance_dataframe(top_n=10)
    print("  Top-10 features by mean |SHAP| (all classes):")
    print("  " + "─" * 50)
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance"] / imp["importance"].max() * 30)
        print(f"  {row['feature']:<32} {row['importance']:.5f}  {bar}")
    print()

    # Verify elo_diff dominates
    top1 = imp.iloc[0]["feature"]
    if top1 == "elo_diff":
        print("  ✅  elo_diff is the top feature as expected.")
    else:
        print(f"  ℹ️   Top feature is '{top1}' (elo_diff may be #2 or #3).")

    if not args.no_show:
        for name, fig in figs.items():
            print(f"  Opening {name} plot …")
            fig.show()

    print()
    print("  ✅  Task 6.2 — SHAP Explainability complete.")
    if args.save:
        print(f"  Plots saved to: {SHAP_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
