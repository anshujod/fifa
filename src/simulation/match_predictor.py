from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.encoder import MATCH_FEATURES_CSV, FeaturePipeline
from src.models.calibration import CalibratedEnsemble, IsotonicCalibrator, PlattCalibrator
from src.models.elo_model import EloModel
from src.models.ensemble import (
    CLASSES, START_YEAR, VAL_YEARS,
    WeightedEnsemble, _DataBundle,
)
from src.models.poisson_model import PoissonGoalModel
from src.models.xgboost_model import XGBoostOutcomeModel

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR    = PROJECT_ROOT / "models" / "saved"

# Neutral venue for all WC 2026 matches (hosted in USA/Canada/Mexico)
WC_NEUTRAL   = True
WC_IMPORTANCE = 1.0    # World Cup match importance weight


class MatchPredictor:
    """
    Predicts match outcomes and simulates scorelines for any two teams.

    Builds a synthetic feature row by combining:
        - Team-level rolling stats from the latest available snapshot
        - Elo ratings
        - H2H statistics
        - Confederation context
    Then passes through the calibrated ensemble for outcome probs,
    and the Poisson model for scoreline simulation.
    """

    def __init__(self):
        self.team_snapshots: dict[str, dict] = {}
        self.h2h_lookup: dict[tuple[str, str], dict] = {}
        self.conf_elo: dict[str, float] = {}
        self.global_home_goals_avg: float = 1.5

        # When True, simulate_scoreline uses direct Poisson sampling
        # (no Dixon-Coles correction, ~200× faster — set by Monte Carlo runner)
        self.fast_mode: bool = False

        self.xgb_model: XGBoostOutcomeModel | None = None
        self.poi_model:  PoissonGoalModel | None     = None
        self.elo_model:  EloModel | None             = None
        self.lgbm_data:  dict | None                 = None
        self.calibrator  = None
        self.ensemble_weights: dict[str, float] = {}
        self.pipeline_features: list[str] = []

    # ─── Build team snapshots from match_features.csv ────────────────────────

    def _build_snapshots(self, mf: pd.DataFrame) -> None:
        """
        For each team, extract the most recent feature values as a snapshot dict.
        Home-perspective features are preferred; away used as fallback.
        """
        played = mf[mf["is_played"] == True].sort_values("date")

        feature_pairs = [
            ("goals_scored_decay",      "home_goals_scored_decay",      "away_goals_scored_decay"),
            ("goals_conceded_decay",     "home_goals_conceded_decay",    "away_goals_conceded_decay"),
            ("win_pct_decay",            "home_win_pct_decay",           "away_win_pct_decay"),
            ("form_score",               "home_form_score",              "away_form_score"),
            ("goals_scored_avg_5",       "home_goals_scored_avg_5",      "away_goals_scored_avg_5"),
            ("goals_conceded_avg_5",     "home_goals_conceded_avg_5",    "away_goals_conceded_avg_5"),
            ("wc_appearances",           "home_wc_appearances",          "away_wc_appearances"),
            ("confederation",            "home_confederation",           "away_confederation"),
            ("elo",                      "home_elo_before",              "away_elo_before"),
            ("matches_played",           "home_matches_played",          "away_matches_played"),
            ("days_since_last_match",    "home_days_since_last_match",   "away_days_since_last_match"),
        ]

        # Get all teams
        all_teams = set(played["home_team"].unique()) | set(played["away_team"].unique())

        for team in all_teams:
            as_home = played[played["home_team"] == team]
            as_away = played[played["away_team"] == team]

            snap: dict = {}
            # Prefer most-recent appearance
            if len(as_home) > 0 and (len(as_away) == 0 or
                    as_home.iloc[-1]["date"] >= as_away.iloc[-1]["date"]):
                row = as_home.iloc[-1]
                for key, hcol, _ in feature_pairs:
                    snap[key] = row.get(hcol, np.nan)
            elif len(as_away) > 0:
                row = as_away.iloc[-1]
                for key, _, acol in feature_pairs:
                    snap[key] = row.get(acol, np.nan)
            else:
                continue

            self.team_snapshots[team] = snap

        # Confederation average Elo
        conf_elos: dict[str, list] = {}
        for snap in self.team_snapshots.values():
            c = snap.get("confederation", "OTHER")
            conf_elos.setdefault(c, []).append(snap.get("elo", 1500))
        self.conf_elo = {c: float(np.mean(v)) for c, v in conf_elos.items()}

        # Global home goals avg (last 3000 played)
        recent = played.tail(3000)
        self.global_home_goals_avg = float(recent["home_goals"].mean())

        log.info("Built snapshots for %d teams", len(self.team_snapshots))

    def _build_h2h(self, mf: pd.DataFrame) -> None:
        """Pre-compute H2H stats for every team pair."""
        played = mf[mf["is_played"] == True].sort_values("date")
        for _, row in played.iterrows():
            h, a = row["home_team"], row["away_team"]
            key = tuple(sorted([h, a]))
            if key not in self.h2h_lookup:
                self.h2h_lookup[key] = {"matches": 0, "h_wins": 0, "h_gd": 0, "h_gf": 0}
            rec = self.h2h_lookup[key]
            rec["matches"] += 1
            rec["h_gf"]    += int(row["home_goals"]) if h < a else int(row["away_goals"])
            gd = int(row["home_goals"]) - int(row["away_goals"])
            rec["h_gd"]    += gd if h < a else -gd
            if row.get("result") == "H":
                rec["h_wins"] += 1 if h < a else 0
            elif row.get("result") == "A":
                rec["h_wins"] += 0 if h < a else 1
        log.info("Pre-computed H2H for %d team pairs", len(self.h2h_lookup))

    # ─── Feature row construction ─────────────────────────────────────────────

    def _get_snap(self, team: str) -> dict:
        """Return team snapshot, falling back to league averages if team unknown."""
        if team in self.team_snapshots:
            return self.team_snapshots[team]
        log.warning("No snapshot for %s — using global averages", team)
        return {
            "elo": 1500, "goals_scored_decay": 1.3, "goals_conceded_decay": 1.3,
            "win_pct_decay": 0.35, "form_score": 1.5, "goals_scored_avg_5": 1.3,
            "goals_conceded_avg_5": 1.3, "wc_appearances": 5, "confederation": "OTHER",
            "matches_played": 50, "days_since_last_match": 10,
        }

    def _make_feature_row(
        self, home: str, away: str, neutral: bool = WC_NEUTRAL
    ) -> pd.DataFrame:
        """
        Construct a single-row DataFrame matching the FeaturePipeline output
        (the 37-feature vector used by XGBoost and LightGBM).
        """
        hs = self._get_snap(home)
        as_ = self._get_snap(away)

        elo_h  = hs.get("elo", 1500)
        elo_a  = as_.get("elo", 1500)
        elo_diff = elo_h - elo_a

        conf_h = hs.get("confederation", "OTHER")
        conf_a = as_.get("confederation", "OTHER")
        conf_elo_diff = self.conf_elo.get(conf_h, 1500) - self.conf_elo.get(conf_a, 1500)

        # Attack/defence strengths
        gl_avg = max(self.global_home_goals_avg, 0.1)
        h_atk  = hs.get("goals_scored_decay",  1.3) / gl_avg
        a_atk  = as_.get("goals_scored_decay",  1.3) / gl_avg
        h_def  = hs.get("goals_conceded_decay", 1.3) / gl_avg
        a_def  = as_.get("goals_conceded_decay", 1.3) / gl_avg

        base_xg = gl_avg if neutral else self.global_home_goals_avg
        xg_h = round(h_atk * a_def * base_xg, 3)
        xg_a = round(a_atk * h_def * base_xg, 3)

        form_diff = hs.get("form_score", 1.5) - as_.get("form_score", 1.5)

        # H2H
        key  = tuple(sorted([home, away]))
        h2h  = self.h2h_lookup.get(key, {})
        n    = h2h.get("matches", 0)
        is_home_first = home < away
        h2h_wp  = h2h.get("h_wins", 0) / max(n, 1)
        h2h_wp  = h2h_wp if is_home_first else 1 - h2h_wp
        h2h_gd  = h2h.get("h_gd", 0) if is_home_first else -h2h.get("h_gd", 0)
        h2h_gd  = h2h_gd / max(n, 1)

        # One-hot confederations (matching FeaturePipeline categories)
        conf_cats  = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"]
        home_conf_ohe = {f"home_conf_{c}": int(conf_h == c) for c in conf_cats}
        away_conf_ohe = {f"away_conf_{c}": int(conf_a == c) for c in conf_cats}

        row = {
            # Elo
            "home_elo_before":            elo_h,
            "away_elo_before":            elo_a,
            "elo_diff":                   elo_diff,
            "confederation_elo_diff":     conf_elo_diff,
            # Poisson
            "home_attack_strength":       round(h_atk, 3),
            "away_attack_strength":       round(a_atk, 3),
            "home_defence_weakness":      round(h_def, 3),
            "away_defence_weakness":      round(a_def, 3),
            "expected_goals_home":        xg_h,
            "expected_goals_away":        xg_a,
            # Rolling form
            "home_form_score_decay":      hs.get("win_pct_decay", 0.4),
            "away_form_score_decay":      as_.get("win_pct_decay", 0.4),
            "feat_form_decay_diff":       hs.get("win_pct_decay", 0.4) - as_.get("win_pct_decay", 0.4),
            "home_goals_scored_decay":    hs.get("goals_scored_decay", 1.3),
            "away_goals_scored_decay":    as_.get("goals_scored_decay", 1.3),
            "home_goals_conceded_decay":  hs.get("goals_conceded_decay", 1.3),
            "away_goals_conceded_decay":  as_.get("goals_conceded_decay", 1.3),
            "home_win_pct_decay":         hs.get("win_pct_decay", 0.4),
            "away_win_pct_decay":         as_.get("win_pct_decay", 0.4),
            # Fixed-window
            "home_goals_scored_avg_5":    hs.get("goals_scored_avg_5", 1.3),
            "away_goals_scored_avg_5":    as_.get("goals_scored_avg_5", 1.3),
            "home_goals_conceded_avg_5":  hs.get("goals_conceded_avg_5", 1.3),
            "away_goals_conceded_avg_5":  as_.get("goals_conceded_avg_5", 1.3),
            # H2H
            "h2h_matches":                n,
            "h2h_win_pct":                round(h2h_wp * 100, 1),
            "h2h_goals_diff":             round(h2h_gd, 2),
            # Context
            "neutral_venue":              int(neutral),
            "match_importance":           WC_IMPORTANCE,
            "home_wc_experience_norm":    0.0,  # dropped (r>0.95) — keep as 0
            "away_wc_experience_norm":    0.0,
            "wc_experience_diff":         hs.get("wc_appearances", 5) - as_.get("wc_appearances", 5),
            **home_conf_ohe,
            **away_conf_ohe,
        }

        # Keep only features the pipeline actually uses (in the right order)
        avail = {k: v for k, v in row.items() if k in self.pipeline_features}
        # Fill any missing features with 0
        full_row = {f: avail.get(f, 0.0) for f in self.pipeline_features}
        return pd.DataFrame([full_row])

    def _make_poisson_row(self, home: str, away: str, neutral: bool = WC_NEUTRAL) -> pd.DataFrame:
        """Build a feature row matching PoissonGoalModel.feature_names."""
        hs  = self._get_snap(home)
        as_ = self._get_snap(away)
        gl_avg = max(self.global_home_goals_avg, 0.1)
        h_atk = hs.get("goals_scored_decay",  1.3) / gl_avg
        a_atk = as_.get("goals_scored_decay",  1.3) / gl_avg
        h_def = hs.get("goals_conceded_decay", 1.3) / gl_avg
        a_def = as_.get("goals_conceded_decay", 1.3) / gl_avg
        base_xg = gl_avg if neutral else self.global_home_goals_avg
        xg_h = round(h_atk * a_def * base_xg, 3)
        xg_a = round(a_atk * h_def * base_xg, 3)

        row = {
            "home_goals_scored_decay":   hs.get("goals_scored_decay",  1.3),
            "away_goals_scored_decay":   as_.get("goals_scored_decay",  1.3),
            "home_goals_conceded_decay": hs.get("goals_conceded_decay", 1.3),
            "away_goals_conceded_decay": as_.get("goals_conceded_decay", 1.3),
            "home_goals_scored_avg_5":   hs.get("goals_scored_avg_5",   1.3),
            "away_goals_scored_avg_5":   as_.get("goals_scored_avg_5",   1.3),
            "home_goals_conceded_avg_5": hs.get("goals_conceded_avg_5",  1.3),
            "away_goals_conceded_avg_5": as_.get("goals_conceded_avg_5",  1.3),
            "home_elo_before":           hs.get("elo", 1500),
            "away_elo_before":           as_.get("elo", 1500),
            "elo_diff":                  hs.get("elo", 1500) - as_.get("elo", 1500),
            "neutral_venue":             int(neutral),
            "match_importance":          WC_IMPORTANCE,
            "expected_goals_home":       xg_h,
            "expected_goals_away":       xg_a,
        }
        full = {f: row.get(f, 0.0) for f in self.poi_model.feature_names}
        return pd.DataFrame([full])

    # ─── Prediction API ───────────────────────────────────────────────────────

    def predict_proba(
        self, home: str, away: str, neutral: bool = WC_NEUTRAL
    ) -> tuple[float, float, float]:
        """
        Returns (p_home_win, p_draw, p_away_win) using the calibrated ensemble.
        """
        feat_row   = self._make_feature_row(home, away, neutral)
        elo_diff   = feat_row["elo_diff"].iloc[0]

        # Collect per-model probas
        xgb_p  = self.xgb_model.predict_proba(feat_row)[CLASSES].values[0]

        lgbm_clf = self.lgbm_data["model"]
        lgbm_le  = self.lgbm_data["label_encoder"]
        lgbm_raw = lgbm_clf.predict_proba(feat_row)
        lgbm_p   = pd.DataFrame(lgbm_raw, columns=list(lgbm_le.classes_))[CLASSES].values[0]

        poi_row  = self._make_poisson_row(home, away, neutral)
        poi_raw  = self.poi_model.predict_proba(poi_row)[CLASSES].values[0]

        elo_raw  = self.elo_model.predict_proba(pd.DataFrame([{"elo_diff": elo_diff}]))
        elo_p    = pd.DataFrame(elo_raw, columns=list(self.elo_model.calibrator.classes_))[CLASSES].values[0]

        # Weighted blend
        w = self.ensemble_weights
        blended = (w["elo"] * elo_p + w["xgb"] * xgb_p +
                   w["poisson"] * poi_raw + w["lgbm"] * lgbm_p)
        blended = np.abs(blended) / np.abs(blended).sum()

        # Calibrate
        cal_proba = self.calibrator.predict_proba(blended.reshape(1, -1))[0]
        cal_proba = np.clip(cal_proba, 1e-7, 1.0)
        cal_proba /= cal_proba.sum()

        idx = {c: i for i, c in enumerate(CLASSES)}
        return float(cal_proba[idx["H"]]), float(cal_proba[idx["D"]]), float(cal_proba[idx["A"]])

    def simulate_scoreline(
        self, home: str, away: str,
        neutral: bool = WC_NEUTRAL,
        rng: Optional[np.random.Generator] = None,
        fast: bool = False,
    ) -> tuple[int, int]:
        """
        Sample a (home_goals, away_goals) scoreline from the Poisson model.

        fast=True  — direct Poisson sampling, skips Dixon-Coles correction.
                     ~200× faster; appropriate for large Monte Carlo runs.
        fast=False — full DC-corrected score-matrix sampling (default).
        """
        poi_row = self._make_poisson_row(home, away, neutral)
        lam_h, lam_a = self.poi_model.predict_lambda(poi_row)
        return self.poi_model.simulate_scoreline(
            float(lam_h[0]), float(lam_a[0]), rng=rng,
            fast=fast or self.fast_mode,
        )

    def predict_lambdas(
        self, home: str, away: str,
        neutral: bool = WC_NEUTRAL,
    ) -> tuple[float, float]:
        """
        Return (λ_home, λ_away) from the Poisson model.
        Used by the knockout simulator to scale rates for extra time.
        """
        poi_row = self._make_poisson_row(home, away, neutral)
        lam_h, lam_a = self.poi_model.predict_lambda(poi_row)
        return float(lam_h[0]), float(lam_a[0])

    def simulate_scoreline_fast(
        self, home: str, away: str,
        neutral: bool = WC_NEUTRAL,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[int, int]:
        """
        Fastest scoreline sampler: reuses cached λs and skips DC correction.
        Same as simulate_scoreline(..., fast=True) but avoids re-parsing args.
        """
        poi_row = self._make_poisson_row(home, away, neutral)
        lam_h, lam_a = self.poi_model.predict_lambda(poi_row)
        _rng = rng if rng is not None else np.random.default_rng()
        return int(_rng.poisson(lam_h[0])), int(_rng.poisson(lam_a[0]))

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        self.xgb_model  = XGBoostOutcomeModel.load()

        self.poi_model  = PoissonGoalModel.load()

        self.lgbm_data  = joblib.load(MODEL_DIR / "lightgbm.joblib")

        # Elo: fit on all competitive data pre-2023
        mf = pd.read_csv(MATCH_FEATURES_CSV)
        played = mf[mf["is_played"] == True]
        train  = played[(played["year"] < 2023) & played["is_competitive"]]
        self.elo_model = EloModel()
        self.elo_model.fit_calibrator(train)

        # Calibrated ensemble weights + calibrator
        cal_data = joblib.load(MODEL_DIR / "ensemble_calibrated.joblib")
        self.ensemble_weights = cal_data["ensemble_weights"]
        self.calibrator       = cal_data["calibrator"]

        # FeaturePipeline feature list
        pipeline = FeaturePipeline()
        pipeline.prepare_training_data(start_year=2010, competitive_only=True)
        # Reconstruct the feature list the pipeline outputs
        base_feats = [
            "home_elo_before", "away_elo_before", "elo_diff", "confederation_elo_diff",
            "home_attack_strength", "away_attack_strength",
            "home_defence_weakness", "away_defence_weakness",
            "expected_goals_home", "expected_goals_away",
            "home_form_score_decay", "away_form_score_decay", "feat_form_decay_diff",
            "home_goals_scored_decay", "away_goals_scored_decay",
            "home_goals_conceded_decay", "away_goals_conceded_decay",
            "home_win_pct_decay", "away_win_pct_decay",
            "home_goals_scored_avg_5", "away_goals_scored_avg_5",
            "home_goals_conceded_avg_5", "away_goals_conceded_avg_5",
            "h2h_matches", "h2h_win_pct", "h2h_goals_diff",
            "neutral_venue", "match_importance",
            "home_wc_experience_norm", "away_wc_experience_norm", "wc_experience_diff",
        ]
        conf_cats = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "OTHER", "UEFA"]
        cat_feats = [f"home_conf_{c}" for c in conf_cats] + [f"away_conf_{c}" for c in conf_cats]
        # use actual XGB feature names as ground truth
        xgb_features = list(self.xgb_model.model.feature_names_in_)
        self.pipeline_features = xgb_features

    @classmethod
    def load(cls) -> "MatchPredictor":
        obj = cls()
        log.info("Loading all component models...")
        obj._load_models()
        log.info("Building team snapshots and H2H lookup...")
        mf = pd.read_csv(MATCH_FEATURES_CSV, parse_dates=["date"])
        obj._build_snapshots(mf)
        obj._build_h2h(mf)
        return obj


# ─── Optional type hint fix ───────────────────────────────────────────────────
from typing import Optional  # noqa: E402 (already imported above in runtime)
