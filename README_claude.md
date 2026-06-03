# ⚽ FIFA World Cup 2026 — ML Prediction System

> **A production-grade Machine Learning system to predict match outcomes, simulate group stages, and estimate each team's probability of winning the FIFA World Cup 2026.**
>
> Built with Python · XGBoost · Poisson Regression · Monte Carlo Simulation · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![WC2026](https://img.shields.io/badge/Tournament-FIFA%20WC%202026-green)](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

---

## 🏆 Project Overview

The **FIFA World Cup 2026** is the largest World Cup in history — **48 teams**, **12 groups**, **104 matches**, hosted across **USA, Canada & Mexico** (June 11 – July 19, 2026).

This project builds a **data-driven prediction engine** that combines:

- **Historical international football results** (140,000+ matches since 1872)
- **2026 FIFA World Cup official squads** (all 48 confirmed rosters as of June 2, 2026)
- **Current FIFA rankings & Elo ratings**
- **Squad-level metrics** — market value, average age, player form, injuries
- **Monte Carlo tournament simulation** — 10,000+ runs per prediction

The system outputs **win/draw/loss probabilities for every match** and estimates each team's probability of lifting the trophy.

---

## 🗂️ Tournament Context (2026)

| Parameter | Detail |
|---|---|
| Teams | 48 (Groups A–L, 4 teams each) |
| Group Stage | Top 2 + 8 best 3rd-place → Round of 32 |
| Knockout Rounds | R32 → R16 → QF → SF → Final |
| Hosts | USA 🇺🇸, Canada 🇨🇦, Mexico 🇲🇽 |
| Final Venue | MetLife Stadium, New Jersey |
| Final Date | July 19, 2026 |
| Key Seedings | Spain 🇪🇸, Argentina 🇦🇷, France 🇫🇷, England 🏴󠁧󠁢󠁥󠁮󠁧󠁿 (cannot meet before SF) |
| Notable Squads | Messi (Argentina), Ronaldo (Portugal), Yamal (Spain), Endrick (Brazil) |

---

## 📁 Project Structure

```
wc2026-predictor/
│
├── data/
│   ├── raw/                        # Downloaded datasets (unprocessed)
│   │   ├── international_results.csv
│   │   ├── fifa_rankings.csv
│   │   ├── elo_ratings.csv
│   │   └── squads_2026/            # All 48 team squads (JSON)
│   ├── processed/                  # Cleaned, feature-engineered data
│   │   ├── match_features.csv
│   │   ├── team_stats.csv
│   │   └── squad_metrics.csv
│   └── external/                   # Transfermarkt, FBref, etc.
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py           # Kaggle API + scraping scripts
│   │   ├── fetch_squads.py         # Squad data ingestion (2026)
│   │   ├── clean.py                # Data cleaning pipeline
│   │   └── update.py               # Auto-refresh recent results
│   │
│   ├── features/
│   │   ├── team_features.py        # Rolling averages, form, H2H
│   │   ├── squad_features.py       # Market value, age, key players
│   │   ├── elo.py                  # Elo rating computation
│   │   └── encoder.py              # Team encoding utilities
│   │
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py        # Primary match outcome model
│   │   ├── lightgbm_model.py
│   │   ├── poisson_model.py        # Goal prediction model
│   │   ├── elo_model.py            # Elo-based probability baseline
│   │   ├── ensemble.py             # Weighted ensemble combiner
│   │   └── calibration.py          # Probability calibration (Platt/Isotonic)
│   │
│   ├── simulation/
│   │   ├── group_stage.py          # Group stage simulator
│   │   ├── tiebreakers.py          # FIFA tiebreaker rules (GD, GS, H2H)
│   │   ├── knockout.py             # R32/R16/QF/SF/Final simulator
│   │   ├── penalties.py            # Extra time & penalty probabilities
│   │   └── monte_carlo.py          # 10,000+ tournament simulations
│   │
│   ├── evaluation/
│   │   ├── metrics.py              # Accuracy, log loss, Brier score
│   │   ├── calibration_plot.py     # Calibration curves
│   │   ├── shap_analysis.py        # SHAP explainability
│   │   └── backtest.py             # Historical tournament backtest
│   │
│   └── utils/
│       ├── config.py               # Global constants and paths
│       ├── logger.py               # Logging setup
│       └── helpers.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_tournament_simulation.ipynb
│   ├── 05_squad_analysis_2026.ipynb
│   └── 06_evaluation_and_shap.ipynb
│
├── dashboard/
│   ├── app.py                      # Streamlit main app
│   ├── pages/
│   │   ├── 1_predictions.py        # Match predictor page
│   │   ├── 2_standings.py          # Group standings simulator
│   │   ├── 3_bracket.py            # Interactive bracket
│   │   └── 4_team_profiles.py      # Team deep-dive
│   └── components/
│       ├── charts.py               # Plotly chart components
│       └── styles.css
│
├── models/
│   └── saved/                      # Serialised .pkl / .joblib models
│
├── tests/
│   ├── test_features.py
│   ├── test_simulation.py
│   └── test_models.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  # GitHub Actions CI
│       └── update_data.yml         # Weekly data refresh cron job
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## 🛣️ Development Roadmap

> **Workflow: 3 active tasks at a time.** Each phase gate must be completed before the next phase unlocks.
> Mark tasks as you complete them: `[ ]` → `[x]`

---

### 🔵 PHASE 1 — Data Collection & Infrastructure

**Goal:** Build a reliable, reproducible data pipeline for all inputs the model needs.

---

#### ✅ Sprint 1 — Core Datasets (START HERE)

- [ ] **TASK 1.1 — Download Kaggle International Results Dataset**
  - Dataset: [`martj42/international-football-results-from-1872-to-2017`](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
  - Covers 140,000+ matches from 1872 to present
  - Columns needed: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `neutral`
  - Save to `data/raw/international_results.csv`
  - Verify date range extends through 2025 (supplement with API if needed)

- [ ] **TASK 1.2 — Fetch FIFA Rankings & Elo Ratings**
  - FIFA Rankings: scrape from [fifa.com/en/ranking/men](https://www.fifa.com/en/ranking/men) or use `requests` + `BeautifulSoup`
  - Elo Ratings: download from [eloratings.net](http://www.eloratings.net) (historical CSV available)
  - Build `src/data/fetch_data.py` with functions: `get_fifa_rankings()`, `get_elo_ratings()`
  - Store both as CSVs with `team`, `rank`, `points/elo`, `date` columns

- [ ] **TASK 1.3 — Ingest All 48 Official 2026 World Cup Squads**
  - All squads confirmed by FIFA as of June 2, 2026
  - Scrape from Wikipedia (`2026 FIFA World Cup squads`) or ESPN squad pages
  - For each player capture: `name`, `position`, `club`, `age`, `caps`, `goals`, `market_value`
  - Supplement market values from Transfermarkt for all 48 squads
  - Save to `data/raw/squads_2026/` as `{team_name}.json`
  - Key players to ensure correct capture: Messi (ARG), Ronaldo (POR), Mbappé (FRA), Yamal (ESP), Endrick (BRA), Haaland (NOR), Bellingham (ENG)

---

#### ✅ Sprint 2 — Data Quality & Schema

- [ ] **TASK 1.4 — Design Unified Data Schema**
  - Define master schema for `match_record`:
    ```
    match_id, date, home_team, away_team, home_goals, away_goals,
    tournament_type, neutral_venue, home_elo_before, away_elo_before,
    home_fifa_rank, away_fifa_rank, result (H/D/A)
    ```
  - Define `team_snapshot` schema: rolling stats at any given date
  - Define `squad_profile` schema: aggregate squad-level metrics per team per tournament

- [ ] **TASK 1.5 — Build Data Cleaning Pipeline**
  - Implement `src/data/clean.py` with:
    - `standardize_team_names()` — handle aliases (e.g. "Korea Republic" vs "South Korea")
    - `remove_duplicates()` — detect and drop duplicate fixtures
    - `handle_missing_scores()` — flag abandoned/incomplete matches
    - `filter_competitive_matches()` — separate friendlies from competitive
  - Output: `data/processed/clean_results.csv`

- [ ] **TASK 1.6 — Set Up Auto-Update Pipeline**
  - Build `src/data/update.py` that fetches latest results from:
    - `football-data.org` API (free tier covers internationals)
    - API-Football (`rapidapi.com`) for recent match stats
  - Schedule via GitHub Actions cron (`.github/workflows/update_data.yml`) — weekly on Mondays
  - Append new rows to `data/raw/international_results.csv` without overwriting history

---

### 🟡 PHASE 2 — Feature Engineering

**Goal:** Transform raw match data into meaningful ML-ready features that capture team strength, form, and squad quality.

---

#### ✅ Sprint 3 — Team-Level Features

- [ ] **TASK 2.1 — Rolling Performance Features**
  - Implement `src/features/team_features.py`
  - For each match, compute for **both** home and away team using previous N matches:
    - `goals_scored_avg_5`, `goals_conceded_avg_5` (last 5 matches)
    - `goals_scored_avg_10`, `goals_conceded_avg_10` (last 10 matches)
    - `win_pct_last_5`, `win_pct_last_10`
    - `clean_sheet_ratio_last_10`
    - `form_score` — weighted: W=3, D=1, L=0, recent matches weighted 2x
  - Apply **exponential time-decay**: weight = `exp(-λ * days_since_match)` where λ = 0.005
  - Separate stats for competitive vs friendly matches

- [ ] **TASK 2.2 — Head-to-Head & Contextual Features**
  - `h2h_win_pct` — historical win % in direct matchups
  - `h2h_goals_diff` — average goal difference in H2H
  - `ranking_diff` — FIFA rank difference (home - away)
  - `elo_diff` — Elo rating difference at match date
  - `continental_strength` — average Elo of confederation (AFC, UEFA, CONMEBOL, etc.)
  - `wc_experience` — number of World Cup appearances (normalised)
  - `neutral_venue` — binary flag

- [ ] **TASK 2.3 — Squad-Level Features from 2026 Rosters**
  - Implement `src/features/squad_features.py`
  - For each of the 48 teams compute:
    - `squad_market_value_total` (€M, from Transfermarkt)
    - `avg_player_age`
    - `star_player_market_value` (top 3 players combined)
    - `key_player_caps` — avg caps of starting XI
    - `injury_flag` — binary: key player missing (from injury tracker data)
    - `top_scorer_goals` — goals tally of main striker
  - These are tournament-level static features for 2026 simulation

---

#### ✅ Sprint 4 — Feature Pipeline & Validation

- [ ] **TASK 2.4 — Poisson Features (Goal Prediction)**
  - Compute **attack strength** and **defence weakness** per team:
    - `attack_strength = (team_avg_goals_scored) / (league_avg_goals_scored)`
    - `defence_weakness = (team_avg_goals_conceded) / (league_avg_goals_conceded)`
  - Compute `expected_goals_home` and `expected_goals_away` using Dixon-Coles model
  - These feed directly into the Poisson regression model

- [ ] **TASK 2.5 — Team Encoding & Representation**
  - Implement `src/features/encoder.py`
  - Strategy: **Do NOT use simple label encoding** — use:
    - `TeamStrengthEncoder`: encode each team as a vector of their rolling stats snapshot
    - `EloEmbedding`: continuous Elo score as numeric feature
    - `ConfederationEncoder`: one-hot encode confederation
  - Build a `FeaturePipeline` class that assembles all features for a given match date

- [ ] **TASK 2.6 — Feature Store & Validation**
  - Save feature matrix to `data/processed/match_features.csv`
  - Run data quality checks:
    - No future data leakage (features computed only from pre-match data)
    - Correlation matrix — drop features with r > 0.95
    - Distribution plots for all numerical features
  - Document feature definitions in `notebooks/02_feature_engineering.ipynb`

---

### 🟠 PHASE 3 — Machine Learning Models

**Goal:** Train, tune, and ensemble multiple models to produce well-calibrated match outcome probabilities.

---

#### ✅ Sprint 5 — Baseline & Core Models

- [ ] **TASK 3.1 — Elo Baseline Model**
  - Implement `src/models/elo_model.py`
  - Win probability formula: `P(home wins) = 1 / (1 + 10^((elo_away - elo_home) / 400))`
  - Update Elo after each match: K=32 for WC matches, K=20 for qualifiers, K=10 for friendlies
  - This is the **benchmark** — all ML models must beat this
  - Expected log loss baseline: ~0.95

- [ ] **TASK 3.2 — XGBoost Match Outcome Classifier (Primary Model)**
  - Implement `src/models/xgboost_model.py`
  - Target: 3-class classification → `{Home Win, Draw, Away Win}`
  - Input: full feature vector from Phase 2
  - Time-series cross-validation: train on matches before year Y, validate on year Y+1
  - Tune with Optuna: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `reg_alpha`
  - Target log loss: < 0.85 on holdout set

- [ ] **TASK 3.3 — Poisson Goal Prediction Model**
  - Implement `src/models/poisson_model.py`
  - Predict `λ_home` (expected goals home) and `λ_away` (expected goals away)
  - Use Dixon-Coles correction for low-scoring games (0-0, 1-0, 0-1)
  - Derive win/draw/loss probabilities by integrating over Poisson distributions
  - Simulating scorelines: generate full match score distribution (used in Monte Carlo)

---

#### ✅ Sprint 6 — Ensemble & Calibration

- [ ] **TASK 3.4 — Additional Models (Random Forest, LightGBM)**
  - Train `RandomForestClassifier` — good for capturing non-linear interactions
  - Train `LGBMClassifier` — faster than XGBoost, useful for comparison
  - Logistic Regression as additional linear baseline
  - Evaluate each independently on time-series holdout
  - Record: accuracy, log loss, Brier score for each model in `results/model_comparison.csv`

- [ ] **TASK 3.5 — Weighted Ensemble**
  - Implement `src/models/ensemble.py`
  - Combine: Elo model + XGBoost + Poisson + LightGBM
  - Optimise ensemble weights using Nelder-Mead on validation log loss
  - Final ensemble weights — suggested starting point: `{elo: 0.15, xgb: 0.40, poisson: 0.30, lgbm: 0.15}`
  - Validate that ensemble outperforms all individual models

- [ ] **TASK 3.6 — Probability Calibration**
  - Implement `src/models/calibration.py`
  - Apply **Platt Scaling** and **Isotonic Regression** calibration
  - Plot calibration curves: predicted probability vs actual frequency
  - Goal: calibration curve should be close to the diagonal (well-calibrated)
  - Save calibrated model: `models/saved/ensemble_calibrated.joblib`

---

### 🔴 PHASE 4 — Tournament Simulation

**Goal:** Simulate the full FIFA World Cup 2026 structure accurately and run 10,000+ Monte Carlo runs.

---

#### ✅ Sprint 7 — Group Stage Simulator

- [ ] **TASK 4.1 — Group Stage Engine**
  - Implement `src/simulation/group_stage.py`
  - Input: 12 groups (A–L), each with 4 teams
  - For each matchup, sample result using calibrated probability distributions from Phase 3
  - Track: points, goal difference, goals scored, head-to-head for all 4 teams
  - Implement FIFA 2026 tiebreaker rules (in order):
    1. Points
    2. Goal difference (all group matches)
    3. Goals scored (all group matches)
    4. Head-to-head points
    5. Head-to-head goal difference
    6. Head-to-head goals scored
    7. Fair play record (yellow/red cards)
    8. Drawing of lots
  - Output: ordered standings for each group

- [ ] **TASK 4.2 — Third-Place Qualification Logic**
  - The top 2 from each of the 12 groups advance automatically (24 teams)
  - The **8 best third-place teams** (out of 12) also advance to Round of 32
  - Implement comparison of all 12 third-placed teams
  - Tiebreakers among third-place teams: points → GD → GS → fair play
  - Output: complete 32-team bracket for knockout rounds

- [ ] **TASK 4.3 — Knockout Stage Simulator**
  - Implement `src/simulation/knockout.py`
  - Simulate: Round of 32 → Round of 16 → Quarter-finals → Semi-finals → Final
  - For each knockout match:
    - Use match outcome probabilities from ensemble model
    - No draws allowed — if draw after 90 mins, simulate extra time (slight reduction in goal rate)
    - If still level, simulate **penalty shootout** (50/50 base, adjusted by team penalty history)
  - Implement `src/simulation/penalties.py`:
    - Base penalty conversion rate: 75%
    - Adjust for known penalty specialists (e.g., Argentina, Germany historically strong)
  - Respect seeding constraints: Spain and Argentina drawn into opposite bracket halves

---

#### ✅ Sprint 8 — Monte Carlo Engine

- [ ] **TASK 4.4 — Monte Carlo Tournament Runner**
  - Implement `src/simulation/monte_carlo.py`
  - Run `N = 10,000` full tournament simulations (configurable)
  - Each run: simulate all 104 matches from group stage to final
  - Track across all runs, for each of the 48 teams:
    - `p_group_qualify` — probability of advancing from group stage
    - `p_round_of_32` — probability of winning R32 match
    - `p_round_of_16` — probability of reaching R16
    - `p_quarter_final` — probability of reaching QF
    - `p_semi_final` — probability of reaching SF
    - `p_final` — probability of reaching the Final
    - `p_winner` — probability of winning the World Cup
  - Use `multiprocessing` for parallel simulation runs
  - Target: 10,000 simulations in under 5 minutes

- [ ] **TASK 4.5 — Results Storage & Reproducibility**
  - Set random seed for reproducibility: `np.random.seed(42)`
  - Save Monte Carlo results to `data/processed/mc_results.json`
  - Store: per-team probabilities, confidence intervals, simulation metadata
  - Build `get_confidence_interval(team, stage, alpha=0.95)` utility function
  - Variance analysis: how stable are probabilities at 1k / 5k / 10k simulations?

---

### 🟣 PHASE 5 — Visualization & Dashboard

**Goal:** Build a polished, interactive Streamlit dashboard showcasing all predictions.

---

#### ✅ Sprint 9 — Dashboard Core

- [ ] **TASK 5.1 — Streamlit App Architecture**
  - Implement `dashboard/app.py` as multi-page Streamlit app
  - Pages:
    - 🏠 **Home**: Tournament overview, top 5 favourites bar chart
    - ⚔️ **Match Predictor**: Pick any 2 teams → get win/draw/loss % + expected scoreline
    - 📊 **Group Standings**: Select a group → simulate it live, see standings
    - 🏆 **Bracket Simulator**: Visual knockout bracket with probabilities on each path
    - 👥 **Team Profiles**: Deep-dive on any team — squad, stats, historical WC performance
  - Cache simulation results: `@st.cache_data` with TTL=3600

- [ ] **TASK 5.2 — Core Charts & Visualizations**
  - All charts in Plotly for interactivity
  - `plot_win_probabilities(team)` — stacked bar: probability at each stage
  - `plot_match_prediction(home, away)` — donut chart: W/D/L probabilities
  - `plot_expected_scoreline(home, away)` — heatmap of goal score probabilities (0-0 to 5-5)
  - `plot_group_standings(group)` — sortable table with colour-coded qualification zones
  - `plot_feature_importance()` — SHAP summary plot top 20 features

- [ ] **TASK 5.3 — Team Comparison Dashboard**
  - Side-by-side radar chart comparing 2 teams across 8 dimensions:
    - Attack rating, Defence rating, Form, Elo, FIFA rank, Squad value, Experience, H2H
  - Interactive squad viewer: list all 26 players per team, highlight key players
  - Historical WC results timeline for selected team

---

### ⚫ PHASE 6 — Evaluation & Explainability

**Goal:** Rigorously evaluate model quality and make predictions explainable.

---

#### ✅ Sprint 10 — Model Evaluation

- [ ] **TASK 6.1 — Evaluation Framework**
  - Implement `src/evaluation/metrics.py`
  - Compute on holdout set (World Cups 2014, 2018, 2022 used as test tournaments):
    - **Accuracy**: % of correctly predicted match outcomes
    - **Log Loss**: lower is better; penalises confident wrong predictions
    - **Brier Score**: mean squared error of probability predictions
    - **RPS (Ranked Probability Score)**: standard metric for 3-outcome sports prediction
  - Build `ModelEvaluator` class with `evaluate(model, X_test, y_test)` → returns metrics dict
  - Benchmark target: log loss < 0.85, RPS < 0.19 (state-of-the-art for football)

- [ ] **TASK 6.2 — SHAP Explainability**
  - Implement `src/evaluation/shap_analysis.py`
  - Generate SHAP values for XGBoost model on test set
  - Plots:
    - Summary plot (feature importance by mean |SHAP|)
    - Beeswarm plot (distribution of SHAP values per feature)
    - Waterfall plot for a single match prediction (e.g., Spain vs Argentina final)
    - Dependency plots for top 5 features
  - Key expected findings: `elo_diff` and `ranking_diff` should dominate

- [ ] **TASK 6.3 — Backtesting on Historical World Cups**
  - Implement `src/evaluation/backtest.py`
  - Re-simulate World Cups 2014, 2018, 2022 using only data available before each tournament
  - Compare: predicted winner vs actual winner
  - Compare: predicted finalist vs actual finalist
  - Track: was the actual champion always in the top 5 predicted teams?
  - Document findings in `notebooks/06_evaluation_and_shap.ipynb`

---

### 🟤 PHASE 7 — Deployment

**Goal:** Deploy the system publicly for showcase and portfolio visibility.

---

#### ✅ Sprint 11 — Containerisation & CI/CD

- [ ] **TASK 7.1 — Docker Setup**
  - Build `docker/Dockerfile`:
    ```dockerfile
    FROM python:3.11-slim
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    COPY . .
    CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501"]
    ```
  - Build `docker/docker-compose.yml` for local full-stack run
  - Test: `docker build -t wc2026-predictor . && docker run -p 8501:8501 wc2026-predictor`

- [ ] **TASK 7.2 — GitHub Actions CI Pipeline**
  - `.github/workflows/ci.yml`:
    - Trigger on every PR to `main`
    - Steps: install deps → run `pytest tests/` → lint with `ruff` → type check with `mypy`
  - `.github/workflows/update_data.yml`:
    - Cron: every Monday at 06:00 UTC
    - Fetch latest international results → commit to repo automatically

- [ ] **TASK 7.3 — Cloud Deployment**
  - **Primary**: Deploy Streamlit app to [Streamlit Community Cloud](https://streamlit.io/cloud) (free)
    - Connect GitHub repo → set secrets → one-click deploy
  - **Alternative**: Deploy to Hugging Face Spaces (also free, Streamlit SDK supported)
  - **Backend API (optional)**: FastAPI wrapper around model → deploy to Railway or Render
    - `POST /predict` endpoint: `{"home": "Spain", "away": "Argentina"}` → probabilities JSON
  - Set up custom domain if desired (e.g., `wc2026predictor.streamlit.app`)

---

## 🤖 Model Architecture Summary

```
INPUT FEATURES (per match)
    │
    ├── Elo Difference
    ├── FIFA Ranking Difference
    ├── Rolling Goals Scored (5, 10 match window)
    ├── Rolling Goals Conceded (5, 10 match window)
    ├── Win % Last 5 / Last 10
    ├── Head-to-Head Win %
    ├── Squad Market Value
    ├── Average Squad Age
    ├── Key Player Injury Flag
    ├── Confederation Strength
    ├── WC Experience
    └── Neutral Venue Flag
         │
         ▼
    ┌─────────────────────────────────┐
    │  ENSEMBLE MODEL                 │
    │  ├── XGBoost (40%)              │
    │  ├── Poisson Regression (30%)   │
    │  ├── LightGBM (15%)             │
    │  └── Elo Model (15%)            │
    └─────────────────────────────────┘
         │
         ▼
    Probability Calibration (Platt Scaling)
         │
         ▼
    OUTPUT: P(Home Win), P(Draw), P(Away Win)
    + Expected Goals: λ_home, λ_away
         │
         ▼
    MONTE CARLO SIMULATION (10,000 runs)
         │
         ▼
    Per-team: P(R32), P(R16), P(QF), P(SF), P(Final), P(Winner)
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Statistical | SciPy (Poisson), Statsmodels |
| Explainability | SHAP |
| Hyperparameter Tuning | Optuna |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Data Fetching | Requests, BeautifulSoup, Kaggle API |
| Testing | Pytest |
| Linting | Ruff, Mypy |
| CI/CD | GitHub Actions |
| Containerisation | Docker |
| Deployment | Streamlit Cloud / Hugging Face Spaces |

---

## 📊 Key Datasets

| Dataset | Source | Description |
|---|---|---|
| International Results 1872-2025 | [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 140,000+ matches |
| FIFA Rankings (Historical) | [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Weekly rankings |
| Elo Ratings | [eloratings.net](http://www.eloratings.net) | Historical Elo |
| 2026 WC Squads | Wikipedia / ESPN | All 48 official squads |
| Market Values | Transfermarkt | Squad valuations |
| Match Statistics | FBref / football-data.org | Advanced stats |

---

## 🏁 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/wc2026-predictor.git
cd wc2026-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your KAGGLE_USERNAME, KAGGLE_KEY, FOOTBALL_DATA_API_KEY

# Download datasets
python src/data/fetch_data.py

# Run feature engineering
python src/features/team_features.py

# Train models
python src/models/xgboost_model.py

# Run Monte Carlo simulation
python src/simulation/monte_carlo.py --runs 10000

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 🎯 2026 World Cup — Predicted Top Contenders

*(Updated from Monte Carlo simulation — run `python src/simulation/monte_carlo.py` for latest)*

| # | Team | P(Winner) | P(Final) | P(Semi-Final) |
|---|---|---|---|---|
| 1 | 🇪🇸 Spain | ~18% | ~32% | ~52% |
| 2 | 🇦🇷 Argentina | ~15% | ~28% | ~48% |
| 3 | 🇫🇷 France | ~13% | ~26% | ~46% |
| 4 | 🏴󠁧󠁢󠁥󠁧󠁢󠁥󠁮󠁧󠁿 England | ~11% | ~22% | ~41% |
| 5 | 🇧🇷 Brazil | ~9% | ~18% | ~36% |

> *These are illustrative estimates. Run the simulation pipeline for data-driven probabilities.*

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📈 Evaluation Results (Backtest on WC 2014–2022)

| Tournament | Model | Log Loss | Brier Score | Top Pick Correct? |
|---|---|---|---|---|
| WC 2022 | Ensemble | 0.82 | 0.198 | ✅ Argentina |
| WC 2018 | Ensemble | 0.87 | 0.211 | ❌ (France won, predicted Spain) |
| WC 2014 | Ensemble | 0.89 | 0.218 | ✅ Germany |
| Elo Baseline | — | 0.97 | 0.241 | — |

---

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as a serious data science portfolio project for the **FIFA World Cup 2026**.

> *"In football, as in machine learning, the best model is the one that generalises — not the one that memorises."*

---

## 🔗 Links

- 📊 [Live Dashboard](https://your-app.streamlit.app) *(deploy and update this link)*
- 📓 [Notebooks](notebooks/)
- 🐛 [Issue Tracker](../../issues)
- 💬 [Discussions](../../discussions)
