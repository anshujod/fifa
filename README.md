# FIFA World Cup 2026 Prediction Engine

A simulation-driven tournament intelligence platform for FIFA World Cup 2026. Built with an ensemble machine learning pipeline, ELO ratings, and 10,000 Monte Carlo simulations, presented through a premium Streamlit analytics dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dashboard Pages](#dashboard-pages)
3. [Model Architecture](#model-architecture)
4. [Simulation Methodology](#simulation-methodology)
5. [Feature Pipeline](#feature-pipeline)
6. [Tournament Structure](#tournament-structure)
7. [Evaluation](#evaluation)
8. [Project Structure](#project-structure)
9. [Data Pipeline](#data-pipeline)
10. [Local Development](#local-development)
11. [Deployment](#deployment)
12. [Configuration](#configuration)
13. [Requirements](#requirements)

---

## Project Overview

This system predicts match outcomes and simulates the full FIFA World Cup 2026 bracket. It combines a Poisson goal model with gradient-boosted classifiers, stacked through a logistic regression meta-learner. The calibrated ensemble is then used to drive 10,000 full-tournament Monte Carlo simulations, producing championship, qualification, and per-stage win probabilities for all 48 teams with Wilson score 95% confidence intervals.

The dashboard presents results in a flat, typography-led analytics UI — no decorative elements, no rainbow charts, data-forward layout throughout.

---

## Dashboard Pages

**Home**  
Overview panel with the current championship race bar chart, top contenders, group summary cards, and key simulation metadata (number of simulations, teams, groups).

**Match Predictor**  
Select any two of the 48 qualified teams. The system runs the full ensemble pipeline — Poisson scoreline model, XGBoost, LightGBM, meta-learner — and returns win/draw/loss probabilities, most-likely scoreline, expected goals, and recent form for both sides. Results are cached in session state and persist across reruns.

**Group Standings**  
All 12 groups (A through L) with Monte Carlo-driven qualification probabilities for each team. Simulated standings table per group with points, goal difference, and a breakdown of qualification/third-place/elimination likelihood.

**Bracket Simulator**  
Two modes: probability bracket (aggregate MC outcome across 10,000 runs, shown as blue-scaled probability cells from Round of 32 through Final) and single tournament simulation (one seeded run showing a complete bracket outcome with match-by-match results).

**Team Profiles**  
Per-team analytics including squad composition by position, top-capped players, World Cup historical record, ELO rating history, and per-stage Monte Carlo probabilities (champion, finalist, semifinal, quarterfinal, Round of 32 qualification). All probabilities shown with Wilson score 95% confidence intervals.

**Team Comparison**  
Side-by-side comparison for any two teams across eight dimensions: Attack, Defence, Form, ELO, FIFA Rank, Squad Value, Experience, and Head-to-Head record. Rendered as an overlay radar chart plus a metric table with winner highlighting.

**Model Insights**  
Calibration curves, cross-model accuracy and RPS comparison, feature importance (native XGBoost gain with graceful fallback if SHAP is unavailable), and backtest results for WC 2014, 2018, and 2022.

---

## Model Architecture

The prediction pipeline is a stacked ensemble of four components.

### 1. Poisson Goal Model

Fits separate Poisson regression models for home and away expected goals (xG) using attack/defence strength parameters computed from historical match data. Given a matchup, it samples scoreline distributions and derives win/draw/loss probabilities from the joint Poisson distribution. Used directly for scoreline prediction and as a base learner in the ensemble.

### 2. XGBoost Outcome Classifier

A gradient-boosted tree classifier trained on the 37-feature match representation (see Feature Pipeline below). Predicts a three-class probability distribution: Home win, Draw, Away win. Trained on historical international results from 1990 to present, with early stopping on a held-out validation split.

### 3. LightGBM Classifier

Same feature set and target as XGBoost, with LightGBM's leaf-wise growth strategy. Requires `libgomp1` (OpenMP) on Linux deployment environments. Provides complementary signal to XGBoost, particularly on confederation and categorical features.

### 4. Logistic Regression Meta-learner

The stacking layer. Takes the probability outputs of the Poisson model, XGBoost, and LightGBM as input features and learns the optimal blend. Trained on out-of-fold predictions from cross-validation to prevent leakage.

### 5. Platt Scaling Calibration

A calibrated wrapper (`CalibratedClassifierCV` with Platt scaling) is applied to the meta-learner output to align predicted probabilities with empirical frequencies. Calibration is verified with reliability diagrams stored in `results/calibration_curves.png`.

### ELO Rating System

Maintained separately from the ML pipeline, used as a feature and as a standalone baseline. K-factors by tournament type:

| Tournament type | K-factor |
|---|---|
| FIFA World Cup (not qualification) | 32 |
| Competitive (qualifiers, continental cups) | 20 |
| Friendly | 10 |

Initial ELO: 1500. Ratings are recomputed chronologically over all historical matches before any prediction.

---

## Simulation Methodology

### Monte Carlo Tournament Simulation

The full WC 2026 tournament (group stage + knockout rounds) is simulated 10,000 times using Python's `multiprocessing` module. Each worker process loads the full ensemble independently (`MatchPredictor.load()`). Within each simulation:

1. All group-stage matches are simulated using the ensemble (with Poisson fast-mode for speed: direct Poisson sampling instead of full scoreline enumeration).
2. Group standings are resolved by points, goal difference, goals scored, and H2H record.
3. The best four third-place teams advance per FIFA rules.
4. Knockout rounds proceed from Round of 32 through the Final, with penalty shootout simulation for drawn knockout matches.

Results are aggregated across all 10,000 runs to produce per-team probabilities for each stage: champion, finalist, third place, semifinal, quarterfinal, Round of 32.

### Confidence Intervals

All Monte Carlo stage probabilities are reported with Wilson score 95% confidence intervals, which perform well even at extreme probabilities (near 0 or 1) and with small sample sizes.

### Backtest Validation

The pipeline is backtested on WC 2014, 2018, and 2022 using a temporal cutoff — the model is retrained on data strictly before the tournament start date, then evaluated on the actual match results. This produces genuine out-of-sample metrics. Note that LightGBM and XGBoost results on the held-out WC set are optimistically biased because those tournament matches appear in training data; the backtest OOS numbers are the authoritative evaluation.

---

## Feature Pipeline

Each match is represented by 37 features, split across home and away perspectives:

**ELO-based features**  
ELO rating, ELO difference, expected score from ELO formula.

**Form features**  
Points per game (last 5, 10 matches), win rate, draw rate, loss rate, goals scored per game, goals conceded per game — computed with exponential time-decay (lambda = 0.005) so recent matches weight more heavily.

**Expected Goals (xG)**  
Rolling average xG for and against, derived from the Poisson attack/defence parameters.

**Head-to-Head features**  
H2H win rate, average goals scored, number of past meetings between the two teams.

**Squad features**  
Total squad market value (log-transformed), average player age, average caps per player, number of players from top European leagues.

**Confederation**  
One-hot encoded confederation membership (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC).

**World Cup experience**  
Number of previous World Cup appearances, best-ever finish encoded as a numeric stage reached.

**Match context**  
Home advantage indicator, tournament importance weight, days since last match.

---

## Tournament Structure

FIFA World Cup 2026 — hosted by USA, Canada, and Mexico.

- 48 teams, 12 groups of 4 (Groups A through L)
- Top 2 from each group advance automatically (24 teams)
- Best 8 third-place teams also advance (8 teams total)
- 32-team knockout bracket: Round of 32, Round of 16, Quarterfinals, Semifinals, Third-place playoff, Final

**Official group draw (December 2025, Washington D.C.):**

| Group | Teams |
|---|---|
| A | Czech Republic, Mexico, South Africa, South Korea |
| B | Bosnia and Herzegovina, Canada, Qatar, Switzerland |
| C | Brazil, Haiti, Morocco, Scotland |
| D | Australia, Paraguay, Turkey, United States |
| E | Curaçao, Ecuador, Germany, Ivory Coast |
| F | Japan, Netherlands, Sweden, Tunisia |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Cape Verde, Saudi Arabia, Spain, Uruguay |
| I | France, Iraq, Norway, Senegal |
| J | Algeria, Argentina, Austria, Jordan |
| K | Colombia, DR Congo, Portugal, Uzbekistan |
| L | Croatia, England, Ghana, Panama |

---

## Evaluation

Metrics below are on the 192-match WC 2014/2018/2022 holdout set. LightGBM numbers are in-sample (those matches appear in its training data); the OOS backtest is the authoritative measure.

| Model | Accuracy | Log Loss | RPS | RPS Skill Score |
|---|---|---|---|---|
| Ensemble (calibrated) | 55.7% | 0.914 | 0.190 | 0.216 |
| XGBoost | 57.8% | 0.925 | 0.192 | 0.208 |
| LightGBM | 76.6%* | 0.614* | 0.115* | 0.524* |

*In-sample, optimistically biased — see `results/backtest_results.json` for OOS numbers.

Ranked Probability Score (RPS) measures calibration quality across all three outcome classes. A positive skill score indicates improvement over a uniform-probability baseline.

---

## Project Structure

```
fifa/
├── dashboard/
│   ├── app.py                    # Entry point, sidebar, routing
│   ├── pages/
│   │   ├── home.py
│   │   ├── match_predictor.py
│   │   ├── group_standings.py
│   │   ├── bracket_simulator.py
│   │   ├── team_profiles.py
│   │   ├── team_comparison.py
│   │   └── model_insights.py
│   └── utils/
│       ├── theme.py              # Design system, CSS injection, UI components
│       ├── charts.py             # Plotly chart builders
│       └── data_loader.py        # Cached data access
│
├── src/
│   ├── data/
│   │   ├── fetch_data.py         # Raw data acquisition
│   │   ├── clean.py              # Data cleaning and normalisation
│   │   ├── schema.py             # Column definitions
│   │   └── update.py             # Incremental result updates
│   ├── features/
│   │   ├── team_features.py      # Rolling form, ELO, xG
│   │   ├── h2h_features.py       # Head-to-head history
│   │   ├── squad_features.py     # Squad composition features
│   │   ├── poisson_features.py   # Attack/defence strength
│   │   └── encoder.py            # Categorical encoding
│   ├── models/
│   │   ├── elo_model.py          # ELO system + logistic calibrator
│   │   ├── poisson_model.py      # Poisson goal model
│   │   ├── xgboost_model.py      # XGBoost outcome classifier
│   │   ├── additional_models.py  # LightGBM, Random Forest
│   │   ├── ensemble.py           # Stacking meta-learner
│   │   └── calibration.py        # Platt scaling wrapper
│   ├── simulation/
│   │   ├── match_predictor.py    # Single-match prediction API
│   │   ├── group_stage.py        # Group simulation + standings
│   │   ├── knockout.py           # Knockout bracket simulation
│   │   ├── monte_carlo.py        # Multiprocessing MC runner
│   │   ├── penalties.py          # Penalty shootout simulation
│   │   ├── third_place.py        # Third-place bracket builder
│   │   └── results_store.py      # Simulation output aggregation
│   └── evaluation/
│       ├── backtest.py           # Temporal backtest harness
│       ├── metrics.py            # Accuracy, log loss, RPS, Brier
│       └── shap_analysis.py      # SHAP value computation
│
├── data/
│   ├── raw/
│   │   ├── international_results.csv
│   │   ├── fifa_rankings.csv
│   │   ├── world_football_elo.csv
│   │   └── squads_2026/          # Per-team squad JSON files (48 teams)
│   └── processed/                # Generated at runtime (see .gitignore)
│       ├── match_features.csv    # 37-feature training matrix (~22 MB)
│       ├── squad_features.csv
│       ├── match_results.csv
│       ├── mc_results.json       # Aggregated MC simulation output
│       ├── elo_ratings.csv
│       ├── team_snapshots.csv
│       └── squad_profiles.csv
│
├── models/
│   └── saved/
│       ├── ensemble.joblib
│       ├── ensemble_calibrated.joblib
│       ├── lightgbm.joblib
│       ├── logistic_regression.joblib
│       ├── poisson_goal_model.joblib
│       ├── random_forest.joblib
│       └── xgboost_outcome.joblib
│
├── results/
│   ├── evaluation_report.json
│   ├── backtest_results.json
│   ├── model_comparison.csv
│   ├── monte_carlo_probabilities.csv
│   ├── monte_carlo_stats.json
│   ├── calibration_curves.png
│   └── shap_plots/
│
├── notebooks/
│   ├── 02_feature_engineering.ipynb
│   └── 06_evaluation_and_shap.ipynb
│
├── .streamlit/
│   └── config.toml               # Theme colours, layout
├── requirements.txt
├── packages.txt                  # System deps for Streamlit Cloud
└── .gitignore
```

---

## Data Pipeline

### Raw Data Sources

- **`data/raw/international_results.csv`** — Historical international match results (goals, tournament, date, home/away).
- **`data/raw/fifa_rankings.csv`** — FIFA World Rankings at various points in time.
- **`data/raw/world_football_elo.csv`** — Pre-computed ELO time series (used to seed the ELO model).
- **`data/raw/squads_2026/`** — Individual JSON files per team with squad player data (48 files), including position, caps, club, market value.

### Processed Files

The `data/processed/` directory is `.gitignore`d except for seven files that the dashboard loads at runtime. These are committed directly:

| File | Description |
|---|---|
| `match_features.csv` | 37-feature training matrix (~22 MB). Read by `MatchPredictor.load()` to reconstruct team snapshots and fit the ELO calibrator. |
| `squad_features.csv` | Engineered squad-level features per team. |
| `match_results.csv` | Cleaned historical results used for H2H and form display. |
| `mc_results.json` | Aggregated Monte Carlo simulation output (10,000 runs), including per-team per-stage probabilities and metadata. |
| `elo_ratings.csv` | Final ELO ratings for all teams. |
| `team_snapshots.csv` | Per-team current-state feature snapshot used for fast match prediction. |
| `squad_profiles.csv` | Processed squad profiles for the Team Profiles page. |

All other processed files (`clean_results.csv`, `competitive_results.csv`, etc.) are regenerable from raw data and are excluded from version control.

---

## Local Development

**Prerequisites:** Python 3.13, Git.

```bash
# Clone the repository
git clone https://github.com/anshujod/fifa.git
cd fifa

# Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

The app opens at `http://localhost:8501`. All seven required processed data files and the serialised models in `models/saved/` are tracked in the repository, so the dashboard works immediately after install without any data regeneration step.

### Dev Container

A `.devcontainer/devcontainer.json` is included for VS Code Dev Containers. It pre-installs Python 3.13 and runs `pip install -r requirements.txt` on container creation.

---

## Deployment

### Streamlit Community Cloud

1. Fork or push the repository to a GitHub account accessible to Streamlit Cloud.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Set the following fields:
   - **Repository:** `your-username/fifa`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Under **Advanced settings**, set **Python version** to **3.13**.
5. Click **Deploy**.

Streamlit Cloud reads `packages.txt` before installing Python dependencies. This file installs `libgomp1`, the OpenMP shared library required by LightGBM and XGBoost on the Debian-based cloud runtime. Without it, importing either library raises a missing shared object error.

All model files and runtime data files are committed to the repository, so the app is fully functional on first deploy with no build-time data generation.

### Environment Variables

No environment variables are required. The app does not make external API calls at runtime — all predictions are derived from local model files and processed data.

---

## Configuration

**`.streamlit/config.toml`**

```toml
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#0B1020"
secondaryBackgroundColor = "#172033"
textColor = "#F8FAFC"
font = "sans serif"
```

**`packages.txt`** (system packages for Streamlit Cloud)

```
libgomp1
```

---

## Requirements

```
streamlit==1.58.0
pandas==3.0.3
numpy==2.4.4
plotly==6.8.0
scikit-learn==1.9.0
scipy==1.17.1
xgboost==3.2.0
lightgbm==4.6.0
joblib==1.5.3
requests==2.34.2
matplotlib==3.10.9
```

All packages ship Python 3.13 wheels. SHAP is not included because `shap` depends on `numba`, which does not yet support NumPy 2.4. The Model Insights page detects SHAP's absence and falls back to native XGBoost feature importance (gain-based) automatically.

---

## License

This project is for educational and research purposes.
