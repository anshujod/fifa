# Product

## Register

product

## Users

Primarily **hiring managers, technical recruiters, and engineering reviewers** evaluating the
author's machine-learning and data-visualization skill — this is a portfolio showcase piece
first. They arrive curious but skeptical, often skimming on limited time, and judge competence
from how the work is presented as much as what it does. A secondary audience is football
analytics enthusiasts who want to explore the predictions themselves.

The job to be done: in a few minutes of exploration, come away convinced the author can build
a rigorous, end-to-end forecasting system *and* present it with professional polish. Every
screen should make the methodology legible and the craft self-evident without requiring the
viewer to read the README.

## Product Purpose

A simulation-driven FIFA World Cup 2026 intelligence platform. It combines a four-model
ensemble (Poisson goal model + XGBoost + LightGBM + Elo) blended as a weighted average whose
weights are tuned to minimize validation log-loss, then runs 10,000 Monte Carlo tournament
simulations to produce championship, qualification, and per-stage win probabilities for all
48 teams, with Wilson 95% confidence intervals. Surfaced through a Streamlit dashboard: Home, Match Predictor, Group Standings,
Bracket Simulator, Team Profiles, Team Comparison, and Model Insights.

Success looks like a reviewer concluding "this person knows what they're doing" — the model
internals (calibration curves, RPS, feature importance, backtests) are not buried; they are a
first-class part of the story. The interface earns trust by showing its work.

## Brand Personality

Authoritative, precise, broadcast-grade. Three words: **rigorous, cinematic, confident.**
The voice is that of a serious sports-intelligence desk — quiet certainty, no hype, numbers
that speak for themselves. Emotionally it should feel expensive and credible, closer to a
financial terminal or a TV broadcast graphics package than a fan app.

## Anti-references

- **Rainbow / chart-junk dashboards.** No garish multi-hue palettes, no decorative gradients
  on data, no chart for chart's sake. Color carries meaning or it isn't used.
- **Toy / gimmicky treatments.** No emoji-as-UI, no playful mascots, no amateur flourishes.
  Flags are the only pictorial element and they're rendered cleanly.
- Generic SaaS template (identical icon-cards + gradient accent) and loud betting-site
  aesthetics (neon, casino energy, aggressive CTAs) are also out.

## Design Principles

1. **Show the work.** Methodology, calibration, and uncertainty are features to surface, not
   details to hide. Confidence intervals and model internals are part of the credibility story.
2. **Data-forward, decoration-back.** Typography and layout do the work; ornament is removed
   until only meaning-bearing elements remain. If a pixel doesn't inform, it goes.
3. **Restraint as signal.** A disciplined palette (midnight canvas, one gold accent for the
   champion, one electric blue for live data) reads as more expert than a colorful one.
4. **Broadcast-grade polish.** Tabular numerals, precise alignment, quiet confident motion.
   The craft of the presentation is itself the portfolio argument.
5. **Legible at a skim.** A time-pressed reviewer should grasp each screen's headline insight
   in seconds, then be able to drill into the rigor underneath.

## Accessibility & Inclusion

Target **WCAG 2.1 AA**: body text ≥4.5:1 and large text ≥3:1 against the dark canvas — verify
muted grays (`#8B98AF`, `#AEB9CC`) wherever they sit on surface tints. Never encode meaning in
color alone; pair probability color scales with numeric labels. Honor
`prefers-reduced-motion` (already wired into the reveal/bar animations) with a static fallback.
Keep flag images decorative with empty `alt`, and ensure data is always available as text, not
only as a chart.
