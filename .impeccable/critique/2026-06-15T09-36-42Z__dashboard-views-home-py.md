---
target: Home page
total_score: 30
p0_count: 0
p1_count: 2
timestamp: 2026-06-15T09-36-42Z
slug: dashboard-views-home-py
---
# Critique — Home (dashboard/views/home.py)

Total: 30/40 (Good)

## Heuristics
1 Visibility 3; 2 Match real-world 4; 3 Control 3; 4 Consistency 3; 5 Error Prevention 3;
6 Recognition 3; 7 Flexibility 3; 8 Aesthetic/Minimalist 2; 9 Error Recovery 3; 10 Help 3.

## Anti-Patterns
Not AI slop. Strong identity (cinematic hero, gold-leader contender board, tabular figures).
Surviving tell: kicker on every section (kept by user choice).
Detector: 7 overused-font (Inter/Space Grotesk — accepted committed brand faces);
1 broken-image (hero <img src=""> empty initial src, ~line 150); 1 em-dash-overuse.
Browser overlay unavailable (Streamlit SPA + sandboxed hero iframe).

## Priority Issues
[P1] Same metric (P(champion)) shown 4 ways: Contenders + Landscape treemap + 48-row table + Groups cards. Distill/differentiate. -> /impeccable distill
[P1] Single-focus/page length: 7 sections; methodology buried mid-scroll. Sequence the arc; progressive-disclose table+groups. -> /impeccable layout
[P2] Survival Curves ships blank when multiselect emptied (if selected: skips). Add empty state. -> /impeccable harden
[P2] Unguarded WC2026_GROUPS import (~line 352) crashes whole page. Guard like other loads. -> /impeccable harden
[P2] Hero flag empty src requests broken image before JS fills. Transparent data-URI placeholder. -> /impeccable polish

## Persona Red Flags
Alex: no keyboard accelerators; must scroll/hunt 48-row table for specific teams.
Sam: hero iframe count-up may not announce to SR; add text fallback. Plotly weak SR but HTML table fallback exists; color not load-bearing alone.
Hiring Manager (project persona): methodology buried mid-scroll / on Model Insights; risk of bounce before seeing rigor.

## Minor
Em-dash density high. Table Champion bars are relative (max-scaled) — may misread as absolute.

## Questions
End Home after the field view with a link to Model Insights? Does the 48-row table belong on the landing page? Is the strongest evidence above the fold?
