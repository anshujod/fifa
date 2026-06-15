---
name: World Cup 2026 Forecasting Terminal
description: A near-black, monospace, single-cyan-accent trading-terminal for FIFA World Cup 2026 forecasts.
colors:
  bg: "#0A0C10"
  bg-2: "#0D1016"
  surface: "#11151C"
  surface-2: "#161B23"
  border: "#FFFFFF14"
  grid: "#FFFFFF08"
  text: "#E6EDF3"
  text-2: "#9BA7B4"
  text-3: "#7D8794"
  accent: "#3AC9E0"
  accent-bright: "#74E2F2"
  accent-deep: "#1C6E7E"
  champion: "#E8A33D"
  champion-bright: "#F4C36B"
  pos: "#3FB950"
  neg: "#F85149"
  warn: "#D29922"
typography:
  display:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "40px"
    fontWeight: 600
    lineHeight: 1.08
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1.12
    letterSpacing: "-0.01em"
  title:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "21px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0"
  data:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "27px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0"
  body:
    fontFamily: "IBM Plex Sans, -apple-system, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  label:
    fontFamily: "IBM Plex Mono, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 500
    lineHeight: 1.2
    letterSpacing: "0.16em"
rounded:
  xs: "2px"
  sm: "4px"
  md: "6px"
  lg: "8px"
spacing:
  xs: "8px"
  sm: "14px"
  md: "18px"
  lg: "24px"
  xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "#06222A"
    rounded: "{rounded.md}"
    padding: "0.55rem 1.4rem"
  button-primary-hover:
    backgroundColor: "{colors.accent-bright}"
    textColor: "#06222A"
  button-secondary:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "0.5rem 1.2rem"
  panel:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "18px 20px"
  chip:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-2}"
    rounded: "{rounded.sm}"
    padding: "7px 14px"
---

# Design System: World Cup 2026 Forecasting Terminal

## 1. Overview

**Creative North Star: "The Forecasting Terminal"**

This is a Bloomberg/trading-terminal for football — the screen an analyst keeps open to watch
the model's read on the tournament. The canvas is near-black graph-paper; data is set in
monospace so every column aligns and every figure reads as a measured quantity, not a
decoration. A single electric cyan is the one accent the system spends — it marks live data,
probability, and active state. Exactly one warm signal (amber) exists, and it marks exactly one
thing per view: the champion / leader. Everything else is neutral.

The system is **flat, gridded, and tabular**. There are no gradient cards, no glows, no
gold-gamer chrome. Depth is one hairline (`rgba(255,255,255,0.08)`) and one tonal step off the
canvas — nothing more. Panels are sharp (4–6px radius), bars are near-square, and motion is
fast and minimal: things appear, they don't perform. The aesthetic argues the author's
engineering credibility on sight — it looks like a tool built by someone who respects data.

It explicitly rejects what came before and what AI reaches for. **No dark-dashboard-with-gold**
(the category reflex for "sports prediction"). **No Inter / Space Grotesk** (the most common
AI-UI pairing of 2026). **No gradient-tinted cards, no decorative glow, no rainbow charts.** If
it looks like a generic SaaS analytics dashboard, the redesign failed.

**Key Characteristics:**
- Near-black graph-paper canvas (`#0A0C10`) with a faint baseline grid
- Monospace everything-that-is-data (IBM Plex Mono); IBM Plex Sans only for prose
- One cyan accent for data/active; one amber signal for the single champion
- Flat surfaces, one hairline border, 4–6px radii, near-square bars
- Tabular columns; figures align down the page
- Fast, minimal motion — appearance, not choreography

## 2. Colors

A near-monochrome terminal palette: a four-step near-black ramp, cool off-white ink, one cyan
data accent, and one amber champion signal. GitHub-dark-grade semantic green/red/amber for
status only.

### Primary
- **Signal Cyan** (`#3AC9E0`, bright `#74E2F2`, deep `#1C6E7E`): The one data accent. Probability
  bars, active nav, primary buttons, chart series, links, the live dot's companion. This is the
  colour the whole system is "in".

### Secondary
- **Champion Amber** (`#E8A33D`, bright `#F4C36B`): The single warm signal. Reserved for the one
  champion / leader per view — the lead contender row, the simulated tournament winner, the
  active favourite. Amber + cyan is a deliberate CRT-terminal duo. (Carried in the `GOLD*`
  constants for code compatibility; it is amber, not gold.)

### Neutral
- **Terminal Black** (`#0A0C10`, panel-black `#0D1016`): The canvas — a cool near-black with a
  1.8%-opacity white grid laid over it.
- **Panel** (`#11151C`, raised `#161B23`): Surfaces and inputs, as flat fills (no gradient).
- **Hairline** (`rgba(255,255,255,0.08)`): The single border weight; brightens to cyan on hover.
- **Ink** (`#E6EDF3`): Primary text and figures.
- **Slate** (`#9BA7B4`): Secondary text, prose on panels.
- **Muted** (`#7D8794`): Labels, captions, meta — the lightest text allowed (≥4.5:1 on canvas).

### Status
- **Green** (`#3FB950`) up/win · **Red** (`#F85149`) down/loss · **Amber** (`#D29922`) warning.
  Semantic only, never decorative.

### Named Rules
**The One-Cyan Rule.** Cyan is the only brand hue for data and interaction. New chart categories
are separated by neutral value steps or shape, not new hues — never reach for a third colour.

**The Single-Signal Rule.** Amber appears once per screen and means "the leader." A second amber
element devalues it; if two things are amber, one is wrong.

## 3. Typography

**Display / Data Font:** IBM Plex Mono (with ui-monospace, monospace)
**Body Font:** IBM Plex Sans (with -apple-system, sans-serif)

**Character:** A monospace-led system. IBM Plex Mono carries headings, numerals, labels and
every figure — its fixed advance is the source of the terminal feel and makes columns self-align.
IBM Plex Sans handles running prose only, so paragraphs stay readable. The pairing reads
"engineering tool," and Plex is far less saturated in AI output than Inter/Space Grotesk.

### Hierarchy
- **Display** (Plex Mono 600, ~40px, line-height 1.08): Hero page title only.
- **Headline** (Plex Mono 600, 30px): Standard page titles (mixed-case, not uppercase — mono
  uppercase at display size overflows).
- **Title** (Plex Mono 600, 21px): Section headings.
- **Data** (Plex Mono 600, 21–40px): KPI values, probabilities, scorelines.
- **Body** (Plex Sans 400, 15px, line-height 1.6): Prose and subtitles, ~65–75ch.
- **Label** (Plex Mono 500, 11px, `0.16em`, UPPERCASE): Kickers, KPI labels, table headers,
  prefixed with a `▸` marker on section eyebrows.

### Named Rules
**The Monospace-Figure Rule.** Every number lives in IBM Plex Mono. Mono is inherently tabular,
so figures align down a column with no extra work — that alignment is the terminal's signature.

**The Two-Family Rule.** Plex Mono for structure and data, Plex Sans for prose. No third
typeface; never uppercase a display-size heading (mono gets too wide and overflows).

## 4. Elevation

Flat. There are no shadows on UI surfaces and no gradients. Depth is exactly two moves: a single
hairline border (`rgba(255,255,255,0.08)`) and one tonal step (`#11151C` panel on `#0A0C10`
canvas). The faint baseline grid behind everything supplies the "graph-paper" depth. The only
shadows in the system are tight drops under flag images so they seat on the panel.

### Shadow Vocabulary
- **Flag seat** (`box-shadow: 0 1px 3px rgba(0,0,0,.5)`): under rectangular flags only.

### Named Rules
**The Flat-Grid Rule.** Surfaces are flat at rest and respond to hover by brightening their
hairline toward cyan — never by lifting, gaining a shadow, or sliding. If a panel needs a shadow
to separate from the canvas, raise its tonal step instead.

## 5. Components

### Buttons
- **Shape:** sharp (6px radius).
- **Primary:** solid Signal Cyan (`#3AC9E0`) with near-black ink (`#06222A`), Plex Mono 600.
- **Secondary:** panel fill with a hairline border, ink text.
- **Hover / Focus:** primary brightens to `#74E2F2`; secondary's hairline shifts to cyan. ~120ms,
  no transform.

### Chips (ensemble pipeline)
- **Style:** small radius (4px), panel fill, hairline, Plex Mono, a cyan dot leading. Joined by a
  muted `+`. Informational, not interactive.

### Panels / Cards
- **Corner Style:** 6px.
- **Background:** flat `#11151C` (no gradient).
- **Border:** 1px hairline; brightens toward cyan on hover.
- **Internal Padding:** 18–20px.
- **Numerals:** Plex Mono, ink.

### Inputs / Fields
- **Style:** panel fill `#11151C`, 1px hairline, 6px radius, Plex Mono for numeric entry.
- **Focus / Slider:** cyan track and handle.

### Navigation (sidebar)
- **Style:** permanently open rail on `rgba(10,12,16,.97)`, Plex Sans labels, 6px rows.
- **States:** hover = faint white wash; active = cyan-tinted fill + cyan hairline + bright ink.
  Logo mark is a cyan-outlined mono square.

### Signature: The Contender Board
The hero ranking. Each row is a mono grid of `pos · flag · name + cyan probability bar · pct`.
Bars are cyan, scaled to the leader. The leader row (`.lead`) is the system's one amber moment:
amber hairline, amber position index, amber bar, amber percentage. Exactly one row is amber —
the most visible expression of the Single-Signal Rule.

## 6. Do's and Don'ts

### Do:
- **Do** set every figure in IBM Plex Mono so columns self-align (the terminal signature).
- **Do** keep cyan as the only data/interaction hue; separate chart categories by value or shape.
- **Do** reserve amber for exactly one champion/leader element per screen.
- **Do** build panels as flat fills with one hairline; brighten the hairline toward cyan on hover
  instead of adding shadow or lift (Flat-Grid Rule).
- **Do** keep muted text at `#7D8794` or lighter-ink and verify ≥4.5:1 on its surface.
- **Do** keep motion fast and minimal (~120–450ms), with a `prefers-reduced-motion` fallback.

### Don't:
- **Don't** reintroduce the dark-dashboard-with-gold look — no gold gradients, no gamer chrome.
  This is the category reflex the redesign exists to escape.
- **Don't** use Inter or Space Grotesk anywhere — the most common AI-UI pairing of 2026.
- **Don't** add gradient-tinted cards, decorative glows, or glassmorphism. Flat only.
- **Don't** build rainbow / chart-junk visuals or spend a third saturated hue (One-Cyan Rule).
- **Don't** let two elements be amber at once; a second amber devalues the champion signal.
- **Don't** uppercase a display-size mono heading — it gets too wide and overflows on tablet.
