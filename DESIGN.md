---
name: World Cup 2026 Prediction Lab
description: Broadcast-grade dark analytics dashboard for FIFA World Cup 2026 forecasting.
colors:
  bg: "#070B14"
  bg-2: "#0B1020"
  surface: "#10182B"
  surface-2: "#16203A"
  border: "#94A3B81A"
  text: "#F4F7FB"
  text-2: "#AEB9CC"
  text-3: "#8B98AF"
  gold: "#E9C46A"
  gold-2: "#F2D98C"
  gold-deep: "#C9A24B"
  accent: "#4C8DFF"
  accent-2: "#7DB0FF"
  success: "#34D399"
  warning: "#F59E0B"
  error: "#F87171"
typography:
  display:
    fontFamily: "Space Grotesk, Inter, sans-serif"
    fontSize: "44px"
    fontWeight: 600
    lineHeight: 1.12
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Space Grotesk, Inter, sans-serif"
    fontSize: "32px"
    fontWeight: 600
    lineHeight: 1.12
    letterSpacing: "-0.03em"
  title:
    fontFamily: "Space Grotesk, Inter, sans-serif"
    fontSize: "23px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  body:
    fontFamily: "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  label:
    fontFamily: "Inter, sans-serif"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.11em"
  metric:
    fontFamily: "Space Grotesk, sans-serif"
    fontSize: "30px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
rounded:
  sm: "10px"
  md: "14px"
  lg: "16px"
  pill: "99px"
spacing:
  xs: "8px"
  sm: "14px"
  md: "18px"
  lg: "22px"
  xl: "46px"
components:
  button-primary:
    backgroundColor: "{colors.gold}"
    textColor: "#1A1407"
    rounded: "{rounded.sm}"
    padding: "0.62rem 1.5rem"
  button-primary-hover:
    backgroundColor: "{colors.gold-2}"
    textColor: "#1A1407"
  button-secondary:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    padding: "0.5rem 1.25rem"
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: "20px 22px"
  metric-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.md}"
    padding: "18px 20px"
  chip:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-2}"
    rounded: "{rounded.pill}"
    padding: "8px 16px"
---

# Design System: World Cup 2026 Prediction Lab

## 1. Overview

**Creative North Star: "The FIFA World Cup 2026 Broadcast Desk"**

This is the on-screen graphics package of a serious televised football-intelligence desk,
translated into a web dashboard. The canvas is a deep midnight blue — the dark of a studio set
just before the broadcast cuts in — and over it sit precise, tabular numbers, a single
FIFA-inspired gold reserved for the champion, and an electric blue that signals live computed
data. It feels expensive, calm, and certain. The work argues for itself: this is a portfolio
piece whose job is to make a reviewer think *"this person knows what they're doing"* within
seconds, so every screen leads with a legible headline insight and lets the rigor (confidence
intervals, calibration, backtests) sit one layer beneath.

The system is **data-forward and decoration-back**. Typography and disciplined layout carry
the design; ornament is stripped until only meaning-bearing elements remain. Depth comes from
tonal layering on a near-black ground, not from heavy shadows or glass. Motion is quiet and
confident — bars grow once, sections reveal once, nothing bounces.

It explicitly rejects two things. **No rainbow / chart-junk:** color encodes meaning (gold =
champion, blue = probability/live, neutral grays = structure) or it is not spent. **No toy or
gimmicky treatment:** no emoji-as-UI, no mascots, no amateur flourishes — rectangular country
flags are the only pictorial element, and they are rendered cleanly with a soft shadow.

**Key Characteristics:**
- Cinematic near-black canvas (`#070B14`) with subtle radial gold/blue atmosphere
- One gold accent, champion-only; one electric blue for live data and probability
- Space Grotesk for numbers and headings, Inter for body — pure weight/role contrast
- Tabular numerals everywhere a figure appears
- Flat-by-default surfaces; borders and tonal tints define cards, not drop shadows
- Quiet, single-pass motion honoring `prefers-reduced-motion`

## 2. Colors

A near-monochrome midnight palette carrying exactly two meaning-bearing accents — gold and
electric blue — over a four-step neutral ramp.

### Primary
- **Champion Gold** (`#E9C46A`, light `#F2D98C`, deep `#C9A24B`): The single brand accent,
  reserved for the tournament leader/champion, the primary CTA, selected sidebar state, and
  section kickers. Its rarity is the entire point. Used as a gradient (`#E9C46A → #C9A24B`) on
  primary buttons and the leading contender's progress fill.

### Secondary
- **Electric Blue** (`#4C8DFF`, light `#7DB0FF`): The "live data" color. Drives probability
  bars, the home-win segment of outcome bars, chart series, and computed-value highlights. Pairs
  with gold as the only two saturated hues in the system.

### Neutral
- **Midnight Canvas** (`#070B14`, raised `#0B1020`): The app background — a deep, slightly blue
  near-black with a faint radial gold/blue glow layered on top.
- **Charcoal Surface** (`#10182B`, raised `#16203A`): Card, panel, and input backgrounds, used
  as low-opacity gradients so they read as tonal lifts off the canvas.
- **Ghost Border** (`rgba(148,163,184,0.10)`): The 1px hairline that defines every card and
  divider. Strengthens to ~0.22 on hover.
- **Ink** (`#F4F7FB`): Primary text and key figures.
- **Slate** (`#AEB9CC`): Secondary text, body copy on surfaces.
- **Muted Slate** (`#8B98AF`): Labels, captions, meta — the lightest text permitted; never
  lighter on body-sized text.

### Status
- **Success Green** (`#34D399`), **Warning Amber** (`#F59E0B`), **Error Red** (`#F87171`):
  Status and delta indicators only. Never decorative.

### Named Rules
**The Champion-Only Rule.** Gold is spent on ≤10% of any screen — the leader, the primary
action, the active nav item, kickers. If gold is on a second team or a generic accent, it has
been devalued. Blue, not gold, is the default data color.

**The Two-Hue Rule.** Gold and blue are the only saturated colors. A third accent is forbidden;
new categories are distinguished by neutral value steps, not new hues. This is what keeps the
system off the rainbow / chart-junk path.

## 3. Typography

**Display Font:** Space Grotesk (with Inter, sans-serif fallback)
**Body Font:** Inter (with -apple-system, BlinkMacSystemFont fallback)

**Character:** A geometric-grotesque display paired with a neutral humanist sans — contrast by
role and weight, never two similar sans-serifs competing. Space Grotesk's slightly mechanical
numerals give figures a broadcast-data feel; Inter keeps prose quiet and legible.

### Hierarchy
- **Display** (Space Grotesk 600, 44px, line-height 1.12, `-0.03em`): Hero page titles only
  (`.wc-header.hero h1`).
- **Headline** (Space Grotesk 600, 32px, `-0.03em`): Standard page titles.
- **Title** (Space Grotesk 600, 23px, `-0.02em`): Section headings (`.wc-section h2`).
- **Metric** (Space Grotesk 600, 27–30px, tabular-nums, `-0.02em`): KPI and probability values.
- **Body** (Inter 400, 15px, line-height 1.6): Subtitles and prose, capped ~65–75ch (`max-width`
  ~680px on headers).
- **Label** (Inter 600, 11px, `0.10–0.16em`, UPPERCASE): KPI labels, kickers, table headers.

### Named Rules
**The Tabular-Figure Rule.** Every number that can change — probabilities, scores, Elo, counts —
uses `font-variant-numeric: tabular-nums` so columns align and values don't jitter on rerun.

**The Two-Family Rule.** Space Grotesk for headings and figures, Inter for everything else. No
third typeface. Hierarchy comes from size and weight within these two, never from new fonts.

## 4. Elevation

Flat by default. Depth is built from **tonal layering on a near-black ground**, not from cast
shadows: surfaces are low-opacity gradient lifts off the canvas, separated by 1px ghost borders.
The only true shadows in the system are small, tight drops under flag images to seat them on the
surface. Atmosphere — not elevation — comes from large, very faint radial gradients (gold top-
right, blue top-left) fixed behind the whole app.

### Shadow Vocabulary
- **Flag seat** (`box-shadow: 0 2px 6px rgba(0,0,0,.45)`): Under rectangular flags only, to lift
  them off the row. Smaller variant (`0 1px 3px`) for sidebar/group flags.
- **Status glow** (animated `box-shadow` pulse on the live dot): The single decorative shadow,
  signalling "model online".

### Named Rules
**The Flat-By-Default Rule.** Cards are flat at rest. On hover they respond by brightening their
border (`.10 → .22 opacity`) and lifting 2px via `transform`, never by gaining a drop shadow. If
a surface needs a shadow to separate from the canvas, the tonal contrast is wrong — fix the
gradient, not the shadow.

## 5. Components

### Buttons
- **Shape:** Softly rounded (10px radius).
- **Primary:** Gold gradient (`#E9C46A → #C9A24B`) with near-black ink (`#1A1407`), weight 700,
  `0.62rem 1.5rem` padding. Reserved for the one key action per view.
- **Secondary:** Charcoal surface (`rgba(22,32,58,.6)`) with a ghost border and ink text.
- **Hover / Focus:** Primary brightens to `#F2D98C → #E9C46A`; secondary lifts 1px and its border
  shifts toward gold (`rgba(233,196,106,.35)`). Transitions ~0.18s.

### Chips (ensemble pipeline)
- **Style:** Pill (99px), charcoal fill, ghost border, 13px weight-600 slate text, a 7px blue
  dot leading each. Connected by a muted `+` separator to read as a model pipeline.
- **State:** Static / informational — chips display the ensemble, they are not interactive filters.

### Cards / Containers
- **Corner Style:** 14–16px radius (`md`/`lg`).
- **Background:** Charcoal surface as a vertical gradient (`rgba(22,32,58,.5) → rgba(16,24,43,.5)`).
- **Shadow Strategy:** None at rest — see Flat-By-Default Rule.
- **Border:** 1px ghost border (`rgba(148,163,184,.10)`), brightening on hover.
- **Internal Padding:** 18–22px.
- **Hover:** Border brightens and the card lifts 2px (KPI/engine cards) or slides 3px right
  (contender rows). Lead/champion cards carry a gold-tinted border and background wash.

### Inputs / Fields
- **Style:** Charcoal fill (`rgba(16,24,43,.8)`), 10px radius, no heavy stroke.
- **Focus:** Slider handles and active tracks use gold; selects open on the dark surface.

### Navigation (sidebar)
- **Style:** Vertical radio list on a darker rail (`rgba(8,12,22,.97)`), 13.5px slate labels,
  9px radius rows, native radio dots hidden.
- **States:** Hover = faint slate wash; active = gold-tinted background + gold border + ink
  weight-600 label. Topped by a gold-mark logo lockup and a pulsing "model online" status.

### Signature: The Contender Board
The hero component — a broadcast-style ranking. Each row is a CSS grid of `position · flag ·
name+probability-bar · percentage`. Bars are blue and scaled to the leader; the leader's row
(`.lead`) flips to a gold border, gold-washed background, gold position number, gold percentage,
and a gold bar fill. Rows reveal in a staggered sequence and the bar grows once from the left.
This is where the Champion-Only Rule is most visible: exactly one row is gold.

## 6. Do's and Don'ts

### Do:
- **Do** keep gold champion-only — the leader, the primary CTA, the active nav item, kickers
  (≤10% of any screen). Default data color is electric blue (`#4C8DFF`).
- **Do** apply `font-variant-numeric: tabular-nums` (via Space Grotesk) to every figure.
- **Do** define cards with a 1px ghost border (`rgba(148,163,184,.10)`) and a tonal gradient
  fill; brighten the border on hover instead of adding a shadow.
- **Do** keep muted text at `#8B98AF` or lighter-ink, and verify ≥4.5:1 on the surface it sits
  on; bump toward `#AEB9CC`/`#F4F7FB` if a label is borderline.
- **Do** pair every probability color with its numeric label — color never carries meaning alone.
- **Do** keep motion single-pass and confident (`cubic-bezier(.22,.7,.2,1)`), with a
  `prefers-reduced-motion` fallback that disables reveals.

### Don't:
- **Don't** build rainbow or chart-junk visuals. No third saturated hue, no decorative gradients
  on data, no chart that doesn't inform.
- **Don't** use toy or gimmicky treatments — no emoji-as-UI, no mascots, no playful flourishes.
  Clean rectangular flags are the only pictorial element.
- **Don't** spend gold on non-champion elements; a second gold item devalues the accent.
- **Don't** add drop shadows to separate surfaces — if a card needs one, the tonal contrast is
  wrong (Flat-By-Default Rule).
- **Don't** introduce a third typeface or pair two similar sans-serifs. Space Grotesk + Inter only.
- **Don't** use light-gray body text on tinted surfaces "for elegance"; it fails contrast and is
  the fastest way to make rigorous work look amateur.
