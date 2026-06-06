# Console responsive redesign

**Date:** 2026-06-05
**Goal:** Make the Sisyphus Console (`/app/`) usable on phones/tablets without
changing the approved desktop look or any data/behavior. Today the console is a
fixed 1280×800 canvas with a JS scale-to-fit transform; on a phone it collapses to
a ~96px sliver (the `.app` flex-shrinks **and** the 0.277 scale is applied on top).

## Approach

Replace the fixed canvas + JS scale with a pure-CSS responsive layout. No changes
to state, data contracts, pk math, or the engine data. Files:
`web/src/styles.css` (layout + media queries) and `web/src/components/App.tsx`
(remove the `scale`/`fit()` transform).

## Breakpoint ramp

- **≥1100px — desktop, unchanged look.** Centered floating card, 2-pane grid
  `286px 1fr`, internal-scroll panes. Change only: `.app` becomes
  `width:min(1280px, calc(100vw-44px)); height:min(800px, calc(100vh-44px))`
  (no transform). Identical at ≥1324px; on smaller laptops it fluidly fits with
  internal scroll instead of shrinking text (a strict improvement).
- **860–1100px — small laptop / tablet landscape.** Still the 2-pane card, but
  in-content `.split`/`.split-even` stack to 1 column and `.statline` goes 4→2.
- **≤860px — phone / tablet portrait.** Single column, natural page scroll:
  - `html,body,#root` height:auto; `.stage` block, no padding; `.app` full-bleed
    (`width:100%`, `height:auto`, `min-height:100vh`, no border/radius/shadow),
    `grid-template-columns:1fr` → rail stacks above main.
  - Rail: nav → horizontal scrollable **chip row** (`.nav` row + overflow-x;
    buttons `width:auto`; `.desc` hidden); `.rail-fields` flex:none + overflow
    visible (no nested scroll); tighter 16px side paddings.
  - Main: `.content` overflow visible (document scrolls); `.mbar` + `.badges`
    wrap; `.tabs` horizontal scroll; `.panel` overflow-x:auto (wide-table safety);
    `.seg button` taller for touch. Charts already scale via SVG `viewBox`.

## Verification

- `npm run build` + `npm run smoke` (all 6 workflows render, 0 errors).
- Playwright at **390 / 768 / 1280 px**: screenshot each; assert `.app` is full
  width (no sliver) on mobile, click through all 6 workflows, 0 console errors.
- `npm run build:pages` → commit → push → live-verify on mobile (app fills width,
  nav chips scroll, a workflow renders, PDF/landing untouched).

## Out of scope

- No change to desktop appearance ≥1324px, to data, or to the landing page.
- No mobile-specific feature work (no collapsible-inputs drawer) — stacked rail is
  enough for v1; revisit if needed.
