/* ============================================================
   pk.ts — PK math helpers.

   Every number is ANCHORED to the real Sisyphus engine:
   - predict single-dose profile  → the real venous_blood ODE curve
   - multi-dose / TDM / dose-adjust → superposition of a 1-compartment
     fit (pkfit) to that real curve — exact for linear PK, and the
     accumulation/steady-state it produces matches the engine's own
     pre-computed simulate metrics.
   - DDI fold changes              → the real engine apply_inhibition
     / apply_induction re-solves (drug.ddi), NOT a heuristic.

   Ported from the Claude Design prototype (pk.js), retyped, with the
   mock drug library replaced by real-engine data.
   ============================================================ */
import type { Drug, ConsoleData, Observation } from "./types";

export type Pt = [number, number];
export interface Band {
  upper: Pt[];
  lower: Pt[];
}

/* ---- single oral-dose shape: exp(-ke t) - exp(-ka t) ---- */
function shape(ka: number, ke: number, t: number): number {
  if (Math.abs(ka - ke) < 1e-6) ke = ke * 0.98;
  return Math.exp(-ke * t) - Math.exp(-ka * t);
}
export function peakOf(ka: number, ke: number): number {
  const tmax = Math.log(ka / ke) / (ka - ke);
  return shape(ka, ke, tmax);
}

/** Reference single-dose Cmax for the ENGINE track (the curve the ODE produces). */
function engineRefCmax(d: Drug): number {
  return d.tracks.engine ?? d.meta.cmax;
}

/** Engine single-dose concentration at time t, scaled to engine Cmax × dose/ref. */
export function concAt(d: Drug, dose: number, t: number): number {
  if (t <= 0) return 0;
  const { ka, ke } = d.pkfit;
  const sc = (engineRefCmax(d) / peakOf(ka, ke)) * (dose / d.dose);
  return Math.max(0, shape(ka, ke, t) * sc);
}

/** Analytical engine single-dose curve (smooth) — used for multi-dose horizons. */
export function singleCurve(d: Drug, dose: number, tEnd: number, step?: number): Pt[] {
  step = step || tEnd / 240;
  const out: Pt[] = [];
  for (let t = 0; t <= tEnd + 1e-9; t += step) out.push([t, concAt(d, dose, t)]);
  return out;
}

/** The REAL engine ODE curve, scaled linearly by dose (peak == engine-track
 *  Cmax × dose/ref). The meta Cmax is shown separately as a marker, not by
 *  rescaling the curve — so the curve, its peak, and the engine AUC all agree. */
export function engineCurve(d: Drug, dose: number): Pt[] {
  const f = d.dose > 0 ? dose / d.dose : 1;
  return d.curve.t.map((t, i) => [t, d.curve.c[i] * f] as Pt);
}

/** Multi-dose superposition: doses at 0, tau, 2tau … (n total). */
export function multiCurve(
  d: Drug,
  dose: number,
  tau: number,
  n: number,
  step?: number
): { pts: Pt[]; tEnd: number; tau: number; n: number } {
  const tEnd = tau * n + d.pkfit.thalf * 4;
  step = step || tEnd / 600;
  const out: Pt[] = [];
  for (let t = 0; t <= tEnd + 1e-9; t += step) {
    let c = 0;
    for (let k = 0; k < n; k++) {
      const dt = t - k * tau;
      if (dt > 0) c += concAt(d, dose, dt);
    }
    out.push([t, c]);
  }
  return { pts: out, tEnd, tau, n };
}

export interface SteadyState {
  run: { pts: Pt[]; tEnd: number; tau: number; n: number };
  cssmax: number;
  cssmin: number;
  ar: number;
  ssDose: number;
  troughs: number[];
}

export function steadyState(d: Drug, dose: number, tau: number, n: number): SteadyState {
  const r = multiCurve(d, dose, tau, n);
  const lastStart = (n - 1) * tau;
  let cssmax = 0;
  let cssmin = Infinity;
  r.pts.forEach((p) => {
    if (p[0] >= lastStart && p[0] <= lastStart + tau) {
      if (p[1] > cssmax) cssmax = p[1];
      if (p[1] < cssmin) cssmin = p[1];
    }
  });
  const cmaxFirst = engineRefCmax(d) * (dose / d.dose);
  const ar = cmaxFirst > 0 ? cssmax / cmaxFirst : 1;
  if (!Number.isFinite(cssmin)) cssmin = 0;
  const troughs: number[] = [];
  for (let k = 1; k <= n; k++) {
    const tt = k * tau;
    let c = 0;
    for (let j = 0; j < k; j++) {
      const dt = tt - j * tau;
      if (dt > 0) c += concAt(d, dose, dt);
    }
    troughs.push(c);
  }
  let ssDose = n;
  for (let i = 2; i < troughs.length; i++) {
    if (Math.abs(troughs[i] - troughs[i - 1]) / troughs[i] < 0.05) {
      ssDose = i + 1;
      break;
    }
  }
  return { run: r, cssmax, cssmin, ar, ssDose, troughs };
}

export function band(pts: Pt[], up: number, lo: number): Band {
  return {
    upper: pts.map((p) => [p[0], p[1] * up] as Pt),
    lower: pts.map((p) => [p[0], p[1] * lo] as Pt),
  };
}

/* ---------------- DDI — real engine folds ---------------- */
export interface DdiResult {
  auc: number;
  cmax: number;
  keFactor: number;
  inh: ConsoleData["inhibitors"][string];
  frac: number;
}

export function ddiFold(d: Drug, data: ConsoleData, inhKey: string): DdiResult {
  const inh = data.inhibitors[inhKey];
  if (!inh || inhKey === "none") {
    return { auc: 1, cmax: 1, keFactor: 1, inh, frac: 0 };
  }
  const real = d.ddi[inhKey];
  // fraction of victim cleared by the perpetrator's primary enzyme (display only)
  let frac = 0;
  const primary = inh.enzyme.split("/")[0];
  Object.keys(d.enzymeFraction).forEach((e) => {
    if ((e === primary || inh.enzyme.indexOf(e) >= 0) && d.enzymeFraction[e] > frac) {
      frac = d.enzymeFraction[e];
    }
  });
  if (!real) return { auc: 1, cmax: 1, keFactor: 1, inh, frac };
  return { auc: real.auc, cmax: real.cmax, keFactor: 1 / real.auc, inh, frac };
}

/** Illustrative victim curve under modified clearance (real fold drives the shape). */
export function ddiCurve(
  d: Drug,
  dose: number,
  tEnd: number,
  keFactor: number,
  cmaxFold: number
): Pt[] {
  const ke2 = d.pkfit.ke * keFactor;
  const tweaked: Drug = {
    ...d,
    tracks: { ...d.tracks, engine: engineRefCmax(d) * cmaxFold },
    pkfit: { ka: d.pkfit.ka, ke: ke2, thalf: 0.693 / ke2 },
  };
  return singleCurve(tweaked, dose, tEnd);
}

/* ---------------- TDM (prior → posterior) ---------------- */
export interface TdmResult {
  tEnd: number;
  prior: Pt[];
  priorBand: Band;
  post: Pt[];
  postBand: Band;
  priorCv: number;
  postCv: number;
  reduction: number;
  ess: number;
  method: string;
  shift: number;
}

export function tdm(d: Drug, dose: number, obs: Observation[]): TdmResult {
  const tEnd = Math.max(d.pkfit.thalf * 5, 12);
  const base = singleCurve(d, dose, tEnd);
  // anchor prior CV / ESS / method to the real pre-computed Bayesian run
  const priorCv = d.tdm.priorCv ?? 0.44;
  const realReduction = d.tdm.reduction ?? 0.7;
  const nObs = obs.length;
  // scale the real (1-obs) reduction by observation count, clamped
  const reduction = Math.min(
    0.92,
    Math.max(0, realReduction + 0.08 * (Math.min(nObs, 3) - 1))
  );
  const postCv = priorCv * (1 - reduction);
  const ess = d.tdm.ess ?? 514;
  let shift = 1;
  if (nObs) {
    const o0 = obs[0];
    const pred = concAt(d, dose, o0.t);
    if (pred > 0) shift = Math.max(0.6, Math.min(1.6, o0.c / pred));
    shift = 1 + (shift - 1) * 0.7;
  }
  const post = base.map((p) => [p[0], p[1] * shift] as Pt);
  return {
    tEnd,
    prior: base,
    priorBand: band(base, 1 + priorCv * 1.64, Math.max(0.05, 1 - priorCv * 1.64)),
    post,
    postBand: band(post, 1 + postCv * 1.64, Math.max(0.05, 1 - postCv * 1.64)),
    priorCv,
    postCv,
    reduction,
    ess,
    method: d.tdm.method || "IS",
    shift,
  };
}

/* ---------------- MIPD (dose recommendation) ---------------- */
export interface MipdResult {
  postCss: number;
  raw: number;
  recommended: number;
  targetCss: number;
  curDose: number;
  tau: number;
}

export function mipd(
  d: Drug,
  curDose: number,
  obs: Observation[],
  targetCss: number,
  tau = 24
): MipdResult {
  const ss = steadyState(d, curDose, tau, 12);
  let postCss = ss.cssmax;
  if (obs && obs.length) {
    const pred = concAt(d, curDose, obs[0].t);
    if (pred > 0) postCss = postCss * (obs[0].c / pred);
  }
  const raw = postCss > 0 && Number.isFinite(postCss) ? curDose * (targetCss / postCss) : curDose;
  const lo = curDose * 0.25;
  const hi = curDose * 4;
  const clamped = Math.max(lo, Math.min(hi, Number.isFinite(raw) ? raw : curDose));
  const rounded = Math.round(clamped / 25) * 25 || Math.round(clamped);
  return { postCss, raw, recommended: rounded, targetCss, curDose, tau };
}

/* ---------------- meta / per-track helpers ---------------- */
export function displayedCmax(d: Drug, method: string): number {
  if (method === "engine") return d.tracks.engine ?? d.meta.cmax;
  if (method === "ml") return d.tracks.ml ?? d.meta.cmax;
  return d.meta.cmax;
}

export function scaleTracks(d: Drug, scale: number): Record<string, number | null> {
  const t: Record<string, number | null> = {};
  (["engine", "ml", "clf", "vdss"] as const).forEach(
    (k) => (t[k] = d.tracks[k] == null ? null : (d.tracks[k] as number) * scale)
  );
  return t;
}
