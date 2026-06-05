/* ============================================================
   types.ts — contracts for the real-engine console data.
   Mirrors the JSON produced by scripts/gen_console_data.py,
   which is itself produced by the real Sisyphus PBPK engine.
   ============================================================ */

export type Confidence = "high" | "medium" | "low";
export type CompoundType = "neutral" | "acid" | "base" | "zwitterion";
export type Route = "oral" | "iv";

/** A single (time, concentration) curve. Parallel arrays for compactness. */
export interface Curve {
  t: number[]; // hours
  c: number[]; // mg/L (venous_blood plasma)
}

/** 1-compartment-with-absorption fit to the real engine curve, used for
 *  client-side multi-dose superposition / dose-response interactivity. */
export interface PkFit {
  ka: number; // /h
  ke: number; // /h
  thalf: number; // h
}

export interface MetaEndpoints {
  cmax: number; // mg/L
  tmax: number; // h
  auc: number; // mg·h/L (AUC0-t)
  thalf: number; // h
}

export interface Tracks {
  engine: number | null;
  ml: number | null;
  clf: number | null;
  vdss: number | null;
  [k: string]: number | null;
}

export type Weights = Tracks;

export interface Disposition {
  clf: number | null; // CL/F, L/h (= dose/AUC for oral)
  vdss: number | null; // L/kg (engine ADME)
  fup: number | null;
  clint: number | null;
}

export interface TdmInfo {
  method: string; // production-routed Bayesian method (SBI / IBIS / IS …)
  ess: number | null; // effective sample size
  priorCv: number | null;
  postCv: number | null;
  reduction: number | null; // fraction (0..1)
}

export interface SimulateInfo {
  interval: number; // h (default regimen)
  nDoses: number;
  cssMax: number | null;
  cssMin: number | null;
  accumRatio: number | null;
  doseToSS: number | null;
  troughs: number[];
}

export interface DdiFold {
  auc: number; // fold change in AUC
  cmax: number; // fold change in Cmax
}

export type DdiMap = Record<string, DdiFold>;

export interface Drug {
  id: string;
  name: string;
  formula: string;
  mw: number;
  smiles: string;
  type: CompoundType;
  dose: number; // reference dose (mg)
  route: Route;
  primaryEnzyme: string;
  enzymeFraction: Record<string, number>;
  confidence: Confidence;
  inDomain: boolean;
  adFlags: string[];
  meta: MetaEndpoints;
  cmax90ci: [number, number]; // conformal 90% PI
  tracks: Tracks;
  weights: Weights;
  disposition: Disposition;
  curve: Curve; // real engine single-dose response at `dose`
  pkfit: PkFit;
  tdm: TdmInfo;
  simulate: SimulateInfo;
  ddi: DdiMap;
  /** Optional UI extras some drugs carry. */
  hasPD?: string;
  fdaCssMax?: number | null;
}

export interface Inhibitor {
  name: string;
  enzyme: string;
  type: string; // "inhibition" | "induction" | "—"
  strength: string | null;
}

export interface ScatterPoint {
  name: string;
  obs: number;
  eng: number;
  ml: number;
  meta: number;
  in_ad: boolean;
}

export interface TrackBlock {
  n: number;
  aafe: number;
  pct_2fold: number;
  pct_3fold?: number;
}

export interface BenchmarkData {
  n_holdout: number;
  overall: { engine: TrackBlock; ml: TrackBlock; meta: TrackBlock };
  in_domain: { n: number; engine: TrackBlock; ml: TrackBlock; meta: TrackBlock };
  scatter: ScatterPoint[];
}

export interface Constants {
  HOLDOUT_AAFE: number;
  INDOMAIN_AAFE: number;
  ENGINE_AAFE: number;
  ML_AAFE: number;
  conformal_factor: number;
}

export interface MetaInfo {
  generated_by: string;
  interpreter: string;
  engine: string;
  notes: string[];
}

export interface ConsoleData {
  meta_info: MetaInfo;
  constants: Constants;
  inhibitors: Record<string, Inhibitor>;
  benchmark: BenchmarkData;
  drugs: Drug[];
}

/* ---------------- UI state ---------------- */
export interface Observation {
  id: number; // stable key for list editing
  t: number;
  c: number;
}

export interface AppState {
  drugId: string;
  dose: number;
  route: Route;
  method: "hybrid" | "engine" | "ml";
  nmc: number;
  interval: number;
  nDoses: number;
  obs: Observation[];
  assayCv: string;
  inhibitor: string;
  targetCss: number;
  benchSet: string;
}

export type WorkflowId =
  | "predict"
  | "simulate"
  | "tdm"
  | "ddi"
  | "dose-adjust"
  | "benchmark";

export interface WorkflowCfg {
  id: WorkflowId;
  label: string;
  desc: string;
  tabs: string[];
}
