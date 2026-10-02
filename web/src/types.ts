/* ============================================================
   types.ts — contracts for the real-engine console data.
   Mirrors the JSON produced by scripts/gen_console_data.py,
   which is itself produced by the real Sisyphus PBPK engine.
   ============================================================ */

export type Confidence = "medium" | "low";
export type CompoundType = "neutral" | "acid" | "base" | "zwitterion";
export type Route = "oral";

/** A single (time, concentration) curve. Parallel arrays for compactness. */
export interface Curve {
  t: number[]; // hours
  c: number[]; // mg/L (venous_blood plasma)
}

/** 1-compartment-with-absorption approximation inferred from engine Tmax
 *  and half-life, used for client-side multi-dose interactivity. */
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
  doseOverAuc0t: number | null; // dose/AUC0–24h, L/h; not terminal CL/F
  vdss: number | null; // L/kg (engine ADME)
  fup: number | null;
  clint: number | null;
}

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
  applicabilityStatus?: "structurally_in_scope" | "flagged";
  inDomain: boolean;
  adFlags: string[];
  executionStatus?: string;
  artifactProvenance?: Record<string, string>;
  meta: MetaEndpoints;
  endpointSources?: Record<string, string | null>;
  cmax90ci: [number, number] | null; // residual 90% PI
  intervalSource?: string | null;
  residualInterval90?: [number, number] | null;
  residualIntervalSource?: string | null;
  parameterInterval90?: [number, number] | null;
  parameterIntervalSource?: string | null;
  tracks: Tracks;
  weights: Weights;
  disposition: Disposition;
  curve: Curve; // real engine single-dose response at `dose`
  pkfit: PkFit;
  engineDiagnostics?: {
    observationNode: string;
    solverSuccess: boolean;
    massBalanceError: number;
  };
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
  n_development: number;
  classification: "retrospective_development_benchmark";
  overall: { engine: TrackBlock; ml: TrackBlock; meta: TrackBlock };
  in_domain: { n: number; engine: TrackBlock; ml: TrackBlock; meta: TrackBlock };
  paired_meta_ml: { ratio: number; ci_95_low: number; ci_95_high: number; n: number };
  scatter: ScatterPoint[];
}

export interface Constants {
  DEVELOPMENT_AAFE: number;
  ENGINE_AAFE: number;
  ML_AAFE: number;
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
  benchmark: BenchmarkData;
  drugs: Drug[];
}

/* ---------------- UI state ---------------- */
export interface AppState {
  drugId: string;
  dose: number;
  route: Route;
  method: "hybrid" | "engine" | "ml";
  benchSet: string;
}

export type WorkflowId = "predict" | "benchmark";

export interface WorkflowCfg {
  id: WorkflowId;
  label: string;
  desc: string;
  tabs: string[];
}
