import type { WorkflowCfg, Drug, Observation } from "../../types";

export const WORKFLOWS: WorkflowCfg[] = [
  { id: "predict", label: "predict", desc: "SMILES → PK", tabs: ["PK Profile", "Tracks", "Body Graph", "Uncertainty", "Log"] },
  { id: "simulate", label: "simulate", desc: "multi-dose", tabs: ["Regimen", "Steady State", "Label Check"] },
  { id: "tdm", label: "tdm", desc: "Bayesian", tabs: ["Posterior", "Diagnostics"] },
  { id: "ddi", label: "ddi", desc: "interactions", tabs: ["Interaction", "Mechanism"] },
  { id: "dose-adjust", label: "dose-adjust", desc: "MIPD", tabs: ["Recommendation"] },
  { id: "benchmark", label: "benchmark", desc: "validation", tabs: ["Holdout N=107", "Prospective", "Tracks"] },
];

// monotonic id source for observation rows (stable React keys)
let _oid = 0;
export const oid = (): number => ++_oid;

export function defObs(d: Drug): Observation[] {
  return [{ id: oid(), t: d.meta.tmax, c: +(d.meta.cmax * 0.92).toFixed(4) }];
}
