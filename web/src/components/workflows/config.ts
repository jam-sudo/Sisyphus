import type { WorkflowCfg } from "../../types";

export const WORKFLOWS: WorkflowCfg[] = [
  { id: "predict", label: "predict", desc: "structure → Cmax", tabs: ["Cmax + Engine", "Tracks", "Body Graph", "Log"] },
  { id: "benchmark", label: "benchmark", desc: "development evidence", tabs: ["Development N=98", "Temporal challenge", "Tracks"] },
];
