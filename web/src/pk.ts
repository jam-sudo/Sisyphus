/* Core Cmax and engine-curve display helpers. */
import type { Drug } from "./types";

export type Pt = [number, number];
export interface Band {
  upper: Pt[];
  lower: Pt[];
}

/** Engine ODE curve from the exact solve represented by this payload. */
export function engineCurve(d: Drug): Pt[] {
  return d.curve.t.map((t, i) => [t, d.curve.c[i]] as Pt);
}

export function displayedCmax(d: Drug, method: string): number {
  if (method === "engine") return d.tracks.engine ?? d.meta.cmax;
  if (method === "ml") return d.tracks.ml ?? d.meta.cmax;
  return d.meta.cmax;
}
