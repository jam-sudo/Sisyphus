/* ============================================================
   data.ts — the data layer.

   Phase 1 (now): a StaticEngineClient reads pre-computed REAL
   Sisyphus-engine outputs from public/data/console_data.json.

   Phase 2 (later): an ApiEngineClient implementing the same
   EngineClient interface will hit a FastAPI backend wrapping
   pipeline.predict for arbitrary SMILES. The UI imports only
   the interface, so swapping clients needs no view changes.
   ============================================================ */
import { useEffect, useState } from "react";
import type { ConsoleData, Drug } from "./types";

const DATA_URL = `${import.meta.env.BASE_URL}data/console_data.json`;
// Set VITE_API_URL at build time to the live engine backend (FastAPI). Empty =
// static tier only (presets); arbitrary-SMILES prediction is then unavailable.
const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");

export interface PredictRequest {
  smiles: string;
  dose_mg: number;
  route: string;
  name?: string;
}

export interface EngineClient {
  /** Pre-computed presets + benchmark + inhibitors + constants (always static). */
  load(): Promise<ConsoleData>;
  /** Is a live backend URL configured at build time? */
  apiConfigured(): boolean;
  /** Probe the live backend; false if absent/unreachable. */
  health(): Promise<boolean>;
  /** Live arbitrary-SMILES prediction → a Drug entry (needs the backend). */
  predict(req: PredictRequest): Promise<Drug>;
}

class SisyphusClient implements EngineClient {
  private cache: ConsoleData | null = null;

  async load(): Promise<ConsoleData> {
    if (this.cache) return this.cache;
    const res = await fetch(DATA_URL, { cache: "no-cache" });
    if (!res.ok) {
      throw new Error(
        `Could not load engine data (${res.status}). Run scripts/gen_console_data.py to (re)generate web/public/data/console_data.json.`
      );
    }
    const data = (await res.json()) as ConsoleData;
    this.cache = data;
    return data;
  }

  apiConfigured(): boolean {
    return !!API_BASE;
  }

  async health(): Promise<boolean> {
    if (!API_BASE) return false;
    try {
      const res = await fetch(`${API_BASE}/health`, { method: "GET" });
      return res.ok;
    } catch {
      return false;
    }
  }

  async predict(req: PredictRequest): Promise<Drug> {
    if (!API_BASE) throw new Error("Live engine is not configured.");
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
    if (!res.ok) {
      let detail = `Prediction failed (${res.status}).`;
      try {
        const e = await res.json();
        if (e && e.detail) detail = e.detail;
      } catch {
        /* ignore */
      }
      throw new Error(detail);
    }
    return (await res.json()) as Drug;
  }
}

export const engineClient: EngineClient = new SisyphusClient();

export interface DataHookState {
  data: ConsoleData | null;
  error: string | null;
  loading: boolean;
}

/** Loads the console data once and exposes loading/error state. */
export function useConsoleData(): DataHookState {
  const [state, setState] = useState<DataHookState>({
    data: null,
    error: null,
    loading: true,
  });
  useEffect(() => {
    let alive = true;
    engineClient
      .load()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .then(undefined, (e: unknown) =>
        alive &&
        setState({
          data: null,
          error: e instanceof Error ? e.message : String(e),
          loading: false,
        })
      );
    return () => {
      alive = false;
    };
  }, []);
  return state;
}

export function drugById(data: ConsoleData, id: string): Drug {
  return data.drugs.find((d) => d.id === id) ?? data.drugs[0];
}
