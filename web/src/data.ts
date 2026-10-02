/* ============================================================
   data.ts — the data layer.

   Presets and development evidence are static JSON. When VITE_API_URL is
   configured, the same client calls the FastAPI core for arbitrary SMILES.
   ============================================================ */
import { useEffect, useState } from "react";
import type { ConsoleData, Drug } from "./types";

const DATA_URL = `${import.meta.env.BASE_URL}data/console_data.json`;
// Set VITE_API_URL at build time to the live engine backend (FastAPI). Empty =
// static tier only (presets); arbitrary-SMILES prediction is then unavailable.
const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
const SUPPORTED_MODEL_MAJOR_MINOR = "0.4";

export interface PredictRequest {
  smiles: string;
  dose_mg: number;
  route: string;
  name?: string;
}

class SisyphusClient {
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
      if (!res.ok) return false;
      const info = (await res.json()) as { version?: string };
      const normalized = String(info.version || "").split("+")[0];
      return normalized === SUPPORTED_MODEL_MAJOR_MINOR || normalized.startsWith(`${SUPPORTED_MODEL_MAJOR_MINOR}.`);
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

export const engineClient = new SisyphusClient();

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
