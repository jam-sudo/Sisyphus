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

/** The seam the live engine will implement in Phase 2. */
export interface EngineClient {
  /** All pre-computed drugs + benchmark + constants. */
  load(): Promise<ConsoleData>;
  /** Phase 2: live arbitrary-SMILES prediction (not wired in the static tier). */
  isLive(): boolean;
}

export class StaticEngineClient implements EngineClient {
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
  isLive() {
    return false;
  }
}

export const engineClient: EngineClient = new StaticEngineClient();

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
