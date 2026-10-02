/* ============================================================
   App.tsx — Sisyphus console shell: rail, nav, run flow, state.
   Loads real-engine data via the data layer; ported from app.jsx.
   ============================================================ */
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { AppState, Drug, WorkflowId } from "../types";
import { useConsoleData, drugById, engineClient } from "../data";
import { Pill } from "./panels";
import { RailInputs } from "./RailInputs";
import { WorkflowView, WORKFLOWS } from "./workflows";

const DEFAULT_STATE: AppState = {
  drugId: "caffeine",
  dose: 100,
  route: "oral",
  method: "hybrid",
  benchSet: "scaffold",
};

const num = (v: unknown, fallback: number): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

/** Coerce an arbitrary (possibly stale/malformed) persisted blob into a valid
 *  AppState — whitelisting keys/types so old localStorage shapes can't crash. */
function sanitize(raw: unknown): AppState {
  const p = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const b = DEFAULT_STATE;
  return {
    drugId: typeof p.drugId === "string" && p.drugId !== "custom" ? p.drugId : b.drugId,
    dose: Math.max(0, num(p.dose, b.dose)),
    route: "oral", // static tier is oral-only (IV arrives with the live engine tier)
    method: p.method === "engine" || p.method === "ml" ? p.method : "hybrid",
    benchSet: typeof p.benchSet === "string" ? p.benchSet : b.benchSet,
  };
}

function loadState(): AppState {
  try {
    const raw = localStorage.getItem("sisyphus_state");
    if (raw) return sanitize(JSON.parse(raw));
  } catch {
    /* ignore */
  }
  return sanitize(null);
}

const RUN_LABELS: Record<WorkflowId, string> = {
  predict: "View prediction",
  benchmark: "View benchmark",
};

export function App() {
  const { data, error, loading } = useConsoleData();

  const [wf, setWf] = useState<WorkflowId>(() => {
    try {
      const stored = localStorage.getItem("sisyphus_wf");
      return WORKFLOWS.some((w) => w.id === stored) ? (stored as WorkflowId) : "predict";
    } catch {
      return "predict";
    }
  });
  const [s, setS] = useState<AppState>(loadState);
  const [tab, setTab] = useState(0);
  const [toast, setToast] = useState<string | null>(null);

  // live engine (arbitrary-SMILES) state
  const [live, setLive] = useState(false);
  const [customDrug, setCustomDrug] = useState<Drug | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [smilesDraft, setSmilesDraft] = useState("");
  const [nameDraft, setNameDraft] = useState("");
  const requestKey = JSON.stringify([s.drugId, s.dose, s.route, smilesDraft.trim(), nameDraft.trim()]);
  const requestKeyRef = useRef(requestKey);
  requestKeyRef.current = requestKey;

  const set = (patch: Partial<AppState>) => setS((prev) => ({ ...prev, ...patch }));

  // probe the live backend once
  useEffect(() => {
    let alive = true;
    engineClient.health().then((ok) => alive && setLive(ok));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("sisyphus_state", JSON.stringify(s));
    } catch {
      /* ignore */
    }
  }, [s]);
  useEffect(() => {
    try {
      localStorage.setItem("sisyphus_wf", wf);
    } catch {
      /* ignore */
    }
  }, [wf]);

  // Static presets are immutable records of one exact engine solve. Never
  // fabricate a new dose by rescaling their tracks or concentration curve.
  useEffect(() => {
    if (!data || s.drugId === "custom") return;
    const preset = drugById(data, s.drugId);
    setS((prev) =>
      prev.dose === preset.dose && prev.route === "oral"
        ? prev
        : { ...prev, dose: preset.dose, route: "oral" }
    );
  }, [data, s.drugId]);

  // A live result is valid only for the exact SMILES+dose request that made it.
  useEffect(() => {
    if (s.drugId === "custom") setCustomDrug(null);
  }, [s.drugId, s.dose, s.route, smilesDraft, nameDraft]);

  if (loading) {
    return (
      <div className="bootscreen">
        <div className="mark" />
        <div className="spin" />
        <div>loading engine data …</div>
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="bootscreen err">
        <div className="mark" />
        <div>Could not load engine data.</div>
        <div style={{ maxWidth: 480, textAlign: "center", lineHeight: 1.6 }}>{error}</div>
        <div>
          Generate it with <code>/opt/miniconda3/bin/python scripts/gen_console_data.py</code>
        </div>
      </div>
    );
  }

  const isCustom = s.drugId === "custom";
  const activeDrug: Drug | null = isCustom ? customDrug : drugById(data, s.drugId);
  const wfCfg = WORKFLOWS.find((x) => x.id === wf) ?? WORKFLOWS[0];
  const tabs = wfCfg.tabs;
  const safeTab = Math.min(tab, tabs.length - 1);
  const customPredictMode = wf === "predict" && isCustom;
  const busy = predicting;

  async function run() {
    if (customPredictMode) {
      const submittedKey = requestKeyRef.current;
      const smiles = smilesDraft.trim();
      if (!smiles) return setPredictError("Enter a SMILES string.");
      if (!live) return setPredictError("Live engine is offline — pick a preset compound.");
      setPredicting(true);
      setPredictError(null);
      try {
        const d = await engineClient.predict({
          smiles,
          dose_mg: s.dose,
          route: s.route,
          name: nameDraft.trim() || undefined,
        });
        if (requestKeyRef.current === submittedKey) {
          setCustomDrug(d);
          setToast("prediction complete · " + d.name);
          setTimeout(() => setToast(null), 1900);
        }
      } catch (e) {
        if (requestKeyRef.current === submittedKey)
          setPredictError(e instanceof Error ? e.message : String(e));
      } finally {
        setPredicting(false);
      }
      return;
    }
    setTab(0);
    setToast(wf === "benchmark" ? `showing development benchmark · N=${data?.benchmark.n_development ?? "?"}` : "showing frozen prediction · " + (activeDrug?.name ?? ""));
    setTimeout(() => setToast(null), 1900);
  }

  function changeWf(id: WorkflowId) {
    setWf(id);
    setTab(0);
  }

  let badges: ReactNode;
  if (wf === "benchmark") {
    badges = (
      <>
        <Pill kind="dom">scaffold split</Pill>
        <Pill kind="mute">seed 42</Pill>
      </>
    );
  } else if (!activeDrug) {
    badges = live ? <Pill kind="mute">awaiting SMILES</Pill> : <Pill kind="warn">live engine offline</Pill>;
  } else {
    badges = (
      <>
        <Pill kind={activeDrug.inDomain ? "dom" : "warn"}>{activeDrug.inDomain ? "in domain" : "out of domain"}</Pill>
        <Pill kind={activeDrug.inDomain ? "dom" : "warn"}>confidence not calibrated</Pill>
      </>
    );
  }

  const runHint = customPredictMode
    ? live ? "live engine prediction" : "live engine offline"
    : wf === "benchmark"
    ? "precomputed bootstrap summary"
    : "frozen reference-dose prediction";

  return (
    <div className="stage">
      <div className="app">
        {/* RAIL */}
        <aside className="rail">
          <div className="rail-top">
            <div className="brand">
              <div className="mark" />
              <div className="nm">Sisyphus</div>
              <div className="ver">v0.4</div>
            </div>
            <div className="navlab">Workflow</div>
            <nav className="nav">
              {WORKFLOWS.map((x) => (
                <button key={x.id} className={wf === x.id ? "on" : ""} onClick={() => changeWf(x.id)}>
                  <span className="g" />
                  {x.label}
                  <span className="desc">{x.desc}</span>
                </button>
              ))}
            </nav>
          </div>
          <div className="rail-fields">
            <RailInputs
              wf={wf}
              s={s}
              set={set}
              data={data}
              live={live}
              smilesDraft={smilesDraft}
              setSmilesDraft={setSmilesDraft}
              nameDraft={nameDraft}
              setNameDraft={setNameDraft}
              predictError={predictError}
            />
          </div>
          <div className="rail-run">
            <button
              className="btn-run"
              onClick={run}
              disabled={busy || (customPredictMode && (!live || !smilesDraft.trim()))}
            >
              {busy ? (
                <>
                  <span className="spin" />
                  {predicting ? "predicting…" : "solving…"}
                </>
              ) : (
                <>{customPredictMode ? "Predict SMILES" : RUN_LABELS[wf]}&nbsp; →</>
              )}
            </button>
            <div className="run-hint">{runHint}</div>
          </div>
        </aside>

        {/* MAIN */}
        <section className="main">
          <div className="mbar">
            <div className="title">
              {wf === "benchmark" ? (
                <div className="dn">
                  Development benchmark <span className="sub">SMILES + dose + route → Cₘₐₓ · not independent</span>
                </div>
              ) : activeDrug ? (
                <div className="dn">
                  {activeDrug.name} <span className="sub">{activeDrug.formula} · {activeDrug.mw}</span>
                </div>
              ) : (
                <div className="dn">
                  Custom compound <span className="sub">enter a SMILES →</span>
                </div>
              )}
            </div>
            <div className="badges">{badges}</div>
          </div>
          <div className="tabs">
            {tabs.map((t, i) => (
              <button key={t} className={safeTab === i ? "on" : ""} onClick={() => setTab(i)}>
                {t}
              </button>
            ))}
          </div>
          <div className={"content" + (busy ? " running" : "")}>
            {wf !== "benchmark" && !activeDrug ? (
              <div className="custom-empty">
                {live
                  ? "Enter a SMILES string in the rail, then hit Predict SMILES."
                  : "Live engine is offline — select a preset compound to explore."}
              </div>
            ) : (
              <WorkflowView wf={wf} s={s} tab={safeTab} running={busy} data={data} drug={activeDrug ?? undefined} />
            )}
          </div>
          <div className="provenance">
            <span className="dot" />
            real Sisyphus engine · {data.meta_info.engine} · {isCustom && activeDrug ? "live prediction" : "precomputed static tier"}
          </div>
        </section>

        {toast && <div className="toast">{toast}</div>}
      </div>
    </div>
  );
}
