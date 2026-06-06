/* ============================================================
   App.tsx — Sisyphus console shell: rail, nav, run flow, state.
   Loads real-engine data via the data layer; ported from app.jsx.
   ============================================================ */
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { AppState, Observation, WorkflowId } from "../types";
import { useConsoleData, drugById } from "../data";
import { Pill } from "./panels";
import { RailInputs } from "./RailInputs";
import { WorkflowView, WORKFLOWS, oid } from "./workflows";

const DEFAULT_STATE: AppState = {
  drugId: "caffeine",
  dose: 100,
  route: "oral",
  method: "hybrid",
  nmc: 1000,
  interval: 24,
  nDoses: 14,
  obs: [{ id: 0, t: 0.673, c: 0.953 }],
  assayCv: "10%",
  inhibitor: "ketoconazole",
  targetCss: 0.9,
  benchSet: "scaffold",
};

const num = (v: unknown, fallback: number): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;

/** Coerce an arbitrary (possibly stale/malformed) persisted blob into a valid
 *  AppState — whitelisting keys/types so old localStorage shapes can't crash. */
function sanitize(raw: unknown): AppState {
  const p = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const b = DEFAULT_STATE;
  let obs: Observation[] = b.obs.map((o) => ({ ...o, id: oid() }));
  if (Array.isArray(p.obs)) {
    const cleaned = p.obs
      .filter((o): o is Record<string, unknown> => !!o && typeof o === "object")
      .map((o) => ({ id: oid(), t: num((o as Record<string, unknown>).t, 1), c: num((o as Record<string, unknown>).c, 0) }));
    if (cleaned.length) obs = cleaned;
  }
  return {
    drugId: typeof p.drugId === "string" ? p.drugId : b.drugId,
    dose: Math.max(0, num(p.dose, b.dose)),
    route: "oral", // static tier is oral-only (IV arrives with the live engine tier)
    method: p.method === "engine" || p.method === "ml" ? p.method : "hybrid",
    nmc: num(p.nmc, b.nmc),
    interval: Math.max(0.25, num(p.interval, b.interval)),
    nDoses: Math.max(2, Math.round(num(p.nDoses, b.nDoses))),
    obs,
    assayCv: typeof p.assayCv === "string" ? p.assayCv : b.assayCv,
    inhibitor: typeof p.inhibitor === "string" ? p.inhibitor : b.inhibitor,
    targetCss: Math.max(0, num(p.targetCss, b.targetCss)),
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
  predict: "Run prediction",
  simulate: "Simulate regimen",
  tdm: "Update posterior",
  ddi: "Apply interaction",
  "dose-adjust": "Recommend dose",
  benchmark: "Run benchmark",
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
  const [running, setRunning] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const runTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = (patch: Partial<AppState>) => setS((prev) => ({ ...prev, ...patch }));

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

  const drug = drugById(data, s.drugId);
  const wfCfg = WORKFLOWS.find((x) => x.id === wf) ?? WORKFLOWS[0];
  const tabs = wfCfg.tabs;
  const safeTab = Math.min(tab, tabs.length - 1);

  function run() {
    setRunning(true);
    if (runTimer.current) clearTimeout(runTimer.current);
    runTimer.current = setTimeout(() => {
      setRunning(false);
      setToast(wf === "benchmark" ? "benchmark complete · N=107" : "prediction complete · " + drug.name);
      setTimeout(() => setToast(null), 1900);
    }, 620);
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
  } else {
    badges = (
      <>
        <Pill kind={drug.inDomain ? "dom" : "warn"}>{drug.inDomain ? "in domain" : "out of domain"}</Pill>
        <Pill kind={drug.confidence === "high" ? "ok" : drug.confidence === "medium" ? "dom" : "warn"}>{drug.confidence}</Pill>
        {drug.hasPD && <span className="pdtag">PK/PD · {drug.hasPD}</span>}
      </>
    );
  }

  const runHint =
    wf === "benchmark"
      ? "10,000 bootstrap resamples"
      : wf === "predict" && s.nmc > 1
      ? "MC N=" + s.nmc.toLocaleString() + " · ~" + (s.nmc / 30).toFixed(0) + " s"
      : "deterministic · ~414 ms";

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
            <RailInputs wf={wf} s={s} set={set} data={data} />
          </div>
          <div className="rail-run">
            <button className="btn-run" onClick={run} disabled={running}>
              {running ? (
                <>
                  <span className="spin" />
                  solving…
                </>
              ) : (
                <>{RUN_LABELS[wf]}&nbsp; →</>
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
                  Holdout validation <span className="sub">SMILES → Cₘₐₓ · external</span>
                </div>
              ) : (
                <div className="dn">
                  {drug.name} <span className="sub">{drug.formula} · {drug.mw}</span>
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
          <div className={"content" + (running ? " running" : "")}>
            <WorkflowView wf={wf} s={s} tab={safeTab} running={running} data={data} />
          </div>
          <div className="provenance">
            <span className="dot" />
            real Sisyphus engine · {data.meta_info.engine} · pre-computed static tier
          </div>
        </section>

        {toast && <div className="toast">{toast}</div>}
      </div>
    </div>
  );
}
