/* ============================================================
   PredictView — SMILES → PK. Curve is the REAL engine ODE
   profile from the same solve; headline Cmax is the
   meta-learner output; the 90% band is a development residual
   interval; tracks/weights are the real applied values.
   ============================================================ */
import type { AppState, Drug } from "../../types";
import { ConcChart } from "../charts";
import { StatLine, TrackBars, BodyGraph, Legend, PipelineLog, fmt, type LogLine } from "../panels";
import { engineCurve, displayedCmax } from "../../pk";

function predictLog(drug: Drug, meta: number): LogLine[] {
  const eng = drug.tracks.engine ?? 0;
  const ml = drug.tracks.ml ?? 0;
  const activeTracks = Object.values(drug.weights).filter((w) => w != null && w > 0).length;
  return [
    { ts: "1", tag: "info", tagText: "chem", msg: `RDKit descriptors · MW <b>${drug.mw}</b> · type <b>${drug.type}</b>` },
    { ts: "2", tag: "ok", tagText: "ad", msg: "applicability domain — " + (drug.inDomain ? "<b>in-domain</b>" : "<b>out-of-domain</b> (flagged)") },
    { ts: "3", tag: "info", tagText: "adme", msg: `fᵤₚ=${fmt(drug.disposition.fup)} · CLᵢₙₜ · Rʙ:ₚ=1.0 · VDₛₛ` },
    { ts: "4", tag: "info", tagText: "ivive", msg: "CLᵢₙₜ contributes to engine clearance" },
    { ts: "5", tag: "ok", tagText: "engine", msg: `PBPK solve · mass-balance err <b>${fmt(drug.engineDiagnostics?.massBalanceError)}</b> · Cₘₐₓ <b>${fmt(eng)}</b>` },
    { ts: "6", tag: "info", tagText: "ml", msg: `direct XGBoost Cₘₐₓ <b>${fmt(ml)}</b>` },
    { ts: "7", tag: "ok", tagText: "meta", msg: `${activeTracks}-track geometric blend → Cₘₐₓ <b>${fmt(meta)} mg/L</b> · confidence <b>not calibrated</b>` },
  ];
}

export function PredictView({ drug, s, tab, running }: { drug: Drug; s: AppState; tab: number; running: boolean }) {
  const displayed = displayedCmax(drug, s.method);
  const cmax = displayed;
  const auc = drug.meta.auc;
  const meta = drug.meta.cmax;
  const engCmax = drug.tracks.engine ?? drug.meta.cmax;
  const activeWeights = Object.entries(drug.weights).filter(([, weight]) => weight != null && weight > 0);
  const curve = engineCurve(drug);
  const tEnd = drug.curve.t[drug.curve.t.length - 1] || Math.max(drug.meta.thalf * 4, 12);
  // does the headline (meta/ml) diverge from the engine-curve peak?
  const headlineDiffersFromCurve = Math.abs(displayed - engCmax) / Math.max(engCmax, 1e-9) > 0.02;
  const residualPi = s.method === "hybrid"
    ? (drug.residualInterval90 ?? drug.cmax90ci)
    : null;

  if (tab === 0)
    return (
      <div className="split">
        <div>
          <div className="panel">
            <h5>
              Plasma concentration · time
              <span className="meta">{s.route} · single dose · deterministic engine curve</span>
            </h5>
            <ConcChart
              series={[{ pts: curve, color: "var(--blue)" }]}
              bands={[]}
              vlines={[{ x: drug.meta.tmax, color: "var(--blue)" }]}
              hlines={headlineDiffersFromCurve ? [{ y: displayed, color: "var(--clay)", dash: "5 4" }] : []}
              points={[{ t: drug.meta.tmax, c: engCmax, color: "var(--blue)" }]}
              xMax={tEnd}
              h={272}
            />
          </div>
          <div style={{ height: 14 }} />
          <StatLine
            items={[
              { k: "C<sub>max</sub>", v: fmt(cmax), u: "mg/L", ci: residualPi ? "nominal 90% development residual band " + fmt(residualPi[0]) + "–" + fmt(residualPi[1]) : "no validated interval for this track" },
              { k: "T<sub>max</sub>", v: fmt(drug.meta.tmax), u: "h" },
              { k: "t½", v: fmt(drug.meta.thalf), u: "h" },
              { k: "AUC<sub>0–t</sub>", v: fmt(auc), u: "mg·h/L" },
            ]}
          />
          <div className="figcap">
            <b>FIG.</b> Engine ODE profile for {drug.name} {s.dose} mg {s.route}; the dot marks the engine-track C<sub>max</sub>.{" "}
            {headlineDiffersFromCurve
              ? <>Dashed line = the <span style={{ fontStyle: "normal" }}>{s.method}</span> C<sub>max</sub> ({fmt(displayed)} mg/L), the production estimate. </>
              : null}
            The Meta band is an empirical development-residual interval. Its observed coverage was 75.6% on the repeatedly used development set; it has no independent calibration guarantee.
          </div>
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Method · {s.method}</h5>
            <p className="note" style={{ margin: 0 }}>
              {s.method === "hybrid" ? (
                <span>
                  <b>Hybrid meta-learner.</b> Geometric blend of {activeWeights.length} active tracks; applied weights below.
                </span>
              ) : s.method === "engine" ? (
                <span>
                  <b>Engine only.</b> PBPK ODE, no ML correction.
                </span>
              ) : (
                <span>
                  <b>ML only.</b> Direct XGBoost C<sub>max</sub>.
                </span>
              )}
            </p>
            <div style={{ marginTop: 12 }}>
              <TrackBars
                drug={{ tracks: drug.tracks, weights: drug.weights, primaryEnzyme: drug.primaryEnzyme }}
                meta={s.method === "hybrid" ? meta : null}
                showWeights
              />
            </div>
          </div>
          <div className="panel">
            <h5>Disposition</h5>
            <div className="kv"><span className="kk">Dose/AUC<sub>0–24h</sub></span><span className="vv">{fmt(drug.disposition.doseOverAuc0t)} L/h</span></div>
            <div className="kv"><span className="kk">V<sub>d</sub> (V<sub>ss</sub>·70)</span><span className="vv">{fmt(drug.disposition.vdss != null ? drug.disposition.vdss * 70 : null)} L</span></div>
            <div className="kv"><span className="kk">f<sub>u,p</sub></span><span className="vv">{fmt(drug.disposition.fup)}</span></div>
            <div className="kv"><span className="kk">k<sub>a</sub></span><span className="vv">{fmt(drug.pkfit.ka)} h⁻¹</span></div>
            <div className="kv"><span className="kk">Mass balance error</span><span className="vv">{fmt(drug.engineDiagnostics?.massBalanceError)}</span></div>
          </div>
        </div>
      </div>
    );

  if (tab === 1)
    return (
      <div className="split">
        <div className="panel">
          <h5>
            {activeWeights.length}-track meta-learner
            <span className="meta">{activeWeights.map(([name, weight]) => `${name} ${((weight ?? 0) * 100).toFixed(0)}%`).join(" / ")}</span>
          </h5>
          <div style={{ marginTop: 4 }}>
            <TrackBars
              drug={{ tracks: drug.tracks, weights: drug.weights, primaryEnzyme: drug.primaryEnzyme }}
              meta={meta}
              showWeights
            />
          </div>
          <div className="divider" style={{ margin: "16px 0" }} />
          <p className="note" style={{ margin: 0 }}>
              Available mechanistic Engine, direct XGBoost C<sub>max</sub> (ML), closed-form CL/F, and conditional VDss tracks use partly different signals. The compound-type-adaptive weights were selected on the original N=107 development cohort and therefore require confirmation on a new blinded holdout.
          </p>
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Per-track C<sub>max</sub></h5>
            {(["engine", "ml", "clf", "vdss"] as const).map((k) => (
              <div className="kv" key={k}>
                <span className="kk" style={{ textTransform: "capitalize" }}>{k === "clf" ? "CL/F" : k === "vdss" ? "VDss" : k}</span>
                <span className="vv">{drug.tracks[k] == null ? "— off" : fmt(drug.tracks[k] as number) + " mg/L"}</span>
              </div>
            ))}
            <div className="kv">
              <span className="kk" style={{ color: "var(--ink)", fontWeight: 600 }}>Meta (blend)</span>
              <span className="vv accent">{fmt(meta)} mg/L</span>
            </div>
          </div>
          <div className="panel">
            <h5>Decorrelation</h5>
            <p className="note" style={{ margin: 0, fontSize: 12 }}>
              Development-set performance does not establish that the blend is optimal. Further weight tuning on this consumed cohort risks adaptive overfitting; improvements should be judged once on the preregistered external holdout.
            </p>
          </div>
        </div>
      </div>
    );

  if (tab === 2)
    return (
      <div className="stack">
        <div className="panel">
          <h5>
            Body graph<span className="meta">34 compartments · identity-blind engine</span>
          </h5>
          <BodyGraph drug={drug} />
          <div className="divider" style={{ margin: "16px 0 12px" }} />
          <Legend
            items={[
              { type: "rect", color: "var(--blue-soft)", label: "blood pool" },
              { type: "rect", color: "var(--clay-soft)", label: "clearing organ" },
              { type: "rect", color: "var(--surface-2)", label: "perfusion-limited" },
              { type: "dash", color: "var(--hair)", label: "permeability-limited" },
            ]}
          />
        </div>
        <div className="split-even">
          <div className="panel">
            <h5>Active clearance site</h5>
            <div className="kv"><span className="kk">Primary route</span><span className="vv accent">{drug.primaryEnzyme}</span></div>
            <div className="kv"><span className="kk">Hepatic model</span><span className="vv">{drug.primaryEnzyme === "renal" ? "GFR filtration" : "well-stirred"}</span></div>
            <div className="kv"><span className="kk">CL<sub>int</sub></span><span className="vv">{fmt(drug.disposition.clint)}</span></div>
          </div>
          <div className="panel">
            <h5>Compile invariant</h5>
            <p className="note" style={{ margin: 0, fontSize: 11.5 }}>
              The ODE is derived from graph <b>topology</b>, never organ identity. Replacing every organ name with a random string yields identical numerics. Compiled once; parameterized many.
            </p>
          </div>
        </div>
      </div>
    );

  // log tab
  return (
    <div className="panel">
      <h5>
        Model stages<span className="meta">order shown · timing not measured</span>
      </h5>
      <PipelineLog running={running} lines={predictLog(drug, meta)} />
    </div>
  );
}
