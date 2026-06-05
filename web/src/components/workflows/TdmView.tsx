/* ============================================================
   TdmView — Bayesian therapeutic drug monitoring.
   Prior/posterior CV, ESS, and routed method are anchored to a
   real bayesian_update run; observation edits adjust the shrink.
   ============================================================ */
import type { AppState, Drug } from "../../types";
import { ConcChart } from "../charts";
import { Legend, Caveat, Pill, fmt, type PillKind } from "../panels";
import { tdm } from "../../pk";

function methodKindOf(method: string): PillKind {
  const m = method.toLowerCase();
  if (m.indexOf("sbi") === 0) return "ok";
  if (m === "ibis") return "warn";
  return "mute";
}

export function TdmView({ drug, s, tab }: { drug: Drug; s: AppState; tab: number }) {
  const r = tdm(drug, s.dose, s.obs);
  const methodDisp = r.method.toUpperCase();
  const methodKind = methodKindOf(r.method);

  if (tab === 0)
    return (
      <div className="split">
        <div>
          <div className="panel">
            <h5>
              Prior → posterior
              <span className="meta">{methodDisp} · {s.obs.length} obs · {s.assayCv} assay CV</span>
            </h5>
            <ConcChart
              series={[
                { pts: r.prior, color: "var(--ink-mute)", dash: "5 4", width: 1.6 },
                { pts: r.post, color: "var(--blue)", width: 2.2 },
              ]}
              bands={[
                { upper: r.priorBand.upper, lower: r.priorBand.lower, color: "var(--hair)", opacity: 0.5 },
                { upper: r.postBand.upper, lower: r.postBand.lower, color: "var(--blue-soft)", opacity: 0.7 },
              ]}
              points={s.obs.map((o) => ({ t: o.t, c: o.c, color: "var(--clay)", ring: true }))}
              xMax={r.tEnd}
              h={272}
            />
            <div style={{ marginTop: 12 }}>
              <Legend
                items={[
                  { type: "dash", color: "var(--ink-mute)", label: "population prior" },
                  { type: "line", color: "var(--blue)", label: "posterior" },
                  { type: "rect", color: "var(--clay)", label: "observation" },
                ]}
              />
            </div>
          </div>
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Posterior refinement</h5>
            <div style={{ display: "flex", gap: 14, marginBottom: 10 }}>
              <div style={{ flex: 1 }}>
                <div className="statcell" style={{ padding: 0 }}>
                  <div className="k">Prior CV</div>
                  <div className="v" style={{ fontSize: 22 }}>{(r.priorCv * 100).toFixed(1)}<span className="u">%</span></div>
                </div>
              </div>
              <div style={{ flex: 1 }}>
                <div className="statcell" style={{ padding: 0 }}>
                  <div className="k">Posterior CV</div>
                  <div className="v" style={{ fontSize: 22, color: "var(--blue)" }}>{(r.postCv * 100).toFixed(1)}<span className="u">%</span></div>
                </div>
              </div>
            </div>
            <div className="trk">
              <span className="tn">reduction</span>
              <span className="bar"><i style={{ width: r.reduction * 100 + "%", background: "var(--teal)" }} /></span>
              <span className="tv" style={{ color: "var(--teal)" }}>{(r.reduction * 100).toFixed(0)}%</span>
            </div>
            <p className="note" style={{ margin: "10px 0 0", fontSize: 10.5, color: "var(--ink-mute)" }}>
              Anchored to one real Bayesian run; the multi-observation shrink is an illustrative interpolation — full re-inference needs the live engine tier.
            </p>
          </div>
          <div className="panel">
            <h5>Sampler diagnostics</h5>
            <div className="kv"><span className="kk">Production routing</span><span className="vv"><Pill kind={methodKind}>{methodDisp}</Pill></span></div>
            <div className="kv"><span className="kk">Effective sample size</span><span className="vv">{fmt(r.ess, 0)}</span></div>
            <div className="kv"><span className="kk">Mean shift</span><span className="vv">{fmt(r.shift, 2)}×</span></div>
            <p className="note" style={{ margin: "10px 0 0", fontSize: 10.5, color: "var(--ink-mute)" }}>
              ESS / CV computed via importance sampling (static tier). Production routes to {methodDisp}.
            </p>
          </div>
        </div>
      </div>
    );

  // diagnostics
  return (
    <div className="stack">
      <div className="panel">
        <h5>
          Method routing<span className="meta">data/sbi/method_routing.json</span>
        </h5>
        <p className="note" style={{ margin: "0 0 14px" }}>
          TDM dispatches one of three Bayesian methods per drug. <b>{drug.name}</b> routes to <Pill kind={methodKind}>{methodDisp}</Pill>{" "}
          {methodKind === "ok"
            ? "— amortized neural posterior, single forward pass (ms), SBC-gate passed."
            : r.method.toLowerCase() === "ibis"
            ? "— iterative importance sampling fallback (SBC gate failed)."
            : "— classical importance sampling (legacy)."}
        </p>
        <table className="btable">
          <thead>
            <tr><th>Method</th><th>Basis</th><th>Inference</th><th>Production</th></tr>
          </thead>
          <tbody>
            <tr className={methodKind === "ok" ? "hl" : ""}><td>SBI</td><td>Neural posterior (spline flow)</td><td>~ms</td><td>12/13</td></tr>
            <tr className={r.method.toLowerCase() === "ibis" ? "hl" : ""}><td>IBIS</td><td>Iterative importance</td><td>closed-loop</td><td>1/13</td></tr>
            <tr className={r.method.toLowerCase() === "is" ? "hl" : ""}><td>IS</td><td>Classical importance</td><td>weighted</td><td>0/13</td></tr>
          </tbody>
        </table>
      </div>
      <div className="panel">
        <h5>ESS health</h5>
        {r.ess < 10 ? (
          <Caveat>
            ESS = <b>{fmt(r.ess, 1)}</b> — particle weight <b>degeneracy</b>. The prior fold exceeds the importance sampler's reliable range; a sequential method (EnKF / particle filter) would be required.
          </Caveat>
        ) : (
          <p className="note" style={{ margin: 0 }}>
            ESS = <b>{fmt(r.ess, 0)}</b> of the prior samples — healthy. A single observation near T<sub>max</sub> carries the most information; points beyond ~4 h post-dose add little.
          </p>
        )}
      </div>
    </div>
  );
}
