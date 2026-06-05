/* ============================================================
   DoseAdjustView — MIPD dose recommendation. Linear dose scaling
   from the real-anchored posterior steady-state to a target Css.
   ============================================================ */
import type { AppState, Drug } from "../../types";
import { ConcChart } from "../charts";
import { Legend, Caveat, fmt } from "../panels";
import { mipd, steadyState } from "../../pk";

export function DoseAdjustView({ drug, s }: { drug: Drug; s: AppState; tab: number }) {
  const tau = 24;
  const m = mipd(drug, s.dose, s.obs, s.targetCss, tau);
  const curRun = steadyState(drug, s.dose, tau, 10);
  const newRun = steadyState(drug, m.recommended, tau, 10);
  const tEnd = curRun.run.tEnd;

  return (
    <div className="split">
      <div>
        <div className="panel">
          <h5>
            Projected steady state at recommended dose
            <span className="meta">target C<sub>ss,max</sub> {fmt(s.targetCss)} mg/L</span>
          </h5>
          <ConcChart
            series={[
              { pts: curRun.run.pts, color: "var(--ink-mute)", dash: "5 4", width: 1.6 },
              { pts: newRun.run.pts, color: "var(--blue)", width: 2 },
            ]}
            hlines={[{ y: s.targetCss, color: "var(--teal)", dash: "5 4" }]}
            xMax={tEnd}
            h={272}
          />
          <div style={{ marginTop: 12 }}>
            <Legend
              items={[
                { type: "dash", color: "var(--ink-mute)", label: "current " + s.dose + " mg" },
                { type: "line", color: "var(--blue)", label: "recommended " + m.recommended + " mg" },
                { type: "dash", color: "var(--teal)", label: "target" },
              ]}
            />
          </div>
        </div>
      </div>
      <div className="stack">
        <div className="panel" style={{ textAlign: "center", padding: "22px 18px" }}>
          <div className="subhead" style={{ marginBottom: 6 }}>Recommended dose</div>
          <div style={{ fontFamily: "var(--serif)", fontSize: 52, fontWeight: 600, letterSpacing: "-0.02em", lineHeight: 1, color: "var(--blue)" }}>
            {m.recommended}
            <span style={{ fontFamily: "var(--mono)", fontSize: 16, color: "var(--ink-mute)", marginLeft: 6 }}>mg</span>
          </div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-mute)", marginTop: 8 }}>
            from {s.dose} mg · q{tau}h · rounded to 25 mg
          </div>
        </div>
        <div className="panel">
          <h5>Computation</h5>
          <div className="kv"><span className="kk">Posterior C<sub>ss,max</sub></span><span className="vv">{fmt(m.postCss)} mg/L</span></div>
          <div className="kv"><span className="kk">Target</span><span className="vv accent">{fmt(m.targetCss)} mg/L</span></div>
          <div className="kv"><span className="kk">Raw scale</span><span className="vv">{fmt(m.raw, 0)} mg</span></div>
          <div className="kv"><span className="kk">Clamped + rounded</span><span className="vv">{m.recommended} mg</span></div>
        </div>
        <div className="panel">
          <Caveat>
            Linear dose scaling assumes <b>non-saturable</b> kinetics. For drugs with nonlinear PK (e.g. phenytoin) this approximation breaks down — use with caution.
          </Caveat>
        </div>
      </div>
    </div>
  );
}
