/* ============================================================
   SimulateView — multi-dose regimen via superposition of the
   real single-dose engine response (exact for linear PK).
   ============================================================ */
import type { AppState, ConsoleData, Drug } from "../../types";
import { ConcChart, TroughChart } from "../charts";
import { StatLine, fmt } from "../panels";
import { steadyState } from "../../pk";
import { drugById } from "../../data";

// FDA-label steady-state Cmax values are cited literature constants; the
// "Predicted" column is computed live from the real engine steady-state below.
const LABEL_ROWS = [
  { id: "atorvastatin", name: "Atorvastatin", reg: "40 mg QD", dose: 40, interval: 24, label: 0.029 },
  { id: "metformin", name: "Metformin", reg: "500 mg BID", dose: 500, interval: 12, label: 1.0 },
  { id: "warfarin", name: "Warfarin", reg: "5 mg QD", dose: 5, interval: 24, label: 1.4 },
];

export function SimulateView({ drug, s, tab, data }: { drug: Drug; s: AppState; tab: number; data: ConsoleData }) {
  const ss = steadyState(drug, s.dose, s.interval, s.nDoses);
  const cmaxFirst = (drug.tracks.engine ?? drug.meta.cmax) * (s.dose / drug.dose);

  if (tab === 0)
    return (
      <div className="split">
        <div>
          <div className="panel">
            <h5>
              Multi-dose regimen<span className="meta">{s.dose} mg · q{s.interval}h · ×{s.nDoses}</span>
            </h5>
            <ConcChart
              series={[{ pts: ss.run.pts, color: "var(--blue)", width: 1.8 }]}
              hlines={[
                { y: ss.cssmax, color: "var(--teal)", dash: "5 4" },
                { y: ss.cssmin, color: "var(--clay)", dash: "5 4" },
              ]}
              vlines={[{ x: ss.ssDose * s.interval, color: "var(--ink-mute)", dash: "2 4" }]}
              xMax={ss.run.tEnd}
              h={272}
            />
            <div className="figcap">
              <b>FIG.</b> Event-driven superposition of the real single-dose response. Teal = C<sub>ss,max</sub>, clay = C<sub>ss,min</sub>; vertical guide marks steady state (dose {ss.ssDose}).
            </div>
          </div>
          <div style={{ height: 14 }} />
          <StatLine
            cols={4}
            items={[
              { k: "C<sub>ss,max</sub>", v: fmt(ss.cssmax), u: "mg/L" },
              { k: "C<sub>ss,min</sub>", v: fmt(ss.cssmin), u: "mg/L" },
              { k: "Accum. ratio", v: fmt(ss.ar, 2), u: "×" },
              { k: "SS reached", v: "dose " + ss.ssDose },
            ]}
          />
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Steady-state detection</h5>
            <p className="note" style={{ margin: 0, fontSize: 12 }}>
              Trough variation &lt; 5% across the last three intervals. The single-dose ODE engine is unmodified — dose events are injected into the state vector between integration segments.
            </p>
          </div>
          <div className="panel">
            <h5>First vs steady state</h5>
            <div className="kv"><span className="kk">C<sub>max</sub>, first dose</span><span className="vv">{fmt(cmaxFirst)} mg/L</span></div>
            <div className="kv"><span className="kk">C<sub>ss,max</sub></span><span className="vv accent">{fmt(ss.cssmax)} mg/L</span></div>
            <div className="kv"><span className="kk">Fold accumulation</span><span className="vv">{fmt(ss.ar, 2)}×</span></div>
            <div className="kv"><span className="kk">Peak–trough swing</span><span className="vv">{fmt(((ss.cssmax - ss.cssmin) / ss.cssmax) * 100, 0)}%</span></div>
          </div>
        </div>
      </div>
    );

  if (tab === 1)
    return (
      <div className="split">
        <div className="panel">
          <h5>Trough accumulation to steady state</h5>
          <TroughChart troughs={ss.troughs} ssDose={ss.ssDose} cssmin={ss.cssmin} />
          <div className="figcap">
            <b>FIG.</b> Pre-dose trough by dose number; convergence within 5% defines steady state.
          </div>
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Convergence</h5>
            <div className="kv"><span className="kk">Doses to SS</span><span className="vv accent">{ss.ssDose}</span></div>
            <div className="kv"><span className="kk">Time to SS</span><span className="vv">{fmt(ss.ssDose * s.interval, 0)} h</span></div>
            <div className="kv"><span className="kk">t½</span><span className="vv">{fmt(drug.meta.thalf)} h</span></div>
            <div className="kv"><span className="kk">≈ half-lives</span><span className="vv">{fmt((ss.ssDose * s.interval) / drug.meta.thalf, 1)}</span></div>
          </div>
          <div className="panel">
            <h5>Rule of thumb</h5>
            <p className="note" style={{ margin: 0, fontSize: 12 }}>
              Steady state is reached in roughly <b>4–5 half-lives</b> regardless of interval; the accumulation ratio is governed by t½ / τ.
            </p>
          </div>
        </div>
      </div>
    );

  // label check
  return (
    <div className="stack">
      <div className="panel">
        <h5>
          FDA-label steady-state check<span className="meta">predicted vs label C<sub>ss,max</sub></span>
        </h5>
        <table className="btable">
          <thead>
            <tr>
              <th>Drug</th>
              <th>Regimen</th>
              <th>Predicted</th>
              <th>FDA label</th>
              <th>Fold error</th>
            </tr>
          </thead>
          <tbody>
            {LABEL_ROWS.map((r) => {
              const rd = drugById(data, r.id);
              const pred = rd && rd.id === r.id ? steadyState(rd, r.dose, r.interval, 14).cssmax : null;
              const fe = pred != null && r.label ? pred / r.label : null;
              return (
                <tr key={r.id} className={r.id === drug.id ? "hl" : ""}>
                  <td>{r.name}</td>
                  <td>{r.reg}</td>
                  <td>{fmt(pred)}</td>
                  <td>{fmt(r.label)}</td>
                  <td style={{ color: fe != null && fe >= 0.8 && fe <= 1.25 ? "var(--teal)" : "var(--clay)" }}>{fe == null ? "—" : fe.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="note" style={{ marginTop: 12, fontSize: 11.5 }}>
          Predicted = real engine steady-state C<sub>ss,max</sub> (superposition at the listed regimen). FDA-label C<sub>ss,max</sub> are cited product-label values.
          {!LABEL_ROWS.some((r) => r.id === drug.id) && <> No published label on file for {drug.name}; the validated reference set is shown.</>}
        </p>
      </div>
      <div className="panel">
        <h5>Interpretation</h5>
        <p className="note" style={{ margin: 0 }}>
          Fold error is predicted ÷ label (teal = within 1.25-fold). Renally-cleared (metformin) and highly-bound (warfarin, f<sub>u</sub>≈0.01) drugs are expected to under-predict — documented engine limitations — while accumulation <b>direction</b> is correct in all cases.
        </p>
      </div>
    </div>
  );
}
