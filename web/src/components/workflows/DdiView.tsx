/* ============================================================
   DdiView — drug-drug interaction. AUC/Cmax fold changes are the
   REAL engine apply_inhibition / apply_induction re-solves; the
   curves are illustrative shapes driven by those real folds.
   ============================================================ */
import type { AppState, ConsoleData, Drug } from "../../types";
import { ConcChart } from "../charts";
import { StatLine, Legend, Caveat, fmt } from "../panels";
import { ddiFold, ddiCurve, singleCurve } from "../../pk";

export function DdiView({ drug, s, tab, data }: { drug: Drug; s: AppState; tab: number; data: ConsoleData }) {
  const fold = ddiFold(drug, data, s.inhibitor);
  const tEnd = Math.max(drug.pkfit.thalf * 5, 14);
  const base = singleCurve(drug, s.dose, tEnd);
  const inhibited = s.inhibitor === "none" ? base : ddiCurve(drug, s.dose, tEnd, fold.keFactor, fold.cmax);
  const baseCmax = drug.meta.cmax * (s.dose / drug.dose);
  const inh = data.inhibitors[s.inhibitor];

  if (tab === 0)
    return (
      <div className="split">
        <div>
          <div className="panel">
            <h5>
              Victim exposure ± perpetrator<span className="meta">{drug.name} + {inh.name}</span>
            </h5>
            <ConcChart
              series={[
                { pts: base, color: "var(--ink-mute)", dash: "5 4", width: 1.7 },
                { pts: inhibited, color: inh.type === "induction" ? "var(--teal)" : "var(--clay)", width: 2.3 },
              ]}
              xMax={tEnd}
              h={272}
            />
            <div style={{ marginTop: 12 }}>
              <Legend
                items={[
                  { type: "dash", color: "var(--ink-mute)", label: drug.name + " alone" },
                  { type: "line", color: inh.type === "induction" ? "var(--teal)" : "var(--clay)", label: "+ " + inh.name },
                ]}
              />
            </div>
          </div>
          <div style={{ height: 14 }} />
          <StatLine
            cols={3}
            items={[
              { k: "AUC fold", v: fmt(fold.auc, 2), u: "×" },
              { k: "C<sub>max</sub> fold", v: fmt(fold.cmax, 2), u: "×" },
              { k: "New C<sub>max</sub>", v: fmt(baseCmax * fold.cmax), u: "mg/L" },
            ]}
          />
        </div>
        <div className="stack">
          <div className="panel">
            <h5>Perpetrator</h5>
            <div className="kv"><span className="kk">Agent</span><span className="vv">{inh.name}</span></div>
            <div className="kv"><span className="kk">Enzyme</span><span className="vv accent">{inh.enzyme}</span></div>
            <div className="kv"><span className="kk">Mechanism</span><span className="vv">{inh.type}</span></div>
            <div className="kv"><span className="kk">Victim f<sub>m</sub>({inh.enzyme.split("/")[0]}) · lit.</span><span className="vv">{fold.frac > 0 ? (fold.frac * 100).toFixed(0) + "%" : "—"}</span></div>
          </div>
          <div className="panel">
            <h5>Clinical magnitude</h5>
            {s.inhibitor === "none" ? (
              <p className="note" style={{ margin: 0 }}>Select a perpetrator to model an interaction.</p>
            ) : Math.abs(Math.log(fold.auc)) > 0.4 ? (
              <p className="note" style={{ margin: 0 }}>
                The engine predicts a <b>{fmt(fold.auc, 2)}×</b> {inh.type === "induction" ? "decrease" : "increase"} in {drug.name} exposure with {inh.name} — a clinically meaningful interaction.
              </p>
            ) : (
              <p className="note" style={{ margin: 0 }}>
                {inh.name} changes {drug.name} exposure only <b>{fmt(fold.auc, 2)}×</b> — minimal. For a strong effect, try a dominant {inh.enzyme.split("/")[0]} substrate (e.g. midazolam + ketoconazole).
              </p>
            )}
            {s.inhibitor !== "none" && Math.abs(Math.log(fold.auc)) > 0.4 && !drug.primaryEnzyme.startsWith("CYP") && (
              <div style={{ marginTop: 12 }}>
                <Caveat>
                  {drug.name} is primarily <b>{drug.primaryEnzyme}</b>-cleared, yet the engine still predicts a CYP interaction. This is the identity-blind engine acting on XGBoost-predicted enzyme affinities — a known <b>over-attribution</b> for non-CYP drugs, surfaced honestly rather than hidden.
                </Caveat>
              </div>
            )}
          </div>
        </div>
      </div>
    );

  // mechanism
  return (
    <div className="stack">
      <div className="panel">
        <h5>How the interaction is applied</h5>
        <p className="note" style={{ margin: "0 0 14px" }}>
          Enzyme abundance is modified <b>before</b> ODE compilation; the engine computes clearance from the adjusted abundance as usual. No engine code path is DDI-aware.
        </p>
        <div className="split-even">
          <div style={{ background: "var(--surface-2)", border: "1px solid var(--hair)", borderRadius: 10, padding: 16 }}>
            <div className="subhead">Competitive inhibition</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 14, color: "var(--ink)", textAlign: "center", padding: "8px 0" }}>
              E<sub>eff</sub> = E<sub>base</sub> / (1 + [I]/K<sub>i</sub>)
            </div>
          </div>
          <div style={{ background: "var(--surface-2)", border: "1px solid var(--hair)", borderRadius: 10, padding: 16 }}>
            <div className="subhead">Induction (E<sub>max</sub>)</div>
            <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--ink)", textAlign: "center", padding: "8px 0" }}>
              E<sub>eff</sub> = E<sub>base</sub>(1 + E<sub>max</sub>[I]/(EC<sub>50</sub>+[I]))
            </div>
          </div>
        </div>
      </div>
      <div className="panel">
        <h5>Preset perpetrators</h5>
        <table className="btable">
          <thead>
            <tr><th>Agent</th><th>Enzyme</th><th>Type</th><th>Strength</th></tr>
          </thead>
          <tbody>
            {Object.keys(data.inhibitors)
              .filter((k) => k !== "none")
              .map((k) => {
                const x = data.inhibitors[k];
                return (
                  <tr key={k} className={k === s.inhibitor ? "hl" : ""}>
                    <td>{x.name}</td>
                    <td>{x.enzyme}</td>
                    <td>{x.type}</td>
                    <td>{x.strength ?? "—"}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
