/* ============================================================
   RailInputs.tsx — contextual rail inputs per workflow.
   Ported from the prototype's RailInputs.
   ============================================================ */
import type { AppState, ConsoleData, WorkflowId } from "../types";
import { drugById } from "../data";
import { concAt } from "../pk";
import { defObs, oid } from "./workflows/config";

type SetFn = (patch: Partial<AppState>) => void;

function DoseInput({ s, set }: { s: AppState; set: SetFn }) {
  return (
    <div style={{ position: "relative" }}>
      <input
        className="num"
        type="number"
        min={0}
        value={s.dose}
        onChange={(e) => set({ dose: Math.max(0, +e.target.value || 0) })}
        style={{ paddingRight: 34 }}
      />
      <span style={{ position: "absolute", right: 11, top: 9, fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-mute)" }}>
        mg
      </span>
    </div>
  );
}

function updObs(s: AppState, set: SetFn, i: number, key: "t" | "c", val: number) {
  const o = s.obs.slice();
  o[i] = { ...o[i], [key]: val };
  set({ obs: o });
}

export function RailInputs({
  wf,
  s,
  set,
  data,
  live,
  smilesDraft,
  setSmilesDraft,
  nameDraft,
  setNameDraft,
  predictError,
}: {
  wf: WorkflowId;
  s: AppState;
  set: SetFn;
  data: ConsoleData;
  live: boolean;
  smilesDraft: string;
  setSmilesDraft: (v: string) => void;
  nameDraft: string;
  setNameDraft: (v: string) => void;
  predictError: string | null;
}) {
  const drug = drugById(data, s.drugId);
  const isCustom = s.drugId === "custom";

  const DrugPicker = (
    <div className="field">
      <label>
        Compound <span className="hintdot">{isCustom ? "custom SMILES" : "SMILES preset"}</span>
      </label>
      <select
        className="sel"
        value={s.drugId}
        onChange={(e) => {
          const v = e.target.value;
          if (v === "custom") {
            if (!smilesDraft.trim()) setSmilesDraft(drug.smiles);
            set({ drugId: "custom" });
          } else {
            const nd = drugById(data, v);
            set({ drugId: v, dose: nd.dose, obs: defObs(nd) });
          }
        }}
      >
        {data.drugs.map((d) => (
          <option key={d.id} value={d.id}>
            {d.name}
          </option>
        ))}
        <option value="custom" disabled={!live}>
          {live ? "✎ Custom SMILES…" : "✎ Custom SMILES (engine offline)"}
        </option>
      </select>
      {isCustom ? (
        <div style={{ marginTop: 8 }}>
          <input
            className="inp smiles"
            value={smilesDraft}
            spellCheck={false}
            placeholder="paste a SMILES string…"
            onChange={(e) => setSmilesDraft(e.target.value)}
            style={{ color: "var(--blue)", fontSize: 11 }}
          />
          <input
            className="inp"
            value={nameDraft}
            placeholder="name (optional)"
            onChange={(e) => setNameDraft(e.target.value)}
            style={{ marginTop: 7, fontSize: 12 }}
          />
          {predictError && (
            <div style={{ marginTop: 7, fontFamily: "var(--mono)", fontSize: 10.5, color: "oklch(0.46 0.1 52)" }}>
              {predictError}
            </div>
          )}
        </div>
      ) : (
        <div className="inp smiles" style={{ marginTop: 8, color: "var(--blue)", fontSize: 11 }}>
          {drug.smiles}
        </div>
      )}
    </div>
  );

  if (wf === "benchmark") {
    return (
      <div>
        <div className="field">
          <label>Holdout set</label>
          <select className="sel" value={s.benchSet} onChange={(e) => set({ benchSet: e.target.value })}>
            <option value="scaffold">Murcko scaffold-stratified</option>
            <option value="temporal">Temporal (FDA NME)</option>
          </select>
        </div>
        <div className="field">
          <label>Random seed</label>
          <input className="num" value="42" readOnly />
        </div>
        <div className="field">
          <label>
            Resamples <span className="hintdot">bootstrap CI</span>
          </label>
          <input className="num" value="10,000" readOnly />
        </div>
        <div className="note" style={{ marginTop: 18, fontSize: 11.5 }}>
          External validation only — the holdout is never used in training or model selection. Re-runs are deterministic on a fixed numerics stack.
        </div>
      </div>
    );
  }

  return (
    <div>
      {DrugPicker}
      {wf === "predict" && (
        <div>
          <div className="row2">
            <div className="field">
              <label>Dose</label>
              <DoseInput s={s} set={set} />
            </div>
            <div className="field">
              <label>
                Route <span className="hintdot">oral · static tier</span>
              </label>
              <div className="seg">
                {(["oral", "iv"] as const).map((r) => (
                  <button
                    key={r}
                    className={s.route === r ? "on" : ""}
                    disabled={r === "iv"}
                    title={r === "iv" ? "IV bolus solves arrive with the live engine tier — the static tier ships oral profiles only" : undefined}
                    style={r === "iv" ? { opacity: 0.4, cursor: "not-allowed" } : undefined}
                    onClick={() => r !== "iv" && set({ route: r })}
                  >
                    {r}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="field">
            <label>Method</label>
            <div className="seg">
              {(["hybrid", "engine", "ml"] as const).map((m) => (
                <button key={m} className={s.method === m ? "on" : ""} onClick={() => set({ method: m })}>
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div className="field">
            <label>
              MC samples <span className="hintdot">uncertainty</span>
            </label>
            <select className="sel" value={s.nmc} onChange={(e) => set({ nmc: +e.target.value })}>
              <option value={1}>deterministic (mean)</option>
              <option value={1000}>N = 1,000</option>
              <option value={2000}>N = 2,000</option>
            </select>
          </div>
        </div>
      )}
      {wf === "simulate" && (
        <div>
          <div className="field">
            <label>Dose</label>
            <DoseInput s={s} set={set} />
          </div>
          <div className="row2">
            <div className="field">
              <label>Interval (h)</label>
              <input className="num" type="number" min={0.25} value={s.interval} onChange={(e) => set({ interval: Math.max(0.25, +e.target.value || 1) })} />
            </div>
            <div className="field">
              <label># Doses</label>
              <input className="num" type="number" value={s.nDoses} onChange={(e) => set({ nDoses: Math.max(2, +e.target.value || 2) })} />
            </div>
          </div>
          <div className="field">
            <label>Quick regimen</label>
            <div className="seg">
              {([["QD", 24], ["BID", 12], ["TID", 8]] as const).map(([lab, h]) => (
                <button key={lab} className={s.interval === h ? "on" : ""} onClick={() => set({ interval: h })}>
                  {lab}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
      {(wf === "tdm" || wf === "dose-adjust") && (
        <div>
          <div className="field">
            <label>Dose</label>
            <DoseInput s={s} set={set} />
          </div>
          <div className="field">
            <label>
              Observations <span className="hintdot">t&nbsp;(h)&nbsp;:&nbsp;conc</span>
            </label>
            <div className="obsrows">
              {s.obs.map((o, i) => (
                <div className="obsrow" key={o.id}>
                  <input className="num" type="number" step="0.1" aria-label={`observation ${i + 1} time (h)`} value={o.t} onChange={(e) => updObs(s, set, i, "t", +e.target.value)} />
                  <input className="num" type="number" step="0.001" aria-label={`observation ${i + 1} concentration (mg/L)`} value={o.c} onChange={(e) => updObs(s, set, i, "c", +e.target.value)} />
                  <button className="del" aria-label="remove observation" onClick={() => set({ obs: s.obs.filter((_, j) => j !== i) })}>
                    ×
                  </button>
                </div>
              ))}
            </div>
            <button
              className="addobs"
              onClick={() => set({ obs: s.obs.concat([{ id: oid(), t: 4, c: +concAt(drug, s.dose, 4).toFixed(4) }]) })}
            >
              + add observation
            </button>
          </div>
          {wf === "tdm" && (
            <div className="field">
              <label>Assay CV</label>
              <div className="seg">
                {["5%", "10%", "20%"].map((c) => (
                  <button key={c} className={s.assayCv === c ? "on" : ""} onClick={() => set({ assayCv: c })}>
                    {c}
                  </button>
                ))}
              </div>
            </div>
          )}
          {wf === "dose-adjust" && (
            <div className="field">
              <label>
                Target C<sub>ss,max</sub> (mg/L)
              </label>
              <input className="num" type="number" step="0.001" value={s.targetCss} onChange={(e) => set({ targetCss: +e.target.value })} />
            </div>
          )}
        </div>
      )}
      {wf === "ddi" && (
        <div>
          <div className="field">
            <label>Victim dose</label>
            <DoseInput s={s} set={set} />
          </div>
          <div className="field">
            <label>Perpetrator</label>
            <select className="sel" value={s.inhibitor} onChange={(e) => set({ inhibitor: e.target.value })}>
              {Object.keys(data.inhibitors).map((k) => (
                <option key={k} value={k}>
                  {data.inhibitors[k].name}
                  {k !== "none" ? " · " + data.inhibitors[k].enzyme : ""}
                </option>
              ))}
            </select>
          </div>
          <div className="note" style={{ marginTop: 16, fontSize: 11.5 }}>
            DDI is applied by adjusting enzyme abundance <b>before</b> ODE compilation — the engine is unmodified.
          </div>
        </div>
      )}
    </div>
  );
}
