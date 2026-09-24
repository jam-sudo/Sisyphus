/* ============================================================
   RailInputs.tsx — contextual rail inputs per workflow.
   Ported from the prototype's RailInputs.
   ============================================================ */
import type { AppState, ConsoleData, WorkflowId } from "../types";
import { drugById } from "../data";

type SetFn = (patch: Partial<AppState>) => void;

function DoseInput({ s, set, editable }: { s: AppState; set: SetFn; editable: boolean }) {
  return (
    <div style={{ position: "relative" }}>
      <input
        className="num"
        type="number"
        min={0.000001}
        max={100000}
        value={s.dose}
        disabled={!editable}
        title={editable ? undefined : "Preset results are locked to the dose that was actually solved"}
        onChange={(e) => set({ dose: Math.min(100000, Math.max(0.000001, +e.target.value || 0.000001)) })}
        style={{ paddingRight: 34 }}
      />
      <span style={{ position: "absolute", right: 11, top: 9, fontFamily: "var(--mono)", fontSize: 11, color: "var(--ink-mute)" }}>
        mg
      </span>
    </div>
  );
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
            set({ drugId: v, dose: nd.dose });
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
          <label>Evidence set</label>
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
          The source-audited N={data.benchmark.n_development} is a repeatedly accessed development benchmark. The temporal N=28 set is consumed. Neither is an independent holdout for the current system.
        </div>
      </div>
    );
  }

  return (
    <div>
      {DrugPicker}
      <div>
          <div className="row2">
            <div className="field">
              <label>Dose</label>
              <DoseInput s={s} set={set} editable={isCustom} />
            </div>
            <div className="field">
              <label>
                Route <span className="hintdot">oral · static tier</span>
              </label>
              <div className="seg">
                {(["oral"] as const).map((r) => (
                  <button
                    key={r}
                    className={s.route === r ? "on" : ""}
                    onClick={() => set({ route: r })}
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
      </div>
    </div>
  );
}
