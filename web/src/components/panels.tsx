/* ============================================================
   panels.tsx — reusable result panels + formatting helpers.
   Ported from the prototype's panels.jsx.
   ============================================================ */
import type { ReactNode } from "react";
import type { Drug } from "../types";

export function fmt(v: number | null | undefined, sig?: number): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 1) return v.toFixed(sig || 2);
  if (a >= 0.01) return v.toFixed(3);
  return v.toExponential(2);
}

export type PillKind = "ok" | "dom" | "warn" | "mute" | "";
export function Pill({ kind, children }: { kind?: PillKind; children: ReactNode }) {
  return (
    <span className={"pill " + (kind || "")}>
      <span className="dot" />
      {children}
    </span>
  );
}

export interface StatItem {
  k: string; // may contain HTML (subscripts)
  v: ReactNode;
  u?: string;
  ci?: string | null;
}
export function Stat({ k, v, u, ci }: StatItem) {
  return (
    <div className="statcell">
      <div className="k" dangerouslySetInnerHTML={{ __html: k }} />
      <div className="v">
        {v}
        {u && <span className="u">{u}</span>}
      </div>
      {ci && <div className="ci">{ci}</div>}
    </div>
  );
}

export function StatLine({ cols, items }: { cols?: number; items: StatItem[] }) {
  return (
    <div className={"statline " + (cols === 3 ? "c3" : cols === 2 ? "c2" : "")}>
      {items.map((it, i) => (
        <Stat key={i} {...it} />
      ))}
    </div>
  );
}

interface TrackBarsDrug {
  tracks: Record<string, number | null>;
  weights: Record<string, number | null>;
  primaryEnzyme?: string;
}
export function TrackBars({
  drug,
  meta,
  showWeights,
}: {
  drug: TrackBarsDrug;
  meta?: number | null;
  showWeights?: boolean;
}) {
  const t = drug.tracks;
  const w = drug.weights;
  const vals = (["engine", "ml", "clf", "vdss"] as const)
    .map((k) => t[k])
    .filter((x): x is number => x != null);
  const max = Math.max.apply(null, vals.concat(meta != null ? [meta] : []));
  const rows = [
    { key: "engine", label: "Engine", color: "var(--blue)" },
    { key: "ml", label: "ML", color: "var(--ink-soft)" },
    { key: "clf", label: "CL/F", color: "var(--teal)" },
    { key: "vdss", label: "VDss", color: "var(--clay)" },
  ];
  return (
    <div>
      {rows.map((r) => {
        const val = t[r.key];
        const wt = w[r.key];
        const off = val == null || wt == null || wt === 0;
        return (
          <div key={r.key} className={"trk" + (showWeights ? " w" : "")}>
            <span className="tn">{r.label}</span>
            <span className="bar">
              <i style={{ width: off ? "0%" : (val! / max) * 100 + "%", background: off ? "var(--hair)" : r.color }} />
            </span>
            <span className="tv" style={off ? { color: "var(--ink-mute)" } : undefined}>
              {val == null ? "off" : fmt(val)}
            </span>
            {showWeights && <span className="tw">{wt == null ? "—" : "·" + (wt * 100).toFixed(0)}</span>}
          </div>
        );
      })}
      {meta != null && (
        <div className={"trk" + (showWeights ? " w" : "")} style={{ borderTop: "1px solid var(--hair)", marginTop: 4, paddingTop: 8 }}>
          <span className="tn" style={{ color: "var(--ink)", fontWeight: 600 }}>Meta</span>
          <span className="bar">
            <i style={{ width: (meta / max) * 100 + "%", background: "var(--ink)" }} />
          </span>
          <span className="tv" style={{ fontWeight: 600 }}>{fmt(meta)}</span>
          {showWeights && <span className="tw">1.0</span>}
        </div>
      )}
    </div>
  );
}

/** Simplified 34-compartment body schematic. */
export function BodyGraph({ drug }: { drug: Drug }) {
  const renal = drug.primaryEnzyme === "renal";
  return (
    <div className="bgraph">
      <div className="bg-lane">
        <span className="lab">absorb</span>
        <span className="nd">stomach</span><span className="arr">→</span>
        <span className="nd">duodenum</span><span className="arr">→</span>
        <span className="nd">jejunum</span><span className="arr">→</span>
        <span className="nd">ileum</span><span className="arr">→</span>
        <span className="nd">colon</span>
      </div>
      <div className="bg-lane">
        <span className="lab">portal</span>
        <span className="nd">gut wall</span>
        <span className="nd">spleen</span>
        <span className="nd">pancreas</span><span className="arr">→</span>
        <span className="nd clear">liver · {drug.primaryEnzyme && drug.primaryEnzyme !== "renal" ? drug.primaryEnzyme : "CYP"}</span>
      </div>
      <div className="bg-lane">
        <span className="lab">central</span>
        <span className="nd blood">lung</span><span className="arr">→</span>
        <span className="nd blood">arterial</span><span className="arr">→</span>
        <span className="nd blood">venous</span>
      </div>
      <div className="bg-lane">
        <span className="lab">perfuse</span>
        <span className="nd">brain</span>
        <span className="nd">heart</span>
        <span className={"nd" + (renal ? " clear" : "")}>kidney{renal ? " · GFR" : ""}</span>
        <span className="nd">muscle</span>
        <span className="nd">adipose</span>
      </div>
      <div className="bg-lane">
        <span className="lab">perm-lim</span>
        <span className="nd perm">skin</span>
        <span className="nd perm">bone</span>
        <span className="nd perm">testes</span>
        <span className="nd perm">spleen-x</span>
      </div>
    </div>
  );
}

export interface LegendItem {
  type?: string; // "rect" | "line" | "dash"
  color: string;
  label: string;
}
export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <div key={i} className="li">
          <span
            className={"sw " + (it.type || "")}
            style={it.type === "line" || it.type === "dash" ? { borderColor: it.color, width: 16 } : { background: it.color }}
          />
          {it.label}
        </div>
      ))}
    </div>
  );
}

export function Caveat({ children }: { children: ReactNode }) {
  return <div className="caveat">{children}</div>;
}

export interface LogLine {
  ts: string;
  tag: string; // "ok" | "info" | "warn"
  tagText: string;
  msg: string; // may contain HTML
}
export function PipelineLog({ lines, running }: { lines: LogLine[]; running?: boolean }) {
  return (
    <div className="log">
      {lines.map((l, i) => (
        <div className="ln" key={i}>
          <span className="ts">{l.ts}</span>
          <span className={"tag " + l.tag}>{l.tagText}</span>
          <span className="msg" dangerouslySetInnerHTML={{ __html: l.msg }} />
        </div>
      ))}
      {running && (
        <div className="ln">
          <span className="ts">…</span>
          <span className="tag info">running</span>
          <span className="msg">solving ODE system …</span>
        </div>
      )}
    </div>
  );
}
