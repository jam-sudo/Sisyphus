/* ============================================================
   charts.tsx — SVG chart components (concentration-time, scatter,
   MC histogram, trough bars). Hand-rolled SVG, no chart library,
   to preserve the scientific-editorial look pixel-for-pixel.
   Ported from the prototype's charts.jsx / inline chart fns.
   ============================================================ */
import type { Pt, Band } from "../pk";

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const exp = Math.floor(Math.log10(v));
  const f = v / Math.pow(10, exp);
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nf * Math.pow(10, exp);
}
function useTicks(max: number, n: number): number[] {
  const step = max / n;
  const arr: number[] = [];
  for (let i = 0; i <= n; i++) arr.push(+(step * i).toPrecision(2));
  return arr;
}
function fmtTick(v: number): string {
  if (v === 0) return "0";
  if (v >= 100) return String(Math.round(v));
  if (v >= 1) return (+v.toFixed(1)).toString();
  if (v >= 0.01) return (+v.toFixed(2)).toString();
  return v.toExponential(0);
}
function sup(d: number): string {
  const map: Record<string, string> = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
  };
  return String(d)
    .split("")
    .map((c) => map[c] || c)
    .join("");
}

export interface Series {
  pts: Pt[];
  color: string;
  width?: number;
  dash?: string;
  opacity?: number;
}
export interface BandSpec extends Band {
  color: string;
  opacity?: number;
}
export interface PointSpec {
  t: number;
  c: number;
  color?: string;
  ring?: boolean;
}
export interface LineSpec {
  x?: number;
  y?: number;
  color: string;
  dash?: string;
  label?: string;
}

export interface ConcChartProps {
  series?: Series[];
  bands?: BandSpec[];
  points?: PointSpec[];
  vlines?: LineSpec[];
  hlines?: LineSpec[];
  xMax?: number;
  yMax?: number;
  xLabel?: string;
  yLabel?: string;
  w?: number;
  h?: number;
}

export function ConcChart(props: ConcChartProps) {
  const w = props.w || 580;
  const h = props.h || 300;
  const ml = 50, mr = 16, mt = 14, mb = 34;
  const pw = w - ml - mr, ph = h - mt - mb;

  const allY: number[] = [];
  (props.series || []).forEach((s) => s.pts.forEach((p) => allY.push(p[1])));
  (props.bands || []).forEach((b) => b.upper.forEach((p) => allY.push(p[1])));
  (props.points || []).forEach((p) => allY.push(p.c));
  const dataYMax = allY.length ? Math.max.apply(null, allY) : 1;
  const yMax = props.yMax || niceMax(dataYMax * 1.12);
  const xMax =
    props.xMax ||
    Math.max.apply(null, (props.series || [{ pts: [[1, 0]] as Pt[] }])[0].pts.map((p) => p[0]));

  const X = (t: number) => ml + (t / xMax) * pw;
  const Y = (c: number) => mt + ph - (Math.max(0, c) / yMax) * ph;

  const xticks = useTicks(xMax, 6);
  const yticks = useTicks(yMax, 4);

  const linePath = (pts: Pt[]) =>
    "M" + pts.map((p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" L");
  const areaPath = (b: Band) => {
    const up = b.upper.map((p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1));
    const lo = b.lower
      .slice()
      .reverse()
      .map((p) => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1));
    return "M" + up.join(" L") + " L" + lo.join(" L") + " Z";
  };

  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img">
      <title>Plasma concentration versus time</title>
      {yticks.map((v, i) => (
        <g key={"y" + i}>
          <line className="grid" x1={ml} y1={Y(v)} x2={w - mr} y2={Y(v)} />
          <text className="tick" x={ml - 8} y={Y(v) + 3} textAnchor="end">{fmtTick(v)}</text>
        </g>
      ))}
      {xticks.map((v, i) => (
        <text key={"x" + i} className="tick" x={X(v)} y={h - 12} textAnchor="middle">{fmtTick(v)}</text>
      ))}
      <line className="axis" x1={ml} y1={mt} x2={ml} y2={mt + ph} />
      <line className="axis" x1={ml} y1={mt + ph} x2={w - mr} y2={mt + ph} />

      {(props.bands || []).map((b, i) => (
        <path key={"b" + i} d={areaPath(b)} fill={b.color} opacity={b.opacity != null ? b.opacity : 0.6} />
      ))}
      {(props.hlines || []).map((l, i) => (
        <line key={"h" + i} x1={ml} y1={Y(l.y!)} x2={w - mr} y2={Y(l.y!)} stroke={l.color} strokeWidth="1" strokeDasharray={l.dash || "4 4"} opacity="0.7" />
      ))}
      {(props.vlines || []).map((l, i) => (
        <line key={"v" + i} x1={X(l.x!)} y1={mt} x2={X(l.x!)} y2={mt + ph} stroke={l.color} strokeWidth="1" strokeDasharray={l.dash || "3 3"} opacity="0.55" />
      ))}
      {(props.series || []).map((s, i) => (
        <path key={"s" + i} d={linePath(s.pts)} fill="none" stroke={s.color} strokeWidth={s.width || 2.2} strokeDasharray={s.dash || "none"} strokeLinejoin="round" opacity={s.opacity != null ? s.opacity : 1} />
      ))}
      {(props.points || []).map((p, i) => (
        <g key={"p" + i}>
          <circle cx={X(p.t)} cy={Y(p.c)} r="4.2" fill={p.color || "var(--clay)"} stroke="var(--paper)" strokeWidth="1.6" />
          {p.ring && <circle cx={X(p.t)} cy={Y(p.c)} r="7.5" fill="none" stroke={p.color || "var(--clay)"} strokeWidth="1" opacity="0.5" />}
        </g>
      ))}
      <text className="alab" x={ml + pw / 2} y={h - 1} textAnchor="middle">{props.xLabel || "time (h)"}</text>
      <text className="alab" transform={`translate(12 ${mt + ph / 2}) rotate(-90)`} textAnchor="middle">{props.yLabel || "conc (mg/L)"}</text>
    </svg>
  );
}

export interface ScatterPt {
  obs: number;
  pred: number;
  inDom: boolean;
}
export function ScatterChart({ points, w = 560, h = 360 }: { points: ScatterPt[]; w?: number; h?: number }) {
  const ml = 50, mr = 16, mt = 14, mb = 36;
  const pw = w - ml - mr, ph = h - mt - mb;
  const lo = -2.4, hi = 1.3; // log10 mg/L
  const L = (v: number) => Math.log10(v);
  const X = (v: number) => ml + ((Math.min(hi, Math.max(lo, L(v))) - lo) / (hi - lo)) * pw;
  const Y = (v: number) => mt + ph - ((Math.min(hi, Math.max(lo, L(v))) - lo) / (hi - lo)) * ph;
  const decades = [-2, -1, 0, 1];

  const diag = (mult: number) => {
    const x1 = Math.pow(10, lo), x2 = Math.pow(10, hi);
    return { x1: X(x1), y1: Y(x1 * mult), x2: X(x2), y2: Y(x2 * mult) };
  };
  const d1 = diag(1), d2u = diag(2), d2l = diag(0.5), d3u = diag(3), d3l = diag(1 / 3);

  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img">
      <title>Predicted versus observed Cmax (log–log)</title>
      {decades.map((d, i) => (
        <g key={i}>
          <line className="grid" x1={X(Math.pow(10, d))} y1={mt} x2={X(Math.pow(10, d))} y2={mt + ph} />
          <line className="grid" x1={ml} y1={Y(Math.pow(10, d))} x2={w - mr} y2={Y(Math.pow(10, d))} />
          <text className="tick" x={X(Math.pow(10, d))} y={h - 12} textAnchor="middle">{"10" + sup(d)}</text>
          <text className="tick" x={ml - 8} y={Y(Math.pow(10, d)) + 3} textAnchor="end">{"10" + sup(d)}</text>
        </g>
      ))}
      <line x1={d3u.x1} y1={d3u.y1} x2={d3u.x2} y2={d3u.y2} stroke="var(--clay)" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />
      <line x1={d3l.x1} y1={d3l.y1} x2={d3l.x2} y2={d3l.y2} stroke="var(--clay)" strokeWidth="1" strokeDasharray="2 4" opacity="0.5" />
      <line x1={d2u.x1} y1={d2u.y1} x2={d2u.x2} y2={d2u.y2} stroke="var(--teal)" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
      <line x1={d2l.x1} y1={d2l.y1} x2={d2l.x2} y2={d2l.y2} stroke="var(--teal)" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
      <line x1={d1.x1} y1={d1.y1} x2={d1.x2} y2={d1.y2} stroke="var(--ink-soft)" strokeWidth="1.3" opacity="0.8" />

      <line className="axis" x1={ml} y1={mt} x2={ml} y2={mt + ph} />
      <line className="axis" x1={ml} y1={mt + ph} x2={w - mr} y2={mt + ph} />

      {points.map((p, i) => (
        <circle key={i} cx={X(p.obs)} cy={Y(p.pred)} r="3.4"
          fill={p.inDom ? "var(--blue)" : "var(--clay)"}
          opacity={p.inDom ? 0.62 : 0.5}
          stroke={p.inDom ? "var(--blue-deep)" : "var(--clay)"} strokeWidth="0.5" />
      ))}
      <text className="alab" x={ml + pw / 2} y={h - 1} textAnchor="middle">observed Cmax (mg/L)</text>
      <text className="alab" transform={`translate(12 ${mt + ph / 2}) rotate(-90)`} textAnchor="middle">predicted (mg/L)</text>
    </svg>
  );
}

/** Diagnostic Monte-Carlo histogram (parameter-uncertainty propagation). */
export function McHistogram({ cmax, cv }: { cmax: number; cv: number }) {
  const w = 580, h = 240, ml = 50, mr = 16, mt = 12, mb = 34;
  const pw = w - ml - mr, ph = h - mt - mb;
  const bins = 22;
  const lo = cmax * 0.4, hi = cmax * 1.9;
  const arr = new Array(bins).fill(0) as number[];
  let seed = Math.round(cmax * 1e4) || 7;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 1000; i++) {
    const z = (rnd() + rnd() + rnd() - 1.5) / 0.5;
    const v = cmax * Math.exp(z * cv);
    const b = Math.floor(((v - lo) / (hi - lo)) * bins);
    if (b >= 0 && b < bins) arr[b]++;
  }
  const max = Math.max.apply(null, arr);
  const bw = pw / bins;
  const fmtv = (v: number) => (v >= 1 ? (+v.toFixed(2)).toString() : v.toFixed(3));
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img">
      <title>Monte-Carlo Cmax distribution (diagnostic)</title>
      <line className="axis" x1={ml} y1={mt + ph} x2={w - mr} y2={mt + ph} />
      {arr.map((v, i) => (
        <rect key={i} x={ml + i * bw + 1} y={mt + ph - (v / max) * ph} width={bw - 2} height={(v / max) * ph} fill="var(--blue)" opacity="0.55" rx="1.5" />
      ))}
      <line x1={ml + ((cmax - lo) / (hi - lo)) * pw} y1={mt} x2={ml + ((cmax - lo) / (hi - lo)) * pw} y2={mt + ph} stroke="var(--ink)" strokeWidth="1.4" strokeDasharray="4 3" />
      {[lo, cmax, hi].map((v, i) => (
        <text key={i} className="tick" x={ml + ((v - lo) / (hi - lo)) * pw} y={h - 12} textAnchor="middle">{fmtv(v)}</text>
      ))}
      <text className="alab" x={ml + pw / 2} y={h - 1} textAnchor="middle">Cmax (mg/L)</text>
      <text className="alab" transform={`translate(12 ${mt + ph / 2}) rotate(-90)`} textAnchor="middle">density</text>
    </svg>
  );
}

/** Trough-by-dose-number bars (steady-state accumulation). */
export function TroughChart({ troughs, ssDose, cssmin }: { troughs: number[]; ssDose: number; cssmin: number }) {
  const w = 580, h = 240, ml = 50, mr = 16, mt = 12, mb = 34;
  const pw = w - ml - mr, ph = h - mt - mb;
  const max = Math.max.apply(null, troughs) * 1.15;
  const bw = pw / troughs.length;
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img">
      <title>Trough concentration by dose number</title>
      <line x1={ml} y1={mt + ph - (cssmin / max) * ph} x2={w - mr} y2={mt + ph - (cssmin / max) * ph} stroke="var(--clay)" strokeWidth="1" strokeDasharray="4 4" opacity="0.7" />
      <line className="axis" x1={ml} y1={mt} x2={ml} y2={mt + ph} />
      <line className="axis" x1={ml} y1={mt + ph} x2={w - mr} y2={mt + ph} />
      {troughs.map((v, i) => {
        const x = ml + i * bw + bw / 2;
        return (
          <g key={i}>
            <rect x={x - bw * 0.32} y={mt + ph - (v / max) * ph} width={bw * 0.64} height={(v / max) * ph} fill={i + 1 >= ssDose ? "var(--teal)" : "var(--blue)"} opacity="0.6" rx="2" />
            {(i % 2 === 0 || i === troughs.length - 1) && (
              <text className="tick" x={x} y={h - 12} textAnchor="middle">{i + 1}</text>
            )}
          </g>
        );
      })}
      <text className="alab" x={ml + pw / 2} y={h - 1} textAnchor="middle">dose number</text>
      <text className="alab" transform={`translate(12 ${mt + ph / 2}) rotate(-90)`} textAnchor="middle">trough (mg/L)</text>
    </svg>
  );
}
