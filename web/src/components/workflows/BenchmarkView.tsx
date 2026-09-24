/* ============================================================
   BenchmarkView — retrospective development evidence. The scatter and per-track
   AAFE are the source-audited N=105 development-benchmark results
   (data/training/4track_holdout_predictions.json).
   ============================================================ */
import type { ConsoleData } from "../../types";
import { ScatterChart, type ScatterPt } from "../charts";
import { Legend, Caveat } from "../panels";

const f1 = (v: number) => (+v.toFixed(1)).toString();
const f2 = (v: number) => (+v.toFixed(2)).toString();

export function BenchmarkView({ tab, data }: { tab: number; data: ConsoleData }) {
  const b = data.benchmark;
  const c = data.constants;

  if (tab === 0) {
    const pts: ScatterPt[] = b.scatter.map((p) => ({ obs: p.obs, pred: p.meta, inDom: p.in_ad }));
    return (
      <div className="split">
        <div className="panel">
          <h5>
            Predicted vs observed C<sub>max</sub><span className="meta">N = {b.n_development} · log–log</span>
          </h5>
          <ScatterChart points={pts} h={360} />
          <div style={{ marginTop: 10 }}>
            <Legend
              items={[
                { type: "rect", color: "var(--blue)", label: "in-domain" },
                { type: "rect", color: "var(--clay)", label: "out-of-domain" },
                { type: "line", color: "var(--ink-soft)", label: "unity" },
                { type: "dash", color: "var(--teal)", label: "2-fold" },
              ]}
            />
          </div>
        </div>
        <div className="stack">
          <div className="panel">
            <h5>AAFE by track</h5>
            <table className="btable">
              <thead>
                <tr><th>Track</th><th>AAFE</th><th>%2-fold</th></tr>
              </thead>
              <tbody>
                <tr className="hl"><td>Meta (prod.)</td><td className="big">{f2(b.overall.meta.aafe)}</td><td>{f1(b.overall.meta.pct_2fold)}%</td></tr>
                <tr><td>Engine</td><td>{f2(b.overall.engine.aafe)}</td><td>{f1(b.overall.engine.pct_2fold)}%</td></tr>
                <tr><td>ML</td><td>{f2(b.overall.ml.aafe)}</td><td>{f1(b.overall.ml.pct_2fold)}%</td></tr>
              </tbody>
            </table>
          </div>
          <div className="panel">
            <Caveat>
              Paired Meta/ML AAFE ratio: <b>{f2(b.paired_meta_ml.ratio)}</b> (95% bootstrap CI {f2(b.paired_meta_ml.ci_95_low)}–{f2(b.paired_meta_ml.ci_95_high)}). The original N=107 cohort informed ~47 system-selection cycles; this audited N={b.n_development} subset is development evidence, and its interval excludes adaptive-selection bias. An outcome-blinded external set is still required.
            </Caveat>
          </div>
        </div>
      </div>
    );
  }

  if (tab === 1)
    return (
      <div className="stack">
        <div className="panel">
          <h5>
            Prospective · FDA NMEs 2024–2025<span className="meta">production-clean · post-FLUX-1 · N=28</span>
          </h5>
          <table className="btable">
            <thead>
              <tr><th>Slice</th><th>AAFE</th><th>95% CI</th><th>%2-fold</th><th>N</th></tr>
            </thead>
            <tbody>
              <tr className="hl"><td>All</td><td className="big">3.29</td><td>2.45–4.48*</td><td>25.0%</td><td>28</td></tr>
              <tr><td>In-domain</td><td>3.32</td><td>diagnostic</td><td>25.0%</td><td>16</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h5>Reading</h5>
          <p className="note" style={{ margin: 0 }}>
            The consumed temporal challenge is <b>worse</b> than the development benchmark ({f2(b.overall.meta.aafe)} → 3.29). New NMEs are markedly harder for the engine, with first-pass <b>bioavailability</b> underprediction prominent. This cohort has now informed diagnosis and cannot serve as the current model's independent holdout. *CI is a read-only diagnostic bootstrap on the current local-stack artifact.
          </p>
        </div>
      </div>
    );

  // tracks tab
  const engW = 100;
  const mlW = Math.round((c.DEVELOPMENT_AAFE === 0 ? 0 : c.ML_AAFE / c.ENGINE_AAFE) * 100);
  const metaW = Math.round((c.DEVELOPMENT_AAFE / c.ENGINE_AAFE) * 100);
  return (
    <div className="stack">
      <div className="panel">
        <h5>Why four tracks</h5>
        <p className="note" style={{ margin: "0 0 14px" }}>
          Each track reaches C<sub>max</sub> through different input channels, so their errors decorrelate. The meta-learner exploits this; no single track wins everywhere.
        </p>
        <div className="trk w"><span className="tn">Engine</span><span className="bar"><i style={{ width: engW + "%", background: "var(--blue)" }} /></span><span className="tv">{f2(c.ENGINE_AAFE)}</span><span className="tw">AAFE</span></div>
        <div className="trk w"><span className="tn">ML</span><span className="bar"><i style={{ width: mlW + "%", background: "var(--ink-soft)" }} /></span><span className="tv">{f2(c.ML_AAFE)}</span><span className="tw">AAFE</span></div>
        <div className="trk w"><span className="tn">Meta</span><span className="bar"><i style={{ width: metaW + "%", background: "var(--ink)" }} /></span><span className="tv">{f2(c.DEVELOPMENT_AAFE)}</span><span className="tw">AAFE</span></div>
      </div>
      <div className="panel">
        <h5>The weakest link</h5>
        <Caveat>
          The XGBoost CL<sub>int</sub> model plateaus at R² ≈ <b>0.24</b> across many documented approaches. More architecture search on this consumed cohort is not independent evidence; progress requires cleaner clinical targets, an outcome-blinded holdout, and a pre-registered hypothesis.
        </Caveat>
      </div>
    </div>
  );
}
