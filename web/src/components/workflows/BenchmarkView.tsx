/* ============================================================
   BenchmarkView — holdout validation. The scatter and per-track
   AAFE are the REAL N=107 holdout results
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
            Predicted vs observed C<sub>max</sub><span className="meta">N = {b.n_holdout} · log–log</span>
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
                <tr><td>Meta, in-domain</td><td>{f2(b.in_domain.meta.aafe)}</td><td>{f1(b.in_domain.meta.pct_2fold)}%</td></tr>
              </tbody>
            </table>
          </div>
          <div className="panel">
            <Caveat>
              The holdout has informed ~47 tuning cycles; a cherry-picking audit scores aggregate risk <b>4.65/10</b>. The headline cannot statistically reject that tuning inflated AAFE — a permanent N50 holdout is planned.
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
              <tr className="hl"><td>All</td><td className="big">3.27</td><td>2.42–4.37</td><td>28.6%</td><td>28</td></tr>
              <tr><td>In-domain</td><td>3.37</td><td>2.06–5.23</td><td>37.5%</td><td>16</td></tr>
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h5>Reading</h5>
          <p className="note" style={{ margin: 0 }}>
            Prospective generalization is <b>worse</b> than retrospective ({f2(b.overall.meta.aafe)} → 3.27), reversing an earlier favorable read that was a small-sample artifact. New NMEs are markedly harder for the engine — the binding constraint shifts from CL<sub>int</sub> to first-pass <b>bioavailability (F)</b>. The gap is directional, not yet statistically separated.
          </p>
        </div>
      </div>
    );

  // tracks tab
  const engW = 100;
  const mlW = Math.round((c.HOLDOUT_AAFE === 0 ? 0 : c.ML_AAFE / c.ENGINE_AAFE) * 100);
  const metaW = Math.round((c.HOLDOUT_AAFE / c.ENGINE_AAFE) * 100);
  return (
    <div className="stack">
      <div className="panel">
        <h5>Why four tracks</h5>
        <p className="note" style={{ margin: "0 0 14px" }}>
          Each track reaches C<sub>max</sub> through different input channels, so their errors decorrelate. The meta-learner exploits this; no single track wins everywhere.
        </p>
        <div className="trk w"><span className="tn">Engine</span><span className="bar"><i style={{ width: engW + "%", background: "var(--blue)" }} /></span><span className="tv">{f2(c.ENGINE_AAFE)}</span><span className="tw">AAFE</span></div>
        <div className="trk w"><span className="tn">ML</span><span className="bar"><i style={{ width: mlW + "%", background: "var(--ink-soft)" }} /></span><span className="tv">{f2(c.ML_AAFE)}</span><span className="tw">AAFE</span></div>
        <div className="trk w"><span className="tn">Meta</span><span className="bar"><i style={{ width: metaW + "%", background: "var(--ink)" }} /></span><span className="tv">{f2(c.HOLDOUT_AAFE)}</span><span className="tw">AAFE</span></div>
      </div>
      <div className="panel">
        <h5>The weakest link</h5>
        <Caveat>
          The XGBoost CL<sub>int</sub> model plateaus at R² ≈ <b>0.24</b> across 41 documented approaches — the bottleneck is assay noise in public hepatocyte data, not model capacity. Bayesian TDM mitigates this per-patient (posterior CV −55%).
        </Caveat>
      </div>
    </div>
  );
}
