import type { AppState, ConsoleData, WorkflowId } from "../../types";
import { drugById } from "../../data";
import { PredictView } from "./PredictView";
import { SimulateView } from "./SimulateView";
import { TdmView } from "./TdmView";
import { DdiView } from "./DdiView";
import { DoseAdjustView } from "./DoseAdjustView";
import { BenchmarkView } from "./BenchmarkView";

export { WORKFLOWS, defObs, oid } from "./config";

export function WorkflowView({
  wf,
  s,
  tab,
  running,
  data,
}: {
  wf: WorkflowId;
  s: AppState;
  tab: number;
  running: boolean;
  data: ConsoleData;
}) {
  const drug = drugById(data, s.drugId);
  if (wf === "predict") return <PredictView drug={drug} s={s} tab={tab} running={running} />;
  if (wf === "simulate") return <SimulateView drug={drug} s={s} tab={tab} data={data} />;
  if (wf === "tdm") return <TdmView drug={drug} s={s} tab={tab} />;
  if (wf === "ddi") return <DdiView drug={drug} s={s} tab={tab} data={data} />;
  if (wf === "dose-adjust") return <DoseAdjustView drug={drug} s={s} tab={tab} />;
  if (wf === "benchmark") return <BenchmarkView tab={tab} data={data} />;
  return null;
}
