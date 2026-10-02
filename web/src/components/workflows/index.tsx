import type { AppState, ConsoleData, Drug, WorkflowId } from "../../types";
import { drugById } from "../../data";
import { PredictView } from "./PredictView";
import { BenchmarkView } from "./BenchmarkView";

export { WORKFLOWS } from "./config";

export function WorkflowView({
  wf,
  s,
  tab,
  running,
  data,
  drug: drugOverride,
}: {
  wf: WorkflowId;
  s: AppState;
  tab: number;
  running: boolean;
  data: ConsoleData;
  drug?: Drug;
}) {
  const drug = drugOverride ?? drugById(data, s.drugId);
  if (wf === "predict") return <PredictView drug={drug} s={s} tab={tab} running={running} />;
  if (wf === "benchmark") return <BenchmarkView tab={tab} data={data} />;
  return null;
}
