# DrugBank Integration — Empirical Findings

**Date:** 2026-03-24

## Key Numbers

| Metric | Value |
|--------|-------|
| Holdout drugs evaluated | 38 / 100 |
| Holdout contamination (TDC vs holdout) | 15 drugs (multi-key catch) |
| fup v2 training samples | 2,753 (TDC 1,614 + DrugBank 1,439 - overlap - holdout) |
| fup v2 CV R² | 0.411 |
| fup v2 CV AAFE | 2.404 |
| logP correction RMSE improvement | 1.587 → 1.343 (-15.4%) |
| Ablation training effect (Silver - Baseline) | -0.009 AAFE (noise) |
| Ablation lookup effect (Gold - Silver) | +0.013 AAFE (noise) |

## What We Learned

1. **ADME 개선이 Cmax AAFE에 전달되지 않는다.** fup R² 개선 + logP RMSE 15% 감소에도
   holdout AAFE는 ±0.01 수준 (노이즈). 근본 원인: meta-learner engine weight 17%.

2. **Meta-learner가 bottleneck.** Engine의 정확도가 개선되어도 83% ML weight가 지배.
   Engine accuracy 개선 → meta-learner 재학습 없이는 최종 Cmax에 반영 안 됨.

3. **CLint R²=0.24는 미해결.** DrugBank에 정량적 CLint 데이터 없음. 가장 큰 error source.

4. **Holdout N=38은 통계적으로 부족.** 100 drugs 중 62개가 dose/SMILES 미비 또는 실패.
   ±0.02 AAFE 변화가 유의미한지 판단 불가.

5. **Error cancellation 확인.** Gold (lookup ON)이 Silver보다 AAFE가 미세 악화.
   Measured values가 calibrated prediction의 error cancellation을 깨뜨림.

## v2 Implications

- **Meta-learner 재학습이 1순위.** Engine이 더 정확해졌다는 signal을 meta-learner에 반영해야 함.
- **Holdout 보강 필요.** N=38 → N=80+ 확보 (dose 정보 추가 또는 DrugBank에서 dose 확보).
- **CLint 개선 경로 필요.** ChEMBL microsomal CLint 데이터? 또는 GNN 아키텍처?
- **Engine-only benchmark 추가.** Meta-learner 없이 engine 단독 AAFE 측정하면
   ADME 개선 효과를 직접 확인 가능.
