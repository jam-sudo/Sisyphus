#!/bin/bash
# Phase 2.0.5 pipeline: train → SBC → routing table
# Run AFTER sbi_generate_multi_drug_data.py completes
set -e

echo "=== Phase 2.0.5: Train amortizer ==="
python3 scripts/sbi_train_multi_drug.py \
    --data data/sbi/multi_drug_train_v2.npz \
    --out models/sbi/multi_drug_nsf_v2.pt \
    --density-estimator nsf \
    --hidden-features 64 \
    --num-transforms 8 \
    --embedding-hidden 32 \
    --batch-size 512 \
    --lr 5e-4 \
    --max-epochs 500 \
    --stop-after-epochs 40

echo ""
echo "=== Phase 2.0.5: Run SBC on 13 validation drugs ==="
python3 scripts/sbi_run_sbc_multi_drug.py \
    --posterior models/sbi/multi_drug_nsf_v2.pt \
    --n-calibration 300 \
    --n-posterior 500 \
    --out data/validation/sbi_sbc_multi_drug_v2.json \
    --coverage-tol 0.10

echo ""
echo "=== Phase 2.0.5: Build routing table ==="
python3 scripts/sbi_build_routing_table.py \
    --sbc data/validation/sbi_sbc_multi_drug_v2.json \
    --out data/sbi/method_routing_v2.json

echo ""
echo "=== Phase 2.0.5 pipeline complete ==="
echo "Check data/validation/sbi_sbc_multi_drug_v2.json for per-drug results"
echo "Check data/sbi/method_routing_v2.json for updated routing"
