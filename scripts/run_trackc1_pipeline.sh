#!/bin/bash
# Track C1 pipeline: train → SBC (hierarchical)
# Run AFTER sbi_generate_hierarchical_data.py completes
set -e

echo "=== Track C1: Train hierarchical amortizer ==="
python3 scripts/sbi_train_hierarchical.py \
    --data data/sbi/hierarchical_train.npz \
    --out models/sbi/hierarchical_nsf.pt \
    --density-estimator nsf \
    --hidden-features 64 \
    --num-transforms 8 \
    --embedding-hidden 32 \
    --batch-size 512 \
    --lr 5e-4 \
    --max-epochs 500 \
    --stop-after-epochs 40

echo ""
echo "=== Track C1: Run hierarchical SBC ==="
python3 scripts/sbi_run_sbc_hierarchical.py \
    --posterior models/sbi/hierarchical_nsf.pt \
    --n-calibration 300 \
    --n-posterior 500 \
    --out data/validation/sbi_sbc_hierarchical.json \
    --coverage-tol 0.10

echo ""
echo "=== Track C1 pipeline complete ==="
echo "Check data/validation/sbi_sbc_hierarchical.json for per-(pop, drug) results"
