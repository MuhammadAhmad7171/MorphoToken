#!/usr/bin/env bash
set -euo pipefail

: "${BRISC_ROOT:?Set BRISC_ROOT to the brisc2025 directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/morphology_ablation}"

python scripts/morphology_ablation.py \
  --stage morphology-ablation \
  --brisc-root "$BRISC_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --split-seed 2026 \
  --morphology-ablation-seeds 2026,3407,5891 \
  --morphology-ablation-variants M0,M1,M2,M3,M4,M5,M6 \
  --class-weights 1,1,1,1.1 \
  --image-size 224 \
  --epochs 80 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr 2.5e-4 \
  --weight-decay 5e-2 \
  --warmup-epochs 5 \
  --grad-clip 1.0 \
  --embed-dim 192 \
  --transformer-depth 4 \
  --heads 6 \
  --dropout 0.10 \
  --prototypes 4,3,3,2 \
  --temperature 0.10 \
  --lambda-mask 0.50 \
  --lambda-proto 0.20 \
  --lambda-sep 0.10 \
  --lambda-div 0.02 \
  --lambda-branch 0.15 \
  --ema-decay 0.995 \
  --separation-margin 0.20 \
  --early-stop 15
