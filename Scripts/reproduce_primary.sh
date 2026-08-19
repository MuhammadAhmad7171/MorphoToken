#!/usr/bin/env bash
set -euo pipefail

: "${BRISC_ROOT:?Set BRISC_ROOT to the brisc2025 directory}"
: "${PMRAM_RAW_ROOT:?Set PMRAM_RAW_ROOT to the raw PMRAM directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/primary_m3}"

python scripts/morphotoken.py \
  --stage final \
  --brisc-root "$BRISC_ROOT" \
  --pmram-raw-root "$PMRAM_RAW_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --seed 2026 \
  --image-size 224 \
  --epochs 80 \
  --batch-size 32 \
  --eval-batch-size 64 \
  --lr 2.5e-4 \
  --weight-decay 5e-2 \
  --warmup-epochs 5 \
  --grad-clip 1.0 \
  --class-weights 1,1,1,1 \
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
