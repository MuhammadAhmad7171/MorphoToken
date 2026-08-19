# Reproducibility protocol

## Primary M3 experiment

The publication-oriented reproduction script fixes the main configuration at:

- seed: 2026
- image size: 224
- ResNet-50 backbone with ImageNet initialization
- embedding dimension: 192
- Transformer depth: 4
- attention heads: 6
- dropout: 0.10
- prototypes: 4/3/3/2
- prototype temperature: 0.10
- AdamW learning rate: 2.5e-4
- weight decay: 0.05
- backbone LR multiplier: 0.10
- warm-up: 5 epochs
- maximum epochs: 80
- early stopping patience: 15
- EMA decay: 0.995
- equal primary class weights: 1/1/1/1
- mask/prototype/separation/diversity/branch loss weights: 0.50/0.20/0.10/0.02/0.15

The best checkpoint is selected by BRISC validation macro-F1 using EMA weights. The selected checkpoint is then frozen for internal/external evaluation.

## Morphology ablation

The controlled operator study fixes the data split independently from model initialization:

- split seed: 2026
- training seeds: 2026, 3407, 5891
- architecture variant: A6
- class weights: 1/1/1/1.1
- variants: M0–M6

The class-weight override is intentionally explicit in `scripts/reproduce_morphology_ablation.sh`; the generic Python CLI keeps neutral `1,1,1,1` defaults so users can run other experiments without silently inheriting the paper-specific ablation weighting.

## TTA semantics

Evaluation with `--tta` computes logits for the original image and its horizontal flip, averages those logits, and then applies softmax. This is **logit averaging**, not probability averaging.

## Preprocessing

The implemented order is:

1. convert to grayscale;
2. optional foreground crop;
3. optional histogram equalization;
4. square pad;
5. replicate grayscale to three channels;
6. resize directly to `--image-size`;
7. for training only: paired flip/affine operations plus image-only color jitter;
8. ImageNet normalization.

The `--resize-size` CLI argument is retained for compatibility with older run configurations but is not used in the current transform.

## Exact deduplication

`--stage final` loads BRISC train/test and PMRAM raw data, computes SHA-256 digests, and retains the first exact copy in the priority order BRISC train → BRISC test → PMRAM. Cross-group exact duplicates are therefore removed before the final training/evaluation path.

Near-duplicate dHash matches are audited separately and are not automatically removed unless the user enables the corresponding failure option.

## Determinism

The scripts seed Python, NumPy, and PyTorch RNGs. Exact bitwise reproduction can still depend on GPU model, CUDA/cuDNN behavior, PyTorch/torchvision version, distributed execution, and hardware-specific kernels. Preserve `run_config.json`, `history.jsonl`, environment metadata, and checkpoint hashes for archival releases.

## Checkpoint release recommendation

For a paper release, publish the selected `best.pt` separately (for example through a GitHub Release or archival repository) and record its SHA-256 checksum. Large checkpoints should not be committed directly to Git history.
