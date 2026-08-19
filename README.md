# MorphoToken

Research implementation of **MorphoToken**, a multi-scale morphology-aware token–prototype model for four-class brain MRI classification.

The repository contains two executable research pipelines:

- `scripts/morphotoken.py` — primary M3 pipeline: BRISC development/internal evaluation, frozen PMRAM external validation, optional Figshare compatibility audit, dataset integrity checks, grid search, and DDP training.
- `scripts/morphology_ablation.py` — controlled A6 morphology study with M0–M6 operator variants and repeated training seeds.

> **Research use only.** This repository is not a medical device and is not intended for clinical diagnosis or treatment decisions.

## Architecture at a glance

The primary model uses a pretrained ResNet-50 backbone, layer-3 and layer-4 features, 192-dimensional projections, morphology-aware feature enrichment, cross-scale fusion at 14×14 resolution, a four-layer Transformer encoder, a mask-supervised token gate, class-specific prototypes, and a parallel discriminative head.

The primary morphology configuration is **M3**: identity features plus morphological gradients with kernel sizes 3, 5, and 7.

## Repository layout

```text
MorphoToken/
├── scripts/
│   ├── morphotoken.py
│   ├── morphology_ablation.py
│   ├── reproduce_primary.sh
│   └── reproduce_morphology_ablation.sh
├── docs/
│   ├── DATASETS.md
│   ├── REPRODUCIBILITY.md
│   └── OUTPUTS.md
├── tests/
│   └── test_repository.py
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-optional.txt
├── .gitignore
├── LICENSE
├── CHANGELOG.md
└── README.md
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For Figshare MATLAB v7.3 support, ablation plots, and optional FLOP profiling:

```bash
pip install -r requirements-optional.txt
```

## Data

Datasets are **not redistributed** in this repository. Obtain BRISC, PMRAM, and (optionally) Cheng/Figshare from their official distribution channels and comply with their licenses/terms.

See [`docs/DATASETS.md`](docs/DATASETS.md) for the directory layout expected by the code.

## Reproduce the primary M3 experiment

Set the two dataset paths and run the provided script:

```bash
export BRISC_ROOT=/path/to/brisc2025
export PMRAM_RAW_ROOT=/path/to/pmram_raw
bash scripts/reproduce_primary.sh
```

Equivalent direct command:

```bash
python scripts/morphotoken.py \
  --stage final \
  --brisc-root /path/to/brisc2025 \
  --pmram-raw-root /path/to/pmram_raw \
  --output-dir runs/primary_m3 \
  --seed 2026 \
  --class-weights 1,1,1,1
```

The `final` stage performs deterministic exact deduplication, trains the primary M3 configuration, selects the best EMA checkpoint by BRISC validation macro-F1, evaluates the frozen BRISC internal test, and then evaluates the frozen checkpoint on raw PMRAM.

## Reproduce the M0–M6 morphology ablation

```bash
export BRISC_ROOT=/path/to/brisc2025
bash scripts/reproduce_morphology_ablation.sh
```

The controlled ablation uses:

- fixed split seed: `2026`
- training seeds: `2026, 3407, 5891`
- architecture: `A6`
- morphology variants: `M0`–`M6`
- class weights: `1,1,1,1.1`
- model selection: BRISC validation macro-F1 only

Variant definitions:

| ID | Morphology branch |
|---|---|
| M0 | No morphology branch |
| M1 | Identity + gradient k=3 |
| M2 | Identity + gradients k=3,5 |
| M3 | Identity + gradients k=3,5,7 |
| M4 | Identity + dilations k=3,5,7 |
| M5 | Identity + erosions k=3,5,7 |
| M6 | Gradients k=3,5,7 without identity |

## Verification stages

The primary pipeline exposes explicit gates that can be run independently:

```bash
python scripts/morphotoken.py --stage verify-data --brisc-root /path/to/brisc2025
python scripts/morphotoken.py --stage verify-preprocess --brisc-root /path/to/brisc2025
python scripts/morphotoken.py --stage verify-model --brisc-root /path/to/brisc2025
```

For end-to-end BRISC training followed by external evaluation, use `--stage all` and provide `--pmram-raw-root`.

## Multi-GPU training

The non-`final` training path supports PyTorch DDP. Example for two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 scripts/morphotoken.py \
  --stage train \
  --brisc-root /path/to/brisc2025 \
  --output-dir runs/ddp_train
```

`final`, `gridsearch`, and `morphology-ablation` are intentionally single-process modes.

## Important implementation details

- **TTA:** the implementation averages original-image and horizontal-flip **logits before softmax**.
- **Image size:** preprocessing resizes directly to `--image-size` (default 224). `--resize-size` is retained only as a legacy compatibility argument and is not part of the current transform path.
- **Masks:** tumor masks supervise the diagnostic token gate during training; masks are not required at inference.
- **Deduplication:** exact byte-identical images are removed deterministically before the main `final` experiment.
- **Figshare:** the code treats Figshare as an optional compatibility/source-overlap analysis, not as independent external evidence.

## Outputs

Runs write JSON/JSONL audit files, checkpoints, split summaries, metrics, and ablation tables/figures. See [`docs/OUTPUTS.md`](docs/OUTPUTS.md).

## Reproducibility notes

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for seeds, checkpoint selection, EMA behavior, TTA semantics, and the distinction between the primary run and morphology ablation.

## Testing

The repository CI performs source compilation and lightweight static integrity tests without requiring datasets or a GPU:

```bash
python -m compileall scripts
python -m unittest discover -s tests -v
```

Full model/data verification requires the datasets and PyTorch dependencies.

## License

Released under the MIT License. Dataset licenses are separate and remain the responsibility of the user.
