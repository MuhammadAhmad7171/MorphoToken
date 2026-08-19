# Dataset layout

The code does not download or redistribute datasets.

## BRISC

Pass `--brisc-root` as the directory containing the BRISC classification and segmentation tasks. The loader expects the following structure:

```text
brisc2025/
├── classification_task/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── pituitary/
│   │   └── no_tumor/
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── no_tumor/
└── segmentation_task/
    ├── train/masks/
    └── test/masks/
```

Tumor-class images are matched to masks by filename stem. A missing tumor mask is treated as an error. `no_tumor` examples may have no mask and are represented by an all-zero target for the auxiliary gate loss.

## PMRAM

Pass `--pmram-raw-root` to a directory containing **raw, non-augmented** PMRAM images arranged under class-identifiable folders. The loader recognizes common names/aliases for glioma, meningioma, pituitary, and normal/no-tumor classes, including resolution-prefixed folders such as `512Glioma`.

The pipeline deliberately rejects paths/content that appear to be augmented when they are supplied as the raw external set.

## Cheng/Figshare (optional)

Pass `--figshare-root` to the original `.mat` collection. `cvind.mat` is ignored. The code supports classic MATLAB files through SciPy and MATLAB v7.3/HDF5 files when `h5py` is installed.

Figshare is handled as a compatibility/source-overlap analysis in this implementation rather than as independent external validation.

## Dataset provenance

Do not commit MRI images, masks, patient information, or third-party dataset archives to this repository unless their licenses explicitly permit redistribution.
