#!/usr/bin/env python3
"""Multi-Scale MorphoToken: BRISC training and frozen PMRAM validation.

Single-file research pipeline with explicit stages:
  verify-data -> verify-preprocess -> verify-model -> train -> external

Launch with torchrun for two-GPU DDP. The script deliberately requires a path
to PMRAM's *raw* images so augmented copies cannot enter external validation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from itertools import product
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps
from scipy.io import loadmat

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision import models as tv_models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid, save_image

ImageFile.LOAD_TRUNCATED_IMAGES = False

CLASSES = ("glioma", "meningioma", "pituitary", "no_tumor")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ALIASES = {
    "glioma": "glioma", "gliomas": "glioma", "gl": "glioma",
    "meningioma": "meningioma", "meningiomas": "meningioma", "menin": "meningioma", "me": "meningioma",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary", "pituitarytumor": "pituitary", "pi": "pituitary",
    "no_tumor": "no_tumor", "notumor": "no_tumor", "no-tumor": "no_tumor", "normal": "no_tumor",
    "healthy": "no_tumor", "no tumor": "no_tumor",
}


@dataclass(frozen=True)
class Sample:
    image: str
    label: int
    mask: Optional[str]
    source: str


@dataclass(frozen=True)
class FigshareSample:
    mat_file: str
    label: int
    patient_id: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("final", "verify-data", "verify-preprocess", "verify-model", "train", "external", "all", "gridsearch"), default="final")
    p.add_argument("--brisc-root", type=Path, required=True, help="Path to brisc2025/ containing classification_task and segmentation_task")
    p.add_argument("--pmram-raw-root", type=Path, default=None, help="Path containing ONLY PMRAM raw images; required for external/all")
    p.add_argument("--figshare-root", type=Path, default=None,
                   help="Optional Cheng/Figshare root containing the original .mat files")
    p.add_argument("--output-dir", type=Path, default=Path("runs/morphotoken"))
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for --stage external; defaults to output-dir/best.pt")
    p.add_argument("--ensemble-checkpoints", type=str, default=None,
                   help="Comma-separated checkpoints for probability-averaged external evaluation")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resize-size", type=int, default=256,
                   help="Legacy compatibility argument; current preprocessing resizes directly to --image-size")
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--eval-batch-size", type=int, default=64, help="Per-GPU evaluation batch size")
    p.add_argument("--workers", type=int, default=8, help="Workers per GPU")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-2)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--class-weights", type=str, default="1,1,1,1",
                   help="Training CE weights: glioma,meningioma,pituitary,no_tumor")
    p.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16")
    p.add_argument("--embed-dim", type=int, default=192)
    p.add_argument("--transformer-depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--backbone", choices=("resnet50",), default="resnet50")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ImageNet pretrained backbone for training/model verification")
    p.add_argument("--backbone-lr-mult", type=float, default=0.10)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0,
                   help="Use 0 for DDP; positive values are supported for single-GPU warm-up")
    p.add_argument("--histogram-equalization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--foreground-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                   help="Average original and horizontal-flip logits before softmax during evaluation")
    p.add_argument("--prototypes", type=str, default="4,3,3,2", help="glioma,meningioma,pituitary,no_tumor")
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--lambda-mask", type=float, default=0.50)
    p.add_argument("--lambda-proto", type=float, default=0.20)
    p.add_argument("--lambda-sep", type=float, default=0.10)
    p.add_argument("--lambda-div", type=float, default=0.02)
    p.add_argument("--lambda-branch", type=float, default=0.15,
                   help="Auxiliary supervision for CNN and prototype classifier branches")
    p.add_argument("--ema-decay", type=float, default=0.995,
                   help="Exponential moving-average decay used for validation and best checkpoint")
    p.add_argument("--separation-margin", type=float, default=0.20)
    p.add_argument("--verify-all-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--near-duplicate-hamming", type=int, default=2)
    p.add_argument("--fail-on-near-duplicates", action="store_true")
    p.add_argument("--allow-cross-dataset-exact-duplicates", action="store_true")
    p.add_argument("--deduplicate-exact", action=argparse.BooleanOptionalAction, default=True,
                   help="Deterministically remove byte-identical repeats, prioritizing BRISC train, then test, then PMRAM")
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument("--grid-lr", type=str, default="1.5e-4,2e-4")
    p.add_argument("--grid-weight-decay", type=str, default="0.03,0.05")
    p.add_argument("--grid-label-smoothing", type=str, default="0.03,0.05")
    p.add_argument("--grid-ema-decay", type=str, default="0.999")
    p.add_argument("--grid-epochs", type=int, default=40,
                   help="Maximum epochs for each validation-only grid trial")
    p.add_argument("--grid-early-stop", type=int, default=10)
    args = p.parse_args()
    args.prototypes = tuple(int(x) for x in args.prototypes.split(","))
    if len(args.prototypes) != len(CLASSES) or min(args.prototypes) < 1:
        p.error("--prototypes must contain four positive integers")
    try:
        args.class_weights = tuple(float(x) for x in args.class_weights.split(","))
    except ValueError:
        p.error("--class-weights must contain four comma-separated numbers")
    if len(args.class_weights) != len(CLASSES) or min(args.class_weights) <= 0:
        p.error("--class-weights must contain four positive numbers")
    args.ensemble_checkpoints = (
        tuple(Path(x.strip()) for x in args.ensemble_checkpoints.split(",") if x.strip())
        if args.ensemble_checkpoints else tuple()
    )
    if args.stage in {"final", "external", "all", "gridsearch"} and args.pmram_raw_root is None:
        p.error("--pmram-raw-root is required for external/all; point it to raw images only")
    if not 0.01 <= args.val_fraction <= 0.40:
        p.error("--val-fraction must be between 0.01 and 0.40")
    if not 0.0 < args.ema_decay < 1.0:
        p.error("--ema-decay must be between 0 and 1")
    if args.grid_epochs < 1 or args.grid_early_stop < 1:
        p.error("--grid-epochs and --grid-early-stop must be positive")
    return args


def ddp_setup() -> Tuple[bool, int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        cuda_device = torch.device(f"cuda:{local_rank}")
        try:
            # Newer PyTorch/NCCL needs this explicit mapping to avoid an
            # ambiguous-device barrier (and, on some systems, a SIGSEGV).
            dist.init_process_group(backend="nccl", init_method="env://", device_id=cuda_device)
        except TypeError:
            # Compatibility for older PyTorch; barrier() still supplies the
            # explicit local device below.
            dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier(device_ids=[torch.cuda.current_device()])


def seed_everything(seed: int, rank: int = 0) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def canonical_class(text: str) -> Optional[str]:
    x = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    direct = ALIASES.get(x) or ALIASES.get(text.lower().strip())
    if direct:
        return direct
    # PMRAM commonly prefixes class directories with the image resolution,
    # e.g. 512Glioma, 512Meningioma, 512Pituitary, and 512Normal.
    without_resolution = re.sub(r"^\d+_?", "", x)
    return ALIASES.get(without_resolution)


def all_images(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def find_class_from_path(path: Path, stop: Path) -> Optional[str]:
    for part in reversed(path.relative_to(stop).parts[:-1]):
        name = canonical_class(part)
        if name:
            return name
    return None


def locate_brisc_root(root: Path) -> Path:
    root = root.resolve()
    candidates = [root] + [p for p in root.rglob("brisc2025") if p.is_dir()]
    for c in candidates:
        if (c / "classification_task" / "train").is_dir() and (c / "segmentation_task" / "train").is_dir():
            return c
    raise FileNotFoundError(f"Could not find BRISC structure below {root}")


def mask_index(root: Path, split: str) -> Dict[str, Path]:
    d = root / "segmentation_task" / split / "masks"
    if not d.is_dir():
        raise FileNotFoundError(d)
    idx: Dict[str, Path] = {}
    for p in all_images(d):
        if p.stem in idx:
            raise RuntimeError(f"Duplicate mask stem: {p.stem}")
        idx[p.stem] = p
    return idx


def collect_brisc(root: Path, split: str) -> List[Sample]:
    croot = root / "classification_task" / split
    midx = mask_index(root, split)
    samples: List[Sample] = []
    for class_name in CLASSES:
        cdir = croot / class_name
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {cdir}")
        for p in all_images(cdir):
            mask = midx.get(p.stem)
            # BRISC does not necessarily provide segmentation masks for healthy
            # (no_tumor) images. PairedTransform converts a missing mask to an
            # all-zero target, which is the correct representation here.
            if mask is None and class_name != "no_tumor":
                raise RuntimeError(f"Missing aligned mask for {p}")
            samples.append(Sample(
                str(p),
                CLASS_TO_IDX[class_name],
                str(mask) if mask is not None else None,
                f"brisc-{split}",
            ))
    return samples


def collect_pmram_raw(root: Path) -> List[Sample]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    low = str(root).lower()
    if "augment" in low and "raw" not in low:
        raise RuntimeError("PMRAM path appears augmented. Point --pmram-raw-root to the raw-image directory.")
    samples: List[Sample] = []
    unknown: List[str] = []
    for p in all_images(root):
        if any("augment" in part.lower() for part in p.relative_to(root).parts):
            raise RuntimeError(f"Augmented PMRAM content found under raw root: {p}")
        cls = find_class_from_path(p, root)
        if cls is None:
            unknown.append(str(p))
            continue
        samples.append(Sample(str(p), CLASS_TO_IDX[cls], None, "pmram-raw"))
    if unknown:
        raise RuntimeError(f"Could not infer class for {len(unknown)} PMRAM images; first: {unknown[0]}")
    if not samples:
        raise RuntimeError(f"No class-organized PMRAM images found below {root}")
    return samples


FIGSHARE_LABEL_TO_CLASS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def _matlab_text(value: np.ndarray) -> str:
    flat = np.asarray(value).reshape(-1)
    if flat.dtype.kind in "ui":
        text = "".join(chr(int(x)) for x in flat if int(x) != 0)
    else:
        text = "".join(str(x) for x in flat)
    return text.strip() or "unknown"


def read_figshare_mat(path: str, load_image: bool = True) -> Tuple[Optional[np.ndarray], int, str]:
    """Read one original Cheng/Figshare MAT sample (classic MAT or v7.3/HDF5)."""
    try:
        data = loadmat(path)
        if "cjdata" not in data:
            raise RuntimeError(f"Missing cjdata structure in {path}")
        cj = data["cjdata"]
        raw_label = int(round(float(np.asarray(cj["label"][0, 0]).reshape(-1)[0])))
        pid = _matlab_text(np.asarray(cj["PID"][0, 0])) if "PID" in cj.dtype.names else Path(path).stem
        image = np.asarray(cj["image"][0, 0]) if load_image else None
    except NotImplementedError:
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("Reading MATLAB v7.3 files requires h5py: pip install h5py") from exc
        with h5py.File(path, "r") as f:
            if "cjdata" not in f:
                raise RuntimeError(f"Missing cjdata structure in {path}")
            cj = f["cjdata"]
            raw_label = int(round(float(np.asarray(cj["label"]).reshape(-1)[0])))
            pid = _matlab_text(np.asarray(cj["PID"])) if "PID" in cj else Path(path).stem
            image = np.asarray(cj["image"]).T if load_image else None
    if raw_label not in FIGSHARE_LABEL_TO_CLASS:
        raise RuntimeError(f"Unexpected Figshare label {raw_label} in {path}")
    label = CLASS_TO_IDX[FIGSHARE_LABEL_TO_CLASS[raw_label]]
    return image, label, pid


def figshare_uint8(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    if a.ndim != 2 or not np.isfinite(a).all() or a.size == 0:
        raise ValueError(f"Invalid Figshare image array: shape={a.shape}")
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def collect_figshare(root: Path) -> List[FigshareSample]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(root.rglob("*.mat"))
    # cvind.mat contains folds, not an image sample.
    files = [p for p in files if p.name.lower() != "cvind.mat"]
    if not files:
        raise RuntimeError(f"No Figshare image .mat files found below {root}")
    samples: List[FigshareSample] = []
    for p in files:
        _, label, pid = read_figshare_mat(str(p), load_image=False)
        samples.append(FigshareSample(str(p), label, pid))
    missing = [c for c in CLASSES[:3] if not any(s.label == CLASS_TO_IDX[c] for s in samples)]
    if missing:
        raise RuntimeError(f"Figshare dataset is missing tumor classes: {missing}")
    return samples


def decoded_dhash_array(array: np.ndarray, size: int = 8) -> int:
    im = Image.fromarray(figshare_uint8(array), mode="L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(im)
    value = 0
    for bit in (a[:, 1:] > a[:, :-1]).ravel():
        value = (value << 1) | int(bit)
    return value


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def dhash(path: str, size: int = 8) -> int:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR))
    bits = a[:, 1:] > a[:, :-1]
    value = 0
    for b in bits.ravel():
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    # bin(...).count works on older Python versions where int.bit_count is
    # unavailable and also handles NumPy integer scalars after the cast.
    return bin(int(a) ^ int(b)).count("1")


def verify_readable(samples: Sequence[Sample], verify_all: bool, seed: int) -> Dict[str, object]:
    chosen = list(samples) if verify_all else random.Random(seed).sample(list(samples), min(128, len(samples)))
    sizes, bad_masks, empty_masks = Counter(), [], []
    for s in chosen:
        try:
            with Image.open(s.image) as im:
                im.load()
                if im.width < 32 or im.height < 32:
                    raise ValueError(f"unreasonably small {im.size}")
                sizes[f"{im.width}x{im.height}"] += 1
            if s.mask:
                with Image.open(s.mask) as m:
                    ma = np.asarray(m.convert("L"))
                if ma.shape[:2] != (im.height, im.width):
                    bad_masks.append((s.image, s.mask, (im.height, im.width), ma.shape))
                if s.label != CLASS_TO_IDX["no_tumor"] and ma.max() == 0:
                    empty_masks.append(s.mask)
        except Exception as e:
            raise RuntimeError(f"Unreadable sample {s.image}: {e}") from e
    if bad_masks:
        raise RuntimeError(f"Image/mask dimension mismatch; first={bad_masks[0]}")
    if empty_masks:
        raise RuntimeError(f"Tumor class has empty mask; first={empty_masks[0]}")
    return {"checked": len(chosen), "unique_sizes": len(sizes), "most_common_sizes": sizes.most_common(10)}


def duplicate_report(groups: Dict[str, Sequence[Sample]], near_threshold: int) -> Dict[str, object]:
    hashes: Dict[str, Dict[str, List[str]]] = {}
    dhashes: Dict[str, Dict[int, List[str]]] = {}
    for name, samples in groups.items():
        sh, dh = defaultdict(list), defaultdict(list)
        for s in samples:
            sh[sha256(s.image)].append(s.image)
            dh[dhash(s.image)].append(s.image)
        hashes[name], dhashes[name] = dict(sh), dict(dh)
    exact_cross = {}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = set(hashes[a]) & set(hashes[b])
            if common:
                exact_cross[f"{a}__{b}"] = [{"a": hashes[a][h], "b": hashes[b][h]} for h in list(common)[:50]]
    near_cross = {}
    if near_threshold >= 0:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # Exact dHash buckets first; then adjacent distances up to the requested small threshold.
                found = []
                for ha, pa in dhashes[a].items():
                    for hb, pb in dhashes[b].items():
                        d = hamming(ha, hb)
                        if d <= near_threshold:
                            found.append({"distance": d, "a": pa[:3], "b": pb[:3]})
                            if len(found) >= 50:
                                break
                    if len(found) >= 50:
                        break
                if found:
                    near_cross[f"{a}__{b}"] = found
    within = {name: sum(len(v) - 1 for v in sh.values() if len(v) > 1) for name, sh in hashes.items()}
    return {"within_exact_duplicate_excess": within, "cross_exact": exact_cross, "cross_near_dhash": near_cross}


def deduplicate_exact(groups: Dict[str, Sequence[Sample]]) -> Tuple[Dict[str, List[Sample]], Dict[str, object]]:
    """Keep the first exact image copy according to insertion/group order."""
    seen: Dict[str, Tuple[str, Sample]] = {}
    cleaned: Dict[str, List[Sample]] = {}
    removed: Dict[str, List[Dict[str, object]]] = {}
    for group_name, samples in groups.items():
        kept: List[Sample] = []
        dropped: List[Dict[str, object]] = []
        for sample in samples:
            digest = sha256(sample.image)
            previous = seen.get(digest)
            if previous is None:
                seen[digest] = (group_name, sample)
                kept.append(sample)
                continue
            kept_group, kept_sample = previous
            if kept_sample.label != sample.label:
                raise RuntimeError(
                    "Exact duplicate has conflicting labels: "
                    f"{kept_sample.image} ({CLASSES[kept_sample.label]}) vs "
                    f"{sample.image} ({CLASSES[sample.label]})"
                )
            dropped.append({
                "removed": sample.image,
                "removed_label": CLASSES[sample.label],
                "kept": kept_sample.image,
                "kept_group": kept_group,
                "sha256": digest,
            })
        cleaned[group_name] = kept
        removed[group_name] = dropped
    return cleaned, {
        "policy": "first copy retained in priority order: brisc_train, brisc_test, pmram_raw",
        "removed_counts": {name: len(items) for name, items in removed.items()},
        "removed": removed,
    }


def class_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    c = Counter(CLASSES[s.label] for s in samples)
    return {k: c.get(k, 0) for k in CLASSES}


def figshare_overlap_audit(figshare: Sequence[FigshareSample], reference_groups: Dict[str, Sequence[Sample]],
                           threshold: int, output: Path) -> None:
    """Audit perceptual overlap without using Figshare for model selection."""
    reference: Dict[str, Dict[int, List[str]]] = {}
    for name, samples in reference_groups.items():
        buckets: Dict[int, List[str]] = defaultdict(list)
        for sample in samples:
            buckets[dhash(sample.image)].append(sample.image)
        reference[name] = dict(buckets)
    matches: Dict[str, List[Dict[str, object]]] = {name: [] for name in reference}
    counts = Counter()
    patients = set()
    for sample in figshare:
        array, label, pid = read_figshare_mat(sample.mat_file, load_image=True)
        fh = decoded_dhash_array(array)
        counts[CLASSES[label]] += 1
        patients.add(pid)
        for group_name, buckets in reference.items():
            for rh, paths in buckets.items():
                distance = hamming(fh, rh)
                if distance <= threshold:
                    matches[group_name].append({
                        "distance": distance, "figshare": sample.mat_file,
                        "figshare_pid": pid, "figshare_class": CLASSES[label],
                        "reference": paths[:3],
                    })
                    break
    report = {
        "figshare_counts": {c: counts.get(c, 0) for c in CLASSES},
        "figshare_patients": len(patients),
        "method": "64-bit decoded-image dHash",
        "hamming_threshold": threshold,
        "candidate_match_counts": {k: len(v) for k, v in matches.items()},
        "candidate_matches": {k: v[:200] for k, v in matches.items()},
        "warning": "Perceptual matches are candidates for manual review; patient independence cannot be proven from image hashes alone.",
    }
    write_json(output, report)


def stratified_split(samples: Sequence[Sample], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    by_class = defaultdict(list)
    for i, s in enumerate(samples):
        by_class[s.label].append(i)
    train, val = [], []
    rng = random.Random(seed)
    for label in range(len(CLASSES)):
        ids = by_class[label]
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_fraction))
        val.extend(ids[:n_val])
        train.extend(ids[n_val:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def verify_data(args: argparse.Namespace, rank: int, distributed: bool) -> Tuple[Path, List[Sample], List[Sample], List[Sample]]:
    root = locate_brisc_root(args.brisc_root)
    train = collect_brisc(root, "train")
    test = collect_brisc(root, "test")
    pmram = collect_pmram_raw(args.pmram_raw_root) if args.pmram_raw_root else []
    counts_before = {"brisc_train": class_counts(train), "brisc_test": class_counts(test), "pmram_raw": class_counts(pmram)}
    dedup_report: Dict[str, object] = {"enabled": False}
    if args.deduplicate_exact:
        ordered_groups: Dict[str, Sequence[Sample]] = {"brisc_train": train, "brisc_test": test}
        if pmram:
            ordered_groups["pmram_raw"] = pmram
        cleaned, dedup_report = deduplicate_exact(ordered_groups)
        dedup_report["enabled"] = True
        train = cleaned["brisc_train"]
        test = cleaned["brisc_test"]
        if pmram:
            pmram = cleaned["pmram_raw"]
    gate_error = [None]
    if is_main(rank):
      try:
        print("[gate:data] verifying files, labels, masks, and duplicates", flush=True)
        report = {
            "brisc_root": str(root), "pmram_raw_root": str(args.pmram_raw_root) if args.pmram_raw_root else None,
            "counts_before_deduplication": counts_before,
            "counts": {"brisc_train": class_counts(train), "brisc_test": class_counts(test), "pmram_raw": class_counts(pmram)},
            "exact_deduplication": dedup_report,
            "readability": {
                "brisc_train": verify_readable(train, args.verify_all_images, args.seed),
                "brisc_test": verify_readable(test, args.verify_all_images, args.seed + 1),
                "pmram_raw": verify_readable(pmram, args.verify_all_images, args.seed + 2) if pmram else None,
            },
        }
        for name, counts in report["counts"].items():
            if name == "pmram_raw" and not pmram:
                continue
            missing = [c for c in CLASSES if counts[c] == 0]
            if missing:
                raise RuntimeError(f"{name} is missing classes: {missing}")
        dup_groups = {"brisc_train": train, "brisc_test": test}
        if pmram:
            dup_groups["pmram_raw"] = pmram
        report["duplicates"] = duplicate_report(dup_groups, args.near_duplicate_hamming)
        cross_exact = report["duplicates"]["cross_exact"]
        if cross_exact and not args.allow_cross_dataset_exact_duplicates:
            write_json(args.output_dir / "dataset_audit_failed.json", report)
            raise RuntimeError(f"Cross-split/dataset exact duplicates detected: {list(cross_exact)}")
        near = report["duplicates"]["cross_near_dhash"]
        if near and args.fail_on_near_duplicates:
            write_json(args.output_dir / "dataset_audit_failed.json", report)
            raise RuntimeError(f"Near-duplicate dHash matches detected: {list(near)}")
        write_json(args.output_dir / "dataset_audit.json", report)
        print(f"[gate:data] PASS: {args.output_dir / 'dataset_audit.json'}", flush=True)
      except Exception as e:
        gate_error[0] = f"{type(e).__name__}: {e}"
    if distributed:
        dist.broadcast_object_list(gate_error, src=0)
    if gate_error[0]:
        raise RuntimeError(f"Dataset verification gate failed: {gate_error[0]}")
    barrier(distributed)
    return root, train, test, pmram


def load_clean_data_for_grid(args: argparse.Namespace) -> Tuple[Path, List[Sample], List[Sample], List[Sample]]:
    """Fast grid-search loader: structural loading plus deterministic exact de-duplication."""
    root = locate_brisc_root(args.brisc_root)
    groups: Dict[str, Sequence[Sample]] = {
        "brisc_train": collect_brisc(root, "train"),
        "brisc_test": collect_brisc(root, "test"),
        "pmram_raw": collect_pmram_raw(args.pmram_raw_root),
    }
    cleaned, dedup = deduplicate_exact(groups)
    train, test, pmram = cleaned["brisc_train"], cleaned["brisc_test"], cleaned["pmram_raw"]
    counts = {"brisc_train": class_counts(train), "brisc_test": class_counts(test), "pmram_raw": class_counts(pmram)}
    for name, values in counts.items():
        missing = [c for c in CLASSES if values[c] == 0]
        if missing:
            raise RuntimeError(f"{name} is missing classes after de-duplication: {missing}")
    write_json(args.output_dir / "grid_data_summary.json", {
        "brisc_root": str(root), "pmram_raw_root": str(args.pmram_raw_root),
        "counts": counts, "exact_deduplication": dedup,
    })
    print(f"[grid:data] loaded clean datasets: train={len(train)} test={len(test)} external={len(pmram)}", flush=True)
    return root, train, test, pmram


def parse_grid_floats(text: str, name: str) -> List[float]:
    try:
        values = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name} grid: {text}") from exc
    if not values or not all(math.isfinite(x) for x in values):
        raise ValueError(f"Invalid {name} grid: {text}")
    return values


class PairedTransform:
    def __init__(self, image_size: int, resize_size: int, train: bool,
                 histogram_equalization: bool = True, foreground_crop: bool = True):
        self.image_size, self.resize_size, self.train = image_size, resize_size, train
        self.histogram_equalization = histogram_equalization
        self.foreground_crop = foreground_crop
        self.jitter = transforms.ColorJitter(brightness=0.12, contrast=0.18)

    @staticmethod
    def _foreground_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        a = np.asarray(image, dtype=np.uint8)
        if a.size == 0 or int(a.max()) < 8:
            return None
        fg = a > max(6, int(0.08 * float(a.max())))
        # Requiring several active pixels per row/column ignores isolated text
        # and scanner annotations better than a plain nonzero bounding box.
        rows = np.flatnonzero(fg.mean(axis=1) > 0.03)
        cols = np.flatnonzero(fg.mean(axis=0) > 0.03)
        if not len(rows) or not len(cols):
            return None
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        left, right = int(cols[0]), int(cols[-1]) + 1
        margin = max(2, int(0.04 * max(bottom - top, right - left)))
        return max(0, left - margin), max(0, top - margin), min(a.shape[1], right + margin), min(a.shape[0], bottom + margin)

    @staticmethod
    def _square_pad(image: Image.Image, fill: int = 0) -> Image.Image:
        w, h = image.size
        side = max(w, h)
        left, top = (side - w) // 2, (side - h) // 2
        return ImageOps.expand(image, (left, top, side - w - left, side - h - top), fill=fill)

    def __call__(self, image: Image.Image, mask: Optional[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        # MRI is intrinsically grayscale. Converting every source identically
        # suppresses dataset-specific colour encoding while preserving use of
        # ImageNet pretrained convolutional filters.
        image = image.convert("L")
        if mask is None:
            mask = Image.new("L", image.size, 0)
        else:
            mask = mask.convert("L")
        if self.foreground_crop:
            box = self._foreground_box(image)
            if box is not None:
                image, mask = image.crop(box), mask.crop(box)
        if self.histogram_equalization:
            image = ImageOps.equalize(image)
        image, mask = self._square_pad(image, 0), self._square_pad(mask, 0)
        image = Image.merge("RGB", (image, image, image))
        image = TF.resize(image, [self.image_size, self.image_size], InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, [self.image_size, self.image_size], InterpolationMode.NEAREST)
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-8.0, 8.0)
            scale = random.uniform(0.94, 1.06)
            tx = int(random.uniform(-0.03, 0.03) * self.image_size)
            ty = int(random.uniform(-0.03, 0.03) * self.image_size)
            image = TF.affine(image, angle, [tx, ty], scale, [0.0, 0.0], InterpolationMode.BILINEAR, fill=0)
            mask = TF.affine(mask, angle, [tx, ty], scale, [0.0, 0.0], InterpolationMode.NEAREST, fill=0)
            image = self.jitter(image)
        x = TF.to_tensor(image)
        x = TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        m = (TF.pil_to_tensor(mask).float() / 255.0 > 0.5).float()
        return x, m


class BrainDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: PairedTransform):
        self.samples, self.transform = list(samples), transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(s.image) as im:
            image = im.copy()
        mask = None
        if s.mask:
            with Image.open(s.mask) as m:
                mask = m.copy()
        x, m = self.transform(image, mask)
        return x, torch.tensor(s.label, dtype=torch.long), m, s.image


class FigshareDataset(Dataset):
    def __init__(self, samples: Sequence[FigshareSample], transform: PairedTransform):
        self.samples, self.transform = list(samples), transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        array, label, pid = read_figshare_mat(s.mat_file, load_image=True)
        image = Image.fromarray(figshare_uint8(array), mode="L")
        x, mask = self.transform(image, None)
        return x, torch.tensor(label, dtype=torch.long), mask, pid


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation without DistributedSampler's padding duplicates."""
    def __init__(self, dataset: Dataset, rank: int, world: int):
        self.n, self.rank, self.world = len(dataset), rank, world

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world))

    def __len__(self) -> int:
        return (self.n - self.rank + self.world - 1) // self.world


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = x.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = x.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (x * std + mean).clamp(0, 1)


def verify_preprocessing(args: argparse.Namespace, train_samples: Sequence[Sample], rank: int, distributed: bool) -> None:
    gate_error = [None]
    if is_main(rank):
      try:
        print("[gate:preprocess] checking aligned stochastic transforms", flush=True)
        ds = BrainDataset(train_samples, build_transform(args, train=True))
        ids = np.linspace(0, len(ds) - 1, min(16, len(ds)), dtype=int)
        panels, stats = [], []
        for i in ids:
            x, y, m, path = ds[int(i)]
            if x.shape != (3, args.image_size, args.image_size) or m.shape != (1, args.image_size, args.image_size):
                raise RuntimeError(f"Bad transformed shapes: x={x.shape}, mask={m.shape}")
            if not torch.isfinite(x).all() or not torch.isfinite(m).all():
                raise RuntimeError(f"Non-finite preprocessing output: {path}")
            if not set(torch.unique(m).tolist()).issubset({0.0, 1.0}):
                raise RuntimeError(f"Mask is not binary after preprocessing: {path}")
            rgb = denormalize(x)
            overlay = rgb * (1 - 0.35 * m) + torch.tensor([1.0, 0.0, 0.0])[:, None, None] * (0.35 * m)
            panels.extend([rgb, overlay.clamp(0, 1)])
            stats.append({"path": path, "class": CLASSES[int(y)], "image_min": float(x.min()), "image_max": float(x.max()), "mask_fraction": float(m.mean())})
        grid = make_grid(panels, nrow=4, padding=2)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, args.output_dir / "preprocessing_audit.png")
        write_json(args.output_dir / "preprocessing_audit.json", stats)
        print(f"[gate:preprocess] PASS: {args.output_dir / 'preprocessing_audit.png'}", flush=True)
      except Exception as e:
        gate_error[0] = f"{type(e).__name__}: {e}"
    if distributed:
        dist.broadcast_object_list(gate_error, src=0)
    if gate_error[0]:
        raise RuntimeError(f"Preprocessing verification gate failed: {gate_error[0]}")
    barrier(distributed)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False),
            nn.GroupNorm(min(8, cout), cout), nn.GELU(),
        )


class MorphologyFeatures(nn.Module):
    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.project = nn.Sequential(nn.Conv2d(channels * 4, out_channels, 1, bias=False), nn.GroupNorm(8, out_channels), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grads = []
        for k in (3, 5, 7):
            dilation = F.max_pool2d(x, k, stride=1, padding=k // 2)
            erosion = -F.max_pool2d(-x, k, stride=1, padding=k // 2)
            grads.append(dilation - erosion)
        return self.project(torch.cat([x] + grads, dim=1))


class MorphoToken(nn.Module):
    def __init__(self, image_size: int, dim: int, depth: int, heads: int, dropout: float,
                 prototype_counts: Sequence[int], temperature: float,
                 backbone_name: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if image_size % 32:
            raise ValueError("image-size must be divisible by 32")
        self.num_classes = len(prototype_counts)
        self.prototype_counts = tuple(prototype_counts)
        self.temperature = temperature
        if backbone_name != "resnet50":
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        try:
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet50(weights=weights)
        except AttributeError:  # older torchvision
            base = tv_models.resnet50(pretrained=pretrained)
        # Layer 3 preserves 14x14 texture/boundary information; layer 4 adds
        # 7x7 high-level semantic context. Keeping both avoids forcing small
        # lesions through a single coarse feature map.
        self.backbone = nn.ModuleDict({
            "stem": nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                  base.layer1, base.layer2),
            "layer3": base.layer3,
            "layer4": base.layer4,
        })
        self.reduce3 = nn.Sequential(nn.Conv2d(1024, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU())
        self.reduce4 = nn.Sequential(nn.Conv2d(2048, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU())
        self.morph3 = MorphologyFeatures(dim, dim)
        self.morph4 = MorphologyFeatures(dim, dim)
        self.fuse3 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU())
        self.fuse4 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU())
        self.scale_fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False), nn.GroupNorm(8, dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(8, dim), nn.GELU(),
        )
        side = image_size // 16
        self.side = side
        self.pos = nn.Parameter(torch.zeros(1, side * side, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        try:
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.cnn_classifier = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, self.num_classes))
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        total = sum(prototype_counts)
        self.prototypes = nn.Parameter(torch.randn(total, dim) * 0.02)
        proto_class = []
        for c, n in enumerate(prototype_counts):
            proto_class.extend([c] * n)
        self.register_buffer("prototype_class", torch.tensor(proto_class, dtype=torch.long), persistent=True)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        stage2 = self.backbone["stem"](x)
        raw3 = self.backbone["layer3"](stage2)
        raw4 = self.backbone["layer4"](raw3)
        local3, local4 = self.reduce3(raw3), self.reduce4(raw4)
        feat3 = self.fuse3(torch.cat([local3, self.morph3(local3)], dim=1))
        feat4 = self.fuse4(torch.cat([local4, self.morph4(local4)], dim=1))
        feat4_up = F.interpolate(feat4, size=feat3.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.scale_fuse(torch.cat([feat3, feat4_up], dim=1)) + 0.5 * (feat3 + feat4_up)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos
        tokens = self.norm(self.encoder(tokens))
        gate_logits = self.gate(tokens).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        tn = F.normalize(tokens, dim=-1)
        pn = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bnd,pd->bnp", tn, pn)
        # Evidence pools diagnostic tokens for every prototype; log-sum-exp pools prototypes per class.
        weights = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        proto_evidence = (weights.unsqueeze(-1) * sim).sum(dim=1) / self.temperature
        proto_logits = []
        for c in range(self.num_classes):
            values = proto_evidence[:, self.prototype_class == c]
            proto_logits.append(torch.logsumexp(values, dim=1) - math.log(values.shape[1]))
        proto_logits = torch.stack(proto_logits, dim=1)
        cnn_logits = self.cnn_classifier(tokens.mean(dim=1))
        alpha = torch.sigmoid(self.fusion_logit)
        logits = alpha * proto_logits + (1.0 - alpha) * cnn_logits
        return {"logits": logits, "proto_logits": proto_logits, "cnn_logits": cnn_logits,
                "fusion_alpha": alpha.expand(x.shape[0], 1),
                "gate": gate, "similarity": sim, "tokens": tokens,
                "gate_logits_map": gate_logits.view(-1, 1, self.side, self.side),
                "gate_map": gate.view(-1, 1, self.side, self.side)}


def morphotoken_loss(out: Dict[str, torch.Tensor], labels: torch.Tensor, masks: torch.Tensor,
                     model: MorphoToken, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    class_weights = out["logits"].new_tensor(args.class_weights)
    ce = F.cross_entropy(out["logits"], labels, weight=class_weights,
                         label_smoothing=args.label_smoothing)
    branch_loss = 0.5 * (
        F.cross_entropy(out["proto_logits"], labels, weight=class_weights,
                        label_smoothing=args.label_smoothing) +
        F.cross_entropy(out["cnn_logits"], labels, weight=class_weights,
                        label_smoothing=args.label_smoothing)
    )
    sim = out["similarity"]
    proto_class = model.prototype_class
    correct_mask = proto_class[None, :] == labels[:, None]
    wrong_mask = ~correct_mask
    gated_sim = sim + 0.2 * out["gate"].clamp_min(1e-6).log().unsqueeze(-1)
    correct = gated_sim.masked_fill(~correct_mask[:, None, :], -1e4).amax(dim=(1, 2))
    wrong = gated_sim.masked_fill(~wrong_mask[:, None, :], -1e4).amax(dim=(1, 2))
    proto_loss = (1.0 - correct).mean()
    sep_loss = F.relu(wrong - correct + args.separation_margin).mean()
    target = F.interpolate(masks, size=out["gate_map"].shape[-2:], mode="nearest")
    tumor = labels != CLASS_TO_IDX["no_tumor"]
    mask_loss = (F.binary_cross_entropy_with_logits(out["gate_logits_map"][tumor], target[tumor])
                 if tumor.any() else ce.new_zeros(()))
    pn = F.normalize(model.prototypes, dim=-1)
    div_terms = []
    for c in range(len(CLASSES)):
        pc = pn[proto_class == c]
        gram = pc @ pc.T
        off = gram[~torch.eye(len(pc), dtype=torch.bool, device=gram.device)]
        if off.numel():
            div_terms.append(F.relu(off - 0.2).mean())
    div_loss = torch.stack(div_terms).mean() if div_terms else ce.new_zeros(())
    total = (ce + args.lambda_branch * branch_loss + args.lambda_mask * mask_loss +
             args.lambda_proto * proto_loss + args.lambda_sep * sep_loss + args.lambda_div * div_loss)
    vals = {"total": total.item(), "ce": ce.item(), "branch": branch_loss.item(), "mask": mask_loss.item(),
            "proto": proto_loss.item(), "sep": sep_loss.item(), "div": div_loss.item()}
    return total, vals


def unwrap(model: nn.Module) -> MorphoToken:
    return model.module if isinstance(model, DDP) else model


def amp_context(args: argparse.Namespace, device: torch.device):
    if device.type != "cuda" or args.amp == "none":
        return nullcontext()
    dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def build_transform(args: argparse.Namespace, train: bool) -> PairedTransform:
    return PairedTransform(args.image_size, args.resize_size, train,
                           args.histogram_equalization, args.foreground_crop)


def build_model(args: argparse.Namespace, pretrained: bool) -> MorphoToken:
    return MorphoToken(args.image_size, args.embed_dim, args.transformer_depth, args.heads,
                       args.dropout, args.prototypes, args.temperature,
                       args.backbone, pretrained)


def verify_model(args: argparse.Namespace, device: torch.device, rank: int, distributed: bool) -> None:
    print(f"[gate:model][rank {rank}] running forward/backward smoke test", flush=True)
    model = build_model(args, args.pretrained).to(device)
    b = 2
    x = torch.randn(b, 3, args.image_size, args.image_size, device=device)
    y = torch.tensor([0, 3], device=device)
    m = torch.zeros(b, 1, args.image_size, args.image_size, device=device)
    m[0, :, 48:160, 62:170] = 1
    with amp_context(args, device):
        out = model(x)
        loss, vals = morphotoken_loss(out, y, m, model, args)
    loss.backward()
    if out["logits"].shape != (b, len(CLASSES)) or out["gate_map"].shape[-2:] != (args.image_size // 16,) * 2:
        raise RuntimeError(f"Unexpected model outputs: {[(k, tuple(v.shape)) for k, v in out.items()]}")
    if not math.isfinite(float(loss.detach())) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("Non-finite loss/gradient in model smoke test")
    n_params = sum(p.numel() for p in model.parameters())
    if is_main(rank):
        write_json(args.output_dir / "model_audit.json", {"parameters": n_params, "output_shapes": {k: list(v.shape) for k, v in out.items()}, "losses": vals})
    del model, x, y, m, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    barrier(distributed)
    if is_main(rank):
        print(f"[gate:model] PASS: {args.output_dir / 'model_audit.json'}", flush=True)


def make_loader(dataset: Dataset, batch_size: int, workers: int, train: bool,
                distributed: bool, rank: int, world: int) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    if distributed and train:
        sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    elif distributed:
        sampler = DistributedEvalSampler(dataset, rank, world)
    else:
        sampler = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler,
                        num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
                        drop_last=train, prefetch_factor=2 if workers > 0 else None)
    return loader, sampler


def confusion_metrics(cm: torch.Tensor) -> Dict[str, object]:
    cm = cm.double()
    support = cm.sum(1)
    tp = cm.diag()
    precision = tp / cm.sum(0).clamp_min(1)
    recall = tp / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    acc = tp.sum() / cm.sum().clamp_min(1)
    return {
        "accuracy": float(acc), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
        "per_class": {CLASSES[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(len(CLASSES))},
        "confusion_matrix": cm.long().tolist(),
    }


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    y_true = y_true.astype(bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace,
             distributed: bool) -> Dict[str, object]:
    model.eval()
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long, device=device)
    nll_sum = torch.zeros((), device=device); brier_sum = torch.zeros((), device=device); n = torch.zeros((), device=device)
    ece_count = torch.zeros(15, device=device); ece_conf = torch.zeros(15, device=device); ece_correct = torch.zeros(15, device=device)
    local_labels, local_probs = [], []
    for x, y, mask, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with amp_context(args, device):
            logits = model(x)["logits"]
            if args.tta:
                logits = 0.5 * (logits + model(torch.flip(x, dims=(-1,)))["logits"])
        probs = logits.float().softmax(1)
        local_labels.append(y.detach().cpu())
        local_probs.append(probs.detach().cpu())
        pred = probs.argmax(1)
        idx = y * len(CLASSES) + pred
        cm += torch.bincount(idx, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))
        nll_sum += F.nll_loss(probs.clamp_min(1e-8).log(), y, reduction="sum")
        onehot = F.one_hot(y, len(CLASSES)).float()
        brier_sum += ((probs - onehot) ** 2).sum(1).sum()
        conf, _ = probs.max(1)
        bins = torch.clamp((conf * 15).long(), max=14)
        ece_count += torch.bincount(bins, minlength=15)
        ece_conf.scatter_add_(0, bins, conf)
        ece_correct.scatter_add_(0, bins, pred.eq(y).float())
        n += y.numel()
    if distributed:
        for t in (cm, nll_sum, brier_sum, ece_count, ece_conf, ece_correct, n):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    metrics = confusion_metrics(cm.cpu())
    metrics["nll"] = float((nll_sum / n.clamp_min(1)).cpu())
    metrics["brier"] = float((brier_sum / n.clamp_min(1)).cpu())
    nonzero = ece_count > 0
    avg_conf = ece_conf[nonzero] / ece_count[nonzero]
    avg_acc = ece_correct[nonzero] / ece_count[nonzero]
    metrics["ece_15bin"] = float(((ece_count[nonzero] / n.clamp_min(1)) * (avg_conf - avg_acc).abs()).sum().cpu())
    packed = (torch.cat(local_labels).numpy(), torch.cat(local_probs).numpy())
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, packed)
        ys = np.concatenate([g[0] for g in gathered])
        ps = np.concatenate([g[1] for g in gathered])
    else:
        ys, ps = packed
    aucs = {CLASSES[c]: binary_auc(ys == c, ps[:, c]) for c in range(len(CLASSES))}
    metrics["auroc_ovr"] = aucs
    metrics["macro_auroc"] = float(np.nanmean(list(aucs.values())))
    return metrics


def three_true_four_pred_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    """Metrics for three tumor ground-truth classes with all four model outputs retained."""
    cm = np.zeros((3, 4), dtype=np.int64)
    for y, p in zip(labels, predictions):
        cm[int(y), int(p)] += 1
    per_class, f1s = {}, []
    for c in range(3):
        tp = int(cm[c, c])
        support = int(cm[c].sum())
        predicted = int(cm[:, c].sum())
        precision = tp / max(1, predicted)
        recall = tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1s.append(f1)
        per_class[CLASSES[c]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1_three_true_classes": float(np.mean(f1s)),
        "false_no_tumor_rate": float(np.mean(predictions == CLASS_TO_IDX["no_tumor"])),
        "per_class": per_class,
        "confusion_matrix_rows_true_tumors_cols_all_predictions": cm.tolist(),
        "prediction_columns": list(CLASSES),
    }


def conditional_three_class_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    pred = probabilities[:, :3].argmax(1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for y, p in zip(labels, pred):
        cm[int(y), int(p)] += 1
    per_class, recalls, f1s = {}, [], []
    for c, name in enumerate(CLASSES[:3]):
        tp = int(cm[c, c]); support = int(cm[c].sum()); predicted = int(cm[:, c].sum())
        precision = tp / max(1, predicted); recall = tp / max(1, support)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        recalls.append(recall); f1s.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"accuracy": float(np.mean(labels == pred)), "macro_f1": float(np.mean(f1s)),
            "balanced_accuracy": float(np.mean(recalls)), "per_class": per_class,
            "confusion_matrix": cm.tolist(), "classes": list(CLASSES[:3])}


@torch.no_grad()
def evaluate_figshare(model: nn.Module, loader: DataLoader, device: torch.device,
                      args: argparse.Namespace, distributed: bool) -> Dict[str, object]:
    model.eval()
    local: List[Tuple[int, np.ndarray, str]] = []
    for x, y, _, pids in loader:
        x = x.to(device, non_blocking=True)
        with amp_context(args, device):
            logits = model(x)["logits"]
            if args.tta:
                logits = 0.5 * (logits + model(torch.flip(x, dims=(-1,)))["logits"])
        probs = logits.float().softmax(1).cpu().numpy()
        local.extend((int(label), prob, str(pid)) for label, prob, pid in zip(y.tolist(), probs, pids))
    if distributed:
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local)
        records = [item for part in gathered for item in part]
    else:
        records = local
    labels = np.asarray([r[0] for r in records], dtype=np.int64)
    probs = np.stack([r[1] for r in records])
    pids = [r[2] for r in records]
    unrestricted = three_true_four_pred_metrics(labels, probs.argmax(1))
    conditional = conditional_three_class_metrics(labels, probs)
    by_patient: Dict[str, List[int]] = defaultdict(list)
    for index, pid in enumerate(pids):
        by_patient[pid].append(index)
    patient_labels, patient_probs = [], []
    for pid, indices in sorted(by_patient.items()):
        unique = set(labels[indices].tolist())
        if len(unique) != 1:
            raise RuntimeError(f"Figshare PID {pid} has conflicting labels: {unique}")
        patient_labels.append(next(iter(unique)))
        patient_probs.append(probs[indices].mean(axis=0))
    py = np.asarray(patient_labels, dtype=np.int64)
    pp = np.stack(patient_probs)
    return {
        "protocol": "frozen BRISC model; Figshare never used for training, tuning, or checkpoint selection",
        "slices": len(records), "patients": len(by_patient),
        "slice_level": {
            "unrestricted_four_output": unrestricted,
            "conditional_three_tumor": conditional,
        },
        "patient_level_mean_probability": {
            "unrestricted_four_output": three_true_four_pred_metrics(py, pp.argmax(1)),
            "conditional_three_tumor": conditional_three_class_metrics(py, pp),
        },
    }


def cosine_schedule(optimizer, epoch: int, step: int, steps: int, args: argparse.Namespace) -> float:
    progress_epoch = epoch + step / max(1, steps)
    if progress_epoch < args.warmup_epochs:
        scale = max(1e-3, progress_epoch / max(1, args.warmup_epochs))
    else:
        t = (progress_epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        scale = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    lr = args.lr * scale
    for g in optimizer.param_groups:
        g["lr"] = lr * float(g.get("lr_scale", 1.0))
    return lr


def make_grad_scaler(device: torch.device, args: argparse.Namespace):
    enabled = device.type == "cuda" and args.amp == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class ModelEMA:
    """Exponential moving average of parameters and floating-point buffers."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup: Optional[Dict[str, torch.Tensor]] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # Warm-up avoids an excessive bias toward the randomly initialized
        # model during the first optimization steps.
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        current = model.state_dict()
        for name, value in current.items():
            if torch.is_floating_point(value):
                self.shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                self.shadow[name].copy_(value)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        if self.backup is not None:
            raise RuntimeError("EMA weights are already applied")
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            raise RuntimeError("EMA restore requested before apply")
        model.load_state_dict(self.backup, strict=True)
        self.backup = None


def save_checkpoint(path: Path, model: nn.Module, optimizer, scaler, epoch: int, best: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "epoch": epoch, "best_macro_f1": best, "classes": CLASSES, "args": vars(args)}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp); tmp.replace(path)


def load_model_checkpoint(path: Path, model: nn.Module, device: torch.device, optimizer=None, scaler=None) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        ckpt = torch.load(path, map_location=device)
    unwrap(model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", -1)) + 1, float(ckpt.get("best_macro_f1", -1))


class ProbabilityEnsemble(nn.Module):
    """Frozen ensemble returning log mean probabilities as compatible logits."""
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        if not models:
            raise ValueError("ProbabilityEnsemble requires at least one model")
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = [model(x)["logits"].float().softmax(dim=1) for model in self.models]
        mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
        return {"logits": mean_probability.clamp_min(1e-8).log()}


def build_external_model(args: argparse.Namespace, checkpoint: Path,
                         device: torch.device) -> Tuple[nn.Module, Tuple[Path, ...]]:
    paths = args.ensemble_checkpoints or (checkpoint,)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoint not found: {missing[0]}")
    models = []
    for path in paths:
        model = build_model(args, pretrained=False).to(device)
        load_model_checkpoint(path, model, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
    ensemble = ProbabilityEnsemble(models).to(device) if len(models) > 1 else models[0]
    return ensemble, paths


def train(args: argparse.Namespace, train_samples: Sequence[Sample], test_samples: Sequence[Sample],
          device: torch.device, distributed: bool, rank: int, local_rank: int, world: int,
          finalize: bool = True) -> Path:
    train_ids, val_ids = stratified_split(train_samples, args.val_fraction, args.seed)
    base_train = BrainDataset(train_samples, build_transform(args, train=True))
    base_eval = BrainDataset(train_samples, build_transform(args, train=False))
    train_ds, val_ds = Subset(base_train, train_ids), Subset(base_eval, val_ids)
    train_eval_ds = Subset(base_eval, train_ids)
    test_ds = BrainDataset(test_samples, build_transform(args, train=False))
    train_loader, train_sampler = make_loader(train_ds, args.batch_size, args.workers, True, distributed, rank, world)
    train_eval_loader, _ = make_loader(train_eval_ds, args.eval_batch_size, args.workers, False, distributed, rank, world)
    val_loader, _ = make_loader(val_ds, args.eval_batch_size, args.workers, False, distributed, rank, world)
    test_loader, _ = make_loader(test_ds, args.eval_batch_size, args.workers, False, distributed, rank, world)
    model = build_model(args, args.pretrained).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False, broadcast_buffers=False)
    core = unwrap(model)
    backbone_params = list(core.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in core.parameters() if id(p) not in backbone_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr_scale": args.backbone_lr_mult},
        {"params": head_params, "lr_scale": 1.0},
    ], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = make_grad_scaler(device, args)
    start, best, bad_epochs = 0, -1.0, 0
    if args.resume:
        start, best = load_model_checkpoint(args.resume, model, device, optimizer, scaler)
    ema = ModelEMA(core, args.ema_decay)
    if is_main(rank):
        write_json(args.output_dir / "split_audit.json", {
            "train": class_counts([train_samples[i] for i in train_ids]),
            "validation": class_counts([train_samples[i] for i in val_ids]),
            "internal_test": class_counts(test_samples), "world_size": world, "per_gpu_batch": args.batch_size,
            "effective_batch": args.batch_size * world, "amp": args.amp,
        })
    best_path = args.output_dir / "best.pt"
    history_path = args.output_dir / "history.jsonl"
    if is_main(rank) and start == 0:
        history_path.write_text("", encoding="utf-8")
    for epoch in range(start, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        backbone_trainable = epoch >= args.freeze_backbone_epochs
        for p in unwrap(model).backbone.parameters():
            p.requires_grad_(backbone_trainable)
        model.train()
        if not backbone_trainable:
            unwrap(model).backbone.eval()
        sums = defaultdict(float); count = 0; tic = time.time()
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y, masks, _) in enumerate(train_loader):
            lr = cosine_schedule(optimizer, epoch, step, len(train_loader), args)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
            with amp_context(args, device):
                out = model(x)
                loss, parts = morphotoken_loss(out, y, masks, unwrap(model), args)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            ema.update(core)
            for k, v in parts.items(): sums[k] += v * y.numel()
            count += y.numel()
        loss_names = ("total", "ce", "branch", "mask", "proto", "sep", "div")
        stats = torch.tensor([count] + [sums[k] for k in loss_names], device=device, dtype=torch.float64)
        if distributed: dist.all_reduce(stats)
        ema.apply(core)
        val_metrics = evaluate(model, val_loader, device, args, distributed)
        row = {"epoch": epoch + 1, "lr": lr, "seconds": time.time() - tic,
               "backbone_trainable": backbone_trainable,
               "validation_weights": "ema", "ema_decay": args.ema_decay,
               "train": {k: float(stats[i + 1] / stats[0].clamp_min(1)) for i, k in enumerate(loss_names)},
               "validation": val_metrics}
        improved = val_metrics["macro_f1"] > best
        if improved: best, bad_epochs = val_metrics["macro_f1"], 0
        else: bad_epochs += 1
        if is_main(rank):
            with open(history_path, "a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
            print(f"epoch={epoch+1:03d} loss={row['train']['total']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} best={best:.4f}", flush=True)
            if improved: save_checkpoint(best_path, model, optimizer, scaler, epoch, best, args)
        ema.restore(core)
        if is_main(rank):
            save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler, epoch, best, args)
        stop = torch.tensor(int(bad_epochs >= args.early_stop), device=device)
        if distributed: dist.broadcast(stop, 0)
        if stop.item(): break
    barrier(distributed)
    if not finalize:
        return best_path
    load_model_checkpoint(best_path, model, device)
    train_metrics = evaluate(model, train_eval_loader, device, args, distributed)
    validation_metrics = evaluate(model, val_loader, device, args, distributed)
    internal = evaluate(model, test_loader, device, args, distributed)
    if is_main(rank):
        write_json(args.output_dir / "brisc_train_metrics.json", train_metrics)
        write_json(args.output_dir / "brisc_validation_metrics.json", validation_metrics)
        write_json(args.output_dir / "brisc_internal_test_metrics.json", internal)
        write_json(args.output_dir / "brisc_final_metrics.json", {
            "train": train_metrics, "validation": validation_metrics, "internal_test": internal,
        })
        print(f"[final:BRISC] train={train_metrics['macro_f1']:.4f} "
              f"validation={validation_metrics['macro_f1']:.4f} internal_test={internal['macro_f1']:.4f}", flush=True)
    barrier(distributed)
    return best_path


def external_validate(args: argparse.Namespace, pmram_samples: Sequence[Sample], checkpoint: Path,
                      device: torch.device, distributed: bool, rank: int, local_rank: int, world: int) -> Dict[str, object]:
    ds = BrainDataset(pmram_samples, build_transform(args, train=False))
    loader, _ = make_loader(ds, args.eval_batch_size, args.workers, False, distributed, rank, world)
    # Checkpoint loading replaces every learned weight; no download is needed.
    model, checkpoint_paths = build_external_model(args, checkpoint, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    metrics = evaluate(model, loader, device, args, distributed)
    if is_main(rank):
        metrics["checkpoint_ensemble"] = [str(path) for path in checkpoint_paths]
        write_json(args.output_dir / "pmram_external_metrics.json", metrics)
        print(f"[external] frozen PMRAM macro-F1={metrics['macro_f1']:.4f}", flush=True)
    barrier(distributed)
    return metrics


def external_validate_figshare(args: argparse.Namespace, samples: Sequence[FigshareSample], checkpoint: Path,
                               device: torch.device, distributed: bool, rank: int,
                               local_rank: int, world: int) -> Dict[str, object]:
    ds = FigshareDataset(samples, build_transform(args, train=False))
    loader, _ = make_loader(ds, args.eval_batch_size, args.workers, False, distributed, rank, world)
    model, checkpoint_paths = build_external_model(args, checkpoint, device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    metrics = evaluate_figshare(model, loader, device, args, distributed)
    if is_main(rank):
        metrics["checkpoint_ensemble"] = [str(path) for path in checkpoint_paths]
        write_json(args.output_dir / "figshare_external_metrics.json", metrics)
        unrestricted = metrics["slice_level"]["unrestricted_four_output"]
        conditional = metrics["slice_level"]["conditional_three_tumor"]
        print(f"[external:figshare] unrestricted_acc={unrestricted['accuracy']:.4f} "
              f"conditional_3class_acc={conditional['accuracy']:.4f}", flush=True)
    barrier(distributed)
    return metrics


def checkpoint_best_score(path: Path) -> float:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return float(checkpoint["best_macro_f1"])


def run_gridsearch(args: argparse.Namespace, train_samples: Sequence[Sample], test_samples: Sequence[Sample],
                   pmram_samples: Sequence[Sample], device: torch.device,
                   distributed: bool, rank: int, local_rank: int, world: int) -> None:
    if distributed:
        raise RuntimeError("Grid search must run as a single process; launch with python, not torchrun")
    spaces = {
        "lr": parse_grid_floats(args.grid_lr, "learning-rate"),
        "weight_decay": parse_grid_floats(args.grid_weight_decay, "weight-decay"),
        "label_smoothing": parse_grid_floats(args.grid_label_smoothing, "label-smoothing"),
        "ema_decay": parse_grid_floats(args.grid_ema_decay, "EMA-decay"),
    }
    if any(x <= 0 for x in spaces["lr"]) or any(x < 0 for x in spaces["weight_decay"]):
        raise ValueError("Grid learning rates must be positive and weight decay non-negative")
    if any(not 0 <= x < 1 for x in spaces["label_smoothing"]):
        raise ValueError("Grid label smoothing values must be in [0, 1)")
    if any(not 0 < x < 1 for x in spaces["ema_decay"]):
        raise ValueError("Grid EMA decay values must be in (0, 1)")
    combinations = list(product(spaces["lr"], spaces["weight_decay"],
                                spaces["label_smoothing"], spaces["ema_decay"]))
    print(f"[grid] {len(combinations)} validation-only trials", flush=True)
    results: List[Dict[str, object]] = []
    trials_root = args.output_dir / "grid_trials"
    for index, (lr, weight_decay, label_smoothing, ema_decay) in enumerate(combinations, 1):
        trial_args = copy.deepcopy(args)
        trial_args.lr = lr
        trial_args.weight_decay = weight_decay
        trial_args.label_smoothing = label_smoothing
        trial_args.ema_decay = ema_decay
        trial_args.epochs = args.grid_epochs
        trial_args.early_stop = args.grid_early_stop
        trial_args.resume = None
        trial_args.output_dir = trials_root / f"trial_{index:03d}"
        seed_everything(args.seed, rank)
        config = {"lr": lr, "weight_decay": weight_decay,
                  "label_smoothing": label_smoothing, "ema_decay": ema_decay}
        print(f"[grid {index:03d}/{len(combinations):03d}] {config}", flush=True)
        checkpoint = train(trial_args, train_samples, test_samples, device,
                           False, rank, local_rank, 1, finalize=False)
        score = checkpoint_best_score(checkpoint)
        result = {"trial": index, "parameters": config, "validation_macro_f1": score,
                  "trial_directory": str(trial_args.output_dir), "checkpoint_retained": False}
        results.append(result)
        write_json(args.output_dir / "grid_results.json", {
            "selection_dataset": "BRISC validation only", "trials": results,
        })
        print(f"[grid {index:03d}] validation_macro_f1={score:.6f}", flush=True)
        # Trial checkpoints contain large optimizer states and are unnecessary
        # after the validation score is recorded; histories remain for audit.
        checkpoint.unlink(missing_ok=True)
        (trial_args.output_dir / "last.pt").unlink(missing_ok=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    winner = max(results, key=lambda item: float(item["validation_macro_f1"]))
    best_parameters = dict(winner["parameters"])
    selection = {
        "selection_dataset": "BRISC validation only",
        "selection_metric": "macro_f1",
        "best_validation_macro_f1": winner["validation_macro_f1"],
        "best_trial": winner["trial"],
        "best_parameters": best_parameters,
        "grid_spaces": spaces,
        "number_of_trials": len(results),
        "trial_epochs": args.grid_epochs,
        "trial_early_stop": args.grid_early_stop,
    }
    write_json(args.output_dir / "best_parameters.json", selection)
    print(f"[grid] winner={best_parameters} validation_macro_f1={winner['validation_macro_f1']:.6f}", flush=True)

    # Retrain the winning configuration from scratch. Internal test and PMRAM
    # are evaluated only here, after parameter selection is finished.
    final_args = copy.deepcopy(args)
    for name, value in best_parameters.items():
        setattr(final_args, name, value)
    final_args.resume = None
    final_args.output_dir = args.output_dir
    seed_everything(args.seed, rank)
    print(f"[grid:final] retraining winner for up to {final_args.epochs} epochs", flush=True)
    best_checkpoint = train(final_args, train_samples, test_samples, device,
                            False, rank, local_rank, 1, finalize=True)
    external_metrics = external_validate(final_args, pmram_samples, best_checkpoint,
                                         device, False, rank, local_rank, 1)
    figshare_metrics = None
    if final_args.figshare_root is not None:
        figshare_samples = collect_figshare(final_args.figshare_root)
        figshare_overlap_audit(figshare_samples,
                               {"brisc_train": train_samples, "brisc_test": test_samples,
                                "pmram_raw": pmram_samples},
                               final_args.near_duplicate_hamming,
                               final_args.output_dir / "figshare_overlap_audit.json")
        figshare_metrics = external_validate_figshare(final_args, figshare_samples, best_checkpoint,
                                                      device, False, rank, local_rank, 1)
    final_metrics = json.loads((args.output_dir / "brisc_final_metrics.json").read_text(encoding="utf-8"))
    write_json(args.output_dir / "gridsearch_final_summary.json", {
        "selection": selection,
        "final_checkpoint": str(best_checkpoint),
        "brisc": final_metrics,
        "pmram_external": external_metrics,
        "figshare_external": figshare_metrics,
    })
    print(f"[grid:complete] best parameters: {args.output_dir / 'best_parameters.json'}", flush=True)
    print(f"[grid:complete] final summary: {args.output_dir / 'gridsearch_final_summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world, device = ddp_setup()
    seed_everything(args.seed, rank)
    if is_main(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "run_config.json", vars(args))
        if world > 1 and world != 2:
            print(f"WARNING: requested DDP world size is {world}, not 2", flush=True)
        print(f"device={device} world_size={world} amp={args.amp}", flush=True)
    barrier(distributed)
    try:
        if args.stage == "final":
            if distributed:
                raise RuntimeError("Final winner mode is configured for one GPU; launch with python, not torchrun")
            _, brisc_train, brisc_test, pmram = load_clean_data_for_grid(args)
            winner = {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "ema_decay": args.ema_decay,
                "architecture": "dual-scale ResNet layer3+layer4 morphology fusion",
                "selected_by": "fixed from single-scale BRISC grid for controlled comparison",
            }
            write_json(args.output_dir / "winner_parameters.json", winner)
            print(f"[final] fixed winner parameters: {winner}", flush=True)
            checkpoint = train(args, brisc_train, brisc_test, device,
                               False, rank, local_rank, 1, finalize=True)
            external_metrics = external_validate(args, pmram, checkpoint,
                                                 device, False, rank, local_rank, 1)
            figshare_metrics = None
            if args.figshare_root is not None:
                figshare_samples = collect_figshare(args.figshare_root)
                figshare_overlap_audit(figshare_samples,
                                       {"brisc_train": brisc_train, "brisc_test": brisc_test,
                                        "pmram_raw": pmram},
                                       args.near_duplicate_hamming,
                                       args.output_dir / "figshare_overlap_audit.json")
                figshare_metrics = external_validate_figshare(
                    args, figshare_samples, checkpoint, device, False, rank, local_rank, 1)
            brisc_metrics = json.loads(
                (args.output_dir / "brisc_final_metrics.json").read_text(encoding="utf-8"))
            write_json(args.output_dir / "final_complete_summary.json", {
                "winner_parameters": winner,
                "checkpoint": str(checkpoint),
                "brisc": brisc_metrics,
                "pmram_external": external_metrics,
                "figshare_external": figshare_metrics,
            })
            print(f"[final] COMPLETE: {args.output_dir / 'final_complete_summary.json'}", flush=True)
            return
        if args.stage == "gridsearch":
            _, brisc_train, brisc_test, pmram = load_clean_data_for_grid(args)
            run_gridsearch(args, brisc_train, brisc_test, pmram,
                           device, distributed, rank, local_rank, world)
            return
        root, brisc_train, brisc_test, pmram = verify_data(args, rank, distributed)
        if args.stage == "verify-data": return
        verify_preprocessing(args, brisc_train, rank, distributed)
        if args.stage == "verify-preprocess": return
        verify_model(args, device, rank, distributed)
        if args.stage == "verify-model": return
        checkpoint = args.checkpoint or args.output_dir / "best.pt"
        if args.stage in {"train", "all"}:
            checkpoint = train(args, brisc_train, brisc_test, device, distributed, rank, local_rank, world)
        if args.stage in {"external", "all"}:
            if not checkpoint.exists(): raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            external_validate(args, pmram, checkpoint, device, distributed, rank, local_rank, world)
            if args.figshare_root is not None:
                figshare_samples = collect_figshare(args.figshare_root)
                if is_main(rank):
                    figshare_overlap_audit(figshare_samples,
                                           {"brisc_train": brisc_train, "brisc_test": brisc_test,
                                            "pmram_raw": pmram},
                                           args.near_duplicate_hamming,
                                           args.output_dir / "figshare_overlap_audit.json")
                barrier(distributed)
                external_validate_figshare(args, figshare_samples, checkpoint, device,
                                           distributed, rank, local_rank, world)
        if is_main(rank): print("ALL REQUESTED STAGES PASSED", flush=True)
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
