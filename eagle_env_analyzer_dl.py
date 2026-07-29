#!/usr/bin/env python3
"""
eagle_env_analyzer_dl.py

Deep-learning semantic segmentation (SegFormer ADE20K) + habitat/environment metrics.

Outputs:
- <output_dir>/per_image.csv
- <output_dir>/per_year_summary.csv
- (optional) <output_dir>/overlays/*.png

Key feature:
- --roi_bottom_frac lets you analyze only the bottom fraction of the image
  (e.g., 0.65 = bottom 65%) to better match classical ROI-based analyses.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

# Torch/Transformers are required in your .venv_dl
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


# -----------------------------
# Utilities
# -----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(input_dir: str, recursive: bool) -> Iterable[Path]:
    p = Path(input_dir)
    if not p.exists():
        return []
    if recursive:
        for f in p.rglob("*"):
            if f.suffix.lower() in IMG_EXTS and f.is_file():
                yield f
    else:
        for f in p.glob("*"):
            if f.suffix.lower() in IMG_EXTS and f.is_file():
                yield f


def safe_relpath(path: Path) -> str:
    """Try to make paths stable for CSV (relative to CWD if possible)."""
    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def parse_exif_datetime_str(s: str) -> Optional[datetime]:
    """
    EXIF datetime typically: 'YYYY:MM:DD HH:MM:SS'
    """
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def exiftool_datetime(exiftool_path: str, img_path: Path) -> Tuple[Optional[datetime], str]:
    """
    Returns (datetime, source_string).
    Prefers DateTimeOriginal, then CreateDate, then ModifyDate.
    """
    exe = Path(exiftool_path)
    if not exe.exists():
        return None, "exif:none:exiftool_missing"

    # Use -s -s -s for raw values, one per line.
    cmd = [
        str(exe),
        "-s",
        "-s",
        "-s",
        "-DateTimeOriginal",
        "-CreateDate",
        "-ModifyDate",
        str(img_path),
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        # exiftool prints values in the same order as tags above (if present)
        tags = ["DateTimeOriginal", "CreateDate", "ModifyDate"]
        for tag, val in zip(tags, lines):
            dt = parse_exif_datetime_str(val)
            if dt is not None:
                return dt, f"exif:exiftool:{tag}"
        return None, "exif:none:no_valid_datetime"
    except Exception:
        return None, "exif:none:exiftool_error"


def filename_year_fallback(path: Path) -> Optional[int]:
    """
    Try to guess year from filename like:
    SONY_ILCE-7RM5_2023-02-17_....jpg  -> 2023
    """
    name = path.name
    # Find first occurrence of 20xx
    for i in range(len(name) - 3):
        chunk = name[i : i + 4]
        if chunk.isdigit():
            y = int(chunk)
            if 1990 <= y <= 2100:
                return y
    return None


def mtime_fallback(path: Path) -> Optional[datetime]:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts)
    except Exception:
        return None


def laplacian_variance(gray: np.ndarray) -> float:
    """
    A lightweight Laplacian variance blur metric without OpenCV.
    gray: float32 array in [0,1], shape (H,W)
    """
    # 2D Laplacian kernel
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    # Convolution (valid area)
    g = gray.astype(np.float32)
    # Pad by 1
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lap = (
        k[0, 0] * gp[:-2, :-2] + k[0, 1] * gp[:-2, 1:-1] + k[0, 2] * gp[:-2, 2:] +
        k[1, 0] * gp[1:-1, :-2] + k[1, 1] * gp[1:-1, 1:-1] + k[1, 2] * gp[1:-1, 2:] +
        k[2, 0] * gp[2:, :-2] + k[2, 1] * gp[2:, 1:-1] + k[2, 2] * gp[2:, 2:]
    )
    return float(np.var(lap))


def hsv_stats_for_mask(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """
    Compute water color stats using HSV within mask.
    rgb: uint8 (H,W,3)
    mask: bool (H,W)
    """
    if mask.sum() < 10:
        return {
            "water_hue_deg_mean": np.nan,
            "water_hue_circ_var": np.nan,
            "water_brightness_mean": np.nan,
            "water_brightness_std": np.nan,
            "water_saturation_mean": np.nan,
            "water_saturation_std": np.nan,
        }

    # Convert to HSV using PIL for correctness; then to numpy
    im = Image.fromarray(rgb, mode="RGB").convert("HSV")
    hsv = np.array(im, dtype=np.float32)  # H,S,V in [0,255]
    h = hsv[..., 0][mask] * (360.0 / 255.0)  # degrees
    s = hsv[..., 1][mask] / 255.0
    v = hsv[..., 2][mask] / 255.0

    # Circular mean/variance for hue
    radians = np.deg2rad(h)
    sinm = np.mean(np.sin(radians))
    cosm = np.mean(np.cos(radians))
    mean_ang = math.atan2(sinm, cosm)
    if mean_ang < 0:
        mean_ang += 2 * math.pi
    hue_mean_deg = float(np.rad2deg(mean_ang))

    R = math.sqrt(sinm * sinm + cosm * cosm)
    circ_var = float(1.0 - R)

    return {
        "water_hue_deg_mean": hue_mean_deg,
        "water_hue_circ_var": circ_var,
        "water_brightness_mean": float(np.mean(v)),
        "water_brightness_std": float(np.std(v)),
        "water_saturation_mean": float(np.mean(s)),
        "water_saturation_std": float(np.std(s)),
    }


# -----------------------------
# Label grouping for ADE20K
# -----------------------------

def build_label_groups(id2label: Dict[int, str]) -> Dict[str, List[int]]:
    """
    Create groups of label IDs for categories we care about.

    Note: ADE20K label names vary; we use keyword matching.
    This is a pragmatic grouping, not perfect taxonomy.
    """
    groups = {
        "sky": [],
        "water": [],
        "vegetation": [],
        "snow_ice": [],
        "ground": [],
        "built": [],
        "other": [],  # we’ll fill later as complement
    }

    sky_kw = ["sky"]
    water_kw = ["water", "river", "sea", "ocean", "lake", "pond", "stream", "waterfall"]
    veg_kw = ["tree", "grass", "plant", "vegetation", "leaf", "forest", "bush", "shrub", "field"]
    snow_kw = ["snow", "ice", "glacier"]
    ground_kw = ["ground", "dirt", "sand", "soil", "mud", "rock", "stone", "gravel", "road", "path", "trail", "mountain"]
    built_kw = ["building", "house", "wall", "bridge", "tower", "dock", "pier", "fence", "pole", "streetlight", "boat", "ship"]

    def match_any(label: str, kws: List[str]) -> bool:
        l = label.lower()
        return any(kw in l for kw in kws)

    all_ids = sorted(list(id2label.keys()))
    assigned = set()

    for i in all_ids:
        label = id2label[i]
        if match_any(label, sky_kw):
            groups["sky"].append(i); assigned.add(i); continue
        if match_any(label, water_kw):
            groups["water"].append(i); assigned.add(i); continue
        if match_any(label, veg_kw):
            groups["vegetation"].append(i); assigned.add(i); continue
        if match_any(label, snow_kw):
            groups["snow_ice"].append(i); assigned.add(i); continue
        if match_any(label, built_kw):
            groups["built"].append(i); assigned.add(i); continue
        if match_any(label, ground_kw):
            groups["ground"].append(i); assigned.add(i); continue

    # Everything else goes to "other"
    groups["other"] = [i for i in all_ids if i not in assigned]
    return groups


def frac_for_group(seg: np.ndarray, ids: List[int]) -> float:
    if len(ids) == 0:
        return 0.0
    m = np.isin(seg, np.array(ids, dtype=seg.dtype))
    return float(m.mean())


# -----------------------------
# Overlay rendering
# -----------------------------

def make_overlay(rgb: np.ndarray, seg: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Create a simple overlay by mapping class ids to a deterministic color palette.
    """
    h, w = seg.shape
    # deterministic pseudo-palette
    colors = np.zeros((seg.max() + 1, 3), dtype=np.uint8)
    for cid in range(colors.shape[0]):
        # simple hash-based color
        r = (cid * 37) % 255
        g = (cid * 91) % 255
        b = (cid * 53) % 255
        colors[cid] = (r, g, b)

    color_mask = colors[seg]
    out = (rgb.astype(np.float32) * (1 - alpha) + color_mask.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# -----------------------------
# Core processing
# -----------------------------

@dataclass
class ImageRecord:
    file: str
    year: int
    month: int
    day_of_year: int
    datetime: str
    datetime_source: str
    width: int
    height: int
    roi_bottom_frac: float

    sky_frac: float
    water_frac: float
    vegetation_frac: float
    snow_ice_frac: float
    ground_frac: float
    built_frac: float
    other_frac: float

    blur_lap_var: float

    water_hue_deg_mean: float
    water_hue_circ_var: float
    water_brightness_mean: float
    water_brightness_std: float
    water_saturation_mean: float
    water_saturation_std: float


def process_one(
    img_path: Path,
    processor: AutoImageProcessor,
    model: AutoModelForSemanticSegmentation,
    device: torch.device,
    groups: Dict[str, List[int]],
    roi_bottom_frac: float,
    exiftool_path: str,
    save_overlay: bool,
    overlay_path: Optional[Path],
) -> ImageRecord:
    # Load image
    img = Image.open(img_path).convert("RGB")
    w0, h0 = img.size

    # ROI crop: bottom fraction
    roi = float(roi_bottom_frac)
    if not (0 < roi <= 1.0):
        raise ValueError("--roi_bottom_frac must be in (0, 1].")

    if roi < 1.0:
        y0 = int(round(h0 * (1.0 - roi)))
        img_roi = img.crop((0, y0, w0, h0))
    else:
        img_roi = img

    # Datetime
    dt, dt_src = exiftool_datetime(exiftool_path, img_path)
    if dt is None:
        y = filename_year_fallback(img_path)
        if y is not None:
            # Use Jan 1 as placeholder
            dt = datetime(y, 1, 1, 0, 0, 0)
            dt_src = "filename:year"
        else:
            dt2 = mtime_fallback(img_path)
            if dt2 is not None:
                dt = dt2
                dt_src = "file:mtime"
            else:
                dt = datetime(1970, 1, 1, 0, 0, 0)
                dt_src = "fallback:epoch"

    year = dt.year
    month = dt.month
    day_of_year = int(dt.strftime("%j"))
    dt_iso = dt.isoformat(timespec="seconds")

    # Blur metric on ROI
    rgb_roi = np.array(img_roi, dtype=np.uint8)
    gray = (0.299 * rgb_roi[..., 0] + 0.587 * rgb_roi[..., 1] + 0.114 * rgb_roi[..., 2]) / 255.0
    blur = laplacian_variance(gray.astype(np.float32))

    # Segmentation inference
    inputs = processor(images=img_roi, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits  # [1, C, h', w']
        # upsample to ROI size
        target_h, target_w = rgb_roi.shape[0], rgb_roi.shape[1]
        logits_up = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
        seg = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)  # (H,W)

    # Fractions
    sky_frac = frac_for_group(seg, groups["sky"])
    water_frac = frac_for_group(seg, groups["water"])
    vegetation_frac = frac_for_group(seg, groups["vegetation"])
    snow_ice_frac = frac_for_group(seg, groups["snow_ice"])
    ground_frac = frac_for_group(seg, groups["ground"])
    built_frac = frac_for_group(seg, groups["built"])
    other_frac = frac_for_group(seg, groups["other"])

    # Water stats using water mask
    water_ids = groups["water"]
    water_mask = np.isin(seg, np.array(water_ids, dtype=np.int32)) if len(water_ids) else np.zeros_like(seg, dtype=bool)
    water_stats = hsv_stats_for_mask(rgb_roi, water_mask)

    # Save overlay
    if save_overlay and overlay_path is not None:
        ov = make_overlay(rgb_roi, seg, alpha=0.45)
        ov.save(overlay_path)

    return ImageRecord(
        file=safe_relpath(img_path),
        year=year,
        month=month,
        day_of_year=day_of_year,
        datetime=dt_iso,
        datetime_source=dt_src,
        width=w0,
        height=h0,
        roi_bottom_frac=roi,

        sky_frac=sky_frac,
        water_frac=water_frac,
        vegetation_frac=vegetation_frac,
        snow_ice_frac=snow_ice_frac,
        ground_frac=ground_frac,
        built_frac=built_frac,
        other_frac=other_frac,

        blur_lap_var=blur,

        water_hue_deg_mean=water_stats["water_hue_deg_mean"],
        water_hue_circ_var=water_stats["water_hue_circ_var"],
        water_brightness_mean=water_stats["water_brightness_mean"],
        water_brightness_std=water_stats["water_brightness_std"],
        water_saturation_mean=water_stats["water_saturation_mean"],
        water_saturation_std=water_stats["water_saturation_std"],
    )


def write_csv(path: Path, rows: List[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ImageRecord.__annotations__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def per_year_summary(rows: List[ImageRecord], out_path: Path) -> None:
    """
    Compute count/mean/std for numeric metrics by year.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy-friendly dict
    years = sorted(set(r.year for r in rows))
    # Numeric fields only (skip file/datetime/datetime_source)
    numeric_fields = [
        "sky_frac", "water_frac", "vegetation_frac", "snow_ice_frac", "ground_frac", "built_frac", "other_frac",
        "blur_lap_var",
        "water_hue_deg_mean", "water_hue_circ_var",
        "water_brightness_mean", "water_brightness_std",
        "water_saturation_mean", "water_saturation_std",
    ]

    # Write wide summary: for each field -> count/mean/std columns
    fieldnames = ["year"]
    for col in numeric_fields:
        fieldnames += [f"{col}_count", f"{col}_mean", f"{col}_std"]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for y in years:
            yr_rows = [r for r in rows if r.year == y]
            row_out = {"year": y}

            for col in numeric_fields:
                vals = np.array([getattr(r, col) for r in yr_rows], dtype=np.float64)
                # ignore NaNs
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    row_out[f"{col}_count"] = 0
                    row_out[f"{col}_mean"] = ""
                    row_out[f"{col}_std"] = ""
                else:
                    row_out[f"{col}_count"] = int(vals.size)
                    row_out[f"{col}_mean"] = float(np.mean(vals))
                    row_out[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size >= 2 else 0.0

            w.writerow(row_out)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder containing images (e.g., photos\\converted)")
    ap.add_argument("--output_dir", required=True, help="Where to write outputs (e.g., outputs_dl)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--max_images", type=int, default=0, help="Process at most N images (0 = no limit)")
    ap.add_argument("--save_overlays", action="store_true", help="Save overlay PNGs for qualitative validation")
    ap.add_argument("--overlay_max", type=int, default=60, help="Max overlays to save")
    ap.add_argument("--exiftool_path", default=r".\exiftool.exe", help="Path to exiftool.exe (default: .\\exiftool.exe)")
    ap.add_argument("--model_name", default="nvidia/segformer-b0-finetuned-ade-512-512", help="HF model name")
    ap.add_argument(
        "--roi_bottom_frac",
        type=float,
        default=1.0,
        help="Analyze only the bottom fraction of the image (e.g., 0.65 = bottom 65%%). 1.0 = full image.",
    )

    args = ap.parse_args()

    input_dir = args.input_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_dir = out_dir / "overlays"
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(iter_images(input_dir, args.recursive)))
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        print("No images found. Check --input_dir.")
        return

    print(f"DL: found {len(image_paths)} images in {input_dir}")
    print("DL: loading model (first run may download weights)...")

    device = torch.device("cpu")

    # Load processor/model
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    # Build label groups
    id2label = model.config.id2label if hasattr(model, "config") and hasattr(model.config, "id2label") else {}
    if not id2label:
        # fallback for safety
        id2label = {i: str(i) for i in range(int(model.config.num_labels))}
    groups = build_label_groups(id2label)

    print("DL: model loaded.")

    rows: List[ImageRecord] = []

    overlays_written = 0
    roi_suffix = ""
    if float(args.roi_bottom_frac) < 1.0:
        roi_suffix = f"_roi{int(round(args.roi_bottom_frac * 100)):02d}"

    total = len(image_paths)
    for i, p in enumerate(image_paths, start=1):
        try:
            overlay_path = None
            if args.save_overlays and overlays_written < int(args.overlay_max):
                overlay_name = f"{p.stem}{roi_suffix}.overlay.png"
                overlay_path = overlay_dir / overlay_name

            rec = process_one(
                img_path=p,
                processor=processor,
                model=model,
                device=device,
                groups=groups,
                roi_bottom_frac=float(args.roi_bottom_frac),
                exiftool_path=args.exiftool_path,
                save_overlay=args.save_overlays and (overlay_path is not None),
                overlay_path=overlay_path,
            )
            rows.append(rec)

            if args.save_overlays and overlay_path is not None:
                overlays_written += 1

            print(f"[{i}/{total}] OK: {p.name}")
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"[{i}/{total}] ERROR: {p.name} -> {e}")

    if not rows:
        print("No results produced.")
        return

    per_image_path = out_dir / "per_image.csv"
    per_year_path = out_dir / "per_year_summary.csv"

    write_csv(per_image_path, rows)
    per_year_summary(rows, per_year_path)

    print("Done. Wrote:")
    print(f"  {per_image_path}")
    print(f"  {per_year_path}")
    if args.save_overlays:
        print(f"Overlays saved: {overlays_written} in {overlay_dir}")


if __name__ == "__main__":
    main()
