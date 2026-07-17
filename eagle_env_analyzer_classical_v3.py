#!/usr/bin/env python3
"""
Classical environment analyzer (v3): more robust than naive color thresholds.
- Uses ROI cropping (bottom fraction) to reduce sky and focus on habitat.
- Adds explicit sky detection + snow/ice detection
- More tolerant vegetation (green + dry grass/brown) and water (blue/cyan + blue-dominance)
- Outputs fractions aligned with DL columns where feasible.

Outputs:
  <output_dir>/per_image.csv
  <output_dir>/per_year_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def iter_images(input_dir: str, recursive: bool) -> Iterable[Path]:
    p = Path(input_dir)
    if not p.exists():
        return []
    if recursive:
        for fp in p.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                yield fp
    else:
        for fp in p.glob("*"):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                yield fp


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def parse_year_from_filename(name: str) -> Optional[int]:
    # Look for a 4-digit year 2010-2035 in filename
    m = re.search(r"(20[1-3][0-9])", name)
    if m:
        return int(m.group(1))
    return None


def exif_datetime_via_exiftool(exiftool_path: str, image_path: Path) -> Tuple[Optional[datetime], str]:
    """
    Try to get DateTimeOriginal/CreateDate/ModifyDate via exiftool.
    Returns (datetime_or_none, source_string).
    """
    exiftool = Path(exiftool_path)
    if not exiftool.exists():
        return None, "none"

    try:
        cmd = [
            str(exiftool),
            "-s3",
            "-DateTimeOriginal",
            "-CreateDate",
            "-ModifyDate",
            str(image_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).splitlines()
        # exiftool returns values in order; may be blank lines
        candidates = [line.strip() for line in out if line.strip()]
        for val in candidates:
            # Format: YYYY:MM:DD HH:MM:SS
            try:
                dt = datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                return dt, "exif:exiftool"
            except Exception:
                pass
        return None, "exif:missing"
    except Exception:
        return None, "exif:error"


def file_mtime_datetime(image_path: Path) -> datetime:
    return datetime.fromtimestamp(image_path.stat().st_mtime)


def rgb_to_hsv_np(rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized RGB->HSV conversion for rgb01 array shape (H, W, 3) in [0,1].
    Returns (H in degrees 0..360, S in 0..1, V in 0..1).
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)

    # Avoid division by zero
    mask = delta > 1e-10

    # Where cmax == r
    idx = mask & (cmax == r)
    h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx])) % 360

    # Where cmax == g
    idx = mask & (cmax == g)
    h[idx] = 60 * (((b[idx] - r[idx]) / delta[idx]) + 2)

    # Where cmax == b
    idx = mask & (cmax == b)
    h[idx] = 60 * (((r[idx] - g[idx]) / delta[idx]) + 4)

    # Saturation
    s = np.zeros_like(cmax)
    nonzero = cmax > 1e-10
    s[nonzero] = delta[nonzero] / cmax[nonzero]

    # Value
    v = cmax

    return h, s, v


def laplacian_variance(gray01: np.ndarray) -> float:
    """
    Simple Laplacian variance as blur metric.
    gray01 shape (H,W) in [0,1]
    """
    # 3x3 Laplacian kernel
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    # Convolve via padding + shifts (fast enough for modest images)
    g = gray01.astype(np.float32)
    gpad = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    lap = (
        k[0, 0]*gpad[:-2, :-2] + k[0, 1]*gpad[:-2, 1:-1] + k[0, 2]*gpad[:-2, 2:] +
        k[1, 0]*gpad[1:-1, :-2] + k[1, 1]*gpad[1:-1, 1:-1] + k[1, 2]*gpad[1:-1, 2:] +
        k[2, 0]*gpad[2:, :-2] + k[2, 1]*gpad[2:, 1:-1] + k[2, 2]*gpad[2:, 2:]
    )
    return float(np.var(lap))


@dataclass
class Thresholds:
    # Sky
    sky_sat_max: float = 0.25
    sky_val_min: float = 0.75

    # Snow/ice (white-ish)
    snow_sat_max: float = 0.18
    snow_val_min: float = 0.80

    # Vegetation: green band + dry grass/brown band
    veg_green_h_min: float = 70.0
    veg_green_h_max: float = 170.0
    veg_dry_h_min: float = 20.0
    veg_dry_h_max: float = 70.0
    veg_sat_min: float = 0.16
    veg_val_min: float = 0.15

    # Water: blue/cyan hues and/or blue-dominance
    water_h_min: float = 160.0
    water_h_max: float = 260.0
    water_sat_min: float = 0.10
    water_val_min: float = 0.10
    blue_dom_r_margin: float = 0.05
    blue_dom_g_margin: float = 0.02
    blue_dom_sat_min: float = 0.05


def classify_pixels(rgb01: np.ndarray, thr: Thresholds) -> Dict[str, np.ndarray]:
    """
    Returns boolean masks for classes on the provided rgb01 image.
    Order matters: sky/snow first, then vegetation/water, then ground.
    """
    h, s, v = rgb_to_hsv_np(rgb01)

    # Basic masks
    snow = (s <= thr.snow_sat_max) & (v >= thr.snow_val_min)

    sky = (s <= thr.sky_sat_max) & (v >= thr.sky_val_min)
    # Make sky a little safer: exclude very white snow-like pixels already flagged
    sky = sky & (~snow)

    veg_green = (h >= thr.veg_green_h_min) & (h <= thr.veg_green_h_max) & (s >= thr.veg_sat_min) & (v >= thr.veg_val_min)
    veg_dry = (h >= thr.veg_dry_h_min) & (h <= thr.veg_dry_h_max) & (s >= thr.veg_sat_min) & (v >= thr.veg_val_min)
    vegetation = (veg_green | veg_dry) & (~sky) & (~snow)

    # Water hue + blue dominance
    water_hue = (h >= thr.water_h_min) & (h <= thr.water_h_max) & (s >= thr.water_sat_min) & (v >= thr.water_val_min)
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    water_blue_dom = (b >= (r + thr.blue_dom_r_margin)) & (b >= (g + thr.blue_dom_g_margin)) & (s >= thr.blue_dom_sat_min)
    water = (water_hue | water_blue_dom) & (~sky) & (~snow)

    # Resolve overlap: if vegetation and water both true, prefer water if hue is in water band
    overlap = vegetation & water
    if np.any(overlap):
        vegetation = vegetation & (~overlap)
        # keep water as is

    # Ground = everything else not labeled
    other = ~(sky | snow | vegetation | water)

    return {
        "sky": sky,
        "snow_ice": snow,
        "vegetation": vegetation,
        "water": water,
        "ground": other,
    }


def frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def load_image_rgb01(path: Path, max_side: int = 1600) -> np.ndarray:
    """
    Load image and downscale for speed (keeps overall proportions).
    Returns float RGB in [0,1], shape (H,W,3).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = 1.0
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def crop_roi(arr: np.ndarray, roi_bottom_frac: float) -> np.ndarray:
    """
    Keep bottom fraction of image (e.g., 0.65 keeps bottom 65%).
    """
    H = arr.shape[0]
    start = int((1.0 - roi_bottom_frac) * H)
    start = max(0, min(H - 1, start))
    return arr[start:, :, :]


def summarize_by_year(df: pd.DataFrame, out_path: Path):
    metric_cols = [
        "sky_frac", "water_frac", "vegetation_frac", "snow_ice_frac", "ground_frac",
        "brightness_mean", "saturation_mean", "blur_lap_var",
    ]
    g = df.groupby("year")
    rows = []
    years = sorted(df["year"].dropna().unique().tolist())
    for y in years:
        row = {"year": int(y), "n_images": int((df["year"] == y).sum())}
        sub = df[df["year"] == y]
        for c in metric_cols:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                row[c + "_mean"] = float(vals.mean())
                row[c + "_std"] = float(vals.std(ddof=1)) if len(vals.dropna()) >= 2 else float("nan")
        rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--roi_bottom_frac", type=float, default=0.65, help="Keep bottom fraction of image (default 0.65)")
    ap.add_argument("--max_side", type=int, default=1600, help="Downscale max side for speed (default 1600)")
    ap.add_argument("--exiftool_path", default=r".\exiftool.exe", help="Path to exiftool.exe")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    thr = Thresholds()

    image_paths = sorted(list(iter_images(args.input_dir, args.recursive)))
    if not image_paths:
        print("No images found. Check --input_dir.")
        return

    rows = []
    for i, fp in enumerate(image_paths, start=1):
        try:
            arr = load_image_rgb01(fp, max_side=args.max_side)
            H, W = arr.shape[0], arr.shape[1]

            # Blur metric (on full image)
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
            blur = laplacian_variance(gray)

            # ROI crop for habitat fractions
            roi = crop_roi(arr, args.roi_bottom_frac)

            masks = classify_pixels(roi, thr)
            sky_frac = frac(masks["sky"])
            snow_ice_frac = frac(masks["snow_ice"])
            vegetation_frac = frac(masks["vegetation"])
            water_frac = frac(masks["water"])
            ground_frac = frac(masks["ground"])

            # Simple brightness/saturation summaries on ROI
            _, s, v = rgb_to_hsv_np(roi)
            brightness_mean = float(v.mean())
            saturation_mean = float(s.mean())

            # Datetime logic
            dt, dt_src = exif_datetime_via_exiftool(args.exiftool_path, fp)
            if dt is None:
                yr = parse_year_from_filename(fp.name)
                if yr is not None:
                    dt = datetime(yr, 1, 1)
                    dt_src = "filename:year"
                else:
                    dt = file_mtime_datetime(fp)
                    dt_src = "file:mtime"
            year = dt.year

            rows.append({
                "file": str(fp).replace("\\", "/"),
                "year": int(year),
                "datetime": dt.isoformat(),
                "datetime_source": dt_src,
                "width": int(W),
                "height": int(H),
                "sky_frac": sky_frac,
                "water_frac": water_frac,
                "vegetation_frac": vegetation_frac,
                "snow_ice_frac": snow_ice_frac,
                # Alias for compatibility if needed
                "snow_frac": snow_ice_frac,
                "ground_frac": ground_frac,
                "built_frac": 0.0,
                "other_frac": 0.0,
                "brightness_mean": brightness_mean,
                "saturation_mean": saturation_mean,
                "blur_lap_var": blur,
            })
            print(f"[{i}/{len(image_paths)}] OK: {fp.name}")
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] FAIL: {fp.name} ({e})")

    df = pd.DataFrame(rows)
    per_image_path = out_dir / "per_image.csv"
    df.to_csv(per_image_path, index=False)

    per_year_path = out_dir / "per_year_summary.csv"
    summarize_by_year(df, per_year_path)

    print("\nDone. Wrote:")
    print(f"  {per_image_path}")
    print(f"  {per_year_path}")


if __name__ == "__main__":
    main()
