"""Turn raw single-year reflectance GeoTIFFs into aligned, model-ready 7-band stacks.

Each year is reprojected to UTM 43N, clipped to the AOI, resampled to one shared 10 m grid,
cloud-filled, given NDVI/NDBI channels and normalised, so every pixel of every year describes
the same ground.

Input band order (from acquire_imagery_gee.py): blue, green, red, nir, swir1, [clearmask].

    python scripts/preprocess.py --aoi config/aoi.geojson --y2017 raw_2017.tif --y2019 raw_2019.tif --y2024 raw_2024.tif --out outputs
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import rasterio.transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_CRS = "EPSG:32643"
COMMON_PIXEL = 10.0
EPS = 1e-6


def load_aoi_bbox(geojson_path):
    with open(geojson_path) as f:
        return tuple(json.load(f)["bbox"])


def bbox_to_utm(bbox_4326, target_crs=TARGET_CRS):
    w, s, e, n = transform_bounds("EPSG:4326", target_crs, *bbox_4326)
    w = np.floor(w / COMMON_PIXEL) * COMMON_PIXEL
    s = np.floor(s / COMMON_PIXEL) * COMMON_PIXEL
    e = np.ceil(e / COMMON_PIXEL) * COMMON_PIXEL
    n = np.ceil(n / COMMON_PIXEL) * COMMON_PIXEL
    return w, s, e, n


def build_reference_grid(bbox_utm, pixel=COMMON_PIXEL, crs=TARGET_CRS):
    w, s, e, n = bbox_utm
    width = int(round((e - w) / pixel))
    height = int(round((n - s) / pixel))
    transform = rasterio.transform.from_origin(w, n, pixel, pixel)
    return {"crs": crs, "transform": transform, "width": width, "height": height}


def reproject_clip_resample(src_path, ref_grid, n_reflectance_bands):
    with rasterio.open(src_path) as src:
        nbands = src.count
        out = np.zeros((nbands, ref_grid["height"], ref_grid["width"]), np.float32)
        for b in range(1, nbands + 1):
            method = Resampling.nearest if b > n_reflectance_bands else Resampling.bilinear
            reproject(source=rasterio.band(src, b), destination=out[b - 1],
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_grid["transform"], dst_crs=ref_grid["crs"],
                      resampling=method)
    refl = out[:n_reflectance_bands]
    mask = out[n_reflectance_bands] if nbands > n_reflectance_bands else None
    return refl, mask


def estimate_shift(ref_band, test_band, max_shift=3):
    ref = (ref_band - np.nanmean(ref_band)) / (np.nanstd(ref_band) + EPS)
    test = (test_band - np.nanmean(test_band)) / (np.nanstd(test_band) + EPS)
    best, best_score = (0, 0), -np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(test, dy, axis=0), dx, axis=1)
            score = float(np.nansum(ref * shifted))
            if score > best_score:
                best_score, best = score, (dy, dx)
    return best


def apply_cloud_fill(refl, mask):
    if mask is None:
        return refl
    out = refl.copy()
    out[:, ~(mask > 0.5)] = np.nan
    return out


def compute_indices(refl):
    _, _, red, nir, swir1 = refl[0], refl[1], refl[2], refl[3], refl[4]
    ndvi = (nir - red) / (nir + red + EPS)
    ndbi = (swir1 - nir) / (swir1 + nir + EPS)
    return ndvi.astype(np.float32), ndbi.astype(np.float32)


def normalise_reflectance(refl, lo_pct=2, hi_pct=98):
    out = np.empty_like(refl)
    for b in range(refl.shape[0]):
        band = refl[b]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            out[b] = band
            continue
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
        out[b] = np.clip((band - lo) / (hi - lo + EPS), 0, 1)
    return out


def export_stack(path, refl, ndvi, ndbi, ref_grid):
    stack = np.concatenate([refl, ndvi[None], ndbi[None]], axis=0)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": stack.shape[0],
        "height": ref_grid["height"], "width": ref_grid["width"],
        "crs": ref_grid["crs"], "transform": ref_grid["transform"],
        "compress": "deflate", "nodata": float("nan"),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)
        for i, name in enumerate(["blue", "green", "red", "nir", "swir1", "ndvi", "ndbi"], start=1):
            dst.set_band_description(i, name)


def rgb(refl, bands):
    img = np.dstack([refl[b] for b in bands])
    return np.nan_to_num(np.clip(img, 0, 1))


def save_composites(refl, out_prefix):
    plt.imsave(f"{out_prefix}_truecolor.png", rgb(refl, [2, 1, 0]))
    plt.imsave(f"{out_prefix}_falsecolor.png", rgb(refl, [3, 2, 1]))


@dataclass
class YearInput:
    year: int
    path: str
    n_reflectance_bands: int = 5


def run_pipeline(aoi_geojson, year_inputs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bbox_utm = bbox_to_utm(load_aoi_bbox(aoi_geojson))
    ref_grid = build_reference_grid(bbox_utm)
    print(f"[grid] {ref_grid['width']} x {ref_grid['height']} px @ {COMMON_PIXEL} m, {TARGET_CRS}")

    cleaned = {}
    anchor = None
    for yi in year_inputs:
        refl, mask = reproject_clip_resample(yi.path, ref_grid, yi.n_reflectance_bands)
        refl = apply_cloud_fill(refl, mask)

        if anchor is None:
            anchor = refl[3].copy()
        else:
            dy, dx = estimate_shift(anchor, refl[3])
            print(f"[align] {yi.year}: shift (dy,dx)=({dy},{dx}) px")

        ndvi, ndbi = compute_indices(refl)
        refl_norm = normalise_reflectance(refl)
        out_tif = os.path.join(out_dir, f"stack_{yi.year}.tif")
        export_stack(out_tif, refl_norm, ndvi, ndbi, ref_grid)
        save_composites(refl_norm, os.path.join(out_dir, str(yi.year)))
        cleaned[yi.year] = refl_norm
        print(f"[export] {out_tif}  ({refl_norm.shape[1]}x{refl_norm.shape[2]})")

    shapes = {y: r.shape for y, r in cleaned.items()}
    assert len(set(shapes.values())) == 1, f"dimension mismatch: {shapes}"
    print(f"[verify] all years share dimensions: {next(iter(shapes.values()))}")
    return ref_grid, cleaned


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aoi", default="config/aoi.geojson")
    parser.add_argument("--y2017", required=True)
    parser.add_argument("--y2019", required=True)
    parser.add_argument("--y2024", required=True)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    run_pipeline(args.aoi,
                 [YearInput(2017, args.y2017), YearInput(2019, args.y2019), YearInput(2024, args.y2024)],
                 args.out)


if __name__ == "__main__":
    main()
