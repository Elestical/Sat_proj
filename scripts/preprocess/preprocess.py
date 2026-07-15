"""
Turn raw single-year reflectance GeoTIFFs into aligned, model-ready 7-band stacks.

Each year is reprojected to UTM 43N, clipped to the AOI, resampled to one shared 10 m grid,
cloud-filled, given NDVI/NDBI channels and normalised, so every pixel of every year describes
the same ground.

Input band order : blue, green, red, nir, swir1, [clearmask].

python scripts/preprocess.py 
--aoi config/aoi.geojson 
--y2017 raw_2017.tif 
--y2019 raw_2019.tif 
--y2024 raw_2024.tif 
--out outputs

"""
import json
import os
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
import rasterio.transform

TARGET_CRS = "EPSG:32643"
COMMON_PIXEL = 10.0
EPS = 1e-6


#reads the same aoi bbox used by the acquire scripts, so preprocessing lines up with what was downloaded
def load_aoi_bbox(geojson_path):
    with open(geojson_path) as f:
        return tuple(json.load(f)["bbox"])


#reprojects the wgs84 bbox to utm meters and snaps it outward to whole pixel boundaries
def bbox_to_utm(bbox_4326, target_crs=TARGET_CRS):
    w, s, e, n = transform_bounds("EPSG:4326", target_crs, *bbox_4326)
    w = np.floor(w / COMMON_PIXEL) * COMMON_PIXEL
    s = np.floor(s / COMMON_PIXEL) * COMMON_PIXEL
    e = np.ceil(e / COMMON_PIXEL) * COMMON_PIXEL
    n = np.ceil(n / COMMON_PIXEL) * COMMON_PIXEL
    return w, s, e, n


#defines the one shared pixel grid every year gets resampled onto, so that multi temporal comparison can take place
def build_reference_grid(bbox_utm, pixel=COMMON_PIXEL, crs=TARGET_CRS):
    w, s, e, n = bbox_utm
    width = int(round((e - w) / pixel))
    height = int(round((n - s) / pixel))
    transform = rasterio.transform.from_origin(w, n, pixel, pixel)
    return {"crs": crs, "transform": transform, "width": width, "height": height}


#every year's imagery pixel alignment
def reproject_clip_resample(src_path, ref_grid, n_reflectance_bands):
    with rasterio.open(src_path) as src:
        nbands = src.count
        #pre allocate an empty output array, with reference grid's height and width
        out = np.zeros((nbands, ref_grid["height"], ref_grid["width"]), np.float32)

        #reprojecting pre band
        for b in range(1, nbands + 1):
            #bilinear for continuous reflectance values
            #nearest for the mask band (keeps it strictly binary, 0 or 1)
            method = Resampling.nearest if b > n_reflectance_bands else Resampling.bilinear 
            reproject(source = rasterio.band(src, b),
                      destination = out[b - 1],
                      src_transform = src.transform,
                      src_crs = src.crs,
                      dst_transform = ref_grid["transform"],
                      dst_crs = ref_grid["crs"],
                      resampling = method)
    refl = out[:n_reflectance_bands]   #the first n_reflectance bands
    mask = out[n_reflectance_bands] if nbands > n_reflectance_bands else None #6th band
    return refl, mask


#blanks out (nan) any pixel the cloud mask didn't mark as clear, so clouded pixels don't pollute later stats
def apply_cloud_fill(refl, mask):
    if mask is None:
        return refl
    out = refl.copy()
    out[:, ~(mask > 0.5)] = np.nan
    return out


#derives ndvi (vegetation index) and ndbi (built-up index) from the reflectance bands
def compute_indices(refl):
    _, _, red, nir, swir1 = refl[0], refl[1], refl[2], refl[3], refl[4]
    ndvi = (nir - red) / (nir + red + EPS)
    ndbi = (swir1 - nir) / (swir1 + nir + EPS)
    return ndvi.astype(np.float32), ndbi.astype(np.float32)


#rescales each band to a 0-1 range using percentile clipping, so outlier pixels don't compress everything else
def normalise_reflectance(refl, lo_pct=2, hi_pct=98):
    out = np.empty_like(refl)
    for b in range(refl.shape[0]):
        band = refl[b]
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            out[b] = band
            continue
        #clip to the 2nd/98th percentile range instead of true min/max, so a few extreme pixels don't dominate the scale
        lo, hi = np.percentile(finite, [lo_pct, hi_pct])
        out[b] = np.clip((band - lo) / (hi - lo + EPS), 0, 1)
    return out


#writes the final 7-band (5 reflectance + ndvi + ndbi) stack to a geotiff, with named band descriptions
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


#one raw input year: which raw geotiff to read, and how many of its bands are reflectance (vs. the trailing mask band)
@dataclass
class YearInput:
    year: int
    path: str
    n_reflectance_bands: int = 5


#the full per-aoi pipeline: build one shared grid, then clean/index/normalise/export each year onto it
def run_pipeline(aoi_geojson, year_inputs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    bbox_utm = bbox_to_utm(load_aoi_bbox(aoi_geojson))
    ref_grid = build_reference_grid(bbox_utm)
    print(f"[grid] {ref_grid['width']} x {ref_grid['height']} px @ {COMMON_PIXEL} m, {TARGET_CRS}")

    cleaned = {}
    for yi in year_inputs:
        refl, mask = reproject_clip_resample(yi.path, ref_grid, yi.n_reflectance_bands)
        refl = apply_cloud_fill(refl, mask)
        ndvi, ndbi = compute_indices(refl)
        refl_norm = normalise_reflectance(refl)
        out_tif = os.path.join(out_dir, f"stack_{yi.year}.tif")
        export_stack(out_tif, refl_norm, ndvi, ndbi, ref_grid)
        cleaned[yi.year] = refl_norm
        print(f"[export] {out_tif}  ({refl_norm.shape[1]}x{refl_norm.shape[2]})")

    #sanity check: every year's output array must share the exact same shape, or later stacking/diffing breaks
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
