"""Aggregate road and built-up growth into a ~1 km cell grid and write per-cell tables and rasters."""
import argparse
import csv
import os

import numpy as np
import rasterio
from rasterio.transform import Affine
from skimage.morphology import skeletonize

OUT = "outputs"
REF = f"{OUT}/pred_2024.tif"
PIX_M = 10.0
NDVI_BAND = 6


def align(path, band, H, W):
    with rasterio.open(path) as s:
        a = s.read(band).astype("float32")
    out = np.full((H, W), np.nan, "float32")
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def built_path(year):
    for name in (f"{OUT}/dw_built_{year}.tif", f"{OUT}/built_{year}.tif"):
        if os.path.exists(name):
            return name
    return f"{OUT}/dw_built_{year}.tif"


def normalise(x):
    peak = x.max()
    return x / peak if peak > 0 else x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-km", type=float, default=1.0)
    parser.add_argument("--road-thr", type=float, default=0.5)
    args = parser.parse_args()
    cell = int(round(args.cell_km * 1000 / PIX_M))

    with rasterio.open(REF) as s:
        transform, crs, H, W = s.transform, s.crs, s.height, s.width

    pred17 = align(f"{OUT}/pred_2017.tif", 1, H, W)
    pred24 = align(f"{OUT}/pred_2024.tif", 1, H, W)
    built17 = align(built_path(2017), 1, H, W)
    built24 = align(built_path(2024), 1, H, W)
    ndvi17 = align(f"{OUT}/stack_2017.tif", NDVI_BAND, H, W)
    ndvi24 = align(f"{OUT}/stack_2024.tif", NDVI_BAND, H, W)

    new_road = (pred24 > args.road_thr) & (pred17 <= args.road_thr)
    skeleton = skeletonize(np.nan_to_num(new_road))
    new_built = (built24 > 0.5) & (built17 < 0.3)
    ndvi_drop = ndvi17 - ndvi24

    nr, nc = (H + cell - 1) // cell, (W + cell - 1) // cell
    road_km = np.zeros((nr, nc), "float32")
    built_km2 = np.zeros((nr, nc), "float32")
    drop = np.zeros((nr, nc), "float32")
    built17_mean = np.zeros((nr, nc), "float32")

    for i in range(nr):
        for j in range(nc):
            ys, xs = slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell)
            road_km[i, j] = skeleton[ys, xs].sum() * PIX_M / 1000.0
            built_km2[i, j] = int(new_built[ys, xs].sum()) * (PIX_M ** 2) / 1e6
            drop[i, j] = np.nanmean(ndvi_drop[ys, xs]) if np.isfinite(ndvi_drop[ys, xs]).any() else 0.0
            built17_mean[i, j] = np.nanmean(built17[ys, xs]) if np.isfinite(built17[ys, xs]).any() else 0.0

    score = 0.5 * normalise(road_km) + 0.5 * normalise(built_km2)

    os.makedirs(OUT, exist_ok=True)
    cell_transform = transform * Affine.scale(cell)
    np.savez(f"{OUT}/growth_cells.npz", road_km=road_km, built_km2=built_km2,
             ndvi_drop=drop, built17=built17_mean, score=score, cell=cell,
             transform=np.array(cell_transform).reshape(-1)[:6], crs=str(crs))

    with open(f"{OUT}/growth_cells.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "row", "col", "cx_utm", "cy_utm",
                         "new_road_km", "new_built_km2", "ndvi_drop", "built17_mean", "growth_score"])
        for i in range(nr):
            for j in range(nc):
                cx, cy = cell_transform * (j + 0.5, i + 0.5)
                writer.writerow([i * nc + j, i, j, round(cx, 1), round(cy, 1),
                                 round(float(road_km[i, j]), 3), round(float(built_km2[i, j]), 4),
                                 round(float(drop[i, j]), 4), round(float(built17_mean[i, j]), 3),
                                 round(float(score[i, j]), 4)])

    profile = dict(driver="GTiff", height=nr, width=nc, count=1, dtype="float32",
                   crs=crs, transform=cell_transform)
    with rasterio.open(f"{OUT}/growth_score.tif", "w", **profile) as dst:
        dst.write(score.astype("float32"), 1)

    print(f"grid {nr}x{nc} cells ({args.cell_km} km each)")
    print(f"total new road  {road_km.sum():.1f} km")
    print(f"total new built {built_km2.sum():.2f} km^2")
    print(f"wrote growth_cells.csv, growth_cells.npz, growth_score.tif")


if __name__ == "__main__":
    main()
