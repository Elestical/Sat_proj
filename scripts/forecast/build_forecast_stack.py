"""Build the yearly driver-channel stack the growth forecaster reads, plus shared grid helpers.

Channels per year: built probability, built density, proximity to the built edge, proximity
to roads, proximity to the airport, NDVI. All on the 10 m reference grid (EPSG:32643).
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import distance_transform_edt, uniform_filter

OUT = "outputs"
REF = f"{OUT}/pred_2024.tif"

YEARS = list(range(2017, 2025))
CH_NAMES = ["built", "built_density", "prox_built", "prox_road", "prox_airport", "ndvi"]
PIX_M = 10.0
NDVI_BAND = 6
AIRPORT_LONLAT = (77.7066, 13.1986)
STACK_YEARS = [2017, 2019, 2024]


def ref_grid():
    with rasterio.open(REF) as s:
        return s.transform, s.crs, s.height, s.width


def align(path, band, H, W):
    with rasterio.open(path) as s:
        a = s.read(band).astype("float32")
        nodata = s.nodata
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    if nodata is not None:
        a[a == np.float32(nodata)] = np.nan
    out = np.full((H, W), np.nan, "float32")
    h, w = min(H, a.shape[0]), min(W, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def proximity(mask, scale_px):
    d = distance_transform_edt(~mask)
    return np.exp(-d / scale_px).astype("float32")


def airport_rc(transform, crs, H, W):
    xs, ys = warp_transform("EPSG:4326", crs, [AIRPORT_LONLAT[0]], [AIRPORT_LONLAT[1]])
    col, row = ~transform * (xs[0], ys[0])
    return int(round(row)), int(round(col))


def built_path(year):
    annual = f"{OUT}/dw_built_{year}.tif"
    if os.path.exists(annual):
        return annual
    if year in (2017, 2024):
        return f"{OUT}/built_{year}.tif"
    raise FileNotFoundError(f"missing {annual} -- run acquire_dw_annual.py to fetch the {year} frame")


def nearest_stack_year(year):
    return min(STACK_YEARS, key=lambda y: abs(y - year))


def nearest_road_pred(year):
    return f"{OUT}/pred_2017.tif" if year <= 2020 else f"{OUT}/pred_2024.tif"


def build_channels(year, grid=None, airport=None):
    transform, crs, H, W = grid or ref_grid()
    if airport is None:
        airport = airport_rc(transform, crs, H, W)

    built = np.clip(np.nan_to_num(align(built_path(year), 1, H, W)), 0.0, 1.0)
    density = uniform_filter(built, size=21, mode="nearest")
    prox_built = proximity(built > 0.5, scale_px=30)
    road = np.nan_to_num(align(nearest_road_pred(year), 1, H, W)) > 0.5
    prox_road = proximity(road, scale_px=30)
    airport_mask = np.zeros((H, W), bool)
    if 0 <= airport[0] < H and 0 <= airport[1] < W:
        airport_mask[airport] = True
    prox_airport = proximity(airport_mask, scale_px=300)
    ndvi = align(f"{OUT}/stack_{nearest_stack_year(year)}.tif", NDVI_BAND, H, W)
    ndvi = np.nan_to_num((ndvi + 1) / 2)

    return np.stack([built, density, prox_built, prox_road, prox_airport, ndvi]).astype("float32")


def build_series(grid=None):
    grid = grid or ref_grid()
    transform, crs, H, W = grid
    airport = airport_rc(transform, crs, H, W)
    frames = [build_channels(year, grid, airport) for year in YEARS]
    return np.stack(frames), grid, airport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    transform, crs, H, W = ref_grid()
    print(f"reference grid {W}x{H} @ {PIX_M} m  {crs}")
    print(f"airport row,col = {airport_rc(transform, crs, H, W)}")

    present = [y for y in YEARS if os.path.exists(f"{OUT}/dw_built_{y}.tif") or y in (2017, 2024)]
    missing = [y for y in YEARS if y not in present]
    print(f"frames present: {present}")
    if missing:
        print(f"missing {missing} -- run acquire_dw_annual.py --years " + " ".join(map(str, missing)))
    if args.check:
        return

    if args.save:
        if missing:
            raise SystemExit("cannot --save until all frames are present")
        series, _, airport = build_series((transform, crs, H, W))
        np.savez_compressed(f"{OUT}/forecast_stack.npz",
                            series=series.astype("float16"), years=np.array(YEARS),
                            channels=np.array(CH_NAMES), airport=np.array(airport),
                            transform=np.array(transform).reshape(-1)[:6], crs=str(crs))
        print(f"saved {OUT}/forecast_stack.npz  shape {series.shape}")


if __name__ == "__main__":
    main()
