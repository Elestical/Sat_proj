"""Per-1 km-cell explanations for the dashboard: driver values, SHAP contributions, growth type.

Trains the transparent suitability model (open land in 2017 that became built by 2024) on the
named driver channels, explains every cell with SHAP, and writes cells.geojson with everything
the dashboard needs to say why each cell grows the way it does.

    python scripts/dashboard/build_cell_explanations.py
"""
import csv
import json
import os
import sys

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.transform import Affine
from sklearn.ensemble import RandomForestClassifier
import shap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "forecast"))
import build_forecast_stack as B

OUT = "outputs"
FEATURES = ["built_density", "prox_built", "prox_road", "prox_airport", "ndvi"]
BUILT_HI, BUILT_LO = 0.5, 0.3
TYPE_NAME = {0: "stable", 1: "expansion", 2: "densification"}
RNG = np.random.default_rng(0)


def built_year(year, H, W):
    return np.clip(np.nan_to_num(B.align(B.built_path(year), 1, H, W)), 0, 1)


def feature_stack(year, grid):
    ch = B.build_channels(year, grid)
    idx = [B.CH_NAMES.index(f) for f in FEATURES]
    return np.stack([ch[i] for i in idx], axis=-1)


def train_suitability(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    neg = RNG.choice(neg, size=min(len(neg), 3 * len(pos)), replace=False)
    idx = RNG.permutation(np.concatenate([pos, neg]))
    return RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1,
                                  class_weight="balanced", random_state=0).fit(X[idx], y[idx])


def cell_means(feats, cell, nr, nc):
    out = np.zeros((nr, nc, feats.shape[-1]), "float32")
    for i in range(nr):
        for j in range(nc):
            block = feats[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell, :]
            out[i, j] = np.nanmean(block.reshape(-1, block.shape[-1]), axis=0)
    return np.nan_to_num(out)


def read_grid(path, nr, nc):
    with rasterio.open(path) as s:
        a = s.read(1).astype("float32")
    out = np.zeros((nr, nc), "float32")
    h, w = min(nr, a.shape[0]), min(nc, a.shape[1])
    out[:h, :w] = a[:h, :w]
    return out


def load_forecast(nr, nc):
    grid = np.zeros((nr, nc), "float32")
    path = f"{OUT}/forecast_cells.csv"
    if not os.path.exists(path):
        return grid
    for row in csv.DictReader(open(path)):
        key = next((k for k in ("new_km2_2030", "new_km2_convlstm_2030", "new_km2_suit_2030") if k in row), None)
        r, c = int(row["row"]), int(row["col"])
        if key and r < nr and c < nc:
            grid[r, c] = float(row[key])
    return grid


def cell_polygon(cell_transform, crs, i, j):
    corners = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1), (j, i)]
    xs, ys = zip(*[cell_transform * (cx, cy) for cx, cy in corners])
    lon, lat = warp_transform(crs, "EPSG:4326", list(xs), list(ys))
    return [[lo, la] for lo, la in zip(lon, lat)]


def main():
    os.makedirs(OUT, exist_ok=True)
    data = np.load(f"{OUT}/growth_cells.npz", allow_pickle=True)
    nr, nc = data["score"].shape
    cell_transform = Affine(*data["transform"])
    crs = str(data["crs"])
    grid = B.ref_grid()
    _, _, H, W = grid

    feats = feature_stack(2017, grid)
    cell = int(data["cell"])
    means = cell_means(feats, cell, nr, nc)

    b17 = built_year(2017, H, W)
    b24 = built_year(2024, H, W)
    candidate = b17 < BUILT_LO
    label = (b24 > BUILT_HI).astype("uint8")
    rf = train_suitability(feats[candidate], label[candidate])

    flat = means.reshape(-1, means.shape[-1])
    suit = rf.predict_proba(flat)[:, 1].reshape(nr, nc)
    sv = np.asarray(shap.TreeExplainer(rf).shap_values(flat))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    shap_cells = sv.reshape(nr, nc, len(FEATURES))

    growth_type = read_grid(f"{OUT}/growth_type.tif", nr, nc)
    hotspot_z = read_grid(f"{OUT}/growth_hotspots_z.tif", nr, nc)
    forecast = load_forecast(nr, nc)

    features = []
    for i in range(nr):
        for j in range(nc):
            gtype = int(round(growth_type[i, j]))
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [cell_polygon(cell_transform, crs, i, j)]},
                "properties": {
                    "id": i * nc + j, "row": i, "col": j,
                    "growth_type": TYPE_NAME[gtype],
                    "new_road_km": round(float(data["road_km"][i, j]), 2),
                    "new_built_km2": round(float(data["built_km2"][i, j]), 3),
                    "built17": round(float(data["built17"][i, j]), 2),
                    "ndvi_drop": round(float(data["ndvi_drop"][i, j]), 3),
                    "hotspot_z": round(float(hotspot_z[i, j]), 2),
                    "forecast_km2": round(float(forecast[i, j]), 3),
                    "suitability": round(float(suit[i, j]), 3),
                    "drivers": {f: round(float(means[i, j, k]), 3) for k, f in enumerate(FEATURES)},
                    "shap": {f: round(float(shap_cells[i, j, k]), 4) for k, f in enumerate(FEATURES)},
                },
            })

    with open(f"{OUT}/cells.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    expansion = sum(1 for ft in features if ft["properties"]["growth_type"] == "expansion")
    print(f"wrote {OUT}/cells.geojson  ({len(features)} cells, {expansion} expansion)")


if __name__ == "__main__":
    main()
