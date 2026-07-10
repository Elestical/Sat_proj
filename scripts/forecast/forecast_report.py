"""Aggregate the ConvLSTM forecast onto the 1 km cell grid and write a per-cell table and figure."""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import build_forecast_stack as B

OUT = B.OUT
PIX_M = 10.0
CELL = int(round(1000 / PIX_M))


def built_now(H, W):
    latest = max(y for y in B.YEARS if os.path.exists(f"{OUT}/dw_built_{y}.tif") or y in (2017, 2024))
    return np.clip(np.nan_to_num(B.align(B.built_path(latest), 1, H, W)), 0, 1), latest


def forecast_path(year):
    for name in (f"{OUT}/forecast_convlstm_{year}_tuned.tif", f"{OUT}/forecast_convlstm_{year}.tif"):
        if os.path.exists(name):
            return name
    return None


def per_cell_new(forecast_mask, base_built, H, W):
    new = forecast_mask & (base_built < 0.5)
    nr, nc = (H + CELL - 1) // CELL, (W + CELL - 1) // CELL
    grid = np.zeros((nr, nc), "float32")
    for i in range(nr):
        for j in range(nc):
            ys, xs = slice(i * CELL, (i + 1) * CELL), slice(j * CELL, (j + 1) * CELL)
            grid[i, j] = int(new[ys, xs].sum()) * (PIX_M ** 2) / 1e6
    return grid


def main():
    _, _, H, W = B.ref_grid()
    base, latest = built_now(H, W)
    print(f"base built year {latest}")

    forecasts = {}
    for year in (2027, 2030):
        path = forecast_path(year)
        if path is not None:
            forecasts[year] = np.nan_to_num(B.align(path, 1, H, W)) > 0.5

    if not forecasts:
        raise SystemExit("no forecast_convlstm_*.tif in outputs/ -- run the ConvLSTM notebook first")

    cell_grids = {}
    for year, mask in forecasts.items():
        cell_grids[year] = per_cell_new(mask, base, H, W)
        print(f"  {year}: +{cell_grids[year].sum():.2f} km^2 new built-up")

    years = sorted(cell_grids)
    nr, nc = cell_grids[years[0]].shape
    with open(f"{OUT}/forecast_cells.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "row", "col"] + [f"new_km2_{y}" for y in years])
        for i in range(nr):
            for j in range(nc):
                writer.writerow([i * nc + j, i, j] + [round(float(cell_grids[y][i, j]), 4) for y in years])
    print(f"wrote forecast_cells.csv ({nr}x{nc} cells)")

    panels = [("observed", latest, base)]
    if 2030 in forecasts:
        panels.append(("forecast", 2030, forecasts[2030].astype("float32")))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (tag, year, image) in zip(axes, panels):
        ax.imshow(image, cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{tag} built {year}")
        ax.axis("off")
    fig.suptitle("Devanahalli built-up: observed vs forecast")
    fig.tight_layout()
    fig.savefig(f"{OUT}/forecast_compare.png", dpi=130)
    plt.close(fig)
    print(f"wrote forecast_compare.png")


if __name__ == "__main__":
    main()
