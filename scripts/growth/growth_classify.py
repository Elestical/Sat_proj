"""Classify each growth cell as stable, expansion (greenfield) or densification (infill)."""
import argparse
import csv

import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

OUT = "outputs"
NAMES = ["stable", "expansion", "densification"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grow-thr", type=float, default=0.10)
    parser.add_argument("--built-thr", type=float, default=0.40)
    args = parser.parse_args()

    data = np.load(f"{OUT}/growth_cells.npz", allow_pickle=True)
    score, built17, drop = data["score"], data["built17"], data["ndvi_drop"]
    transform = Affine(*data["transform"])
    crs = str(data["crs"])
    nr, nc = score.shape

    growth_type = np.zeros((nr, nc), "uint8")
    growing = score >= args.grow_thr
    growth_type[growing & (built17 >= args.built_thr)] = 2
    growth_type[growing & (built17 < args.built_thr)] = 1

    counts = {name: int((growth_type == k).sum()) for k, name in enumerate(NAMES)}
    print("cell counts:", counts)
    print(f"new built-up in expansion cells:     {float(data['built_km2'][growth_type == 1].sum()):.2f} km^2")
    print(f"new built-up in densification cells: {float(data['built_km2'][growth_type == 2].sum()):.2f} km^2")

    profile = dict(driver="GTiff", height=nr, width=nc, count=1, dtype="uint8", crs=crs, transform=transform)
    with rasterio.open(f"{OUT}/growth_type.tif", "w", **profile) as dst:
        dst.write(growth_type, 1)

    cmap = ListedColormap(["#e8e8e8", "#d7191c", "#fdae61"])
    plt.figure(figsize=(8, 7))
    plt.imshow(growth_type, cmap=cmap, vmin=0, vmax=2)
    plt.xticks([])
    plt.yticks([])
    plt.title("Growth type per 1 km cell")
    plt.legend(handles=[Patch(color="#e8e8e8", label="stable"),
                        Patch(color="#d7191c", label="expansion (greenfield)"),
                        Patch(color="#fdae61", label="densification (infill)")],
               loc="lower right", framealpha=0.9, fontsize=8)
    plt.savefig(f"{OUT}/growth_type.png", dpi=140, bbox_inches="tight")
    plt.close()

    with open(f"{OUT}/growth_classified.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "growth_score", "built17_mean", "ndvi_drop", "growth_type"])
        for i in range(nr):
            for j in range(nc):
                writer.writerow([i, j, round(float(score[i, j]), 4), round(float(built17[i, j]), 3),
                                 round(float(drop[i, j]), 4), NAMES[growth_type[i, j]]])
    print(f"wrote growth_type.tif, growth_type.png, growth_classified.csv")


if __name__ == "__main__":
    main()
