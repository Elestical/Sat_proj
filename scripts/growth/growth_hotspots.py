"""Find statistically significant growth clusters with the Getis-Ord Gi* hotspot statistic."""
import csv

import numpy as np
import rasterio
from rasterio.transform import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"


def window_sum(a):
    padded = np.pad(a, 1, mode="constant")
    total = np.zeros_like(a, dtype="float64")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += padded[1 + dy:1 + dy + a.shape[0], 1 + dx:1 + dx + a.shape[1]]
    return total


def getis_ord(x):
    x = x.astype("float64")
    n = x.size
    mean = x.mean()
    std = x.std()
    wsum = window_sum(x)
    wcount = window_sum(np.ones_like(x))
    denom = std * np.sqrt((n * wcount - wcount ** 2) / (n - 1))
    denom[denom == 0] = np.nan
    return (wsum - mean * wcount) / denom


def main():
    data = np.load(f"{OUT}/growth_cells.npz", allow_pickle=True)
    score = data["score"]
    transform = Affine(*data["transform"])
    crs = str(data["crs"])
    nr, nc = score.shape

    z = getis_ord(score)
    print(f"Getis-Ord Gi* on {nr * nc} cells:")
    print(f"  hot spots  (z>=+1.96): {int((z >= 1.96).sum())}")
    print(f"  cold spots (z<=-1.96): {int((z <= -1.96).sum())}")

    profile = dict(driver="GTiff", height=nr, width=nc, count=1, dtype="float32", crs=crs, transform=transform)
    with rasterio.open(f"{OUT}/growth_hotspots_z.tif", "w", **profile) as dst:
        dst.write(np.nan_to_num(z).astype("float32"), 1)

    limit = np.nanmax(np.abs(z))
    plt.figure(figsize=(8, 7))
    im = plt.imshow(z, cmap="RdBu_r", vmin=-limit, vmax=limit)
    plt.colorbar(im, fraction=0.046, shrink=0.85, label="Gi* z-score")
    ys, xs = np.where(z >= 1.96)
    plt.scatter(xs, ys, s=12, facecolors="none", edgecolors="black", linewidths=0.8, label="hot (95%)")
    plt.xticks([])
    plt.yticks([])
    plt.title("Growth hotspots (Getis-Ord Gi*)")
    if len(xs):
        plt.legend(loc="lower right", fontsize=8, framealpha=0.9)
    plt.savefig(f"{OUT}/growth_hotspots.png", dpi=140, bbox_inches="tight")
    plt.close()

    with open(f"{OUT}/growth_hotspots.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "growth_score", "gi_z", "class"])
        for i in range(nr):
            for j in range(nc):
                zc = z[i, j]
                label = "hot" if zc >= 1.96 else ("cold" if zc <= -1.96 else "ns")
                writer.writerow([i, j, round(float(score[i, j]), 4),
                                 round(float(zc), 3) if np.isfinite(zc) else "nan", label])
    print(f"wrote growth_hotspots_z.tif, growth_hotspots.png, growth_hotspots.csv")


if __name__ == "__main__":
    main()
