"""Draw predicted 2024 roads and new-since-2017 roads over the true-colour image.

    python scripts/render_predictions.py --pred2017 outputs/pred_2017.tif --pred2024 outputs/pred_2024.tif
"""
import argparse

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "outputs"
STACK24 = f"{OUT}/stack_2024.tif"


def truecolor():
    with rasterio.open(STACK24) as s:
        rgb = np.transpose(s.read([3, 2, 1]).astype("float32"), (1, 2, 0))
    finite = rgb[np.isfinite(rgb)]
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    return np.nan_to_num(np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1))


def read(path, thr):
    with rasterio.open(path) as s:
        return s.read(1) > thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred2017", required=True)
    parser.add_argument("--pred2024", required=True)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--out", default="outputs/pred_roads_figure.png")
    args = parser.parse_args()

    rgb = truecolor()
    p17 = read(args.pred2017, args.thr)
    p24 = read(args.pred2024, args.thr)
    new = p24 & ~p17

    left = rgb.copy()
    left[p24] = [1, 0, 0]
    right = rgb.copy()
    right[p17] = [0, 0.45, 1]
    right[new] = [1, 0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(left)
    axes[0].set_title(f"predicted roads 2024 (red), thr {args.thr}")
    axes[1].imshow(right)
    axes[1].set_title("road growth: 2017 (blue) vs new by 2024 (red)")
    axes[1].legend(handles=[Patch(color=[0, 0.45, 1], label="road in 2017"),
                            Patch(color=[1, 0, 0], label="new since 2017")],
                   loc="lower right", framealpha=0.9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("U-Net road detection - Devanahalli AOI (Sentinel-2 10 m)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
