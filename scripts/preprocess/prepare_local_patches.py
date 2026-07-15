"""
Cut the Sentinel-2 stack + OSM road mask into 256x256 patches with a spatial train/val split.

Bands are saved R,G,B,NIR,SWIR so img[:3] is true-colour RGB. The right-hand strip of the
AOI is held out for validation so val patches never overlap train patches.

python scripts/prepare_local_patches.py

"""
import os

import numpy as np
import rasterio

OUT_STACK = "outputs/stack_2024.tif"
ROADS = "outputs/osm_roads_2024_mask.tif"
OUT = "data/patches/local"

PATCH = 256
STRIDE = 128
VAL_FRAC = 0.28


def stretch_bands(arr):
    out = np.empty_like(arr, dtype="float32")
    for b in range(arr.shape[0]):
        finite = arr[b][np.isfinite(arr[b])]
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        out[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(out)


def main():
    with rasterio.open(OUT_STACK) as s:
        img = s.read([3, 2, 1, 4, 5]).astype("float32")
        H, W = s.height, s.width
    with rasterio.open(ROADS) as r:
        mask = (r.read(1) == 1).astype("uint8")

    img = stretch_bands(img)
    split_col = int(W * (1 - VAL_FRAC))
    print(f"image {H}x{W}, split at col {split_col} (right {VAL_FRAC:.0%} = val)")

    for sub in ("train", "val"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    counts = {"train": 0, "val": 0}
    road_px = {"train": 0, "val": 0}
    total_px = {"train": 0, "val": 0}

    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):
            if x >= split_col:
                sub = "val"
            elif x + PATCH <= split_col:
                sub = "train"
            else:
                continue

            patch = img[:, y:y + PATCH, x:x + PATCH]
            patch_mask = mask[y:y + PATCH, x:x + PATCH]
            if patch[:3].max() < 1e-4:
                continue

            np.savez_compressed(f"{OUT}/{sub}/p_{y:04d}_{x:04d}.npz",
                                img=patch.astype("float32"), mask=patch_mask.astype("uint8"))
            counts[sub] += 1
            road_px[sub] += int(patch_mask.sum())
            total_px[sub] += patch_mask.size

    for sub in ("train", "val"):
        frac = road_px[sub] / max(total_px[sub], 1)
        print(f"{sub}: {counts[sub]} patches, road pixel fraction {frac:.3%}")
    print(f"wrote patches to {OUT}/")


if __name__ == "__main__":
    main()
