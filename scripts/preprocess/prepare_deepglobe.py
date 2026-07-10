"""Downsample DeepGlobe 0.5 m road tiles 16x to ~8 m patches that match Sentinel-2 road width.

    python scripts/prepare_deepglobe.py --src data/deepglobe/train --val-frac 0.1
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

OUT = "data/patches/deepglobe"
FACTOR = 16
PATCH = 1024 // FACTOR
KEEP_EMPTY_FRAC = 0.15


def downsample(img_arr, mask_arr):
    nh, nw = img_arr.shape[0] // FACTOR, img_arr.shape[1] // FACTOR
    img = np.asarray(Image.fromarray(img_arr).resize((nw, nh), Image.BILINEAR), dtype="float32") / 255.0
    mask = (np.asarray(Image.fromarray(mask_arr).resize((nw, nh), Image.NEAREST)) > 127).astype("uint8")
    return img, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder of *_sat.jpg / *_mask.png")
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    sats = sorted(glob.glob(f"{args.src}/*_sat.jpg"))
    if not sats:
        raise SystemExit(f"no *_sat.jpg found in {args.src}")
    print(f"{len(sats)} DeepGlobe tiles; downsampling {FACTOR}x to {PATCH}x{PATCH}")
    for sub in ("train", "val"):
        os.makedirs(f"{OUT}/{sub}", exist_ok=True)

    counts = {"train": 0, "val": 0}
    road_px = {"train": 0, "val": 0}
    rng = np.random.default_rng(0)
    val_every = max(int(round(1 / args.val_frac)), 1)

    for n, sat in enumerate(sats):
        mask_path = sat.replace("_sat.jpg", "_mask.png")
        if not os.path.exists(mask_path):
            continue
        img = np.asarray(Image.open(sat).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        img_s, mask_s = downsample(img, mask)

        if mask_s.sum() == 0 and rng.random() > KEEP_EMPTY_FRAC:
            continue

        sub = "val" if n % val_every == 0 else "train"
        chw = np.transpose(img_s, (2, 0, 1))
        np.savez_compressed(f"{OUT}/{sub}/d{n:05d}.npz",
                            img=chw.astype("float32"), mask=mask_s.astype("uint8"))
        counts[sub] += 1
        road_px[sub] += int(mask_s.sum())
        if n % 500 == 0:
            print(f"  {n}/{len(sats)}  ->  train {counts['train']} val {counts['val']}")

    for sub in ("train", "val"):
        total = counts[sub] * PATCH * PATCH
        print(f"{sub}: {counts[sub]} patches, road pixel fraction {road_px[sub]/max(total,1):.3%}")
    print(f"wrote patches to {OUT}/")


if __name__ == "__main__":
    main()
