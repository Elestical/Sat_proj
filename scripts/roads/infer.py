"""Run a trained road model over a full Sentinel-2 stack and write a georeferenced probability raster.

    python scripts/infer.py --stack outputs/stack_2024.tif --ckpt outputs/road_model.pt --out outputs/pred_2024.tif
"""
import argparse
import os
import sys

import numpy as np
import rasterio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.unet import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PATCH = 256
STRIDE = 192


def stretch_bands(arr):
    out = np.empty_like(arr, dtype="float32")
    for b in range(arr.shape[0]):
        finite = arr[b][np.isfinite(arr[b])]
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        out[b] = np.clip((arr[b] - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(out)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    channels = 3 if "smp" in args.ckpt else args.in_channels
    model = build_model(args.ckpt, channels)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()

    with rasterio.open(args.stack) as s:
        img = stretch_bands(s.read([3, 2, 1, 4, 5]).astype("float32"))[:channels]
        profile = s.profile
        H, W = s.height, s.width

    prob = np.zeros((H, W), "float32")
    cover = np.zeros((H, W), "float32")
    ys = sorted(set(list(range(0, max(H - PATCH, 0) + 1, STRIDE)) + [H - PATCH]))
    xs = sorted(set(list(range(0, max(W - PATCH, 0) + 1, STRIDE)) + [W - PATCH]))
    for y in ys:
        for x in xs:
            tile = img[:, y:y + PATCH, x:x + PATCH]
            t = torch.from_numpy(tile[None]).float().to(DEVICE)
            p = torch.sigmoid(model(t)).cpu().numpy()[0, 0]
            prob[y:y + PATCH, x:x + PATCH] += p
            cover[y:y + PATCH, x:x + PATCH] += 1
    prob /= np.maximum(cover, 1)

    profile.update(count=1, dtype="float32", nodata=None)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(prob.astype("float32"), 1)
    print(f"wrote {args.out}  (mean prob {prob.mean():.3f}, >0.5 covers {100*(prob>0.5).mean():.1f}%)")


if __name__ == "__main__":
    main()
