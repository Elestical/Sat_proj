"""Score one or more road-model checkpoints on a validation patch folder.

    python scripts/eval.py --data data/patches/local/val scratch=outputs/road_model.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.unet import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_patches(folder, in_channels):
    xs, ys = [], []
    for f in sorted(glob.glob(f"{folder}/*.npz")):
        data = np.load(f)
        xs.append(data["img"][:in_channels])
        ys.append(data["mask"])
    return np.stack(xs), np.stack(ys)


@torch.no_grad()
def score(model, X, Y):
    model.eval()
    tp = fp = fn = 0
    for i in range(0, len(X), 8):
        x = torch.from_numpy(X[i:i + 8]).float().to(DEVICE)
        p = (torch.sigmoid(model(x)) > 0.5).float().cpu().numpy()[:, 0]
        y = Y[i:i + 8]
        tp += float(((p == 1) & (y == 1)).sum())
        fp += float(((p == 1) & (y == 0)).sum())
        fn += float(((p == 0) & (y == 1)).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return iou, f1, precision, recall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="validation patch folder")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("ckpts", nargs="+", help="label=path.pt entries")
    args = parser.parse_args()

    print(f"{'model':28s} {'IoU':>7s} {'F1':>7s} {'Prec':>7s} {'Recall':>7s}")
    print("-" * 60)
    for entry in args.ckpts:
        label, path = entry.split("=", 1)
        channels = 3 if "smp" in label else args.in_channels
        model = build_model(label, channels)
        X, Y = load_patches(args.data, channels)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        iou, f1, precision, recall = score(model, X, Y)
        print(f"{label:28s} {iou:7.3f} {f1:7.3f} {precision:7.3f} {recall:7.3f}")


if __name__ == "__main__":
    main()
