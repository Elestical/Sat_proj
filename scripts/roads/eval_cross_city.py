"""Cross-city validation: does the road model generalise to a city it never trained on?

The model was only ever validated on the home AOI, so its transfer to a different city was
never measured. This scores a checkpoint on a home validation set and on an unseen-city patch
set, reporting both strict pixel IoU/F1 and the fair buffered completeness/correctness (1 px
tolerance, the project's road metric). A buffered F1 on the unseen city near the home score
means the model transfers rather than overfits.

    python scripts/roads/eval_cross_city.py --home data/patches/local/val --away data/patches/orr scratch=outputs/road_model.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.unet import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_patches(folder, in_channels):
    xs, ys = [], []
    for f in sorted(glob.glob(f"{folder}/*.npz")):
        d = np.load(f)
        xs.append(d["img"][:in_channels])
        ys.append(d["mask"])
    return np.stack(xs).astype("float32"), np.stack(ys).astype("uint8")


@torch.no_grad()
def evaluate(model, X, Y, tol=1):
    model.eval()
    tp = fp = fn = 0
    comp_hit = corr_hit = truth = pred_tot = 0
    st = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    for i in range(0, len(X), 8):
        x = torch.from_numpy(X[i:i + 8]).to(DEVICE)
        p = (torch.sigmoid(model(x)) > 0.5).cpu().numpy()[:, 0].astype(bool)
        y = Y[i:i + 8].astype(bool)
        tp += float((p & y).sum())
        fp += float((p & ~y).sum())
        fn += float((~p & y).sum())
        for pk, yk in zip(p, y):
            comp_hit += (yk & binary_dilation(pk, st)).sum()
            corr_hit += (pk & binary_dilation(yk, st)).sum()
            truth += yk.sum()
            pred_tot += pk.sum()
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    comp = comp_hit / max(truth, 1)
    corr = corr_hit / max(pred_tot, 1)
    bf1 = 2 * comp * corr / max(comp + corr, 1e-6)
    return dict(iou=iou, f1=f1, bf1=bf1, bcomp=comp, bcorr=corr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True, help="home-city validation patch folder")
    parser.add_argument("--away", required=True, help="unseen-city patch folder")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("ckpts", nargs="+", help="label=path.pt entries")
    args = parser.parse_args()

    print(f"{'model / setting':40s} {'IoU':>6} {'F1':>6} | {'bufF1':>6} {'bComp':>6} {'bCorr':>6}")
    print("-" * 80)
    for entry in args.ckpts:
        label, path = entry.split("=", 1)
        channels = 3 if "smp" in label else args.in_channels
        model = build_model(label, channels)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.to(DEVICE)
        for tag, folder in (("home", args.home), ("unseen city", args.away)):
            X, Y = load_patches(folder, channels)
            r = evaluate(model, X, Y)
            print(f"{label + '  ->  ' + tag:40s} {r['iou']:6.3f} {r['f1']:6.3f} | "
                  f"{r['bf1']:6.3f} {r['bcomp']:6.3f} {r['bcorr']:6.3f}")
        home = evaluate(model, *load_patches(args.home, channels))
        away = evaluate(model, *load_patches(args.away, channels))
        delta = 100 * (away["bf1"] - home["bf1"]) / max(home["bf1"], 1e-6)
        print(f"  generalisation (buffered F1): home {home['bf1']:.3f}  ->  unseen {away['bf1']:.3f} ({delta:+.0f}%)")


if __name__ == "__main__":
    main()
