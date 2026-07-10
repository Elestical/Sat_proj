"""Train a road-segmentation U-Net on patch folders (BCE + Dice loss, best val IoU checkpointed).

    python scripts/train.py --model scratch --data data/patches/local --epochs 40 --out outputs/road_model.pt
    python scripts/train.py --model scratch --data data/patches/local --epochs 1 --overfit
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.unet import build_model

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class PatchDS(Dataset):
    def __init__(self, folder, in_channels=3, augment=False):
        self.files = sorted(glob.glob(f"{folder}/*.npz"))
        self.channels = in_channels
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img, mask = img[:, :, ::-1], mask[:, ::-1]
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1, :], mask[::-1, :]
        k = np.random.randint(4)
        if k:
            img = np.rot90(img, k, axes=(1, 2))
            mask = np.rot90(mask, k)
        if np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.85, 1.15), 0, 1)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, i):
        data = np.load(self.files[i])
        img, mask = data["img"][:self.channels], data["mask"]
        if self.augment:
            img, mask = self._augment(img, mask)
        return (torch.from_numpy(img.astype("float32")),
                torch.from_numpy(mask.astype("float32"))[None])


def dice_bce_loss(logits, target, eps=1.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        p = (torch.sigmoid(model(x.to(DEVICE))) > 0.5).float().cpu()
        tp += float((p * y).sum())
        fp += float((p * (1 - y)).sum())
        fn += float(((1 - p) * y).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return iou, f1


def overfit_one_batch(model, opt, loader):
    x, y = next(iter(loader))
    x, y = x.to(DEVICE), y.to(DEVICE)
    for i in range(60):
        opt.zero_grad()
        loss = dice_bce_loss(model(x), y)
        loss.backward()
        opt.step()
        if i % 15 == 0 or i == 59:
            print(f"  overfit step {i:3d}  loss {loss.item():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["scratch", "smp"], default="scratch")
    parser.add_argument("--data", required=True, help="folder with train/ and val/ subdirs")
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default=None, help="checkpoint to warm-start from")
    parser.add_argument("--out", default="outputs/model.pt")
    parser.add_argument("--overfit", action="store_true", help="fit one batch as a sanity check")
    args = parser.parse_args()
    if args.model == "smp":
        args.in_channels = 3

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = build_model(args.model, args.in_channels, encoder_weights="imagenet").to(DEVICE)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=DEVICE))
        print(f"warm-started from {args.init}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"device={DEVICE} model={args.model} in_ch={args.in_channels} params={params:.1f}M")

    train_ds = PatchDS(f"{args.data}/train", args.in_channels, augment=not args.overfit)
    val_ds = PatchDS(f"{args.data}/val", args.in_channels)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    print(f"train {len(train_ds)} patches, val {len(val_ds)} patches")

    if args.overfit:
        overfit_one_batch(model, opt, train_loader)
        return

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = dice_bce_loss(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        iou, f1 = evaluate(model, val_loader)
        flag = ""
        if iou > best:
            best = iou
            torch.save(model.state_dict(), args.out)
            flag = "  <- best, saved"
        print(f"epoch {epoch:3d}  loss {total/len(train_ds):.4f}  val IoU {iou:.3f}  F1 {f1:.3f}{flag}")
    print(f"best val IoU {best:.3f}  ->  {args.out}")


if __name__ == "__main__":
    main()
