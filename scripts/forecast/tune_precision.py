"""Sweep the ConvLSTM decision threshold for the best-F1 operating point and save a tightened forecast."""
import os
import sys

import numpy as np
import rasterio
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_forecast_stack as B

OUT = "outputs"
CKPT = "outputs/convlstm_forecast.pt"
K = 3
DEVICE = "cpu"


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid, k=3):
        super().__init__()
        self.hid = hid
        self.conv = nn.Conv2d(in_ch + hid, 4 * hid, k, padding=k // 2)

    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        i, f, o, g = i.sigmoid(), f.sigmoid(), o.sigmoid(), g.tanh()
        c = f * c + i * g
        return o * c.tanh(), c


class ConvLSTMForecaster(nn.Module):
    def __init__(self, in_ch, hids=(48, 64)):
        super().__init__()
        self.cells = nn.ModuleList()
        prev = in_ch
        for hid in hids:
            self.cells.append(ConvLSTMCell(prev, hid))
            prev = hid
        self.head = nn.Conv2d(prev, 1, 1)

    def forward(self, seq):
        batch, steps, _, h, w = seq.shape
        hs = [torch.zeros(batch, cell.hid, h, w) for cell in self.cells]
        cs = [torch.zeros(batch, cell.hid, h, w) for cell in self.cells]
        for t in range(steps):
            x = seq[:, t]
            for i, cell in enumerate(self.cells):
                hs[i], cs[i] = cell(x, hs[i], cs[i])
                x = hs[i]
        return self.head(x)


def scores(pred, true):
    hits = int((pred & true).sum())
    misses = int((~pred & true).sum())
    false_alarms = int((pred & ~true).sum())
    figure_of_merit = hits / max(hits + misses + false_alarms, 1)
    precision = hits / max(hits + false_alarms, 1)
    recall = hits / max(hits + misses, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return figure_of_merit, precision, recall, f1


def built(year, h, w):
    return np.clip(np.nan_to_num(B.align(B.built_path(year), 1, h, w)), 0, 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    series, grid, _ = B.build_series()
    transform, crs, H, W = grid
    steps, channels = series.shape[0], series.shape[1]
    ser = torch.from_numpy(series)
    print(f"series {series.shape}, loading {CKPT}")

    model = ConvLSTMForecaster(channels)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(model(ser[steps - 1 - K:steps - 1].unsqueeze(0)))[0, 0].numpy()
    b23 = built(2023, H, W)
    b24 = built(2024, H, W)
    open23 = b23 < 0.5
    new_true = (b24 > 0.5) & open23

    print(f"\n{'thr':>5} {'FoM':>6} {'prec':>6} {'recall':>6} {'F1':>6}")
    best_f1, best_thr = -1.0, 0.5
    sweep = []
    for thr in np.arange(0.30, 0.91, 0.05):
        pred = (prob > thr) & open23
        fom, precision, recall, f1 = scores(pred, new_true)
        sweep.append((thr, precision, recall, f1))
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
        print(f"{thr:5.2f} {fom:6.3f} {precision:6.3f} {recall:6.3f} {f1:6.3f}")
    print(f"\nbest-F1 threshold = {best_thr:.2f}")

    sweep = np.array(sweep)
    plt.figure(figsize=(7, 4))
    plt.plot(sweep[:, 0], sweep[:, 1], label="precision", marker="o")
    plt.plot(sweep[:, 0], sweep[:, 2], label="recall", marker="o")
    plt.plot(sweep[:, 0], sweep[:, 3], label="F1", marker="o")
    plt.axvline(best_thr, color="k", ls="--", lw=0.8, label=f"chosen {best_thr:.2f}")
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.legend()
    plt.title("ConvLSTM precision/recall vs threshold (held-out 2024)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/precision_sweep.png", dpi=140)
    plt.close()

    roll = [series[t] for t in range(steps)]
    built_now = b24.copy()
    forecasts = {}
    for year in range(2025, 2031):
        with torch.no_grad():
            change = torch.sigmoid(model(torch.from_numpy(np.stack(roll[-K:])).unsqueeze(0)))[0, 0].numpy()
        built_now = np.maximum(built_now, (change > best_thr).astype("float32"))
        frame = series[-1].copy()
        frame[0] = built_now
        roll.append(frame)
        forecasts[year] = built_now.copy()

    profile = dict(driver="GTiff", height=H, width=W, count=1, dtype="float32",
                   crs=crs, transform=transform, nodata=0)
    for year in (2027, 2030):
        new_area = ((forecasts[year] > 0.5) & (b24 < 0.5)).sum() * 1e-4
        print(f"{year}: +{new_area:.2f} km^2 new built-up")
        with rasterio.open(f"{OUT}/forecast_convlstm_{year}_tuned.tif", "w", **profile) as dst:
            dst.write(forecasts[year].astype("float32"), 1)

    print(f"wrote precision_sweep.png, forecast_convlstm_2027/2030_tuned.tif (threshold {best_thr:.2f})")


if __name__ == "__main__":
    main()
