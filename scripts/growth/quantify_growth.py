"""Report new road length (km) and area (km^2) between 2017 and 2024, cross-checked against OSM.

    python scripts/quantify_growth.py --pred2017 outputs/pred_2017.tif --pred2024 outputs/pred_2024.tif
"""
import argparse

import rasterio
from skimage.morphology import skeletonize

OUT = "outputs"
OSM17 = f"{OUT}/osm_roads_2017_mask.tif"
OSM24 = f"{OUT}/osm_roads_2024_mask.tif"
PIX_M = 10.0


def read(path):
    with rasterio.open(path) as s:
        return s.read(1)


def km_length(mask):
    return int(skeletonize(mask).sum()) * PIX_M / 1000.0


def km2_area(mask):
    return int(mask.sum()) * (PIX_M ** 2) / 1e6


def report(tag, m17, m24):
    new = m24 & ~m17
    print(f"\n[{tag}]")
    print(f"  2017 road area  {km2_area(m17):7.2f} km^2   length {km_length(m17):7.1f} km")
    print(f"  2024 road area  {km2_area(m24):7.2f} km^2   length {km_length(m24):7.1f} km")
    print(f"  new since 2017  {km2_area(new):7.2f} km^2   length {km_length(new):7.1f} km")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred2017", required=True)
    parser.add_argument("--pred2024", required=True)
    parser.add_argument("--thr", type=float, default=0.5)
    args = parser.parse_args()

    p17 = read(args.pred2017) > args.thr
    p24 = read(args.pred2024) > args.thr
    report(f"U-Net prediction (thr {args.thr})", p17, p24)

    o17 = read(OSM17) == 1
    o24 = read(OSM24) == 1
    report("OSM ground-truth", o17, o24)

    iou = int((p24 & o24).sum()) / (int((p24 | o24).sum()) + 1e-6)
    print(f"\nmodel-vs-OSM 2024 road IoU: {iou:.3f}")


if __name__ == "__main__":
    main()
