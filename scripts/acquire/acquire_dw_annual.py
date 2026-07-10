"""Downloads a yearly Dynamic World built-up probability series (2017-2024) to outputs/."""

import argparse
import os

import ee
import geemap

EE_PROJECT = os.environ.get("GEE_PROJECT", "your-ee-project")
AOI_BBOX = [77.62, 13.14, 77.75, 13.25]
TARGET_CRS = "EPSG:32643"
SCALE = 10
DW = "GOOGLE/DYNAMICWORLD/V1"
OUT = "outputs"


def window(year):
    return f"{year}-11-01", f"{year + 1}-03-15"


def init():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def built_prob(start, end, region):
    col = ee.ImageCollection(DW).filterBounds(region).filterDate(start, end).select("built")
    print(f"  {start[:4]} Dynamic World scenes: {col.size().getInfo()}")
    return col.mean().clip(region)


def download_local(image, filename, region):
    image = image.toFloat()
    try:
        geemap.download_ee_image(image, filename, region=region, crs=TARGET_CRS, scale=SCALE)
    except AttributeError:
        geemap.ee_export_image(image, filename=filename, scale=SCALE, region=region, crs=TARGET_CRS)
    print(f"  downloaded: {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2017, 2025)))
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    init()
    region = ee.Geometry.Rectangle(AOI_BBOX)
    for year in args.years:
        out = f"{OUT}/dw_built_{year}.tif"
        if os.path.exists(out):
            print(f"  skip {year} (exists)")
            continue
        download_local(built_prob(*window(year), region), out, region)


if __name__ == "__main__":
    main()
