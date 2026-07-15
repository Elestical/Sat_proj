"""
Downloads a yearly Dynamic World built-up probability series [2017-2024] to outputs/ .
"""

import argparse
import os

#gee python api
import ee

#companion to the ee library
import geemap

EE_PROJECT = os.environ.get("GEE_PROJECT", "temp-ee-project")
AOI_BBOX = [77.62, 13.14, 77.75, 13.25]
TARGET_CRS = "EPSG:32643"

#10m/pixel
SCALE = 10

#dynamic world: google's built-up probability dataset
DW = "GOOGLE/DYNAMICWORLD/V1"
OUT = "outputs"


#returns the same nov-mar dry season window used for the sentinel-2 pipeline, for a given year
def window(year):
    return f"{year}-11-01", f"{year + 1}-03-15"


#checks the earth engine session
def init():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def built_prob(start, end, region):
    #filters dynamic world to the aoi/date window, keeping only the "built" probability band
    #dynamic world is also a unet
    #outputs land cover class probabilities per pixel (water, trees crops, built-up, ....)
    col = ee.ImageCollection(DW).filterBounds(region).filterDate(start, end).select("built")
    print(f"  {start[:4]} Dynamic World scenes: {col.size().getInfo()}")

    #per pixel mean built-probability across all scenes in the window
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
    #lets the years be overridden from the command line, defaults to the full 2017-2024 span
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2017, 2025)))
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    init()
    region = ee.Geometry.Rectangle(AOI_BBOX)
    for year in args.years:
        out = f"{OUT}/dw_built_{year}.tif"
        if os.path.exists(out):
            #skip years already downloaded so reruns don't redo finished work
            print(f"  skip {year} (exists)")
            continue
        download_local(built_prob(*window(year), region), out, region)


if __name__ == "__main__":
    main()
