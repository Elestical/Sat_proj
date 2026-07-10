"""
Download cloud-masked dry-season Sentinel-2 median composites (2017/2019/2024) from Earth Engine.

Single sensor for all years (COPERNICUS/S2_SR_HARMONIZED at 10 m)
the harmonized collection keeps reflectance on one scale across the 2022 processing-baseline change.

"""

import os

import ee
import geemap

EE_PROJECT = os.environ.get("GEE_PROJECT", "your-ee-project")
AOI_BBOX = [77.62, 13.14, 77.75, 13.25]
WINDOWS = {
    2017: ("2017-11-01", "2018-03-15"),
    2019: ("2018-12-01", "2019-03-31"),
    2024: ("2024-11-01", "2025-03-15"),
}
TARGET_CRS = "EPSG:32643"
SCALE = 10
REFL_SCALE = 0.0001
MAX_SCENE_CLOUD = 20


def init():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def aoi():
    return ee.Geometry.Rectangle(AOI_BBOX)


def sentinel2_sr(start, end, region):
    def mask_and_scale(img):
        scl = img.select("SCL")
        clear = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)).Not()
        sr = img.select(["B2", "B3", "B4", "B8", "B11"]).multiply(REFL_SCALE)
        return (sr.updateMask(clear)
                .addBands(clear.rename("clearmask").toFloat())
                .copyProperties(img, ["system:time_start"]))

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD))
           .map(mask_and_scale))
    print(f"  {start[:4]} Sentinel-2 candidate scenes: {col.size().getInfo()}")
    composite = col.median()
    refl = composite.select(["B2", "B3", "B4", "B8", "B11"],
                            ["blue", "green", "red", "nir", "swir1"])
    return refl.addBands(composite.select("clearmask")).clip(region)


def cloudiest_scene_truecolor(start, end, region, rgb_bands=("B4", "B3", "B2")):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    worst = ee.Image(col.sort("CLOUDY_PIXEL_PERCENTAGE", False).first())
    return worst.select(list(rgb_bands)).multiply(REFL_SCALE).clip(region)


def download_local(image, filename, region):
    image = image.toFloat()
    try:
        geemap.download_ee_image(image, filename, region=region, crs=TARGET_CRS, scale=SCALE)
    except AttributeError:
        geemap.ee_export_image(image, filename=filename, scale=SCALE, region=region, crs=TARGET_CRS)
    print(f"  downloaded: {filename}")


def main():
    init()
    region = aoi()
    for year in (2017, 2019, 2024):
        download_local(sentinel2_sr(*WINDOWS[year], region), f"raw_{year}.tif", region)

    start, end = WINDOWS[2024]
    download_local(cloudiest_scene_truecolor(start, end, region), "cloudy_2024.tif", region)
    download_local(sentinel2_sr(start, end, region).select(["red", "green", "blue"]),
                   "clean_2024.tif", region)


if __name__ == "__main__":
    main()
