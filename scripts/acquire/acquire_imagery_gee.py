"""
Downloads cloud-masked, dry-season Sentinel-2 median composites for 2017, 2019, and 2024 from Earth Engine.

Uses a single sensor collection for all years (COPERNICUS/S2_SR_HARMONIZED, 10 m resolution).
The harmonized collection keeps reflectance values on a consistent scale across the 2022
processing-baseline change, so composites from different years remain directly comparable.

"""

import os

#gee python api
import ee

#gompanion to the ee library
import geemap

EE_PROJECT = os.environ.get("GEE_PROJECT", "temp-ee-project")
AOI_BBOX = [77.62, 13.14, 77.75, 13.25]

#november to march windows (dry season)

WINDOWS = {
    2017: ("2017-11-01", "2018-03-15"),
    2019: ("2018-12-01", "2019-03-31"),
    2024: ("2024-11-01", "2025-03-15"),
}
TARGET_CRS = "EPSG:32643"

#10m/pixel
SCALE = 10
#to be multiplied by surface reflectance of sentinel-2
REFL_SCALE = 0.0001
#cloud cover threshold of 20%
MAX_SCENE_CLOUD = 20

#checks the earth engine session
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
        #scene classification layer has each pixel in category code like "vegetation", "water", "cloud","cloud shadow"
        scl = img.select("SCL")
        clear = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)).Not()
        #pixel doesnt have cloud/shadow/cirrus/snow

        # B2,B3,B4,B8,B11 -> blue, green, red, near infra red(nir), short wave infrared (swir)
        sr = img.select(["B2", "B3", "B4", "B8", "B11"]).multiply(REFL_SCALE)
        return (sr.updateMask(clear)
                .addBands(clear.rename("clearmask").toFloat())
                .copyProperties(img, ["system:time_start"]))
    #update mask invalidates the pixels that violate the criteria
    #adds the mask as a band
    #properties measure the timestamp metadata

    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD)) #drops the whole scene with 20% or more cloud cover
           .map(mask_and_scale))  #applied the per pizel cleaning function described above
    
    print(f"  {start[:4]} Sentinel-2 candidate scenes: {col.size().getInfo()}")

    composite = col.median()
    #takes the per pixel median across all could-masked scenes

    refl = composite.select(["B2", "B3", "B4", "B8", "B11"],
                            ["blue", "green", "red", "nir", "swir1"])
    
    #renames the bands and also attaches the mask as the 6th band
    return refl.addBands(composite.select("clearmask")).clip(region)


def cloudiest_scene_truecolor(start, end, region, rgb_bands=("B4", "B3", "B2")):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    worst = ee.Image(col.sort("CLOUDY_PIXEL_PERCENTAGE", False).first())
    return worst.select(list(rgb_bands)).multiply(REFL_SCALE).clip(region)

    #picking the cloudies day we could find, with cloud coverage of greater than 60%,
    # and extract its true colors

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
