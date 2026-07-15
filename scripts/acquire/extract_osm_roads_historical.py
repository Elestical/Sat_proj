"""

Rasterise historical OSM roads for the AOI on a past date via the ohsome API.

"""


import argparse
import io
import json

import numpy as np

#reads/writes vector data (the road geometries) with crs support
import geopandas as gpd
import requests
import rasterio
import rasterio.transform

#turns vector shapes into a raster mask
from rasterio.features import rasterize

#reprojects a bbox between crs's without touching the full dataset
from rasterio.warp import transform_bounds

AOI_GEOJSON = "config/aoi.geojson"
TARGET_CRS = "EPSG:32643"

#10m pixel grid, matching the satellite imagery resolution
PIXEL = 10.0

#ohsome: an api that can query osm data as it existed at a past date
OHSOME = "https://api.ohsome.org/v1/elements/geometry"

#width of the road around the center line
ROAD_BUFFER_M = 15.0


#reads the aoi bbox straight out of aoi.geojson so this script always matches the same region
def load_bbox():
    with open(AOI_GEOJSON) as f:
        return json.load(f)["bbox"]


#builds a pixel grid (transform + width/height) covering the bbox, snapped to whole pixels
def ref_grid_from_bbox(bbox):
    #bbox comes in as lon/lat (epsg:4326), reproject to meters first
    west, south, east, north = transform_bounds("EPSG:4326", TARGET_CRS, *bbox)

    #snap outward to whole pixel boundaries so the raster grid aligns cleanly
    west = np.floor(west / PIXEL) * PIXEL
    south = np.floor(south / PIXEL) * PIXEL
    east = np.ceil(east / PIXEL) * PIXEL
    north = np.ceil(north / PIXEL) * PIXEL
    width = int(round((east - west) / PIXEL))
    height = int(round((north - south) / PIXEL))
    transform = rasterio.transform.from_origin(west, north, PIXEL, PIXEL)
    return transform, width, height


#queries ohsome for every osm road (highway=*) that existed at the given historical date
def fetch_ohsome_roads(bbox, date):
    w, s, e, n = bbox
    params = {
        "bboxes": f"{w},{s},{e},{n}",
        "filter": "highway=* and type:way",
        "time": date,
        "properties": "tags",
    }
    print(f"Querying ohsome for highway=* at {date} ...")
    response = requests.post(OHSOME, data=params, timeout=180)
    response.raise_for_status()

    #ohsome returns geojson bytes, read straight into a geodataframe
    gdf = gpd.read_file(io.BytesIO(response.content))
    print(f"  features returned: {len(gdf)}")
    return gdf


def main():
    parser = argparse.ArgumentParser()
    #which historical snapshot date to pull roads for, defaults to the 2017 baseline year
    parser.add_argument("--date", default="2017-01-01", help="ISO-8601 snapshot date")
    args = parser.parse_args()
    tag = args.date[:4]

    bbox = load_bbox()
    gdf = fetch_ohsome_roads(bbox, args.date)

    #osm data comes back in wgs84, reproject to the project's meters-based crs
    gdf = gdf.set_crs("EPSG:4326").to_crs(TARGET_CRS)

    transform, width, height = ref_grid_from_bbox(bbox)

    #buffer each road line into a polygon strip so it rasterizes to more than a 1px-wide line
    shapes = [(geom.buffer(ROAD_BUFFER_M), 1) for geom in gdf.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    print(f"  road mask {width}x{height}, road px = {int(mask.sum())}")

    profile = dict(driver="GTiff", dtype="uint8", count=1, height=height,
                   width=width, crs=TARGET_CRS, transform=transform, compress="deflate")
    mask_path = f"outputs/osm_roads_{tag}_mask.tif"
    geo_path = f"outputs/osm_roads_{tag}.geojson"

    #write both the rasterized mask (for model input/comparison) and the raw road vectors (for reference)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask[None])
    gdf[["geometry"]].to_file(geo_path, driver="GeoJSON")
    print(f"  wrote {mask_path} and {geo_path}")


if __name__ == "__main__":
    main()
