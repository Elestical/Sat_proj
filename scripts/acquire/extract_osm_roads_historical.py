"""
Rasterise historical OSM roads for the AOI on a past date via the ohsome API.
"""


import argparse
import io
import json

import numpy as np
import geopandas as gpd
import requests
import rasterio
import rasterio.transform
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

AOI_GEOJSON = "config/aoi.geojson"
TARGET_CRS = "EPSG:32643"
PIXEL = 10.0
OHSOME = "https://api.ohsome.org/v1/elements/geometry"
ROAD_BUFFER_M = 15.0


def load_bbox():
    with open(AOI_GEOJSON) as f:
        return json.load(f)["bbox"]


def ref_grid_from_bbox(bbox):
    west, south, east, north = transform_bounds("EPSG:4326", TARGET_CRS, *bbox)
    west = np.floor(west / PIXEL) * PIXEL
    south = np.floor(south / PIXEL) * PIXEL
    east = np.ceil(east / PIXEL) * PIXEL
    north = np.ceil(north / PIXEL) * PIXEL
    width = int(round((east - west) / PIXEL))
    height = int(round((north - south) / PIXEL))
    transform = rasterio.transform.from_origin(west, north, PIXEL, PIXEL)
    return transform, width, height


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
    gdf = gpd.read_file(io.BytesIO(response.content))
    print(f"  features returned: {len(gdf)}")
    return gdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2017-01-01", help="ISO-8601 snapshot date")
    args = parser.parse_args()
    tag = args.date[:4]

    bbox = load_bbox()
    gdf = fetch_ohsome_roads(bbox, args.date)
    gdf = gdf.set_crs("EPSG:4326").to_crs(TARGET_CRS)

    transform, width, height = ref_grid_from_bbox(bbox)
    shapes = [(geom.buffer(ROAD_BUFFER_M), 1) for geom in gdf.geometry if geom is not None]
    mask = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    print(f"  road mask {width}x{height}, road px = {int(mask.sum())}")

    profile = dict(driver="GTiff", dtype="uint8", count=1, height=height,
                   width=width, crs=TARGET_CRS, transform=transform, compress="deflate")
    mask_path = f"outputs/osm_roads_{tag}_mask.tif"
    geo_path = f"outputs/osm_roads_{tag}.geojson"
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask[None])
    gdf[["geometry"]].to_file(geo_path, driver="GeoJSON")
    print(f"  wrote {mask_path} and {geo_path}")


if __name__ == "__main__":
    main()
