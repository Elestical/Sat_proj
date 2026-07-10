# Satellite Urbanisation Pipeline

From raw Sentinel-2 imagery to a road-detection model, a quantified map of how infrastructure
grew, and a forecast of where it grows next. Worked example: the Devanahalli / KIA airport
corridor, Bengaluru. The area of interest is a single config file.

## Pipeline

| Stage | Scripts | Output |
|---|---|---|
| Acquire | `acquire/` | Sentinel-2 stacks, Dynamic World built-up, OSM road masks |
| Preprocess | `preprocess/` | one common 10 m grid; patches for training |
| Road model | `roads/` | U-Net road masks per year (DeepGlobe pretrain + Sentinel fine-tune) |
| Growth | `growth/` | per-1 km-cell growth, expansion vs infill, Getis-Ord hotspots |
| Forecast | `forecast/` + `notebooks/forecast_convlstm_colab.ipynb` | ConvLSTM built-up forecast (2027/2030) |

Change detection (`notebooks/change_detection_colab.ipynb`) predicts new-since-2017 roads directly.

## Structure

```
config/        aoi.geojson                 area of interest
models/        unet.py                     from-scratch U-Net
scripts/       acquire preprocess roads growth forecast
notebooks/     road pipeline, change detection, ConvLSTM forecaster (Colab)
data/ outputs/ downloads + results (gitignored)
```
