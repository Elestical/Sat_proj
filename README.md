# Satellite Urbanisation Pipeline

From raw Sentinel-2 imagery to a road-detection model, a quantified map of how infrastructure
grew, and a forecast of where it grows next. Worked example: the Devanahalli / KIA airport
corridor, Bengaluru. The area of interest is a single config file.

## Pipeline

| Stage | Scripts | Output |
|---|---|---|
| Acquire | `acquire/` | Sentinel-2 stacks, Dynamic World built-up, OSM road masks |
| Preprocess | `preprocess/` | one common 10 m grid; patches for training |
| Road model | `roads/` | U-Net road masks per year (DeepGlobe pretrain + Sentinel fine-tune); `eval_cross_city.py` scores transfer to an unseen city |
| Growth | `growth/` | per-1 km-cell growth, expansion vs infill, Getis-Ord hotspots |
| Forecast | `forecast/` + `notebooks/forecast_convlstm_colab.ipynb` | ConvLSTM built-up forecast (2027/2030); `tune_precision.py` picks the F1-optimal threshold |
| Dashboard | `dashboard/` | interactive Leaflet map; click a 1 km cell for its SHAP driver contributions and forecast |

Change detection (`notebooks/change_detection_colab.ipynb`) predicts new-since-2017 roads directly.

The dashboard reads `cells.geojson` from `build_cell_explanations.py` (per-cell suitability + SHAP)
and layers the built-up, road, new-road and forecast rasters into a self-contained `dashboard.html`.

## Structure

```
config/        aoi.geojson                 area of interest
models/        unet.py                     from-scratch U-Net
scripts/       acquire preprocess roads growth forecast dashboard
notebooks/     road pipeline, change detection, ConvLSTM forecaster (Colab)
data/ outputs/ downloads + results (gitignored)
```
