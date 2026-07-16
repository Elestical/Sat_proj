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

## Setup

1. Python 3.10+ with a virtual environment:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Google Earth Engine (for the acquire stage): sign up at earthengine.google.com, then

   ```
   earthengine authenticate
   export GEE_PROJECT=your-ee-project-id
   ```

3. Kaggle (optional, only for the DeepGlobe pretraining download): place your API token at
   `~/.kaggle/kaggle.json`.

4. GPU steps (road-model training, change detection, the ConvLSTM forecaster) run in the
   Colab notebooks under `notebooks/`; everything else runs locally on CPU. Run every
   command below from the repository root; results land in `outputs/`.

## Run

```
# 1. acquire: Sentinel-2 composites, Dynamic World built-up series, OSM road labels
python scripts/acquire/acquire_imagery_gee.py
python scripts/acquire/acquire_dw_annual.py
python scripts/acquire/extract_osm_roads_historical.py --date 2017-08-01   # repeat for 2019, 2024

# 2. preprocess: one shared 10 m grid, 7-band stacks, training patches
python scripts/preprocess/preprocess.py --aoi config/aoi.geojson \
    --y2017 raw_2017.tif --y2019 raw_2019.tif --y2024 raw_2024.tif --out outputs
python scripts/preprocess/prepare_deepglobe.py --src data/deepglobe/train   # optional pretrain set
python scripts/preprocess/prepare_local_patches.py

# 3. road model: train (local or Colab), then predict each year
python scripts/roads/train.py --model scratch --data data/patches/local --epochs 40 --out outputs/road_model.pt
python scripts/roads/infer.py --stack outputs/stack_2017.tif --ckpt outputs/road_model.pt --out outputs/pred_2017.tif
python scripts/roads/infer.py --stack outputs/stack_2024.tif --ckpt outputs/road_model.pt --out outputs/pred_2024.tif
python scripts/roads/eval.py --data data/patches/local/val scratch=outputs/road_model.pt

# 4. growth: headline numbers, 1 km cells, growth type, hotspots
python scripts/growth/quantify_growth.py --pred2017 outputs/pred_2017.tif --pred2024 outputs/pred_2024.tif
python scripts/growth/render_predictions.py --pred2017 outputs/pred_2017.tif --pred2024 outputs/pred_2024.tif
python scripts/growth/growth_zonal.py
python scripts/growth/growth_classify.py
python scripts/growth/growth_hotspots.py

# 5. forecast: check inputs, train the ConvLSTM in Colab, then tighten and report
python scripts/forecast/build_forecast_stack.py --check
#   -> run notebooks/forecast_convlstm_colab.ipynb (Colab GPU), download its outputs into outputs/
python scripts/forecast/tune_precision.py
python scripts/forecast/forecast_report.py

# 6. dashboard: per-cell SHAP explanations, then the interactive map
python scripts/dashboard/build_cell_explanations.py
python scripts/dashboard/build_dashboard.py
open outputs/dashboard/dashboard.html
```
