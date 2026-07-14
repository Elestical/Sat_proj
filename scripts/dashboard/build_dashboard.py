"""Smart Map Dashboard: every layer on one interactive map, click a cell to see why it grew.

Renders the built-up, road, new-road and forecast layers as image overlays, true-colour Sentinel-2
imagery for each period as switchable backdrops, and embeds the clickable 1 km cell grid
(cells.geojson from build_cell_explanations.py). Writes a self-contained dashboard.html with two
panel tabs: Explore (per-cell SHAP explanations) and Pipeline (how the whole system works).

    python scripts/dashboard/build_dashboard.py
"""
import json
import os

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/dashboard"
LAYER_DIR = f"{OUT}/layers"
CELLS = "outputs/cells.geojson"

LAYERS = [
    ("built",    "Built-up 2024",       "outputs/dw_built_2024.tif",                "built"),
    ("roads",    "Road network 2024",   "outputs/pred_2024.tif",                    "mask"),
    ("newroads", "New roads 2017-2024", None,                                       "newroad"),
    ("forecast", "Forecast 2030",       "outputs/forecast_convlstm_2030_tuned.tif", "forecast"),
]

SAT_YEARS = [2017, 2019, 2024]
RGB_BANDS = (3, 2, 1)


def read(path, band=1):
    with rasterio.open(path) as s:
        a = s.read(band).astype("float32")
        nodata = s.nodata
    a[~np.isfinite(a)] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    if nodata is not None:
        a[a == np.float32(nodata)] = np.nan
    return a


def read_geo(path, band=1):
    with rasterio.open(path) as s:
        return read(path, band), s.transform, s.crs, s.height, s.width


def to_wgs84(arr, transform, crs, h, w):
    left, bottom, right, top = array_bounds(h, w, transform)
    dst_transform, dw, dh = calculate_default_transform(crs, "EPSG:4326", w, h, left, bottom, right, top)
    dst = np.full((dh, dw), np.nan, "float32")
    reproject(arr, dst, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs="EPSG:4326",
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    west, south, east, north = array_bounds(dh, dw, dst_transform)
    return dst, (south, west, north, east)


def colour(dst, kind):
    h, w = dst.shape
    rgba = np.zeros((h, w, 4), "float32")
    v = np.nan_to_num(dst)
    if kind == "mask":
        rgba[v > 0.5] = [0.20, 0.45, 0.90, 0.75]
    elif kind == "newroad":
        rgba[v > 0.5] = [0.90, 0.15, 0.15, 0.9]
    elif kind == "forecast":
        rgba[v > 0.5] = [0.55, 0.15, 0.75, 0.7]
    elif kind == "built":
        rgba[v > 0.5] = [0.85, 0.55, 0.20, 0.45]
    return np.clip(rgba, 0, 1)


def new_roads():
    p17, t, crs, h, w = read_geo("outputs/pred_2017.tif")
    p24 = read("outputs/pred_2024.tif")
    arr = ((np.nan_to_num(p24) > 0.5) & (np.nan_to_num(p17) <= 0.5)).astype("float32")
    return arr, t, crs, h, w


def satellite_rgb(year):
    """True-colour Sentinel-2 for one year, reprojected to WGS84 as an RGBA image."""
    path = f"outputs/stack_{year}.tif"
    with rasterio.open(path) as s:
        transform, crs, h, w = s.transform, s.crs, s.height, s.width
    planes = []
    for band in RGB_BANDS:
        dst, bounds = to_wgs84(read(path, band), transform, crs, h, w)
        planes.append(dst)
    rgb = np.stack(planes, axis=-1)
    alpha = np.isfinite(rgb).all(axis=-1).astype("float32")
    rgb = np.nan_to_num(np.clip(rgb, 0, 1))
    # mild brightness lift so the composite is legible under the analysis overlays
    rgb = np.clip(rgb * 1.25, 0, 1)
    return np.dstack([rgb, alpha]), bounds


def main():
    os.makedirs(LAYER_DIR, exist_ok=True)

    satellites = []
    for year in SAT_YEARS:
        if not os.path.exists(f"outputs/stack_{year}.tif"):
            continue
        rgba, (south, west, north, east) = satellite_rgb(year)
        plt.imsave(f"{LAYER_DIR}/sat_{year}.png", rgba)
        satellites.append({"id": f"sat{year}", "label": f"Satellite {year}",
                           "file": f"layers/sat_{year}.png",
                           "bounds": [[south, west], [north, east]]})
        print(f"  satellite {year}")

    overlays = []
    for lid, label, path, kind in LAYERS:
        arr, t, crs, h, w = new_roads() if lid == "newroads" else read_geo(path)
        dst, (south, west, north, east) = to_wgs84(arr, t, crs, h, w)
        plt.imsave(f"{LAYER_DIR}/{lid}.png", colour(dst, kind))
        overlays.append({"id": lid, "label": label, "file": f"layers/{lid}.png",
                         "bounds": [[south, west], [north, east]],
                         "default": lid in ("built", "newroads")})
        print(f"  overlay {lid}")

    cells = json.load(open(CELLS))
    write_html(overlays, satellites, cells)
    print(f"wrote {OUT}/dashboard.html  ({len(cells['features'])} clickable cells, "
          f"{len(satellites)} satellite periods)")


def write_html(overlays, satellites, cells):
    lats = [c for f in cells["features"] for lon, c in f["geometry"]["coordinates"][0]]
    lons = [lon for f in cells["features"] for lon, c in f["geometry"]["coordinates"][0]]
    cy, cx = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    html = (TEMPLATE.replace("__OVERLAYS__", json.dumps(overlays))
                    .replace("__SATELLITES__", json.dumps(satellites))
                    .replace("__CELLS__", json.dumps(cells))
                    .replace("__CY__", str(cy)).replace("__CX__", str(cx)))
    with open(f"{OUT}/dashboard.html", "w") as f:
        f.write(html)


TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Devanahalli Corridor - Smart Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{height:100%;margin:0;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
  #map{position:absolute;top:0;left:0;right:360px;bottom:0}
  #panel{position:absolute;top:0;right:0;width:360px;bottom:0;background:#14161c;color:#e8ecf1;
    overflow-y:auto;padding:0 0 20px;box-sizing:border-box;box-shadow:-2px 0 12px rgba(0,0,0,.4)}
  .head{padding:16px 18px 0}
  #panel h2{margin:0 0 2px;font-size:17px}
  #panel .sub{color:#9fb2c6;font-size:12px;margin-bottom:12px}
  .tabs{display:flex;gap:4px;padding:0 18px;border-bottom:1px solid #23262f;position:sticky;top:0;
    background:#14161c;z-index:5}
  .tab{padding:8px 12px;font-size:12px;font-weight:600;color:#8aa0b6;cursor:pointer;
    border-bottom:2px solid transparent;user-select:none}
  .tab:hover{color:#c7d2de}
  .tab.on{color:#fff;border-bottom-color:#d7451c}
  .body{padding:14px 18px}
  .tag{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12px;font-weight:600;color:#fff}
  .reason{background:#1c1f27;border-radius:8px;padding:10px 12px;font-size:13px;line-height:1.5;margin:12px 0}
  .sec{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#8aa0b6;margin:16px 0 6px}
  .bar{display:flex;align-items:center;height:22px;margin:3px 0;font-size:12px}
  .bar .name{width:118px;flex:0 0 auto;color:#c7d2de}
  .bar .track{flex:1;position:relative;height:14px}
  .bar .fill{position:absolute;top:0;height:14px;border-radius:3px}
  .bar .val{width:44px;flex:0 0 auto;text-align:right;color:#9fb2c6;font-variant-numeric:tabular-nums}
  .stat{display:flex;justify-content:space-between;font-size:13px;padding:4px 0;border-bottom:1px solid #23262f}
  .stat .k{color:#9fb2c6}
  #hint{color:#7f93a8;font-size:13px;margin-top:30px;text-align:center}
  .leaflet-control-layers{font-size:13px}
  .info{display:inline-block;width:13px;height:13px;line-height:13px;text-align:center;border-radius:50%;
    background:#2b3140;color:#9fb2c6;font-size:9px;font-weight:700;font-style:italic;margin-left:5px;
    cursor:help;position:relative;vertical-align:middle}
  #tip{position:fixed;display:none;max-width:230px;background:#0a0c10;color:#dfe6ee;padding:8px 10px;
    border-radius:6px;font-size:11px;font-weight:400;font-style:normal;line-height:1.45;z-index:1000;
    box-shadow:0 3px 14px rgba(0,0,0,.6);pointer-events:none;white-space:normal}
  .stage{background:#1a1d25;border:1px solid #23262f;border-radius:8px;margin:8px 0;overflow:hidden}
  .stage.on{border-color:#d7451c}
  .stage .top{display:flex;align-items:center;gap:10px;padding:10px 12px;cursor:pointer;user-select:none}
  .stage .top:hover{background:#1f232c}
  .stage .num{flex:0 0 22px;height:22px;line-height:22px;text-align:center;border-radius:50%;
    background:#2b3140;color:#c7d2de;font-size:11px;font-weight:700}
  .stage.on .num{background:#d7451c;color:#fff}
  .stage .t{flex:1;font-size:13px;font-weight:600}
  .stage .t small{display:block;font-weight:400;color:#8aa0b6;font-size:11px;margin-top:1px}
  .stage .chev{color:#6b7c8f;font-size:11px}
  .stage .detail{display:none;padding:0 12px 12px;font-size:12px;line-height:1.55;color:#c7d2de}
  .stage.on .detail{display:block}
  .stage .detail .kv{display:flex;justify-content:space-between;padding:3px 0;
    border-bottom:1px solid #23262f;font-size:11px}
  .stage .detail .kv span:first-child{color:#8aa0b6}
  .stage .detail .out{margin-top:8px;font-size:11px;color:#8aa0b6}
  .flow{font-size:11px;color:#8aa0b6;line-height:1.6;background:#1c1f27;border-radius:8px;
    padding:10px 12px;margin-bottom:6px}
</style></head>
<body>
<div id="map"></div>
<div id="panel">
  <div class="head">
    <h2>Devanahalli corridor</h2>
    <div class="sub">Sentinel-2, 2017 to 2030 forecast.</div>
  </div>
  <div class="tabs">
    <div class="tab on" data-tab="explore">Explore</div>
    <div class="tab" data-tab="pipeline">Pipeline</div>
  </div>
  <div class="body">
    <div id="explore">
      <div id="detail"><div id="hint">No cell selected.<br>Click a cell on the map.</div></div>
    </div>
    <div id="pipeline" style="display:none"></div>
  </div>
</div>
<script>
const OVERLAYS = __OVERLAYS__;
const SATELLITES = __SATELLITES__;
const CELLS = __CELLS__;
const NAMES = {prox_road:"Road proximity", built_density:"Built density", prox_built:"Built edge",
               prox_airport:"Airport proximity", ndvi:"Vegetation (NDVI)"};
const TYPE_COLOR = {expansion:"#d7451c", densification:"#e0902a", stable:"#5a6b7a"};
const TIPS = {
  shap:"SHAP shows how much each driver pushed this cell's predicted development up or down, in the model's own units.",
  growth_type:"Expansion = greenfield land that became built. Densification = already-built land that grew. Stable = little change.",
  suitability:"The model's probability (0 to 1) that this open cell develops.",
  prox_road:"How close the cell is to the road network. Road proximity is the strongest growth predictor.",
  built_density:"How built-up the cell's neighbourhood already is; growth clusters near existing development.",
  prox_built:"Proximity to the edge of existing built-up land, where expansion tends to start.",
  prox_airport:"How close the cell is to KIA airport, the corridor's growth engine.",
  ndvi:"Vegetation greenness. Open or lightly vegetated land is available to develop.",
  new_road_km:"Length of road built here between 2017 and 2024 (centreline km).",
  new_built_km2:"Area that turned from non-built to built between 2017 and 2024.",
  forecast_km2:"Predicted new built-up area in this cell by 2030 (ConvLSTM forecast).",
  hotspot_z:"Getis-Ord Gi* z-score. Above +1.96 is a statistically significant growth cluster (95%).",
  cell:"The cell's row and column in the 1 km grid."};
function info(k){ return TIPS[k] ? `<span class="info" data-tip="${TIPS[k]}">i</span>` : ''; }

const map = L.map('map', {center:[__CY__, __CX__], zoom:13});
const basemap = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
  {attribution:'&copy; OpenStreetMap &copy; CARTO', maxZoom:19}).addTo(map);

// Satellite imagery per period, switchable (radio). "Map only" leaves the dark basemap bare.
const satLayers = {};
const bases = {"Map only": L.layerGroup().addTo(map)};
SATELLITES.forEach(s => {
  const layer = L.imageOverlay(s.file, s.bounds, {opacity:1});
  satLayers[s.id] = layer;
  bases[s.label] = layer;
});

const overlayLayers = {};
const overlayDict = {};
OVERLAYS.forEach(o => {
  const layer = L.imageOverlay(o.file, o.bounds, {opacity:0.85});
  if (o.default) layer.addTo(map);
  overlayLayers[o.id] = layer;
  overlayDict[o.label] = layer;
});

let selected = null;
function cellStyle(f){return {color:"#ffffff", weight:0.6, opacity:0.35, fillColor:TYPE_COLOR[f.properties.growth_type], fillOpacity:0.12};}
const cellLayer = L.geoJSON(CELLS, {
  style: cellStyle,
  onEachFeature: (f, lyr) => {
    lyr.on('mouseover', () => { if(lyr!==selected) lyr.setStyle({weight:1.6, opacity:0.9}); });
    lyr.on('mouseout',  () => { if(lyr!==selected) cellLayer.resetStyle(lyr); });
    lyr.on('click', () => {
      if (selected) cellLayer.resetStyle(selected);
      selected = lyr; lyr.setStyle({color:"#ffd54a", weight:2.4, opacity:1, fillOpacity:0.25});
      showTab('explore');
      showCell(f.properties);
    });
  }
}).addTo(map);
overlayLayers['cells'] = cellLayer;
overlayDict["1 km cells (click)"] = cellLayer;

L.control.layers(bases, overlayDict, {collapsed:false, position:'topleft'}).addTo(map);

function reason(p){
  const s = p.shap;
  const keys = Object.keys(s).sort((a,b)=>Math.abs(s[b])-Math.abs(s[a]));
  const top = keys[0], up = s[top] >= 0;
  const phrase = {prox_road:"close to the road network", built_density:"near dense existing development",
    prox_built:"on the built-up edge", prox_airport:"near the airport", ndvi:"open/vegetated land"}[top];
  const gt = p.growth_type;
  let lead;
  if (gt==="expansion") lead = `Greenfield in 2017 (built ${p.built17}) that has grown - <b>expansion</b> onto new land.`;
  else if (gt==="densification") lead = `Already built-up in 2017 that grew further - <b>densification</b> (infill).`;
  else lead = `Little net change - <b>stable</b>.`;
  const driver = up ? `The model leans on it being <b>${phrase}</b>` : `It is held back by <b>not</b> being ${phrase}`;
  return `${lead} ${driver}; overall development suitability <b>${p.suitability}</b>.`;
}

function shapBars(s){
  const keys = Object.keys(s).sort((a,b)=>Math.abs(s[b])-Math.abs(s[a]));
  const maxAbs = Math.max(...keys.map(k=>Math.abs(s[k])), 1e-6);
  return keys.map(k=>{
    const v = s[k], w = Math.abs(v)/maxAbs*46, pos = v>=0;
    const fill = `left:50%;width:${w}%;background:#3fa66a`;
    const neg  = `right:50%;width:${w}%;background:#c0453f`;
    return `<div class="bar"><span class="name">${NAMES[k]}${info(k)}</span>
      <span class="track"><span class="fill" style="${pos?fill:neg}"></span>
      <span style="position:absolute;left:50%;top:-1px;width:1px;height:16px;background:#3a4150"></span></span>
      <span class="val">${v>=0?'+':''}${v}</span></div>`;
  }).join('');
}

function stat(k,v,tip){return `<div class="stat"><span class="k">${k}${info(tip)}</span><span>${v}</span></div>`;}

function showCell(p){
  const hot = Math.abs(p.hotspot_z) >= 1.96 ? (p.hotspot_z>0?" (hot 95%)":" (cold 95%)") : "";
  document.getElementById('detail').innerHTML =
    `<span class="tag" style="background:${TYPE_COLOR[p.growth_type]}">${p.growth_type}</span>${info('growth_type')}
     <div class="reason">${reason(p)}</div>
     <div class="sec">Why - driver contributions (SHAP)${info('shap')}</div>
     <div class="sub" style="margin:-2px 0 6px">green pushes toward growth, red against</div>
     ${shapBars(p.shap)}
     <div class="sec">This cell</div>
     ${stat("Suitability", p.suitability, 'suitability')}
     ${stat("New road", p.new_road_km+" km", 'new_road_km')}
     ${stat("New built-up", p.new_built_km2+" km2", 'new_built_km2')}
     ${stat("Forecast by 2030", p.forecast_km2+" km2", 'forecast_km2')}
     ${stat("Hotspot z", p.hotspot_z+hot, 'hotspot_z')}
     ${stat("Cell", "row "+p.row+", col "+p.col, 'cell')}`;
}

// ---------------- Pipeline tab ----------------
// Each stage explains one step and, when opened, switches the map to the layers it produced.
const STAGES = [
  {n:1, t:"Acquire", s:"Sentinel-2, Dynamic World, OpenStreetMap",
   d:"Dry-season median composites of Sentinel-2 Level-2A are pulled from Google Earth Engine for 2017 to 2024. Cloudy pixels are masked and the per-pixel median removes what is left. Dynamic World gives an annual built-up probability, and the OpenStreetMap history gives period-matched road labels.",
   kv:[["Sensor","Sentinel-2 L2A, 10 m"],["Years","2017 - 2024"],["Labels","OSM history (ohsome)"]],
   show:{sat:"sat2024", overlays:[]}},
  {n:2, t:"Preprocess", s:"One locked grid for every layer",
   d:"Every year is reprojected to UTM 43N, clipped to one locked area of interest and resampled onto a single shared grid, so a pixel means the same place in every year. Co-registration is checked by cross-correlation (the measured shift was zero pixels). NDVI and NDBI are computed and appended.",
   kv:[["Grid","1424 x 1234 @ 10 m"],["CRS","EPSG:32643"],["Bands","7 (B,G,R,NIR,SWIR,NDVI,NDBI)"]],
   show:{sat:"sat2017", overlays:[]}},
  {n:3, t:"Road model", s:"U-Net, resolution-bridged",
   d:"Public road datasets are 0.5 m, where a road is 20-40 px wide; Sentinel-2 is 10 m, where a road is 1-2 px. DeepGlobe is downsampled 16x to bridge that gap, used to pretrain a U-Net, which is then fine-tuned on local Sentinel-2 patches labelled with OpenStreetMap.",
   kv:[["IoU","0.459"],["Buffered F1","0.81"],["2024 network","965 km vs OSM 960 km"]],
   show:{sat:null, overlays:["roads"]}},
  {n:4, t:"Change", s:"What is new since 2017",
   d:"Roads are predicted for 2017 and 2024 and differenced to get the new network. A direct Siamese change model was also trained, conditioned on the past map and using a recall-weighted loss, because change pixels are under 1% of the image.",
   kv:[["New road","584 km"],["New built-up","16.5 km2"],["Direct change IoU","0.288"]],
   show:{sat:null, overlays:["newroads"]}},
  {n:5, t:"Growth", s:"Expansion, densification, hotspots",
   d:"Predictions are aggregated onto a 1 km grid. Each cell is classified as expansion (growth on land that was open in 2017), densification (infill of already-built land) or stable, and the Getis-Ord Gi* statistic finds statistically significant growth clusters.",
   kv:[["Cells","195 (13 x 15)"],["Expansion","150 cells"],["Hotspots","19 (95%)"]],
   show:{sat:null, overlays:["built","cells"]}},
  {n:6, t:"Forecast", s:"ConvLSTM to 2030",
   d:"A ConvLSTM learns the growth dynamic from the eight annual built-up frames and rolls forward to 2030. It predicts the change mask rather than the built-up state, because predicting the state collapses to copying last year's map. The decision threshold is tuned to the F1-optimal point.",
   kv:[["Figure of Merit","0.350"],["Threshold","0.75 (tuned)"],["New built-up by 2030","+21 km2"]],
   show:{sat:null, overlays:["forecast"]}},
  {n:7, t:"Explain", s:"SHAP, per cell",
   d:"SHAP fairly splits the credit for each prediction across the named drivers, giving both a global ranking and a per-cell reason. Road proximity and built density come top across five attribution methods and two independent model families, and removing the road driver collapses predicted growth by 23%.",
   kv:[["Top driver","Road proximity"],["Then","Built density"],["Counterfactual","-23% growth"]],
   show:{sat:null, overlays:["cells"]}},
  {n:8, t:"Dashboard", s:"You are here",
   d:"Every layer above is reprojected and stitched onto this map. Switch the backdrop between the Sentinel-2 imagery of each period to see the corridor change, toggle any analysis layer, and click a 1 km cell to see why the model thinks it grows the way it does.",
   kv:[["Layers","4 + 3 satellite periods"],["Clickable cells","195"]],
   show:{sat:null, overlays:["built","newroads","cells"]}}
];

function applyStage(st){
  Object.values(satLayers).forEach(l => map.removeLayer(l));
  if (st.show.sat && satLayers[st.show.sat]) satLayers[st.show.sat].addTo(map);
  Object.entries(overlayLayers).forEach(([id, l]) => {
    if (st.show.overlays.includes(id)) l.addTo(map); else map.removeLayer(l);
  });
}

function buildPipeline(){
  const flow = `<div class="flow">Acquire &rarr; Preprocess &rarr; Road model &rarr; Change &rarr;
    Growth &rarr; Forecast &rarr; Explain &rarr; Dashboard.<br>
    Click a step to read it and switch the map to what that step produced.</div>`;
  const cards = STAGES.map((st, i) => `
    <div class="stage" data-i="${i}">
      <div class="top">
        <span class="num">${st.n}</span>
        <span class="t">${st.t}<small>${st.s}</small></span>
        <span class="chev">&#9662;</span>
      </div>
      <div class="detail">
        <div>${st.d}</div>
        ${st.kv.map(([k,v])=>`<div class="kv"><span>${k}</span><span>${v}</span></div>`).join('')}
        <div class="out">Map switched to this step's layers.</div>
      </div>
    </div>`).join('');
  const el = document.getElementById('pipeline');
  el.innerHTML = flow + cards;
  el.querySelectorAll('.stage').forEach(card => {
    card.querySelector('.top').addEventListener('click', () => {
      const open = card.classList.contains('on');
      el.querySelectorAll('.stage').forEach(c => c.classList.remove('on'));
      if (!open) {
        card.classList.add('on');
        applyStage(STAGES[+card.dataset.i]);
      }
    });
  });
}

function showTab(name){
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('on', t.dataset.tab === name));
  document.getElementById('explore').style.display = name === 'explore' ? 'block' : 'none';
  document.getElementById('pipeline').style.display = name === 'pipeline' ? 'block' : 'none';
}
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => showTab(t.dataset.tab));
});
buildPipeline();

setTimeout(() => map.invalidateSize(), 150);
const tip = document.createElement('div'); tip.id = 'tip'; document.body.appendChild(tip);
document.addEventListener('mouseover', e => {
  const el = e.target.closest('.info');
  if (!el) return;
  tip.textContent = el.dataset.tip;
  tip.style.display = 'block';
  const r = el.getBoundingClientRect();
  let left = r.left - tip.offsetWidth - 8;
  if (left < 8) left = Math.min(r.right + 8, window.innerWidth - tip.offsetWidth - 8);
  let top = Math.min(r.top - 4, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = left + 'px'; tip.style.top = Math.max(8, top) + 'px';
});
document.addEventListener('mouseout', e => { if (e.target.closest('.info')) tip.style.display = 'none'; });

</script>
</body></html>"""


if __name__ == "__main__":
    main()
