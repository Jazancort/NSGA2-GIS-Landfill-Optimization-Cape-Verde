# =============================================================================
# LANDFILL SITE OPTIMIZATION — NSGA-II + EXHAUSTIVE SEARCH BASELINE
# Ilha do Fogo, Cabo Verde
#
# Authors: (see paper)
# Description: Multi-objective GIS-based landfill site selection using NSGA-II.
#              Validated against exhaustive enumeration of all feasible solutions.
#
# Usage:
#   python landfill_nsga2_optimization.py
#
# Results are saved to: results/results-YYYY-MM-DD_HH-MM-SS/
#   (a new timestamped folder is created on every run)
# =============================================================================

"""End-to-end landfill siting workflow using GIS criteria and NSGA-II.

This script builds candidate-site objectives, runs constrained multi-objective
optimization, validates solutions against exhaustive enumeration, and exports
publication-ready figures and reports.
"""

# =============================================================================
# SECTION 1 — INITIAL SETUP
# =============================================================================

import os
import time
import json
import math
import pickle
import warnings
from datetime import datetime
from itertools import combinations
from math import comb

import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, box, LineString
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import geopy.distance

from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.sampling import Sampling
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# --- Reproducibility ---
# For 100% reproducibility also run:  $env:PYTHONHASHSEED="1"  (Windows PowerShell)
GLOBAL_SEED = 1
np.random.seed(GLOBAL_SEED)
SEEDS_LIST = list(range(1, 31))

# --- Results directory: New code/results/results-YYYY-MM-DD_HH-MM-SS ---
_run_ts     = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results', f'results-{_run_ts}')
FIGS_DIR    = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGS_DIR, exist_ok=True)

def get_output_path(filename):
    """Return absolute path for files saved in the run output directory."""
    return os.path.join(RESULTS_DIR, filename)

def get_fig_path(filename):
    """Return absolute path for files saved in the figures subdirectory."""
    return os.path.join(FIGS_DIR, filename)

# --- IEEE figure style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# --- Constants ---
UTM_CRS    = 'EPSG:32626'   # UTM Zone 26N (Cape Verde)
WGS84      = 'EPSG:4326'
K_SELECT   = 3              # Number of landfill sites per solution
MIN_DIST_KM = 10.0          # Minimum pairwise distance constraint (km)
POP_SIZE   = 150            # NSGA-II population size
N_GEN      = 200            # NSGA-II generations
N_RUNS     = 30             # Independent runs
N_JOBS     = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free

print(f'Parameters: k={K_SELECT}, min_dist={MIN_DIST_KM} km, pop={POP_SIZE}, gen={N_GEN}, runs={N_RUNS}')
print(f'Parallel workers: {N_JOBS} / {multiprocessing.cpu_count()} CPU cores')
print(f'Results directory: {os.path.abspath(RESULTS_DIR)}')


# =============================================================================
# SECTION 1.5 — GIS PROCESSING (skipped if cached CSV exists)
# =============================================================================

GIS_CACHE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_cache')
GIS_CACHE_FILE = os.path.join(GIS_CACHE_DIR, 'df_real_data_cached.csv')

_DATA_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..'

_existing_csvs = [
    os.path.join(_DATA_ROOT, 'optimization_criteria_data_524-fixed_penalized-organized.csv'),
    os.path.join(_DATA_ROOT, 'optimization_criteria_data_524-fixed_penalized.csv'),
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3_standardized-organized.csv'),
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3_standardized.csv'),
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3.csv'),
    GIS_CACHE_FILE,
]
_cache_found = any(os.path.exists(f) for f in _existing_csvs)

if _cache_found:
    print('GIS PROCESSING SKIPPED — cached CSV found.')
else:
    print('='*70)
    print('NO CACHED DATA FOUND — Running full GIS pipeline')
    print('='*70)

    import rasterio
    import rasterio.errors
    from rasterio.warp import transform as rasterio_transform
    import osmnx as ox
    from scipy.ndimage import sobel

    def get_nearest_osm_dist(lat, lon, tags, max_dist=30000,
                              point_crs='EPSG:4326', target_crs='EPSG:32626'):
        """Return nearest OSM feature distance (meters) for a point and tag query."""
        try:
            pt = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs=point_crs).to_crs(target_crs)
            ox.settings.use_cache = True
            ox.settings.log_console = False
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=max_dist)
            if gdf.empty:
                return float('nan')
            gdf_proj = gdf.to_crs(target_crs)
            dists = gdf_proj.geometry.distance(pt.geometry.iloc[0])
            return dists.min() if not dists.empty else float('nan')
        except Exception:
            return float('nan')

    def get_raster_value(lat, lon, raster_path):
        """Sample one raster pixel value at the given WGS84 point; return NaN if invalid."""
        try:
            with rasterio.open(raster_path) as src:
                xs, ys = rasterio_transform('EPSG:4326', src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
                if not (src.bounds.left <= x <= src.bounds.right and
                        src.bounds.bottom <= y <= src.bounds.top):
                    return float('nan')
                row, col = src.index(x, y)
                if not (0 <= row < src.height and 0 <= col < src.width):
                    return float('nan')
                val = float(src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                if src.nodata is not None and abs(val - src.nodata) < 1e-6:
                    return float('nan')
                return val
        except Exception:
            return float('nan')

    def get_aspect_at_point(lat, lon, dem_path):
        """Estimate terrain aspect (degrees) from DEM gradients at a WGS84 point."""
        try:
            with rasterio.open(dem_path) as src:
                xs, ys = rasterio_transform('EPSG:4326', src.crs, [lon], [lat])
                x, y = xs[0], ys[0]
                b = src.bounds
                if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
                    return float('nan')
                dem = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    dem = np.where(dem == src.nodata, np.nan, dem)
                dx = sobel(dem, axis=1, mode='reflect') / (8 * src.transform.a)
                dy = sobel(dem, axis=0, mode='reflect') / (8 * abs(src.transform.e))
                aspect = (90.0 - np.degrees(np.arctan2(dx, -dy)) + 360.0) % 360.0
                flat = (np.abs(dx) < 1e-9) & (np.abs(dy) < 1e-9)
                aspect[flat] = -1.0
                r, c = src.index(x, y)
                if 0 <= r < aspect.shape[0] and 0 <= c < aspect.shape[1]:
                    v = aspect[r, c]
                    if src.nodata is not None and abs(v - src.nodata) < 1e-6:
                        return float('nan')
                    return float(v)
                return float('nan')
        except Exception:
            return float('nan')

    def get_distance_to_feature(lat, lon, feature_gdf, point_crs='EPSG:4326'):
        """Return minimum distance from point to a feature layer geometry in layer CRS units."""
        try:
            if feature_gdf is None or feature_gdf.empty:
                return float('nan')
            pt = gpd.GeoDataFrame([{'geometry': Point(lon, lat)}], crs=point_crs).to_crs(feature_gdf.crs)
            return feature_gdf.geometry.distance(pt.geometry.iloc[0]).min()
        except Exception:
            return float('nan')

    # Step 1 — Candidate grid
    island_gdf = gpd.read_file(os.path.join(_DATA_ROOT, 'geologia_fogo.shp')).to_crs(UTM_CRS)
    island_dissolved = island_gdf.dissolve()
    xmin, ymin, xmax, ymax = island_dissolved.total_bounds
    gx = np.arange(np.floor(xmin), np.ceil(xmax), 1000)
    gy = np.arange(np.floor(ymin), np.ceil(ymax), 1000)
    polys = [box(x, y, x + 1000, y + 1000) for x in gx for y in gy]
    gdf_grid = gpd.GeoDataFrame(geometry=polys, crs=UTM_CRS)
    gdf_grid = gpd.sjoin(gdf_grid, island_dissolved, how='inner', predicate='intersects')
    gdf_grid = gdf_grid.drop_duplicates(subset=['geometry']).reset_index(drop=True)

    MIN_LAND_COVERAGE = 0.50
    grid_intersections = []
    for idx, row in gdf_grid.iterrows():
        cell_geom = row.geometry
        intersection = cell_geom.intersection(island_dissolved.geometry.iloc[0])
        if intersection.is_empty:
            continue
        coverage_ratio = intersection.area / cell_geom.area
        if coverage_ratio >= MIN_LAND_COVERAGE:
            grid_intersections.append({'geometry': intersection.centroid,
                                       'coverage_ratio': coverage_ratio})

    gdf_grid_filtered = gpd.GeoDataFrame(grid_intersections, crs=UTM_CRS)
    centroids_in_island = gdf_grid_filtered.geometry.within(island_dissolved.geometry.iloc[0])
    gdf_grid_filtered = gdf_grid_filtered[centroids_in_island].reset_index(drop=True)
    candidates_wgs84 = gdf_grid_filtered[['geometry']].to_crs(WGS84)

    # Step 2 — Vector datasets
    METRIC_CRS = UTM_CRS
    low_perm_gdf, pa_gdf_proj = None, None
    try:
        geo_full = gpd.read_file(os.path.join(_DATA_ROOT, 'geologia_com_permeabilidade_calculada.shp'))
        if geo_full.crs is None:
            geo_full.set_crs(WGS84, inplace=True)
        low_perm_gdf = geo_full[geo_full['permeabili'] == 'Baixa'].to_crs(METRIC_CRS)
    except Exception:
        pass
    try:
        pa_gdf = gpd.read_file(os.path.join(_DATA_ROOT, 'area_protegida.shp'))
        if pa_gdf.crs is None:
            pa_gdf.set_crs(WGS84, inplace=True)
        pa_gdf_proj = pa_gdf.to_crs(METRIC_CRS)
    except Exception:
        pass

    # Step 3 — Multi-criteria extraction
    DEM_ELEVATION = os.path.join(_DATA_ROOT, 'elevation_utm.tif')
    DEM_SLOPE     = os.path.join(_DATA_ROOT, 'declive_gdal_graus.tif')
    osm_criteria = {
        'dist_urban_core':  {'tags': {'landuse': ['residential', 'commercial', 'retail', 'industrial']}},
        'dist_aerodrome':   {'tags': {'aeroway': 'aerodrome'}},
        'dist_streams':     {'tags': {'waterway': ['stream', 'river', 'canal']}},
        'dist_roads':       {'tags': {'highway': ['motorway', 'trunk', 'primary', 'secondary',
                                                   'tertiary', 'unclassified', 'residential',
                                                   'service', 'road', 'track', 'path']}},
        'dist_wells':       {'tags': {'man_made': ['water_well', 'borehole', 'petroleum_well',
                                                    'water_tap', 'water_works']}},
        'dist_cultivation': {'tags': {'landuse': ['farmland', 'farmyard', 'orchard',
                                                   'vineyard', 'allotments']}},
        'dist_coastline':   {'tags': {'natural': 'coastline'}},
    }
    real_data = []
    t_gis = time.time()
    for i, (_, pt) in enumerate(candidates_wgs84.iterrows()):
        lat, lon = pt.geometry.y, pt.geometry.x
        rec = {'latitude': lat, 'longitude': lon}
        rec['slope_degrees'] = round(v, 4) if pd.notna(v := get_raster_value(lat, lon, DEM_SLOPE)) else np.nan
        rec['slope_aspect']  = round(v, 4) if pd.notna(v := get_aspect_at_point(lat, lon, DEM_ELEVATION)) else np.nan
        for cname, cinfo in osm_criteria.items():
            d = get_nearest_osm_dist(lat, lon, cinfo['tags'])
            rec[cname] = round(d, 4) if pd.notna(d) else np.nan
        d = get_distance_to_feature(lat, lon, low_perm_gdf)
        rec['dist_low_permeability'] = round(d, 4) if pd.notna(d) else np.nan
        d = get_distance_to_feature(lat, lon, pa_gdf_proj)
        rec['dist_protected_area'] = round(d, 4) if pd.notna(d) else np.nan
        real_data.append(rec)
        if (i + 1) % 50 == 0 or i == 0:
            pct = 100 * (i + 1) / len(candidates_wgs84)
            print(f'  GIS: {pct:.0f}% ({i+1}/{len(candidates_wgs84)}) ...', end='\r')

    df_gis = pd.DataFrame(real_data)
    df_gis.rename(columns={
        'latitude': 'latitude (degrees)', 'longitude': 'longitude (degrees)',
        'slope_degrees': 'slope (degrees)', 'slope_aspect': 'slope_aspect (degrees)',
        'dist_urban_core': 'dist_urban_core (m)', 'dist_aerodrome': 'dist_aerodrome (m)',
        'dist_streams': 'dist_streams (m)', 'dist_roads': 'dist_roads (m)',
        'dist_wells': 'dist_wells (m)', 'dist_cultivation': 'dist_cultivation (m)',
        'dist_coastline': 'dist_coastline (m)',
        'dist_low_permeability': 'dist_low_permeability (m)',
        'dist_protected_area': 'dist_protected_area (m)',
    }, inplace=True)

    os.makedirs(GIS_CACHE_DIR, exist_ok=True)
    df_gis.to_csv(GIS_CACHE_FILE, index=False)
    print(f'\n  GIS cache saved: {GIS_CACHE_FILE}  ({len(df_gis)} sites)')


# =============================================================================
# SECTION 2 — DATA LOADING
# =============================================================================

# Load island shapefile
gdf_island   = gpd.read_file(os.path.join(_DATA_ROOT, 'geologia_fogo.shp'))
gdf_island_utm = gdf_island.to_crs(UTM_CRS)
print(f'Island shapefile: {len(gdf_island)} features')

# Protected area
gdf_protected = gpd.read_file(os.path.join(_DATA_ROOT, 'area_protegida.shp'))
if gdf_protected.crs != WGS84:
    gdf_protected = gdf_protected.to_crs(WGS84)
zona_proibida = gdf_protected.geometry.union_all()
print(f'Protected area: {len(gdf_protected)} polygons')

# Candidate data — try files in order of preference
CANDIDATE_FILES = [
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3.csv'),
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3_standardized.csv'),
    os.path.join(_DATA_ROOT, 'criteria_data_for_optimization_524-3_standardized-organized.csv'),
    os.path.join(_DATA_ROOT, 'optimization_criteria_data_524-fixed_penalized.csv'),
    os.path.join(_DATA_ROOT, 'optimization_criteria_data_524-fixed_penalized-organized.csv'),
    GIS_CACHE_FILE,
]

df = None
for fpath in CANDIDATE_FILES:
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath, delimiter=';')
            if len(df.columns) < 5:
                df = pd.read_csv(fpath)
            print(f'Loaded: {os.path.basename(fpath)}  ({len(df)} sites, {len(df.columns)} cols)')
            break
        except Exception as e:
            print(f'  Failed {fpath}: {e}')

if df is None:
    raise FileNotFoundError('No candidate CSV found. Run GIS processing first.')

# Standardise column names
NAME_MAP = {
    'latitude (degrees)': 'latitude', 'longitude (degrees)': 'longitude',
    'slope (degrees)': 'Slope (degrees)', 'slope_aspect (degrees)': 'Aspect (degrees)',
    'aspect (degrees)': 'Aspect (degrees)',
    'dist_urban_core (m)': 'Urban Areas (m)', 'distance_to_urban_areas (m)': 'Urban Areas (m)',
    'dist_aerodrome (m)': 'Aerodrome (m)', 'distance_to_aerodrome (m)': 'Aerodrome (m)',
    'dist_streams (m)': 'Rivers (m)', 'distance_to_rivers (m)': 'Rivers (m)',
    'dist_roads (m)': 'Road Network (m)', 'distance_to_roads (m)': 'Road Network (m)',
    'dist_wells (m)': 'Wells/Boreholes (m)', 'distance_to_water_infrastructure (m)': 'Wells/Boreholes (m)',
    'dist_cultivation (m)': 'Cultivated Areas (m)', 'distance_to_cultivated_areas (m)': 'Cultivated Areas (m)',
    'dist_coastline (m)': 'Coastline (m)', 'distance_to_coastline (m)': 'Coastline (m)',
    'dist_low_permeability (m)': 'Permeability (Distance) (m)',
    'distance_to_low_permeability_zones (m)': 'Permeability (Distance) (m)',
    'dist_protected_area (m)': 'Protected Areas (m)',
    'distance_to_protected_areas (m)': 'Protected Areas (m)',
}
df.rename(columns=NAME_MAP, inplace=True)

# AHP weights (Andrade & Barbosa, 2015, Table 5)
AHP_WEIGHTS = {
    'Urban Areas (m)': 0.166, 'Aerodrome (m)': 0.102,
    'Permeability (Distance) (m)': 0.152, 'Protected Areas (m)': 0.066,
    'Coastline (m)': 0.069, 'Rivers (m)': 0.091,
    'Slope (degrees)': 0.113, 'Road Network (m)': 0.030,
    'Aspect (degrees)': 0.075, 'Wells/Boreholes (m)': 0.055,
    'Cultivated Areas (m)': 0.081,
}
SCENARIO_WEIGHTS = {'Cost': 0.295, 'Impact': 0.356, 'Safety': 0.349}

print(f'AHP sum={sum(AHP_WEIGHTS.values()):.3f}  |  shape={df.shape}')


# =============================================================================
# SECTION 3 — PRE-PROCESSING
# =============================================================================

valid_cols = [c for c in AHP_WEIGHTS if c in df.columns]
MIN_MAX = {c: (df[c].min(), df[c].max()) for c in valid_cols}

def _safe_norm(v, mn, mx):
    """Min-max normalize with neutral fallback (0.5) for non-finite inputs."""
    if not np.isfinite(v):
        return 0.5
    rng = mx - mn
    return 0.0 if rng <= 0 else (v - mn) / rng

def _safe_norm_inv(v, mn, mx):
    """Inverse min-max normalize with neutral fallback (0.5) for non-finite inputs."""
    if not np.isfinite(v):
        return 0.5
    rng = mx - mn
    return 0.0 if rng <= 0 else (mx - v) / rng

def cost_function(row):
    """Compute the cost objective from normalized permeability, slope, and road distance."""
    score = 0.0
    for crit in ('Permeability (Distance) (m)', 'Slope (degrees)', 'Road Network (m)'):
        score += AHP_WEIGHTS[crit] * _safe_norm(row[crit], *MIN_MAX[crit])
    return score

def impact_function(row):
    """Compute environmental impact with inverse-distance criteria plus aspect penalty."""
    score = 0.0
    for crit in ('Protected Areas (m)', 'Coastline (m)', 'Rivers (m)', 'Wells/Boreholes (m)'):
        score += AHP_WEIGHTS[crit] * _safe_norm_inv(row[crit], *MIN_MAX[crit])
    asp = row['Aspect (degrees)']
    norm_asp = 0 if (np.isfinite(asp) and 90 <= asp <= 180) else (0.5 if not np.isfinite(asp) else 1)
    score += AHP_WEIGHTS['Aspect (degrees)'] * norm_asp
    return score

def safety_function(row):
    """Compute safety objective using normalized distances to urban, aerodrome, and cultivated areas."""
    score = 0.0
    for crit in ('Urban Areas (m)', 'Aerodrome (m)', 'Cultivated Areas (m)'):
        score += AHP_WEIGHTS[crit] * _safe_norm(row[crit], *MIN_MAX[crit])
    return score

df['CostDef']   = df.apply(cost_function, axis=1)
df['ImpactDef'] = df.apply(impact_function, axis=1)
df['SafetyDef'] = df.apply(safety_function, axis=1)

obj_cols = ['CostDef', 'ImpactDef', 'SafetyDef']
assert df[obj_cols].apply(np.isfinite).all().all(), "Non-finite objectives found!"
print('Objectives computed — all finite.')

# Block protected areas
count_blocked = 0
for idx, row in df.iterrows():
    pt = Point(row['longitude'], row['latitude'])
    if pt.within(zona_proibida):
        df.at[idx, 'CostDef']   = 1.0
        df.at[idx, 'ImpactDef'] = 1.0
        df.at[idx, 'SafetyDef'] = 0.0
        count_blocked += 1
print(f'Protected area blocking: {count_blocked} sites blocked.')

# Haversine distance matrix
def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance (km) between two latitude/longitude points."""
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

lats = df['latitude'].values
lons = df['longitude'].values
n_sites = len(df)
dist_matrix = np.zeros((n_sites, n_sites))
for i in range(n_sites):
    for j in range(i+1, n_sites):
        d = haversine(lats[i], lons[i], lats[j], lons[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

upper = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
print(f'Distance matrix {n_sites}×{n_sites}: min={upper.min():.2f} km, max={upper.max():.2f} km')


# =============================================================================
# SECTION 4 — OPTIMIZATION PROBLEM DEFINITION
# =============================================================================

class SelectAreasRobust(ElementwiseProblem):
    """Multi-objective landfill site selection (cardinality-constrained)."""
    def __init__(self, df_data, n_select=3):
        super().__init__(n_var=len(df_data), n_obj=3, xl=0, xu=1, type_var=bool)
        self.df = df_data
        self.n_select = n_select

    def _evaluate(self, x, out, *args, **kwargs):
        if x.sum() != self.n_select:
            out['F'] = [1e5, 1e5, 1e5]
            return
        sel = self.df.iloc[np.where(x)[0]]
        out['F'] = [sel['CostDef'].sum(), sel['ImpactDef'].sum(), -sel['SafetyDef'].sum()]


class SelectAreasWithMinDistance(Problem):
    """Multi-objective problem — VECTORIZED (processes entire population at once).

    Optimizations over ElementwiseProblem:
      1. Objectives via matrix multiply X @ costs: single O(pop×n) BLAS call.
      2. Distance constraint vectorised across the whole population via numpy
         fancy indexing — no per-individual Python loop.
      3. Objective arrays cached as float64 at construction time.
    Expected speedup: 5–15× per generation.
    """
    def __init__(self, df_data, n_select=3, min_dist_km=10.0, dm=None):
        n = len(df_data)
        super().__init__(n_var=n, n_obj=3, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
        self.n_select  = n_select
        self.min_dist  = min_dist_km
        self.dm        = dm
        self._costs    = df_data['CostDef'].values.astype(np.float64)
        self._impacts  = df_data['ImpactDef'].values.astype(np.float64)
        self._safeties = df_data['SafetyDef'].values.astype(np.float64)
        from itertools import combinations as _comb
        self._pairs = [(a, b) for a, b in _comb(range(n_select), 2)]

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=bool)
        n_pop = X.shape[0]
        k     = self.n_select

        F1 =  X @ self._costs
        F2 =  X @ self._impacts
        F3 = -(X @ self._safeties)

        X_int = X.view(np.uint8)
        top_k = np.argpartition(X_int, -k, axis=1)[:, -k:]
        top_k = np.sort(top_k, axis=1)

        dists = np.full(n_pop, np.inf)
        for a, b in self._pairs:
            d = self.dm[top_k[:, a], top_k[:, b]]
            np.minimum(dists, d, out=dists)
        G = self.min_dist - dists

        wrong_k = X.sum(axis=1) != k
        if wrong_k.any():
            F1[wrong_k] = 1e5
            F2[wrong_k] = 1e5
            F3[wrong_k] = 1e5
            G[wrong_k]  = 1e5

        out['F'] = np.column_stack([F1, F2, F3])
        out['G'] = G[:, np.newaxis]


class BinarySamplingFixedK(Sampling):
    """Binary sampler that activates exactly k sites per individual."""
    def __init__(self, k, n_var):
        super().__init__()
        self.k, self.n_var = k, n_var

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, self.n_var), False, dtype=bool)
        for i in range(n_samples):
            X[i, np.random.choice(self.n_var, self.k, replace=False)] = True
        return X


class DistanceAwareSampling(Sampling):
    """Binary sampler that tries to satisfy minimum pairwise distance at initialization."""
    def __init__(self, k, n_var, dm, min_dist_km, max_attempts=100):
        super().__init__()
        self.k, self.n_var, self.dm = k, n_var, dm
        self.min_dist, self.max_attempts = min_dist_km, max_attempts

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, self.n_var), False, dtype=bool)
        for i in range(n_samples):
            for _ in range(self.max_attempts):
                idx = np.random.choice(self.n_var, self.k, replace=False)
                ok = all(self.dm[idx[a], idx[b]] >= self.min_dist
                         for a in range(self.k) for b in range(a+1, self.k))
                if ok:
                    break
            X[i, idx] = True
        return X


class FixKRepair(Repair):
    """Repair operator enforcing exactly k selected sites after variation."""
    def __init__(self, k):
        super().__init__()
        self.k = k

    def _do(self, problem, X, **kwargs):
        for i in range(len(X)):
            n_sel = X[i].sum()
            if n_sel > self.k:
                on = np.where(X[i])[0]
                X[i, np.random.choice(on, int(n_sel - self.k), replace=False)] = False
            elif n_sel < self.k:
                off = np.where(~X[i])[0]
                X[i, np.random.choice(off, int(self.k - n_sel), replace=False)] = True
        return X


class HVCallback(Callback):
    """Collect population objective arrays each generation for HV post-analysis."""
    def __init__(self):
        super().__init__()
        self.data['F'] = []

    def notify(self, algorithm):
        self.data['F'].append(algorithm.pop.get('F'))


print('Optimization classes defined (SelectAreasWithMinDistance: VECTORIZED).')


# =============================================================================
# SECTION 5 — NSGA-II EXECUTION (30 INDEPENDENT RUNS — PARALLEL)
# =============================================================================

def _run_single(run_id, seed_i, df_data, k_select, min_dist_km, dm, pop_size, n_gen):
    """Execute one independent NSGA-II run. Designed to be called in parallel."""
    np.random.seed(seed_i)
    problem_local = SelectAreasWithMinDistance(
        df_data, n_select=k_select, min_dist_km=min_dist_km, dm=dm
    )
    algo = NSGA2(
        pop_size=pop_size,
        sampling=DistanceAwareSampling(k_select, len(df_data), dm, min_dist_km),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=1.0 / len(df_data)),
        repair=FixKRepair(k_select),
        eliminate_duplicates=True,
    )
    cb = HVCallback()
    t0 = time.time()
    res = minimize(problem_local, algo, ('n_gen', n_gen), seed=seed_i, verbose=False, callback=cb)
    t_run = time.time() - t0

    F_valid = res.F[res.F[:, 0] < 1e4] if res.F is not None else np.array([])
    hv_temp = float(HV(ref_point=np.max(F_valid, axis=0) + 1e-3)(F_valid)) if len(F_valid) > 0 else 0.0

    return {'F': res.F, 'X': res.X, 'hist': cb.data['F'], 'time': t_run, 'hv_temp': hv_temp}


problem = SelectAreasWithMinDistance(df, n_select=K_SELECT,
                                     min_dist_km=MIN_DIST_KM, dm=dist_matrix)

nsga2_cache_file = get_output_path('nsga2_all_runs_cache.pkl')

if os.path.exists(nsga2_cache_file):
    print('Loading cached NSGA-II results ...')
    with open(nsga2_cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    all_runs         = cached_data['all_runs']
    run_times        = cached_data['run_times']
    time_nsga2_total = cached_data['time_nsga2_total']
    time_nsga2_avg   = cached_data['time_nsga2_avg']
    time_nsga2_std   = cached_data['time_nsga2_std']
    print(f'  Loaded {len(all_runs)} runs | total={time_nsga2_total:.1f}s')
else:
    print('='*70)
    print(f'NSGA-II EXECUTION: {N_RUNS} independent runs ({N_JOBS} parallel workers)')
    print('='*70)

    t_start_all = time.time()

    all_runs = Parallel(n_jobs=N_JOBS, verbose=10, prefer='processes')(
        delayed(_run_single)(
            run_id, SEEDS_LIST[run_id],
            df, K_SELECT, MIN_DIST_KM, dist_matrix, POP_SIZE, N_GEN
        )
        for run_id in range(N_RUNS)
    )

    time_nsga2_total = time.time() - t_start_all
    run_times        = [r['time'] for r in all_runs]
    time_nsga2_avg   = np.mean(run_times)
    time_nsga2_std   = np.std(run_times)

    # Print per-run summary
    for run_id, r in enumerate(all_runs):
        best_cost = r['F'][:, 0].min() if r['F'] is not None else np.nan
        pct    = 100 * (run_id + 1) / N_RUNS
        filled = int(30 * (run_id + 1) / N_RUNS)
        bar    = '█' * filled + '░' * (30 - filled)
        print(f'  [{bar}] Run {run_id+1:2d}/{N_RUNS} ({pct:5.1f}%) | '
              f'Seed: {SEEDS_LIST[run_id]} | Time: {r["time"]:5.1f}s | '
              f'Solutions: {len(r["F"]):3d} | HV: {r["hv_temp"]:.4f} | '
              f'Best cost: {best_cost:.4f}')

    print(f'\nAll {N_RUNS} runs done. Total={time_nsga2_total:.1f}s, '
          f'Avg={time_nsga2_avg:.1f}s ± {time_nsga2_std:.1f}s')

    cache_data = {'all_runs': all_runs, 'run_times': run_times,
                  'time_nsga2_total': time_nsga2_total,
                  'time_nsga2_avg': time_nsga2_avg,
                  'time_nsga2_std': time_nsga2_std}
    with open(nsga2_cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f'Cache saved: {nsga2_cache_file}')

# HV statistics with common reference point
all_F_concat = np.vstack([r['F'] for r in all_runs if r['F'] is not None])
all_F_valid  = all_F_concat[all_F_concat[:, 0] < 1e4]
epsilon      = 1e-3
REF_POINT    = np.max(all_F_valid, axis=0) + epsilon
hv_indicator = HV(ref_point=REF_POINT)

hv_matrix = np.zeros((N_RUNS, N_GEN))
for i, r in enumerate(all_runs):
    for g, Fg in enumerate(r['hist']):
        if g >= N_GEN:
            break
        Fv = Fg[Fg[:, 0] < 1e4] if Fg is not None else np.array([])
        hv_matrix[i, g] = hv_indicator(Fv) if len(Fv) > 0 else 0.0

hv_mean   = hv_matrix.mean(axis=0)
hv_std    = hv_matrix.std(axis=0)
hv_gens   = np.arange(1, N_GEN + 1)
hv_finals = np.array([hv_indicator(r['F'][r['F'][:, 0] < 1e4]) for r in all_runs])

print(f'HV  mean={hv_finals.mean():.6f} ± {hv_finals.std():.6f}  '
      f'best={hv_finals.max():.6f}')

best_run_idx = np.argmax(hv_finals)
F_best  = all_runs[best_run_idx]['F']
X_best  = all_runs[best_run_idx]['X']
mask_valid = F_best[:, 0] < 1e4
F_pareto = F_best[mask_valid]
X_pareto = X_best[mask_valid]
print(f'Best run #{best_run_idx+1}: {len(F_pareto)} Pareto solutions, HV={hv_finals[best_run_idx]:.6f}')

hv_meta = {
    'generations': hv_gens.tolist(), 'hv_mean': hv_mean.tolist(), 'hv_std': hv_std.tolist(),
    'hv_finals': hv_finals.tolist(), 'run_times': run_times,
    'run_times_stats': {'mean': float(np.mean(run_times)), 'std': float(np.std(run_times)),
                        'min': float(min(run_times)), 'max': float(max(run_times)),
                        'total': float(time_nsga2_total)},
    'reference_point': REF_POINT.tolist(), 'n_runs': N_RUNS,
    'algorithm_config': {'pop': POP_SIZE, 'gen': N_GEN, 'k': K_SELECT, 'min_dist_km': MIN_DIST_KM},
    'best_run_index': int(best_run_idx),
}
with open(get_output_path('hypervolume_metadata_complete.json'), 'w') as f:
    json.dump(hv_meta, f, indent=2)

run_stats_df = pd.DataFrame({
    'run_id': range(1, N_RUNS + 1), 'seed': SEEDS_LIST, 'time_seconds': run_times,
    'n_solutions': [len(r['F'][r['F'][:, 0] < 1e4]) for r in all_runs],
    'hypervolume': hv_finals, 'hv_temp': [r['hv_temp'] for r in all_runs],
    'best_cost':   [r['F'][r['F'][:, 0] < 1e4][:, 0].min()
                    if len(r['F'][r['F'][:, 0] < 1e4]) > 0 else np.nan for r in all_runs],
    'best_impact': [r['F'][r['F'][:, 0] < 1e4][:, 1].min()
                    if len(r['F'][r['F'][:, 0] < 1e4]) > 0 else np.nan for r in all_runs],
    'best_safety': [r['F'][r['F'][:, 0] < 1e4][:, 2].min()
                    if len(r['F'][r['F'][:, 0] < 1e4]) > 0 else np.nan for r in all_runs],
})
run_stats_df.to_csv(get_output_path('nsga2_run_statistics.csv'), index=False)
print('Run statistics saved.')


# =============================================================================
# SECTION 6 — EXHAUSTIVE SEARCH (VALIDATION BASELINE)
# =============================================================================

def is_pareto_efficient_fast(costs):
    """Return boolean mask of non-dominated points using sort-and-sweep filtering."""
    n = len(costs)
    if n == 0:
        return np.array([], dtype=bool)
    print(f'  Pareto filter: {n:,} points ...')
    t0 = time.time()
    obj_sum = costs.sum(axis=1)
    order = np.argsort(obj_sum)
    remaining_idx   = order.copy()
    remaining_costs = costs[order]
    pareto_indices  = []
    iteration = 0
    while len(remaining_idx) > 0:
        iteration += 1
        pareto_indices.append(remaining_idx[0])
        best = remaining_costs[0]
        rest_costs = remaining_costs[1:]
        rest_idx   = remaining_idx[1:]
        if len(rest_costs) == 0:
            break
        keep = ~(np.all(best <= rest_costs, axis=1) & np.any(best < rest_costs, axis=1))
        remaining_costs = rest_costs[keep]
        remaining_idx   = rest_idx[keep]
        if iteration <= 3 or iteration % 100 == 0:
            print(f'    iter {iteration}: {len(pareto_indices)} Pareto pts | '
                  f'{len(remaining_idx):,} remaining | {time.time()-t0:.1f}s')
    print(f'  Done: {len(pareto_indices)} solutions in {time.time()-t0:.1f}s')
    result = np.zeros(n, dtype=bool)
    result[pareto_indices] = True
    return result


costs_arr    = df['CostDef'].values.astype(np.float64)
impacts_arr  = df['ImpactDef'].values.astype(np.float64)
safeties_arr = df['SafetyDef'].values.astype(np.float64)

total_combos = comb(n_sites, K_SELECT)
print(f'Exhaustive search: C({n_sites},{K_SELECT}) = {total_combos:,}')

t_ex_start = time.time()
feasible_indices   = []
feasible_objectives = []
total_checked = 0

for combo in combinations(range(n_sites), K_SELECT):
    total_checked += 1
    ok = True
    for a in range(K_SELECT):
        for b in range(a+1, K_SELECT):
            if dist_matrix[combo[a], combo[b]] < MIN_DIST_KM:
                ok = False
                break
        if not ok:
            break
    if ok:
        feasible_indices.append(combo)
        feasible_objectives.append([
            costs_arr[list(combo)].sum(),
            impacts_arr[list(combo)].sum(),
            -safeties_arr[list(combo)].sum(),
        ])
    if total_checked % 5_000_000 == 0:
        pct = 100 * total_checked / total_combos
        print(f'  {total_checked:,} ({pct:.1f}%) | feasible={len(feasible_indices):,} | '
              f'{time.time()-t_ex_start:.0f}s')

time_enum = time.time() - t_ex_start
feasible_F   = np.array(feasible_objectives)
feasible_idx = np.array(feasible_indices)
print(f'Phase 1: {time_enum:.1f}s  | feasible={len(feasible_F):,} '
      f'({100*len(feasible_F)/total_checked:.2f}%)')

valid_feas_mask = np.all(np.isfinite(feasible_F), axis=1)
if (~valid_feas_mask).any():
    feasible_F   = feasible_F[valid_feas_mask]
    feasible_idx = feasible_idx[valid_feas_mask]

print('Phase 2: Pareto front ...')
t_pareto   = time.time()
pareto_mask = is_pareto_efficient_fast(feasible_F)
time_pareto = time.time() - t_pareto
true_pareto_F   = feasible_F[pareto_mask]
true_pareto_idx = feasible_idx[pareto_mask]

# Unified reference point (NSGA-II + Exhaustive)
nsga2_F_valid    = all_F_concat[all_F_concat[:, 0] < 1e4]
combined_F       = np.vstack([nsga2_F_valid, true_pareto_F])
UNIFIED_REF_POINT = np.max(combined_F, axis=0) + 1e-3
hv_indicator_unified = HV(ref_point=UNIFIED_REF_POINT)

MAX_HV_POINTS = 10000
if len(true_pareto_F) > MAX_HV_POINTS:
    np.random.seed(GLOBAL_SEED)
    sample_idx = np.random.choice(len(true_pareto_F), size=MAX_HV_POINTS, replace=False)
    hv_exhaustive = hv_indicator_unified(true_pareto_F[sample_idx])
else:
    try:
        hv_exhaustive = hv_indicator_unified(true_pareto_F)
    except Exception:
        np.random.seed(GLOBAL_SEED)
        hv_exhaustive = hv_indicator_unified(
            true_pareto_F[np.random.choice(len(true_pareto_F), 5000, replace=False)])

hv_nsga2_unified = hv_indicator_unified(F_pareto)
hv_finals_unified = np.array([
    hv_indicator_unified(r['F'][r['F'][:, 0] < 1e4]) if len(r['F'][r['F'][:, 0] < 1e4]) > 0 else 0.0
    for r in all_runs
])

time_exhaustive = time.time() - t_ex_start
print(f'Exhaustive done: {time_exhaustive:.1f}s | true Pareto={len(true_pareto_F):,}')
print(f'HV exhaustive={hv_exhaustive:.6f}  NSGA-II={hv_nsga2_unified:.6f}  '
      f'gap={100*(hv_exhaustive-hv_nsga2_unified)/hv_exhaustive:.2f}%')

ex_df = pd.DataFrame(true_pareto_F, columns=['Cost', 'Impact', 'Neg_Safety'])
ex_df['Safety'] = -ex_df['Neg_Safety']
for j in range(K_SELECT):
    ex_df[f'idx{j+1}'] = true_pareto_idx[:, j]
ex_df.to_csv(get_output_path('exhaustive_search_pareto.csv'), index=False)


# =============================================================================
# SECTION 7 — RESULTS ANALYSIS
# =============================================================================

hv_nsga2_best = hv_finals_unified.max()
gap_pct      = 100 * (hv_exhaustive - hv_nsga2_best) / hv_exhaustive
capture_rate = 100 - gap_pct
speedup      = time_exhaustive / time_nsga2_avg

F_real = F_pareto.copy()
F_real[:, 2] = -F_real[:, 2]

idx_best_cost   = np.argmin(F_real[:, 0])
idx_best_impact = np.argmin(F_real[:, 1])
idx_best_safety = np.argmax(F_real[:, 2])
extreme_labels  = ['Lowest Cost', 'Lowest Impact', 'Highest Safety']
extreme_idxs    = [idx_best_cost, idx_best_impact, idx_best_safety]

low_cost_sites   = np.where(X_pareto[idx_best_cost])[0]
low_impact_sites = np.where(X_pareto[idx_best_impact])[0]
high_safety_sites = np.where(X_pareto[idx_best_safety])[0]

# Best compromise (unified normalization)
combined_for_range = np.vstack([F_pareto, true_pareto_F])
unified_min = combined_for_range.min(axis=0)
unified_max = combined_for_range.max(axis=0)
unified_range = unified_max - unified_min
unified_range[unified_range == 0] = 1.0

def _unified_ws(F_block):
    """Compute weighted score on a jointly normalized objective block."""
    fnorm = (F_block - unified_min) / unified_range
    return (SCENARIO_WEIGHTS['Cost']   * fnorm[:, 0] +
            SCENARIO_WEIGHTS['Impact'] * fnorm[:, 1] +
            SCENARIO_WEIGHTS['Safety'] * fnorm[:, 2])

ws          = _unified_ws(F_pareto)
best_sc_idx = np.argmin(ws)
bc_sites    = np.where(X_pareto[best_sc_idx])[0]

ex_ws       = _unified_ws(true_pareto_F)
ex_best_idx = np.argmin(ex_ws)
ex_F_real   = true_pareto_F.copy()
ex_F_real[:, 2] = -ex_F_real[:, 2]

ex_bc_sites  = sorted(int(s) for s in true_pareto_idx[ex_best_idx])
ex_bc_cost   = ex_F_real[ex_best_idx, 0]
ex_bc_impact = ex_F_real[ex_best_idx, 1]
ex_bc_safety = ex_F_real[ex_best_idx, 2]
ex_bc_ws     = ex_ws[ex_best_idx]

nsga_bc_sites  = sorted(bc_sites.tolist())
nsga_bc_cost   = F_real[best_sc_idx, 0]
nsga_bc_impact = F_real[best_sc_idx, 1]
nsga_bc_safety = F_real[best_sc_idx, 2]
nsga_bc_ws     = ws[best_sc_idx]

# Triangle geometry
bc_coords = [(df.iloc[s]['latitude'], df.iloc[s]['longitude']) for s in bc_sites]
d12 = geopy.distance.geodesic(bc_coords[0], bc_coords[1]).km
d13 = geopy.distance.geodesic(bc_coords[0], bc_coords[2]).km
d23 = geopy.distance.geodesic(bc_coords[1], bc_coords[2]).km
sides = sorted([d12, d13, d23])
sp = sum(sides) / 2
triangle_area = math.sqrt(max(sp*(sp-sides[0])*(sp-sides[1])*(sp-sides[2]), 0))
aspect_ratio  = sides[2] / sides[0] if sides[0] > 0 else float('inf')
spread_avg    = np.mean(sides)

MUNICIPIOS = {
    'Sao Filipe': (14.895, -24.495),
    'Mosteiros': (14.858, -24.348),
    'Santa Catarina': (14.915, -24.287),
}

def nearest_municipality(lat, lon):
    """Return municipality label with the smallest geodesic distance to a point."""
    return min(MUNICIPIOS, key=lambda m: geopy.distance.geodesic((lat, lon), MUNICIPIOS[m]).km)

mun_set = set()
for s in bc_sites:
    mun_set.add(nearest_municipality(df.iloc[s]['latitude'], df.iloc[s]['longitude']))

pareto_df = pd.DataFrame({'Cost': F_real[:, 0], 'Impact': F_real[:, 1], 'Safety': F_real[:, 2]})
for j in range(K_SELECT):
    pareto_df[f'idx{j+1}'] = [np.where(X_pareto[i])[0][j]
                               if j < len(np.where(X_pareto[i])[0]) else -1
                               for i in range(len(X_pareto))]
pareto_df.to_csv(get_output_path('nsga2_pareto_front_results.csv'), index=False)

sites_match = set(nsga_bc_sites) == set(ex_bc_sites)
obj_close   = (np.isclose(nsga_bc_cost, ex_bc_cost, atol=1e-6) and
               np.isclose(nsga_bc_impact, ex_bc_impact, atol=1e-6) and
               np.isclose(nsga_bc_safety, ex_bc_safety, atol=1e-6))
ws_close    = np.isclose(nsga_bc_ws, ex_bc_ws, atol=1e-6)

print(f'Best compromise: sites={nsga_bc_sites}  '
      f'Cost={nsga_bc_cost:.4f}, Impact={nsga_bc_impact:.4f}, Safety={nsga_bc_safety:.4f}')
if sites_match and ws_close:
    print('✓ MATCH: NSGA-II found the true optimal compromise.')
elif sites_match:
    print('✓ SITES MATCH')
else:
    print(f'✗ DIFF: WS_NSGA-II={nsga_bc_ws:.6f}, WS_Exhaustive={ex_bc_ws:.6f}')

# --- Multi-scenario best compromise (Table 13) ---
SCENARIOS = {
    'S1_Regional':      {'Cost': 0.295, 'Impact': 0.356, 'Safety': 0.349},
    'S2_Economic':      {'Cost': 0.600, 'Impact': 0.200, 'Safety': 0.200},
    'S3_Environmental': {'Cost': 0.200, 'Impact': 0.600, 'Safety': 0.200},
    'S4_Social':        {'Cost': 0.200, 'Impact': 0.200, 'Safety': 0.600},
}
scenario_results = {}
print('\n--- SCENARIO ANALYSIS (Table 13) ---')
for _sname, _sw in SCENARIOS.items():
    _fnorm = (F_pareto - unified_min) / unified_range
    _ws_s  = (_sw['Cost']   * _fnorm[:, 0] +
              _sw['Impact'] * _fnorm[:, 1] +
              _sw['Safety'] * _fnorm[:, 2])
    _idx_s  = int(np.argmin(_ws_s))
    _sites_s = sorted(np.where(X_pareto[_idx_s])[0].tolist())
    _coords_s = [(df.iloc[s]['latitude'], df.iloc[s]['longitude']) for s in _sites_s]
    _munis_s  = [nearest_municipality(lat, lon) for lat, lon in _coords_s]
    scenario_results[_sname] = {
        'sites'    : _sites_s,
        'Cost'     : float(F_real[_idx_s, 0]),
        'Impact'   : float(F_real[_idx_s, 1]),
        'Safety'   : float(F_real[_idx_s, 2]),
        'ws'       : float(_ws_s[_idx_s]),
        'coords'   : _coords_s,
        'municipalities': _munis_s,
        'weights'  : _sw,
    }
    print(f'  {_sname}: sites={_sites_s}, '
          f'Cost={F_real[_idx_s,0]:.4f}, '
          f'Impact={F_real[_idx_s,1]:.4f}, '
          f'Safety={-F_pareto[_idx_s,2]:.4f}, '
          f'munis={_munis_s}')

with open(get_output_path('scenario_analysis.json'), 'w') as _f:
    json.dump(scenario_results, _f, indent=2)


# =============================================================================
# SECTION 7.5 — DECISION MAKING: CONSTRAINED vs UNCONSTRAINED COMPARISON
# =============================================================================
# spread_avg is already computed above from the best-compromise triangle
spread_constrained = spread_avg

_unc_cache = get_output_path('unconstrained_baseline.pkl')
if os.path.exists(_unc_cache):
    print('Loading unconstrained baseline from cache…')
    with open(_unc_cache, 'rb') as _f:
        _unc_cache_data = pickle.load(_f)
    spread_avg_unconstrained = _unc_cache_data['spread_avg_unconstrained']
    hv_unconstrained         = _unc_cache_data['hv_unconstrained']
else:
    print('Running unconstrained NSGA-II (5 runs, 150 gen) for baseline…')
    _prob_unc = SelectAreasRobust(df, n_select=K_SELECT)
    _algo_unc = NSGA2(
        pop_size=POP_SIZE,
        sampling=BinarySamplingFixedK(K_SELECT, len(df)),
        crossover=TwoPointCrossover(),
        mutation=BitflipMutation(prob=1.0 / len(df)),
        eliminate_duplicates=True,
    )
    _hv_unc_runs = []
    _spreads_unc = []
    for _seed in SEEDS_LIST[:5]:
        np.random.seed(_seed)
        _res_unc = minimize(_prob_unc, _algo_unc, ('n_gen', N_GEN), seed=_seed, verbose=False)
        _F_unc = _res_unc.F[_res_unc.F[:, 0] < 1e4] if _res_unc.F is not None else np.array([])
        if len(_F_unc) == 0:
            continue
        _hv_unc_runs.append(float(HV(ref_point=np.max(_F_unc, axis=0) + 1e-3)(_F_unc)))
        # pick best-compromise from unconstrained front and compute spread
        _ws_unc = ((_F_unc - _F_unc.min(0)) / (_F_unc.max(0) - _F_unc.min(0) + 1e-12)).sum(1)
        _bc_unc = np.argmin(_ws_unc)
        _bc_idx_unc = np.where(_res_unc.X[_bc_unc])[0]
        _bc_c_unc = [(df.iloc[s]['latitude'], df.iloc[s]['longitude']) for s in _bc_idx_unc]
        if len(_bc_c_unc) >= 3:
            _d12u = geopy.distance.geodesic(_bc_c_unc[0], _bc_c_unc[1]).km
            _d13u = geopy.distance.geodesic(_bc_c_unc[0], _bc_c_unc[2]).km
            _d23u = geopy.distance.geodesic(_bc_c_unc[1], _bc_c_unc[2]).km
            _spreads_unc.append(np.mean([_d12u, _d13u, _d23u]))
    spread_avg_unconstrained = float(np.mean(_spreads_unc)) if _spreads_unc else 0.0
    hv_unconstrained         = float(np.max(_hv_unc_runs)) if _hv_unc_runs else 0.0
    with open(_unc_cache, 'wb') as _f:
        pickle.dump({'spread_avg_unconstrained': spread_avg_unconstrained,
                     'hv_unconstrained': hv_unconstrained}, _f)
    print(f'  Unconstrained spread={spread_avg_unconstrained:.3f} km, HV={hv_unconstrained:.6f}')

# Derived metrics
_spread_delta   = spread_constrained - spread_avg_unconstrained
spread_improvement_pct = (100.0 * _spread_delta / spread_avg_unconstrained
                          if spread_avg_unconstrained > 0 else 0.0)
hv_reduction_pct       = (100.0 * (hv_unconstrained - hv_nsga2_best) / hv_unconstrained
                          if hv_unconstrained > 0 else 0.0)
tradeoff_ratio         = (abs(spread_improvement_pct) / abs(hv_reduction_pct)
                          if abs(hv_reduction_pct) > 1e-9 else float('inf'))

if spread_improvement_pct >= 5 and hv_reduction_pct <= 15:
    _interp = (f'The min-dist constraint ({MIN_DIST_KM} km) improved geographic spread by '
               f'{spread_improvement_pct:.1f}% ({spread_constrained - spread_avg_unconstrained:+.2f} km) '
               f'with only {hv_reduction_pct:.1f}% HV loss — a favourable trade-off '
               f'(ratio {tradeoff_ratio:.2f}:1).')
elif spread_improvement_pct < 0:
    _interp = (f'Geographic spread decreased by {abs(spread_improvement_pct):.1f}% under '
               f'the {MIN_DIST_KM} km constraint; consider relaxing the distance requirement.')
else:
    _interp = (f'The {MIN_DIST_KM} km constraint yielded a {spread_improvement_pct:.1f}% spread '
               f'improvement at the cost of {hv_reduction_pct:.1f}% HV reduction '
               f'(trade-off ratio {tradeoff_ratio:.2f}:1).')

_dm_metrics = {
    'spread_constrained_km'   : spread_constrained,
    'spread_unconstrained_km' : spread_avg_unconstrained,
    'spread_improvement_pct'  : spread_improvement_pct,
    'hv_constrained_best'     : hv_nsga2_best,
    'hv_unconstrained'        : hv_unconstrained,
    'hv_reduction_pct'        : hv_reduction_pct,
    'tradeoff_ratio'          : tradeoff_ratio,
    'interpretation'          : _interp,
}
with open(get_output_path('decision_making_metrics.json'), 'w') as _f:
    json.dump(_dm_metrics, _f, indent=2)
print(f'Decision Making: spread {spread_avg_unconstrained:.2f}→{spread_constrained:.2f} km '
      f'(+{spread_improvement_pct:.1f}%), HV -{hv_reduction_pct:.1f}%, ratio {tradeoff_ratio:.2f}:1')


# =============================================================================
# SECTION 8 — CORRELATION ANALYSIS PREP
# =============================================================================

exclude = {'latitude', 'longitude', 'CostDef', 'ImpactDef', 'SafetyDef'}
criteria_cols = [c for c in df.columns
                 if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
FRIENDLY = {
    'Slope (degrees)': 'Slope', 'Aspect (degrees)': 'Aspect',
    'Urban Areas (m)': 'D. Urban', 'Aerodrome (m)': 'D. Aerodrome',
    'Rivers (m)': 'D. Rivers', 'Road Network (m)': 'D. Roads',
    'Wells/Boreholes (m)': 'D. Wells', 'Cultivated Areas (m)': 'D. Cultivated',
    'Coastline (m)': 'D. Coastline', 'Permeability (Distance) (m)': 'D. Permeability',
    'Protected Areas (m)': 'D. Protected',
}
corr_matrix = df[criteria_cols].corr(method='spearman')
pairs = []
for i, c1 in enumerate(criteria_cols):
    for j, c2 in enumerate(criteria_cols):
        if i < j:
            rho = corr_matrix.loc[c1, c2]
            if abs(rho) >= 0.5:
                pairs.append((FRIENDLY.get(c1, c1), FRIENDLY.get(c2, c2), rho))
pairs.sort(key=lambda x: abs(x[2]), reverse=True)

df['Municipality'] = df.apply(lambda r: nearest_municipality(r['latitude'], r['longitude']), axis=1)


# =============================================================================
# SECTION 9 — FIGURES
# =============================================================================

# --- 9.1 Hypervolume convergence ---
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(hv_gens, hv_mean, 'b-', linewidth=2, label='Mean HV')
ax.fill_between(hv_gens, hv_mean - hv_std, hv_mean + hv_std,
                alpha=0.25, color='steelblue', label='±1 Std')
ax.axhline(y=hv_exhaustive, color='r', ls='--', lw=1.2,
           label=f'True Pareto HV = {hv_exhaustive:.4f}')
hv90 = 0.9 * hv_mean[-1]
conv_gen = next((g for g, h in zip(hv_gens, hv_mean) if h >= hv90), None)
if conv_gen:
    ax.axvline(x=conv_gen, color='green', ls=':', lw=1, alpha=0.7,
               label=f'90% gen {conv_gen}')
ax.set_xlabel('Generation'); ax.set_ylabel('Hypervolume')
ax.set_title(f'Hypervolume Convergence ({N_RUNS} Runs)')
ax.legend(loc='lower right')
fig.savefig(get_fig_path('fig1_hypervolume_convergence.png'))
plt.close()
print('Fig 1 saved.')

# --- 9.2 3D Pareto front ---
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(F_real[:, 0], F_real[:, 1], F_real[:, 2],
                c=F_real[:, 2], cmap='RdYlGn', s=40, alpha=0.8)
ax.scatter(*F_real[best_sc_idx], c='red', marker='*', s=250,
           edgecolors='black', linewidths=1.5, label='Best Compromise', zorder=10)
for eidx, (mk, col, lab) in zip(extreme_idxs,
        [('D', 'green', 'Min Cost'), ('s', 'blue', 'Min Impact'), ('^', 'orange', 'Max Safety')]):
    ax.scatter(*F_real[eidx], c=col, marker=mk, s=100,
               edgecolors='black', linewidths=1, label=lab, zorder=9)
ax.set_xlabel('Cost (min)'); ax.set_ylabel('Impact (min)'); ax.set_zlabel('Safety (max)')
ax.set_title('3D Pareto Front'); ax.legend(loc='upper left', fontsize=8)
fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10, label='Safety')
fig.savefig(get_fig_path('fig2_pareto_front_3d.png'))
plt.close()
print('Fig 2 saved.')

# --- 9.3 Pairwise projections ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panel_labels = ['(a)', '(b)', '(c)']
for ax, plabel, (i, j, xl, yl, col) in zip(axes, panel_labels,
        [(0, 1, 'Cost', 'Impact', 'steelblue'),
         (0, 2, 'Cost', 'Safety', 'forestgreen'),
         (1, 2, 'Impact', 'Safety', 'purple')]):
    ax.scatter(F_real[:, i], F_real[:, j], c=col, s=25, alpha=0.6)
    for eidx, (mk, mcol) in zip(extreme_idxs,
                                 [('D', 'green'), ('s', 'blue'), ('^', 'orange')]):
        ax.scatter(F_real[eidx, i], F_real[eidx, j], c=mcol, marker=mk, s=80,
                   edgecolors='black', linewidths=0.8, zorder=5)
    ax.scatter(F_real[best_sc_idx, i], F_real[best_sc_idx, j],
               c='red', marker='*', s=160, edgecolors='black', linewidths=1, zorder=6)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(f'{plabel} {xl} vs. {yl}')
legend_els = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='green', ms=8, label='Min Cost'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='blue', ms=8, label='Min Impact'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='orange', ms=8, label='Max Safety'),
    Line2D([0],[0], marker='*', color='w', markerfacecolor='red', ms=12, label='Best Compromise'),
]
axes[0].legend(handles=legend_els, fontsize=7)
fig.tight_layout()
fig.savefig(get_fig_path('fig3_pareto_projections.png'))
plt.close()
print('Fig 3 saved.')

# --- 9.4 Spearman heatmap ---
corr_friendly = corr_matrix.rename(columns=FRIENDLY, index=FRIENDLY)
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_friendly, dtype=bool), k=1)
sns.heatmap(corr_friendly, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.7, 'label': 'Spearman ρ'}, ax=ax)
ax.set_title('Spearman Rank Correlation Matrix')
fig.tight_layout()
fig.savefig(get_fig_path('fig4_spearman_heatmap.png'))
plt.close()
print('Fig 4 saved.')

# --- 9.5 k-sensitivity ---
k_values = [2, 3, 4, 5, 6, 7, 8, 10]
k_results = []
for k in k_values:
    prob_k = SelectAreasRobust(df, n_select=k)
    algo_k = NSGA2(pop_size=POP_SIZE,
                   sampling=BinarySamplingFixedK(k=k, n_var=len(df)),
                   crossover=TwoPointCrossover(),
                   mutation=BitflipMutation(prob=1.0/len(df)),
                   repair=FixKRepair(k=k),
                   eliminate_duplicates=True)
    t0 = time.time()
    res_k = minimize(prob_k, algo_k, ('n_gen', N_GEN), seed=GLOBAL_SEED, verbose=False)
    tk = time.time() - t0
    Fv = res_k.F[res_k.F[:, 0] < 1e4]
    Xv = res_k.X[res_k.F[:, 0] < 1e4]
    hv_k = hv_norm_k = bc_cost_k = bc_imp_k = bc_safe_k = avg_d = 0.0
    if len(Fv) > 0:
        ref_k   = np.max(Fv, axis=0) + np.abs(np.max(Fv, axis=0)) * 0.01 + 1e-6
        ideal_k = np.array([Fv[:, 0].min(), Fv[:, 1].min(), Fv[:, 2].min()])
        hv_k    = float(HV(ref_point=ref_k)(Fv))
        max_vol = np.prod(np.abs(ref_k - ideal_k))
        hv_norm_k = hv_k / max_vol if max_vol > 0 else 0.0
        Fv_range = Fv.max(axis=0) - Fv.min(axis=0)
        Fv_range[Fv_range == 0] = 1.0
        Fv_norm  = (Fv - Fv.min(axis=0)) / Fv_range
        ws_k     = (SCENARIO_WEIGHTS['Cost']   * Fv_norm[:, 0] +
                    SCENARIO_WEIGHTS['Impact'] * Fv_norm[:, 1] +
                    SCENARIO_WEIGHTS['Safety'] * Fv_norm[:, 2])
        bi       = np.argmin(ws_k)
        sel_k    = np.where(Xv[bi])[0]
        bc_cost_k = Fv[bi, 0]; bc_imp_k = Fv[bi, 1]; bc_safe_k = -Fv[bi, 2]
        coords_k = df.iloc[sel_k][['latitude', 'longitude']].values
        dists_k  = [haversine(*coords_k[i], *coords_k[j])
                    for i in range(len(coords_k)) for j in range(i+1, len(coords_k))]
        avg_d = float(np.mean(dists_k)) if dists_k else 0.0
    k_results.append({'k': k, 'HV': hv_k, 'HV_norm': hv_norm_k,
                       'avg_cost_per_site':   bc_cost_k / k,
                       'avg_impact_per_site': bc_imp_k / k,
                       'avg_safety_per_site': bc_safe_k / k,
                       'avg_dist': avg_d, 'n_sol': len(Fv), 'time_s': tk})
    print(f'  k={k:2d}  HV_norm={hv_norm_k:.4f}  #sol={len(Fv)}  t={tk:.1f}s')

k_df = pd.DataFrame(k_results)
k_df.to_csv(get_output_path('k_sensitivity_results.csv'), index=False)

# --- Comprehensive 6-panel k-sensitivity figure ---
k_arr   = k_df['k'].values
hv_arr  = k_df['HV_norm'].values
eff_arr = hv_arr / k_arr                           # HV per facility
geo_arr = k_df['avg_dist'].values                  # geographic spread
sol_arr = k_df['n_sol'].values                     # solution diversity
elbow_k = int(k_df.loc[k_df['HV_norm'].idxmax(), 'k'])

# HV growth rate: % change vs previous k
hv_growth = np.zeros(len(k_arr))
for _i in range(1, len(k_arr)):
    prev = hv_arr[_i - 1]
    hv_growth[_i] = 100.0 * (hv_arr[_i] - prev) / prev if prev > 0 else 0.0

# Normalise metrics for heatmap (panel f)
def _norm_row(v):
    """Normalize a 1D array to [0, 1] with zero-vector fallback for constant input."""
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn) if mx > mn else np.zeros_like(v)

heatmap_data = np.vstack([
    _norm_row(hv_arr),           # HV Quality
    _norm_row(geo_arr),          # Geo Spread
    _norm_row(eff_arr),          # Efficiency
])
heatmap_labels = ['HV Quality', 'Geo Spread', 'Efficiency']

_C_RED   = '#e74c3c'
_C_GREEN = '#27ae60'

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Comprehensive k-Value Sensitivity Analysis', fontweight='bold', fontsize=13)

# (a) Normalized Hypervolume
ax = axes[0, 0]
ax.plot(k_arr, hv_arr, 'o-', color='steelblue', lw=2.5, ms=8)
ax.axvline(x=K_SELECT, color=_C_RED,   ls='--', lw=1.5, label=f'Current k={K_SELECT}')
ax.axvline(x=elbow_k,  color=_C_GREEN, ls='--', lw=1.5, label=f'Elbow k={elbow_k}')
ax.set_xlabel('k (Number of Facilities)'); ax.set_ylabel('Hypervolume')
ax.set_title('(a) Hypervolume vs. k'); ax.legend(fontsize=8)

# (b) HV per Facility efficiency
ax = axes[0, 1]
ax.plot(k_arr, eff_arr, 'o-', color='#f39c12', lw=2.5, ms=8)
ax.axvline(x=K_SELECT, color=_C_RED,   ls='--', lw=1.5)
ax.axvline(x=elbow_k,  color=_C_GREEN, ls='--', lw=1.5)
ax.set_xlabel('k (Number of Facilities)'); ax.set_ylabel('HV / k (Efficiency)')
ax.set_title('(b) Efficiency: HV per Facility')

# (c) Geographic Distribution
ax = axes[0, 2]
ax.plot(k_arr, geo_arr, 'o-', color=_C_GREEN, lw=2.5, ms=8)
ax.axvline(x=K_SELECT, color=_C_RED,   ls='--', lw=1.5)
ax.axvline(x=elbow_k,  color=_C_GREEN, ls='--', lw=1.5)
ax.set_xlabel('k (Number of Facilities)'); ax.set_ylabel('Avg. Distance (km)')
ax.set_title('(c) Geographic Distribution')

# (d) Diminishing Returns (HV growth rate %)
ax = axes[1, 0]
bar_colors = ['#b388ff' if v >= 50 else '#ce93d8' for v in hv_growth]
ax.bar(k_arr, hv_growth, color=bar_colors, edgecolor='white', width=0.6)
ax.axhline(y=50, color='black', ls='--', lw=1.5, label='50% growth threshold')
ax.set_xlabel('k (Number of Facilities)'); ax.set_ylabel('HV Growth Rate (%)')
ax.set_title('(d) Diminishing Returns Analysis'); ax.legend(fontsize=8)

# (e) Solution Diversity (Pareto front size)
ax = axes[1, 1]
ax.bar(k_arr, sol_arr, color='#00897b', edgecolor='white', width=0.6)
ax.axhline(y=POP_SIZE, color=_C_RED, ls='--', lw=1.5, label='Max population')
ax.set_xlabel('k (Number of Facilities)'); ax.set_ylabel('Pareto Front Size')
ax.set_title('(e) Solution Diversity'); ax.legend(fontsize=8)

# (f) Multi-Criteria Normalized Comparison (heatmap)
ax = axes[1, 2]
im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
               vmin=0, vmax=1, interpolation='nearest')
ax.set_xticks(range(len(k_arr))); ax.set_xticklabels(k_arr, fontsize=8)
ax.set_yticks(range(len(heatmap_labels))); ax.set_yticklabels(heatmap_labels, fontsize=8)
ax.set_xlabel('k (Number of Facilities)')
ax.set_title('(f) Multi-Criteria Normalized Comparison')
for _r in range(heatmap_data.shape[0]):
    for _c in range(heatmap_data.shape[1]):
        _val = heatmap_data[_r, _c]
        ax.text(_c, _r, f'{_val:.2f}', ha='center', va='center',
                fontsize=7, color='black' if 0.25 < _val < 0.75 else 'white')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
_comp_path = get_fig_path('k_sensitivity_comprehensive_analysis.png')
fig.savefig(_comp_path, dpi=300, bbox_inches='tight')
fig.savefig(get_fig_path('fig5_k_sensitivity.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Fig 5 saved (comprehensive): {_comp_path}')

# --- 9.6 Geographic Decision Map (Fig 9.4 — Publication-Ready) ---
# Style: clean point markers, hatched Pareto-optimal region, north arrow, scale bar.
# Matches the cartographic style from optimization_map_viz.py / CELULA_9_4_CODIGO.py.
print('=' * 70)
print('FIGURE 9.4: Geographic Decision Map (clean + terrain versions)')
print('=' * 70)
try:
    import matplotlib.patheffects as _pe
    from shapely.geometry import MultiPoint as _MultiPoint

    ISLAND_SHP = os.path.join(_DATA_ROOT, 'geologia_fogo.shp')

    # ── helpers ────────────────────────────────────────────────────────────
    def _ll2utm(lat, lon):
        pt = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326').to_crs(UTM_CRS)
        return float(pt.iloc[0].x), float(pt.iloc[0].y)

    def _sites_utm(indices):
        out = []
        for i in indices:
            row = df.iloc[int(i)]
            x, y = _ll2utm(float(row['latitude']), float(row['longitude']))
            out.append((x, y, int(i)))
        return out

    def _smart_offset(x, y, cx, cy, all_pts, W, H,
                      push=0.07, nudge=0.03):
        """Push label radially outward; prefer quadrant with fewest neighbours."""
        dx, dy = x - cx, y - cy
        mag = math.hypot(dx, dy) or 1.0
        base_ox = (dx / mag) * W * push
        base_oy = (dy / mag) * H * push
        # Secondary nudge perpendicular to avoid cluster overlap
        perp_x, perp_y = -dy / mag, dx / mag
        # Count neighbours in each perpendicular direction
        lx, ly = x + base_ox, y + base_oy
        left_n  = sum(1 for px, py, _ in all_pts
                      if math.hypot(px - (lx - perp_x * W * nudge),
                                    py - (ly - perp_y * H * nudge)) < W * 0.12)
        right_n = sum(1 for px, py, _ in all_pts
                      if math.hypot(px - (lx + perp_x * W * nudge),
                                    py - (ly + perp_y * H * nudge)) < W * 0.12)
        side = -1 if left_n < right_n else 1
        return (base_ox + side * perp_x * W * nudge,
                base_oy + side * perp_y * H * nudge)

    def _annotate_site(ax, x, y, lx, ly, label,
                       fontsize, fcolor, ecolor, zbase=20):
        """Annotate with arrow + opaque white-background coloured box."""
        ax.annotate(
            label,
            xy=(x, y), xytext=(lx, ly),
            arrowprops=dict(arrowstyle='-',
                            color=ecolor, lw=1.1,
                            shrinkA=6, shrinkB=4),
            fontsize=fontsize, fontweight='bold', color=fcolor,
            ha='center', va='center',
            zorder=zbase,
            bbox=dict(boxstyle='round,pad=0.40',
                      facecolor='white', edgecolor=ecolor,
                      linewidth=1.4, alpha=0.97),
        )

    def _draw_cartography(ax, bnds, W, H,
                          nb_color='black', sb_color='black'):
        """North arrow (top-right) + scale bar (bottom-right)."""
        na_x  = bnds[2] + W * 0.010
        na_y  = bnds[3] - H * 0.055
        arr_l = H * 0.055
        ax.annotate('', xy=(na_x, na_y),
                    xytext=(na_x, na_y - arr_l),
                    arrowprops=dict(arrowstyle='-|>', lw=2.2,
                                   color=nb_color, mutation_scale=18),
                    zorder=30)
        ax.text(na_x, na_y + arr_l * 0.30, 'N',
                fontsize=13, fontweight='bold',
                ha='center', color=nb_color, zorder=30)
        sc_len = 5000
        sb_xe  = bnds[2] - W * 0.018
        sb_xs  = sb_xe - sc_len
        sb_y   = bnds[1] + H * 0.038
        h      = H * 0.007
        ax.fill_between([sb_xs, sb_xs + sc_len / 2],
                        [sb_y - h, sb_y - h], [sb_y + h, sb_y + h],
                        color=sb_color, zorder=30)
        ax.fill_between([sb_xs + sc_len / 2, sb_xe],
                        [sb_y - h, sb_y - h], [sb_y + h, sb_y + h],
                        color='white', zorder=30)
        ax.plot([sb_xs, sb_xe], [sb_y - h, sb_y - h], color=sb_color, lw=1, zorder=31)
        ax.plot([sb_xs, sb_xe], [sb_y + h, sb_y + h], color=sb_color, lw=1, zorder=31)
        ax.plot([sb_xs, sb_xs], [sb_y - h, sb_y + h], color=sb_color, lw=1, zorder=31)
        ax.plot([sb_xe, sb_xe], [sb_y - h, sb_y + h], color=sb_color, lw=1, zorder=31)
        ax.text(sb_xs, sb_y - H * 0.013, '0',
                ha='center', fontsize=8.5, fontweight='bold',
                color=sb_color, zorder=31)
        ax.text(sb_xe, sb_y - H * 0.013, '5 km',
                ha='center', fontsize=8.5, fontweight='bold',
                color=sb_color, zorder=31)

    # Collect all sites that appear in ANY extreme solution (multi-criteria sites)
    _all_extreme_sites = (set(low_cost_sites.tolist()) |
                          set(low_impact_sites.tolist()) |
                          set(high_safety_sites.tolist()))
    _multicriteria_sites = {s for s in bc_sites.tolist() if s in _all_extreme_sites}

    def _build_legend(has_multicriteria, loc='upper left'):
        _els = [
            Line2D([0],[0], marker='*', color='w', markerfacecolor='#8E44AD',
                   markeredgecolor='#4A235A', markersize=15,
                   label='Best Compromise (BC)'),
            Line2D([0],[0], marker='o', color='w', markerfacecolor='#2980B9',
                   markeredgecolor='#1A5276', markersize=11,
                   label='Min. Cost'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='#27AE60',
                   markeredgecolor='#1D6A39', markersize=11,
                   label='Min. Impact'),
            Line2D([0],[0], marker='D', color='w', markerfacecolor='#E67E22',
                   markeredgecolor='#A04000', markersize=11,
                   label='Max. Safety'),
        ]
        if has_multicriteria:
            _els.append(
                Line2D([0],[0], marker='*', color='w', markerfacecolor='#8E44AD',
                       markeredgecolor='#E67E22', markersize=15,
                       markeredgewidth=3.0,
                       label='Multi-criteria feasible site')
            )
        _els += [
            mpatches.Patch(facecolor='#F2E0FF', edgecolor='#9B59B6',
                           linewidth=1.4, linestyle='--', hatch='//',
                           label='Pareto-optimal region'),
            Line2D([0],[0], marker='s', color='w', markerfacecolor='#1C2833',
                   markersize=10, label='Settlements'),
        ]
        return _els

    def _draw_sites(ax, bc_coords, cost_coords, imp_coords, safe_coords,
                    bc_set, safe_set, cx, cy, W, H,
                    all_pts, multicriteria_set=None,
                    push_normal=0.07, push_bc=0.10):
        """Draw all solution markers + annotated labels."""
        if multicriteria_set is None:
            multicriteria_set = set()

        # Min. Cost
        for _x, _y, _sid in cost_coords:
            if _sid in bc_set:
                continue
            ax.scatter(_x, _y, s=180, c='#2980B9', marker='o',
                       edgecolors='#1A5276', linewidths=1.2, zorder=10)
            ox, oy = _smart_offset(_x, _y, cx, cy, all_pts, W, H,
                                   push=push_normal)
            _annotate_site(ax, _x, _y, _x + ox, _y + oy,
                           f'#{_sid}', 9.5, '#1A5276', '#2980B9')

        # Min. Impact
        for _x, _y, _sid in imp_coords:
            if _sid in bc_set:
                continue
            ax.scatter(_x, _y, s=180, c='#27AE60', marker='s',
                       edgecolors='#1D6A39', linewidths=1.2, zorder=10)
            ox, oy = _smart_offset(_x, _y, cx, cy, all_pts, W, H,
                                   push=push_normal)
            _annotate_site(ax, _x, _y, _x + ox, _y + oy,
                           f'#{_sid}', 9.5, '#1D6A39', '#27AE60')

        # Max. Safety (non-BC)
        for _x, _y, _sid in safe_coords:
            if _sid in bc_set:
                continue
            ax.scatter(_x, _y, s=210, c='#E67E22', marker='D',
                       edgecolors='#A04000', linewidths=1.2, zorder=10)
            ox, oy = _smart_offset(_x, _y, cx, cy, all_pts, W, H,
                                   push=push_normal)
            _annotate_site(ax, _x, _y, _x + ox, _y + oy,
                           f'#{_sid}', 9.5, '#A04000', '#E67E22')

        # Best Compromise — orange ring if multi-criteria, then star on top
        for _x, _y, _sid in bc_coords:
            # Orange ring for multi-criteria feasible sites
            ax.scatter(_x, _y, s=800, facecolors='none',
                       edgecolors='#E67E22', linewidths=3.2, zorder=11)
            ax.scatter(_x, _y, s=360, c='#8E44AD', marker='*',
                       edgecolors='#4A235A', linewidths=1.0, zorder=12)
            ox, oy = _smart_offset(_x, _y, cx, cy, all_pts, W, H,
                                   push=push_bc)
            _annotate_site(ax, _x, _y, _x + ox, _y + oy,
                           f'#{_sid}', 11.5, '#4A235A', '#8E44AD', zbase=22)

    # ── shared data ─────────────────────────────────────────────────────────
    _bc_coords   = _sites_utm(bc_sites)
    _cost_coords = _sites_utm(low_cost_sites)
    _imp_coords  = _sites_utm(low_impact_sites)
    _safe_coords = _sites_utm(high_safety_sites)

    _bc_set   = {c[2] for c in _bc_coords}
    _safe_set = {c[2] for c in _safe_coords}
    _all_pts  = _bc_coords + _cost_coords + _imp_coords + _safe_coords

    _bc_pts = _MultiPoint([(x, y) for x, y, _ in _bc_coords])
    _pareto_poly = _bc_pts.convex_hull

    _settlements = {
        'São Filipe':    (14.8967, -24.4958, 'left'),
        'Mosteiros':     (15.0391, -24.3347, 'right'),
        'Sta. Catarina': (14.9145, -24.2870, 'right'),
    }

    _island94 = gpd.read_file(ISLAND_SHP).to_crs(UTM_CRS)
    _bnds = _island94.total_bounds
    _W  = _bnds[2] - _bnds[0]
    _H  = _bnds[3] - _bnds[1]
    _cx = (_bnds[0] + _bnds[2]) / 2
    _cy = (_bnds[1] + _bnds[3]) / 2
    # ── VERSION A — CLEAN (white / geological background) ──────────────────
    print('  Rendering clean version ...')
    _figA, _axA = plt.subplots(figsize=(11, 12), dpi=300)
    _axA.set_aspect('equal')
    _axA.axis('off')

    _island94.plot(ax=_axA, facecolor='#F6F5F0', edgecolor='#3A3A3A',
                   linewidth=0.9, zorder=1)

    # Pareto region
    if (not _pareto_poly.is_empty
            and _pareto_poly.geom_type in ('Polygon', 'MultiPolygon')):
        _px, _py = _pareto_poly.exterior.xy
        _axA.fill(_px, _py, facecolor='#F2E0FF', alpha=0.45,
                  edgecolor='#9B59B6', linewidth=1.6,
                  linestyle='--', hatch='//', zorder=2)

    _draw_sites(_axA, _bc_coords, _cost_coords, _imp_coords, _safe_coords,
                _bc_set, _safe_set, _cx, _cy, _W, _H, _all_pts,
                multicriteria_set=_multicriteria_sites)

    # Settlements
    for _nm, (_lt, _ln, _al) in _settlements.items():
        _sx, _sy = _ll2utm(_lt, _ln)
        _axA.scatter(_sx, _sy, s=80, c='#1C2833', marker='s', zorder=15,
                     edgecolors='white', linewidths=0.8)
        _sxoff = _W * 0.013 * (-1 if _al == 'left' else 1)
        _sha   = 'right' if _al == 'left' else 'left'
        _axA.text(_sx + _sxoff, _sy, _nm,
                  fontsize=9.5, fontstyle='italic', fontweight='semibold',
                  ha=_sha, va='center', zorder=16,
                  bbox=dict(facecolor='white', alpha=0.88,
                            edgecolor='none', boxstyle='round,pad=0.25'))

    _draw_cartography(_axA, _bnds, _W, _H)
    _axA.legend(handles=_build_legend(bool(_multicriteria_sites)), loc='upper left',
                fontsize=10, frameon=True, edgecolor='#AAAAAA',
                facecolor='white', framealpha=0.97,
                title='Decision Map', title_fontsize=11,
                handlelength=2.2, handleheight=1.8,
                borderpad=1.0, labelspacing=0.65)
    _figA.tight_layout(pad=0.8)
    _outA = get_fig_path('fig_9_4_clean')
    _figA.savefig(_outA + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    _figA.savefig(_outA + '.svg', format='svg', bbox_inches='tight', facecolor='white')
    plt.close(_figA)
    print('  Clean version saved: ' + _outA + '.png')

    # ── VERSION B — TERRAIN (OpenTopoMap via contextily) ───────────────────
    print('  Rendering terrain version (contextily) ...')
    try:
        import contextily as cx

        # Work in Web Mercator (EPSG:3857) for basemap tiles
        _CRS_WM  = 'EPSG:3857'
        _island_wm = _island94.to_crs(_CRS_WM)
        _bnds_wm   = _island_wm.total_bounds
        _Wwm  = _bnds_wm[2] - _bnds_wm[0]
        _Hwm  = _bnds_wm[3] - _bnds_wm[1]
        _cxwm = (_bnds_wm[0] + _bnds_wm[2]) / 2
        _cywm = (_bnds_wm[1] + _bnds_wm[3]) / 2

        def _ll2wm(lat, lon):
            pt = gpd.GeoSeries([Point(lon, lat)], crs='EPSG:4326').to_crs(_CRS_WM)
            return float(pt.iloc[0].x), float(pt.iloc[0].y)

        def _sites_wm(indices):
            out = []
            for i in indices:
                row = df.iloc[int(i)]
                x, y = _ll2wm(float(row['latitude']), float(row['longitude']))
                out.append((x, y, int(i)))
            return out

        _bc_wm   = _sites_wm(bc_sites)
        _cost_wm = _sites_wm(low_cost_sites)
        _imp_wm  = _sites_wm(low_impact_sites)
        _safe_wm = _sites_wm(high_safety_sites)
        _all_wm  = _bc_wm + _cost_wm + _imp_wm + _safe_wm

        _bc_pts_wm   = _MultiPoint([(x, y) for x, y, _ in _bc_wm])
        _pareto_wm   = _bc_pts_wm.convex_hull

        _figB, _axB = plt.subplots(figsize=(11, 12), dpi=300)
        _axB.set_aspect('equal')
        _axB.axis('off')

        # Island outline on top of tiles (transparent fill)
        _island_wm.plot(ax=_axB, facecolor='none', edgecolor='#222222',
                        linewidth=1.2, zorder=3)

        # Add OpenTopoMap basemap (shows mountain paths, contours, roads)
        cx.add_basemap(_axB,
                       crs=_CRS_WM,
                       source=cx.providers.OpenTopoMap,
                       zoom='auto',
                       zorder=1)

        # Clip view to island + small margin
        _mg = max(_Wwm, _Hwm) * 0.04
        _axB.set_xlim(_bnds_wm[0] - _mg, _bnds_wm[2] + _mg)
        _axB.set_ylim(_bnds_wm[1] - _mg, _bnds_wm[3] + _mg)

        # Pareto region (Web Mercator coords)
        if (not _pareto_wm.is_empty
                and _pareto_wm.geom_type in ('Polygon', 'MultiPolygon')):
            _pwx, _pwy = _pareto_wm.exterior.xy
            _axB.fill(_pwx, _pwy, facecolor='#F2E0FF', alpha=0.35,
                      edgecolor='#9B59B6', linewidth=1.8,
                      linestyle='--', hatch='//', zorder=4)

        # Re-use draw helper (pass Web Mercator versions)
        _draw_sites(_axB, _bc_wm, _cost_wm, _imp_wm, _safe_wm,
                    _bc_set, _safe_set, _cxwm, _cywm, _Wwm, _Hwm, _all_wm,
                    multicriteria_set=_multicriteria_sites,
                    push_normal=0.075, push_bc=0.11)

        # Settlements
        for _nm, (_lt, _ln, _al) in _settlements.items():
            _sx, _sy = _ll2wm(_lt, _ln)
            _axB.scatter(_sx, _sy, s=80, c='#111111', marker='s', zorder=15,
                         edgecolors='white', linewidths=0.8)
            _sxoff = _Wwm * 0.013 * (-1 if _al == 'left' else 1)
            _sha   = 'right' if _al == 'left' else 'left'
            _axB.text(_sx + _sxoff, _sy, _nm,
                      fontsize=9.5, fontstyle='italic', fontweight='semibold',
                      ha=_sha, va='center', zorder=16,
                      bbox=dict(facecolor='white', alpha=0.90,
                                edgecolor='none', boxstyle='round,pad=0.25'))

        _draw_cartography(_axB, _bnds_wm, _Wwm, _Hwm,
                          nb_color='black', sb_color='black')
        _axB.legend(handles=_build_legend(bool(_multicriteria_sites)), loc='upper left',
                    fontsize=10, frameon=True, edgecolor='#AAAAAA',
                    facecolor='white', framealpha=0.97,
                    title='Decision Map', title_fontsize=11,
                    handlelength=2.2, handleheight=1.8,
                    borderpad=1.0, labelspacing=0.65)
        _figB.tight_layout(pad=0.8)
        _outB = get_fig_path('fig_9_4_terrain')
        _figB.savefig(_outB + '.png', dpi=300, bbox_inches='tight',
                      facecolor='white')
        _figB.savefig(_outB + '.svg', format='svg', bbox_inches='tight',
                      facecolor='white')
        plt.close(_figB)
        print('  Terrain version saved: ' + _outB + '.png')

    except ImportError:
        print('  contextily not installed — terrain version skipped.')
        print('  Install with:  pip install contextily')
    except Exception as _eB:
        import traceback as _tb2; _tb2.print_exc()
        print('  Terrain version failed: ' + str(_eB))

    # Keep backward-compatible alias pointing to clean version
    import shutil as _sh
    _sh.copy2(_outA + '.png', get_fig_path('fig_9_4_geographic_dispersion') + '.png')
    print('Fig 9.4 done.')

except Exception as _e94:
    import traceback as _tb; _tb.print_exc()
    print('  Fig 9.4 skipped: ' + str(_e94))


# =============================================================================
# SECTION 10 — DATA EXPORT
# =============================================================================

pareto_df.to_csv(get_output_path('nsga2_pareto_front_results.csv'), index=False)

validation = {
    'nsga2': {
        'hv_best': float(hv_nsga2_best), 'hv_mean': float(hv_finals.mean()),
        'hv_std': float(hv_finals.std()), 'n_runs': N_RUNS,
        'avg_time_s': float(time_nsga2_avg),
        'best_compromise_sites': bc_sites.tolist(),
        'best_compromise_objectives': {
            'cost': float(F_real[best_sc_idx, 0]),
            'impact': float(F_real[best_sc_idx, 1]),
            'safety': float(F_real[best_sc_idx, 2]),
        },
    },
    'exhaustive': {'hv': float(hv_exhaustive), 'n_pareto': len(true_pareto_F),
                   'time_s': float(time_exhaustive)},
    'comparison': {'gap_pct': float(gap_pct), 'capture_rate_pct': float(capture_rate),
                   'speedup': float(speedup)},
    'triangle': {'area_km2': float(triangle_area), 'aspect_ratio': float(aspect_ratio),
                 'avg_spread_km': float(spread_avg),
                 'sides_km': [float(s) for s in sides]},
    'parameters': {'k': K_SELECT, 'min_dist_km': MIN_DIST_KM,
                   'pop_size': POP_SIZE, 'n_gen': N_GEN, 'seed': GLOBAL_SEED},
}
with open(get_output_path('validation_metrics.json'), 'w') as f:
    json.dump(validation, f, indent=2)

site_info = []
for s in bc_sites:
    row = df.iloc[s]
    site_info.append({
        'site_id': int(s), 'latitude': float(row['latitude']),
        'longitude': float(row['longitude']), 'cost': float(row['CostDef']),
        'impact': float(row['ImpactDef']), 'safety': float(row['SafetyDef']),
        'municipality': nearest_municipality(row['latitude'], row['longitude']),
    })
pd.DataFrame(site_info).to_csv(get_output_path('best_compromise_sites.csv'), index=False)

# --- results_summary.txt ---
_sep = '=' * 70
_lines = [
    _sep,
    'LANDFILL SITE OPTIMIZATION -- RESULTS SUMMARY',
    'Run timestamp        : ' + _run_ts,
    'Results directory    : ' + os.path.abspath(RESULTS_DIR),
    _sep,
    '',
    '--- CONFIGURATION ---',
    '  k (sites/solution) : ' + str(K_SELECT),
    '  Min pairwise dist  : ' + str(MIN_DIST_KM) + ' km',
    '  Population size    : ' + str(POP_SIZE),
    '  Generations        : ' + str(N_GEN),
    '  Independent runs   : ' + str(N_RUNS),
    '  Parallel workers   : ' + str(N_JOBS) + ' / ' + str(multiprocessing.cpu_count()) + ' cores',
    '  Global seed        : ' + str(GLOBAL_SEED),
    '  Candidate sites    : ' + str(n_sites),
    '  Blocked (protected): ' + str(count_blocked),
    '',
    '--- NSGA-II TIMING ---',
    '  Total wall time    : {:.1f} s  ({:.2f} min)'.format(time_nsga2_total, time_nsga2_total / 60),
    '  Avg per run        : {:.1f} s +/- {:.1f} s'.format(time_nsga2_avg, time_nsga2_std),
    '  Fastest run        : {:.1f} s  (run #{})'.format(min(run_times), run_times.index(min(run_times)) + 1),
    '  Slowest run        : {:.1f} s  (run #{})'.format(max(run_times), run_times.index(max(run_times)) + 1),
    '',
    '--- EXHAUSTIVE SEARCH ---',
    '  Total combinations : C({},{}) = {:,}'.format(n_sites, K_SELECT, comb(n_sites, K_SELECT)),
    '  Feasible solutions : {:,}'.format(len(feasible_F)),
    '  True Pareto size   : {:,}'.format(len(true_pareto_F)),
    '  Exhaustive time    : {:.1f} s  ({:.2f} min)'.format(time_exhaustive, time_exhaustive / 60),
    '  Speedup (Exh/NSGA2): {:.1f}x'.format(speedup),
    '',
    '--- HYPERVOLUME ---',
    '  NSGA-II HV best    : {:.6f}'.format(hv_nsga2_best),
    '  NSGA-II HV mean    : {:.6f} +/- {:.6f}'.format(hv_finals_unified.mean(), hv_finals_unified.std()),
    '  Exhaustive HV      : {:.6f}'.format(hv_exhaustive),
    '  Gap                : {:.4f} %'.format(gap_pct),
    '  Capture rate       : {:.4f} %'.format(capture_rate),
    '  Best run           : #{}  (seed={})'.format(best_run_idx + 1, SEEDS_LIST[best_run_idx]),
    '  HV 90% conv. gen   : {}'.format(conv_gen if conv_gen else 'N/A'),
    '',
    '--- BEST COMPROMISE -- NSGA-II ---',
    '  Sites              : ' + str(nsga_bc_sites),
    '  Cost               : {:.6f}'.format(nsga_bc_cost),
    '  Impact             : {:.6f}'.format(nsga_bc_impact),
    '  Safety             : {:.6f}'.format(nsga_bc_safety),
    '  Weighted score     : {:.6f}'.format(nsga_bc_ws),
    '  Municipalities     : ' + str(sorted(mun_set)),
    '',
    '--- BEST COMPROMISE -- EXHAUSTIVE ---',
    '  Sites              : ' + str(ex_bc_sites),
    '  Cost               : {:.6f}'.format(ex_bc_cost),
    '  Impact             : {:.6f}'.format(ex_bc_impact),
    '  Safety             : {:.6f}'.format(ex_bc_safety),
    '  Weighted score     : {:.6f}'.format(ex_bc_ws),
    '',
    '--- MATCH ASSESSMENT ---',
    '  Sites match        : ' + str(sites_match),
    '  Objectives close   : ' + str(obj_close),
    '  WS close           : ' + str(ws_close),
    ('  RESULT: NSGA-II found the true optimal compromise.' if sites_match and ws_close
     else '  RESULT: NSGA-II did NOT match exhaustive optimum.'),
    '',
    '--- SPATIAL GEOMETRY (BEST COMPROMISE TRIANGLE) ---',
    '  Side lengths (km)  : ' + str([round(s, 3) for s in sides]),
    '  Triangle area (km2): {:.3f}'.format(triangle_area),
    '  Aspect ratio       : {:.3f}'.format(aspect_ratio),
    '  Avg spread (km)    : {:.3f}'.format(spread_avg),
    '',
    '--- DECISION MAKING: CONSTRAINED vs UNCONSTRAINED (k={}) ---'.format(K_SELECT),
    '  Spread (unconstrained) : {:.3f} km'.format(spread_avg_unconstrained),
    '  Spread (constrained)   : {:.3f} km'.format(spread_constrained),
    '  Spread improvement     : {:.2f} %  ({:+.2f} km)'.format(
        spread_improvement_pct, spread_constrained - spread_avg_unconstrained),
    '  HV (unconstrained)     : {:.6f}'.format(hv_unconstrained),
    '  HV (constrained best)  : {:.6f}'.format(hv_nsga2_best),
    '  HV reduction           : {:.2f} %'.format(hv_reduction_pct),
    '  Trade-off ratio        : {:.2f}:1  (spread gain / HV loss)'.format(tradeoff_ratio),
    '  Interpretation         : ' + _dm_metrics['interpretation'],
    '',
    '--- SCENARIO ANALYSIS (Table 13) ---',
    '  {:>20}  {:>8}  {:>8}  {:>8}  {}'.format('Scenario', 'Cost', 'Impact', 'Safety', 'Sites'),
    '  ' + '-' * 70,
] + [
    '  {:>20}  {:>8.4f}  {:>8.4f}  {:>8.4f}  {}'.format(
        _sn,
        scenario_results[_sn]['Cost'],
        scenario_results[_sn]['Impact'],
        scenario_results[_sn]['Safety'],
        str(scenario_results[_sn]['sites']))
    for _sn in scenario_results
] + [
    '',
    '--- PER-RUN STATISTICS ---',
    '  {:>4}  {:>5}  {:>8}  {:>5}  {:>10}  {:>10}'.format('Run', 'Seed', 'Time(s)', '#Sol', 'HV', 'BestCost'),
    '  ' + '-' * 55,
]
for _ri, _r in enumerate(all_runs):
    _fv = _r['F'][_r['F'][:, 0] < 1e4]
    _bcv = _fv[:, 0].min() if len(_fv) > 0 else float('nan')
    _lines.append('  {:>4}  {:>5}  {:>8.1f}  {:>5}  {:>10.6f}  {:>10.6f}'.format(
        _ri + 1, SEEDS_LIST[_ri], _r['time'], len(_fv), hv_finals_unified[_ri], _bcv))

_lines += ['', '--- OUTPUT FILES ---']
for _fn in sorted(os.listdir(RESULTS_DIR)):
    _fp = os.path.join(RESULTS_DIR, _fn)
    if os.path.isfile(_fp):
        _lines.append('  {:<45} {:>8.1f} KB'.format(_fn, os.path.getsize(_fp) / 1024))
_lines.append('  figures/')
for _fn in sorted(os.listdir(FIGS_DIR)):
    _fp = os.path.join(FIGS_DIR, _fn)
    _lines.append('    {:<43} {:>8.1f} KB'.format(_fn, os.path.getsize(_fp) / 1024))

_lines += ['', _sep, 'END OF SUMMARY', _sep]
_summary = '\n'.join(_lines)

with open(get_output_path('results_summary.txt'), 'w', encoding='utf-8') as _sf:
    _sf.write(_summary)

# --- results_report.pdf ---
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from PIL import Image as PILImage

    _pdf_path = get_output_path('results_report.pdf')
    _doc = SimpleDocTemplate(
        _pdf_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    _W = A4[0] - 4*cm  # usable width

    _base  = getSampleStyleSheet()
    _sT    = ParagraphStyle('Title',   parent=_base['Title'],   fontSize=16, spaceAfter=6,  textColor=colors.HexColor('#1a237e'))
    _sH1   = ParagraphStyle('H1',      parent=_base['Heading1'], fontSize=12, spaceBefore=14, spaceAfter=4, textColor=colors.HexColor('#283593'))
    _sMono = ParagraphStyle('Mono',    parent=_base['Normal'],  fontName='Courier', fontSize=7.5, leading=11, spaceAfter=2)
    _sNorm = ParagraphStyle('Normal2', parent=_base['Normal'],  fontSize=9,  spaceAfter=3)
    _sSub  = ParagraphStyle('Sub',     parent=_base['Normal'],  fontSize=8,  textColor=colors.grey, spaceAfter=8)
    _sFigC = ParagraphStyle('FigCap',  parent=_base['Normal'],  fontSize=8,  textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10)

    def _hr():
        return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#9fa8da'), spaceAfter=6)

    def _h1(txt):
        return Paragraph(txt, _sH1)

    def _mono(txt):
        return Paragraph(txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), _sMono)

    def _fit_image(path, max_w, max_h):
        with PILImage.open(path) as _im:
            iw, ih = _im.size
        ratio = min(max_w / iw, max_h / ih)
        return RLImage(path, width=iw * ratio, height=ih * ratio)

    _story = []

    # --- Cover ---
    _story.append(Spacer(1, 1.5*cm))
    _story.append(Paragraph('Landfill Site Optimization', _sT))
    _story.append(Paragraph('NSGA-II + Exhaustive Search &mdash; Results Report', _sT))
    _story.append(Spacer(1, 0.3*cm))
    _story.append(Paragraph('Run: ' + _run_ts, _sSub))
    _story.append(Paragraph('Directory: ' + os.path.abspath(RESULTS_DIR), _sSub))
    _story.append(_hr())

    # --- Section 1: Configuration ---
    _story.append(_h1('1. Configuration'))
    _cfg_data = [
        ['Parameter', 'Value'],
        ['k (sites per solution)', str(K_SELECT)],
        ['Min pairwise distance', str(MIN_DIST_KM) + ' km'],
        ['Population size', str(POP_SIZE)],
        ['Generations', str(N_GEN)],
        ['Independent runs', str(N_RUNS)],
        ['Parallel workers', '{} / {} cores'.format(N_JOBS, multiprocessing.cpu_count())],
        ['Global seed', str(GLOBAL_SEED)],
        ['Candidate sites', str(n_sites)],
        ['Blocked (protected area)', str(count_blocked)],
    ]
    _cfg_tbl = Table(_cfg_data, colWidths=[_W * 0.55, _W * 0.45])
    _cfg_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
    ]))
    _story.append(_cfg_tbl)
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 2: Timing & Performance ---
    _story.append(_h1('2. Timing &amp; Performance'))
    _tim_data = [
        ['Metric', 'Value'],
        ['NSGA-II total wall time', '{:.1f} s  ({:.2f} min)'.format(time_nsga2_total, time_nsga2_total / 60)],
        ['NSGA-II avg per run', '{:.1f} s +/- {:.1f} s'.format(time_nsga2_avg, time_nsga2_std)],
        ['NSGA-II fastest run', '{:.1f} s  (run #{})'.format(min(run_times), run_times.index(min(run_times)) + 1)],
        ['NSGA-II slowest run', '{:.1f} s  (run #{})'.format(max(run_times), run_times.index(max(run_times)) + 1)],
        ['Exhaustive total time', '{:.1f} s  ({:.2f} min)'.format(time_exhaustive, time_exhaustive / 60)],
        ['Total combinations', 'C({},{}) = {:,}'.format(n_sites, K_SELECT, comb(n_sites, K_SELECT))],
        ['Feasible solutions', '{:,}'.format(len(feasible_F))],
        ['True Pareto size', '{:,}'.format(len(true_pareto_F))],
        ['Speedup (Exh / NSGA-II avg)', '{:.1f}x'.format(speedup)],
    ]
    _tim_tbl = Table(_tim_data, colWidths=[_W * 0.55, _W * 0.45])
    _tim_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
    ]))
    _story.append(_tim_tbl)
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 3: Hypervolume ---
    _story.append(_h1('3. Hypervolume Analysis'))
    _hv_data = [
        ['Metric', 'Value'],
        ['NSGA-II HV (best run)',  '{:.6f}'.format(hv_nsga2_best)],
        ['NSGA-II HV (mean)',      '{:.6f} +/- {:.6f}'.format(hv_finals_unified.mean(), hv_finals_unified.std())],
        ['Exhaustive HV',          '{:.6f}'.format(hv_exhaustive)],
        ['Gap',                    '{:.4f} %'.format(gap_pct)],
        ['Capture rate',           '{:.4f} %'.format(capture_rate)],
        ['Best run',               '#{} (seed={})'.format(best_run_idx + 1, SEEDS_LIST[best_run_idx])],
        ['HV 90% convergence gen', str(conv_gen) if conv_gen else 'N/A'],
    ]
    _hv_tbl = Table(_hv_data, colWidths=[_W * 0.55, _W * 0.45])
    _hv_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
    ]))
    _story.append(_hv_tbl)
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 3.5: Decision Making ---
    _story.append(_h1('4. Decision Making: Constrained vs Unconstrained (k={})'.format(K_SELECT)))
    _dm_data = [
        ['Metric', 'Unconstrained', 'Constrained ({} km min dist)'.format(MIN_DIST_KM)],
        ['Avg pairwise spread (km)',
         '{:.3f}'.format(spread_avg_unconstrained),
         '{:.3f}'.format(spread_constrained)],
        ['Spread improvement',
         '—',
         '{:+.2f} km  ({:.1f}%)'.format(
             spread_constrained - spread_avg_unconstrained, spread_improvement_pct)],
        ['Hypervolume (best run)',
         '{:.6f}'.format(hv_unconstrained),
         '{:.6f}'.format(hv_nsga2_best)],
        ['HV reduction',
         '—',
         '{:.2f}%'.format(hv_reduction_pct)],
        ['Trade-off ratio',
         '—',
         '{:.2f}:1  (spread gain / HV loss)'.format(tradeoff_ratio)],
    ]
    _dm_tbl = Table(_dm_data, colWidths=[_W * 0.38, _W * 0.28, _W * 0.34])
    _dm_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('FONTNAME', (2, 2), (2, 2), 'Helvetica-Bold'),
        ('TEXTCOLOR', (2, 2), (2, 2), colors.HexColor('#1b5e20')),
        ('FONTNAME', (2, 4), (2, 4), 'Helvetica-Bold'),
        ('TEXTCOLOR', (2, 4), (2, 4), colors.HexColor('#b71c1c')),
        ('FONTNAME', (2, 5), (2, 5), 'Helvetica-Bold'),
        ('TEXTCOLOR', (2, 5), (2, 5), colors.HexColor('#1a237e')),
    ]))
    _story.append(_dm_tbl)
    _story.append(Spacer(1, 0.25*cm))
    _story.append(Paragraph(
        '<i>' + _dm_metrics['interpretation'] + '</i>', _sNorm))
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 3.6: Scenario Analysis ---
    _story.append(_h1('5. Scenario Analysis (Table 13)'))
    _sc_hdr = [['Scenario', 'Weights (C/I/S)', 'Sites', 'Cost', 'Impact', 'Safety', 'Municipalities']]
    _sc_rows = []
    for _sn, _sv in scenario_results.items():
        _sc_rows.append([
            _sn,
            '{:.2f}/{:.2f}/{:.2f}'.format(_sv['weights']['Cost'], _sv['weights']['Impact'], _sv['weights']['Safety']),
            str(_sv['sites']),
            '{:.4f}'.format(_sv['Cost']),
            '{:.4f}'.format(_sv['Impact']),
            '{:.4f}'.format(_sv['Safety']),
            ', '.join(_sv['municipalities']),
        ])
    _sc_tbl = Table(_sc_hdr + _sc_rows,
                    colWidths=[_W*0.16, _W*0.13, _W*0.18, _W*0.08, _W*0.08, _W*0.08, _W*0.29])
    _sc_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING',   (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 3),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    _story.append(_sc_tbl)
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 4: Best Compromise ---
    _story.append(_h1('6. Best Compromise Solution'))
    _bc_data = [
        ['', 'NSGA-II', 'Exhaustive'],
        ['Sites (grid indices)', str(nsga_bc_sites), str(ex_bc_sites)],
        ['Cost',   '{:.6f}'.format(nsga_bc_cost),   '{:.6f}'.format(ex_bc_cost)],
        ['Impact', '{:.6f}'.format(nsga_bc_impact), '{:.6f}'.format(ex_bc_impact)],
        ['Safety', '{:.6f}'.format(nsga_bc_safety), '{:.6f}'.format(ex_bc_safety)],
        ['Weighted score', '{:.6f}'.format(nsga_bc_ws), '{:.6f}'.format(ex_bc_ws)],
        ['Match', ('YES' if sites_match and ws_close else 'NO'), ''],
    ]
    _bc_tbl = Table(_bc_data, colWidths=[_W * 0.38, _W * 0.31, _W * 0.31])
    _bc_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('TEXTCOLOR', (1, 6), (1, 6),
         colors.HexColor('#2e7d32') if sites_match and ws_close else colors.HexColor('#c62828')),
        ('FONTNAME', (1, 6), (1, 6), 'Helvetica-Bold'),
    ]))
    _story.append(_bc_tbl)
    _story.append(Spacer(1, 0.3*cm))
    _story.append(Paragraph(
        'Municipalities: ' + str(sorted(mun_set)) +
        ' | Triangle area: {:.3f} km2 | Avg spread: {:.3f} km | Aspect ratio: {:.3f}'.format(
            triangle_area, spread_avg, aspect_ratio), _sNorm))
    _story.append(Spacer(1, 0.4*cm))

    # --- Section 5: Per-run table ---
    _story.append(_h1('7. Per-Run Statistics'))
    _run_hdr = [['Run', 'Seed', 'Time (s)', '# Sol', 'HV', 'Best Cost']]
    _run_rows = []
    for _ri, _r in enumerate(all_runs):
        _fv = _r['F'][_r['F'][:, 0] < 1e4]
        _bcv = '{:.6f}'.format(_fv[:, 0].min()) if len(_fv) > 0 else 'nan'
        _run_rows.append([
            str(_ri + 1), str(SEEDS_LIST[_ri]),
            '{:.1f}'.format(_r['time']), str(len(_fv)),
            '{:.6f}'.format(hv_finals_unified[_ri]), _bcv
        ])
    _run_tbl = Table(_run_hdr + _run_rows,
                     colWidths=[_W*0.08, _W*0.08, _W*0.14, _W*0.1, _W*0.3, _W*0.3])
    _run_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3949ab')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f5f5f5'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#c5cae9')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('LEFTPADDING',  (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING',   (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 3),
    ]
    # highlight best run
    _run_style.append(('BACKGROUND', (0, best_run_idx + 1), (-1, best_run_idx + 1),
                        colors.HexColor('#c8e6c9')))
    _run_tbl.setStyle(TableStyle(_run_style))
    _story.append(_run_tbl)

    # --- Section 7: Figures ---
    _story.append(PageBreak())
    _story.append(_h1('8. Figures'))
    _figs = [
        ('fig1_hypervolume_convergence.png', 'Fig. 1 — Hypervolume Convergence ({} runs)'.format(N_RUNS)),
        ('fig2_pareto_front_3d.png',         'Fig. 2 — 3D Pareto Front'),
        ('fig3_pareto_projections.png',       'Fig. 3 — Pairwise Pareto Projections'),
        ('fig4_spearman_heatmap.png',         'Fig. 4 — Spearman Rank Correlation Matrix'),
        ('fig5_k_sensitivity.png',            'Fig. 5 — k-Value Sensitivity Analysis (Comprehensive)'),
        ('fig_9_4_geographic_dispersion.png', 'Fig. 6 — Geographic Dispersion Map (Best Compromise)'),
    ]
    for _fname, _caption in _figs:
        _fpath = os.path.join(FIGS_DIR, _fname)
        if os.path.exists(_fpath):
            _story.append(_fit_image(_fpath, _W, 14*cm))
            _story.append(Paragraph(_caption, _sFigC))

    _doc.build(_story)
    print('PDF report saved: ' + _pdf_path)
except Exception as _epdf:
    print('  PDF generation skipped: ' + str(_epdf))

print('\n' + _sep)
print('ALL OUTPUTS SAVED')
print('Results directory: ' + os.path.abspath(RESULTS_DIR))
print(_sep)
print(_summary)
print('\nDONE.')
