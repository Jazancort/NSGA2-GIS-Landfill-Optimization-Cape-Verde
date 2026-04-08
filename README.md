# Landfill Site Optimization — NSGA-II + Exhaustive Validation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17704387.svg)](https://doi.org/10.5281/zenodo.17704387)

Multi-objective GIS-based selection of landfill sites for Ilha do Fogo, Cabo Verde.  
The script applies NSGA-II (30 independent runs) and validates results against a full
exhaustive enumeration of all feasible combinations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Required Input Files](#3-required-input-files)
4. [Dependencies](#4-dependencies)
5. [Configuration](#5-configuration)
6. [How to Run](#6-how-to-run)
7. [Output Structure](#7-output-structure)
8. [Algorithm Design](#8-algorithm-design)
9. [Objective Functions](#9-objective-functions)
10. [AHP Weights](#10-ahp-weights)
11. [Performance Notes](#11-performance-notes)
12. [Reproducibility Checklist](#12-reproducibility-checklist)
13. [Data Availability Statement](#13-data-availability-statement)
14. [Code Availability and Citation](#14-code-availability-and-citation)
15. [Manuscript-to-Repository Mapping](#15-manuscript-to-repository-mapping)
16. [Limitations and Threats to Validity](#16-limitations-and-threats-to-validity)
17. [References](#17-references)

---

## 1. Project Overview

The optimization problem selects **k = 3** candidate landfill sites from a 1 km × 1 km island grid
while simultaneously minimizing three conflicting objectives:

| Objective | Description                         | Direction |
| --------- | ----------------------------------- | --------- |
| Cost      | Infrastructure + distance cost      | minimize  |
| Impact    | Environmental & social impact       | minimize  |
| Safety    | Distance from urban/sensitive areas | maximize  |

Constraints:

- Exactly `k` sites must be selected.
- Minimum pairwise geodesic distance ≥ 10 km between selected sites.
- Sites within protected areas are automatically penalized.

A **Decision Making** comparison (Section 7.5) additionally runs a short unconstrained
NSGA-II baseline to quantify how much the minimum-distance constraint improves geographic
spread versus the hypervolume loss it causes.

### Research Highlights

- Multi-objective optimization with NSGA-II for landfill siting.
- GIS-integrated evaluation using 11 geospatial criteria.
- Real-world case study for Ilha do Fogo, Cabo Verde.
- AHP-based weighting and best-compromise selection for decision support.
- Open-source implementation in Python with reproducible experiment artifacts.

### Workflow at a Glance

1. Spatial grid generation and candidate filtering.
2. Multi-criteria extraction and objective construction.
3. NSGA-II optimization and Pareto front discovery.
4. Best-compromise ranking and exhaustive-search validation.

---

## 2. Directory Structure

```
NSGA2-GIS-Landfill-Optimization-Cape-Verde/
│
├── landfill_nsga2_optimization.py   ← main script
├── README.md                        ← this file
├── requirements.txt                 ← pip dependencies
│
├── results/
│   └── results-YYYY-MM-DD_HH-MM-SS/    ← created on each run
│       ├── figures/
│       │   ├── fig1_hypervolume_convergence.png
│       │   ├── fig2_pareto_front_3d.png
│       │   ├── fig3_pareto_projections.png
│       │   ├── fig4_spearman_heatmap.png
│       │   ├── fig5_k_sensitivity.png
│       │   ├── fig_9_4_clean.png
│       │   ├── fig_9_4_terrain.png
│       │   └── fig_9_4_geographic_dispersion.png
│       ├── nsga2_all_runs_cache.pkl
│       ├── unconstrained_baseline.pkl
│       ├── nsga2_run_statistics.csv
│       ├── nsga2_pareto_front_results.csv
│       ├── exhaustive_search_pareto.csv
│       ├── k_sensitivity_results.csv
│       ├── best_compromise_sites.csv
│       ├── hypervolume_metadata_complete.json
│       ├── validation_metrics.json
│       ├── decision_making_metrics.json
│       ├── results_summary.txt
│       └── results_report.pdf
│
├── criteria_data_for_optimization_524-3.csv   ← primary data source (see §3)
├── area_protegida.shp                         ← protected areas polygons
├── geologia_fogo.shp                          ← island boundary
└── geologia_com_permeabilidade_calculada.shp  ← geology + permeability
```

> **Note:** The results folder name uses a full **date + time** timestamp
> (e.g., `results-2024-05-01_14-32-07`). Running the script twice in the same second
> will overwrite the same folder — space runs at least 1 second apart if needed.

---

## 3. Required Input Files

The script looks for candidate data files in the following priority order (first found wins):

| Priority | Filename                                                          |
| -------- | ----------------------------------------------------------------- |
| 1        | `criteria_data_for_optimization_524-3.csv`                        |
| 2        | `criteria_data_for_optimization_524-3_standardized.csv`           |
| 3        | `criteria_data_for_optimization_524-3_standardized-organized.csv` |
| 4        | `optimization_criteria_data_524-fixed_penalized.csv`              |
| 5        | `optimization_criteria_data_524-fixed_penalized-organized.csv`    |
| 6        | `results_cache/df_real_data_cached.csv` (GIS cache)               |

If **none** of these files exist, the full GIS pipeline runs automatically (see §8.1).
Additional shapefiles required for GIS mode:

- `elevation_utm.tif` — DEM (elevation raster)
- `declive_gdal_graus.tif` — slope raster (degrees)
- `geologia_com_permeabilidade_calculada.shp` — permeability zones

Optional legacy data bundle (external link):

- Google Drive: https://drive.google.com/drive/folders/1I382_NkI-qb3mvULdjaBCKli6nT6XPyY?usp=sharing
- Use this only as a convenience source. For publication-grade reproducibility, always
  report exact dataset version/date and file checksums in the manuscript.

---

## 4. Dependencies

Install all dependencies at once using the provided requirements file:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas geopandas shapely scipy geopy matplotlib seaborn \
            pymoo joblib rasterio osmnx reportlab pillow contextily
```

Tested with:

- Python 3.10+
- pymoo 0.6.x
- geopandas 0.14.x
- joblib 1.3.x
- osmnx 1.9.x (only required if GIS pipeline runs)
- reportlab 4.x (required for PDF report generation)
- contextily (optional — required for terrain basemap in Fig 9.4)

---

## 5. Configuration

All tunable constants are at the top of `landfill_nsga2_optimization.py`:

```python
GLOBAL_SEED  = 1      # master random seed
K_SELECT     = 3      # number of sites to select
MIN_DIST_KM  = 10.0   # minimum inter-site distance (km)
POP_SIZE     = 150    # NSGA-II population size
N_GEN        = 200    # NSGA-II generations per run
N_RUNS       = 30     # independent runs
```

---

## 6. How to Run

```bash
cd NSGA2-GIS-Landfill-Optimization-Cape-Verde
python landfill_nsga2_optimization.py
```

The script prints a progress bar for each run and a summary at the end.  
All results land in `results/results-YYYY-MM-DD_HH-MM-SS/`.

**Reproducibility:** For exact bit-for-bit replication across platforms also set:

```powershell
# Windows PowerShell
$env:PYTHONHASHSEED = "1"
python landfill_nsga2_optimization.py
```

**Caching:** On subsequent runs the unconstrained baseline (`unconstrained_baseline.pkl`)
and the NSGA-II runs (`nsga2_all_runs_cache.pkl`) are loaded from disk instead of
recomputed — significantly faster re-runs for reporting purposes.

---

## 7. Output Structure

### CSV files

| File                             | Contents                                                       |
| -------------------------------- | -------------------------------------------------------------- |
| `nsga2_pareto_front_results.csv` | Full NSGA-II Pareto front (Cost, Impact, Safety, site indices) |
| `exhaustive_search_pareto.csv`   | True Pareto front from exhaustive enumeration                  |
| `nsga2_run_statistics.csv`       | Per-run metrics (HV, time, #solutions, best objectives)        |
| `k_sensitivity_results.csv`      | HV and per-site metrics for k = 2…10                           |
| `best_compromise_sites.csv`      | Coordinates & objectives of the 3 recommended sites            |

### JSON files

| File                                 | Contents                                                             |
| ------------------------------------ | -------------------------------------------------------------------- |
| `hypervolume_metadata_complete.json` | HV time-series (mean ± std), reference point, all run HVs            |
| `validation_metrics.json`            | Gap %, capture rate, speedup, triangle geometry                      |
| `decision_making_metrics.json`       | Constrained vs unconstrained comparison: spread, HV, trade-off ratio |

### Text / PDF reports

| File                  | Contents                                                           |
| --------------------- | ------------------------------------------------------------------ |
| `results_summary.txt` | Human-readable summary of all metrics and key results              |
| `results_report.pdf`  | Full formatted report with tables and embedded figures (ReportLab) |

### Figures (PNG, 300 DPI)

| File                                | Description                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `fig1_hypervolume_convergence.png`  | Mean HV ± std over generations vs. true Pareto HV                                      |
| `fig2_pareto_front_3d.png`          | 3D Pareto front with extreme & best-compromise markers                                 |
| `fig3_pareto_projections.png`       | Pairwise 2D projections (Cost–Impact, Cost–Safety, Impact–Safety)                      |
| `fig4_spearman_heatmap.png`         | Spearman rank-correlation matrix of criteria                                           |
| `fig5_k_sensitivity.png`            | 4-panel k-sensitivity analysis (HV norm, per-site efficiency, coverage, marginal gain) |
| `fig_9_4_clean.png`                 | Geographic dispersion map — clean white background, point markers, hatched Pareto hull |
| `fig_9_4_terrain.png`               | Same map with contextily terrain basemap (mountain paths visible)                      |
| `fig_9_4_geographic_dispersion.png` | Alias copy of the clean version (used in PDF report)                                   |

### Pickle cache files

| File                         | Contents                                               |
| ---------------------------- | ------------------------------------------------------ |
| `nsga2_all_runs_cache.pkl`   | All 30 NSGA-II run results (F, X, HV history, timings) |
| `unconstrained_baseline.pkl` | Unconstrained baseline spread & HV (5-run average)     |

---

## 8. Algorithm Design

### 8.1 GIS Pipeline (automatic, skipped when cache exists)

1. Generate a 1 km × 1 km grid clipped to the island boundary.
2. Filter cells with ≥ 50 % land coverage.
3. For each centroid, query OSM for distances to urban areas, aerodrome, rivers,
   roads, wells, cultivated areas, coastline.
4. Extract DEM slope and aspect from raster files.
5. Compute distance to low-permeability zones and protected areas.
6. Cache to `results_cache/df_real_data_cached.csv`.

### 8.2 NSGA-II Configuration

- **Sampling:** `DistanceAwareSampling` — generates initial population where all
  solutions already satisfy the minimum distance constraint.
- **Crossover:** Two-Point Crossover (binary representation).
- **Mutation:** Bit-flip with probability `1 / n_vars`.
- **Repair:** `FixKRepair` — ensures exactly `k` bits are set after crossover/mutation.
- **Problem:** `SelectAreasWithMinDistance` (vectorized `Problem` subclass).
  Processes the full population in a single batch:
  - Objectives via `X @ costs` (BLAS matrix multiply).
  - Distance constraint via numpy fancy indexing on the pre-computed Haversine matrix.
- **Parallelism:** 30 independent runs executed via `joblib.Parallel` using all available
  CPU cores minus one.

### 8.3 Exhaustive Search

Full enumeration of C(n, k) combinations with the minimum-distance filter applied inline.
Pareto filtering uses `is_pareto_efficient_fast`: O(n log n) sort-and-sweep algorithm.

### 8.4 Best Compromise Selection

Weighted-sum over the normalized Pareto front using scenario weights:

| Scenario | Weight |
| -------- | ------ |
| Cost     | 0.295  |
| Impact   | 0.356  |
| Safety   | 0.349  |

Normalization range is computed jointly from NSGA-II and exhaustive Pareto fronts
to ensure fair comparison.

### 8.5 Decision Making: Constrained vs Unconstrained (Section 7.5)

After the main optimization, a short unconstrained NSGA-II run (5 seeds × N_GEN generations,
using `SelectAreasRobust` with no minimum-distance penalty) establishes a baseline.
Four metrics are computed and saved to `decision_making_metrics.json`:

| Metric                 | Definition                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| Geographic spread      | Mean pairwise geodesic distance among best-compromise sites                |
| Spread improvement (%) | `100 × (spread_constrained − spread_unconstrained) / spread_unconstrained` |
| HV reduction (%)       | `100 × (HV_unconstrained − HV_constrained_best) / HV_unconstrained`        |
| Trade-off ratio        | `\|spread improvement %\| / \|HV reduction %\|`                            |

The unconstrained result is cached to `unconstrained_baseline.pkl` and reused on
subsequent runs.

### 8.6 Practical Interpretation of Main Outputs

- Pareto front points represent feasible 3-site combinations where no objective can
  improve without degrading at least one other objective.
- `fig2_pareto_front_3d.png` shows the global trade-off structure among Cost, Impact,
  and Safety.
- `fig_9_4_clean.png` and `fig_9_4_terrain.png` summarize geographic allocation for
  extreme solutions and the final best-compromise solution.
- In map interpretation:
  - Low Cost tends to favor infrastructure/topography efficiency.
  - Low Impact prioritizes environmental distancing.
  - High Safety increases separation from population/sensitive exposure.
  - Best Compromise balances all objectives under scenario weights.

---

## 9. Objective Functions

```
Cost(S)   = Σ w_i · norm(permeability_i) + w_j · norm(slope_j) + w_k · norm(road_k)
Impact(S) = Σ w_i · norm_inv(protected_i) + … + w_j · asp_penalty(j)
Safety(S) = Σ w_i · norm(urban_i) + w_j · norm(aerodrome_j) + w_k · norm(cultivated_k)
```

where `norm(v) = (v - min) / (max - min)` and  
`norm_inv(v) = (max - v) / (max - min)`.  
Non-finite values default to 0.5 (neutral).

---

## 10. AHP Weights

Derived from Andrade & Barbosa (2015), Table 5 (consistency ratio < 0.1):

| Criterion        | Weight    |
| ---------------- | --------- |
| Urban Areas      | 0.166     |
| Permeability     | 0.152     |
| Slope            | 0.113     |
| Impact           | 0.102     |
| Cultivated Areas | 0.081     |
| Aspect           | 0.075     |
| Coastline        | 0.069     |
| Protected Areas  | 0.066     |
| Rivers           | 0.091     |
| Wells/Boreholes  | 0.055     |
| Road Network     | 0.030     |
| **Sum**          | **1.000** |

---

## 11. Performance Notes

- **Vectorized evaluation (`Problem`):** The `SelectAreasWithMinDistance` class processes
  the entire population (150 individuals) in a single call per generation — typically
  5–15× faster than the element-wise variant.
- **Parallel runs:** `joblib.Parallel` distributes the 30 independent NSGA-II runs across
  all available CPU cores (minus one), significantly reducing wall-clock time.
- **Distance matrix:** Pre-computed once at startup (O(n²) Haversine, n ≈ 524).
- **Exhaustive search:** ~2.9 M combinations for n = 524, k = 3; feasibility filter
  reduces this to ~tens of thousands. Pareto filter uses an O(n log n) sort-sweep.
- **Unconstrained baseline:** 5 short runs only; cached after first execution so
  subsequent runs do not incur extra computation.
- Typical runtime: NSGA-II ~2–5 min total (30 runs), exhaustive ~1–3 min,
  unconstrained baseline ~20–60 s (first run only).

---

## 12. Reproducibility Checklist

Use the checklist below before claiming full reproducibility in a manuscript submission.

- Fix Python hash seed and global RNG seed:
  - `GLOBAL_SEED = 1` is already set in the script.
  - Also export `PYTHONHASHSEED=1` before execution.
- Keep dependency versions fixed:
  - Install from `requirements.txt`.
  - Archive the exact environment used in the paper (for example with `pip freeze > requirements-locked.txt`).
- Archive code identity:
  - Save the commit hash used for experiments (`git rev-parse HEAD`).
- Archive data identity:
  - Record checksums for all input CSV/SHP/TIF files (for example with `Get-FileHash` on Windows).
- Archive runtime metadata:
  - Keep `results_summary.txt`, `validation_metrics.json`, `hypervolume_metadata_complete.json`, and `decision_making_metrics.json`.
- Report hardware:
  - Include CPU model, RAM, OS version, and Python version in the manuscript's experimental setup section.

---

## 13. Data Availability Statement

The optimization pipeline depends on local geospatial datasets listed in Section 3.

- If your target journal requires open data, publish the exact input datasets in a trusted repository (for example Zenodo/Figshare) and provide a persistent identifier.
- If legal or institutional restrictions apply, provide a restricted-access statement in the manuscript and specify a contact process for qualified requests.
- The manuscript should explicitly report dataset version/date, CRS assumptions, and any preprocessing steps applied before running this code.

---

## 14. Code Availability and Citation

- Source code license: MIT (see `LICENSE`).
- Zenodo archive DOI: https://doi.org/10.5281/zenodo.17704387
- Suggested software citation (replace URL/version placeholders with final metadata before submission):

```bibtex
@software{nsga2_fogo_landfill,
  title   = {NSGA-II GIS Landfill Optimization for Ilha do Fogo, Cabo Verde},
  author  = {Azancort Neto, Julio Leite},
  year    = {2024},
  publisher = {GitHub},
  url     = {<REPOSITORY_URL>},
  note    = {Version: <COMMIT_OR_RELEASE_TAG>}
}

@misc{azancort_neto_zenodo_2026,
  author = {Azancort Neto, Julio Leite},
  title  = {NSGA-II GIS Landfill Optimization for Ilha do Fogo, Cabo Verde},
  year   = {2026},
  doi    = {10.5281/zenodo.17704387},
  url    = {https://doi.org/10.5281/zenodo.17704387}
}
```

- If an associated journal paper exists, cite both the paper and the software repository.

---

## 15. Manuscript-to-Repository Mapping

To align a manuscript (IEEE Access style) with this repository:

- Problem formulation and constraints: Sections 1, 8, 9, and 10 of this README.
- Algorithm configuration and computational setup: Sections 5 and 8.
- Validation protocol (NSGA-II vs exhaustive): Sections 7 and 8.3.
- Decision-making analysis (constrained vs unconstrained): Section 8.5 and `decision_making_metrics.json`.
- Figures and tables for reproducibility: `results/.../figures/`, CSV/JSON artifacts, and `results_report.pdf`.

---

## 16. Limitations and Threats to Validity

- Results depend on the quality and completeness of GIS/OSM layers at collection time.
- Weighted-sum compromise ranking depends on chosen scenario weights.
- Hypervolume is reference-point dependent; comparisons should always report the reference point definition.
- Runtime and convergence behavior vary with hardware and dependency versions.
- For strict cross-platform replication, fixed seeds alone may be insufficient without pinned binary/library builds.

---

## 17. References

Selected high-relevance references from the LaTeX `.bib` list for this study:

1. Deb, K., Pratap, A., Agarwal, S., and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
2. Saaty, R. W. (1987). The analytic hierarchy process-what it is and how it is used. *Mathematical Modelling*, 9(3-5), 161-176.
3. Malczewski, J. (2006). GIS-based multicriteria decision analysis: A survey of the literature. *International Journal of Geographical Information Science*, 20(7), 703-726.
4. Andrade, A. J. B., and Barbosa, N. P. P. (2015). Combination of the AHP method and GIS in the selection of areas with potential for landfill installation: Case study of Fogo Island, Republic of Cape Verde. *Revista de Geografia (UFPE)*, 32(2), 248-266.
5. Demesouka, O. E., Vavatsikos, A. P., and Anagnostopoulos, K. P. (2013). Siting MSW landfills with a spatial multiple criteria analysis methodology. *Waste Management*, 33(7), 1190-1206.
6. Spigolon, L. M. G., Giannotti, M., Larocca, A. P., Russo, M. A. T., and Souza, N. C. (2018). Landfill siting based on optimisation, multiple decision analysis, and geographic information system analyses. *Waste Management and Research*, 36(7), 606-615.
7. Rezaeisabzevar, Y., Bazargan, A., and Zohourian, B. (2020). Landfill site selection using multi criteria decision making: Influential factors for comparing locations. *Journal of Environmental Sciences*, 93, 170-184.
8. Eiselt, H. A., and Marianov, V. (2014). A bi-objective model for the location of landfills for municipal solid waste. *European Journal of Operational Research*, 235(1), 187-194.
9. Chen, B., Zhang, R., Long, S., and Sakdanuphab, R. (2024). A multi-objective multi-period low-carbon location-routing problem: improved NSGA-II approach. *IEEE Access*, 12, 51590-51605.

Additional bibliography items from the manuscript can be kept in the LaTeX file; the list above prioritizes methodological and domain-specific references directly connected to this repository.
