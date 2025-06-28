# NSGA-II GIS Landfill Optimization: Fogo Island, Cape Verde

🔍 A multi-objective optimization framework for sustainable landfill site selection on Fogo Island, Cape Verde, using GIS and NSGA-II.

[cite_start]This repository contains the code, methodology, and results for a case study applying a multi-objective optimization genetic algorithm (NSGA-II) to solve the complex problem of landfill site allocation[cite: 7, 113]. [cite_start]The project integrates Geographic Information Systems (GIS) analysis to evaluate and identify optimal locations on Fogo Island, Cape Verde [cite: 6][cite_start], by balancing three conflicting objectives: cost minimization, environmental protection, and social safety[cite: 7, 36].

## 🌟 Key Features

* [cite_start]**Multi-Objective Optimization:** Utilizes the NSGA-II algorithm to find a set of optimal solutions (a Pareto Front) rather than a single answer, enabling flexible and informed decision-making[cite: 7, 34].
* [cite_start]**GIS Integration:** Processes and analyzes 11 distinct geospatial criteria, including digital elevation models (DEM), geology, land use (OSM), and infrastructure to evaluate 524 candidate sites[cite: 7, 8, 157].
* [cite_start]**Real-World Case Study:** Applies the methodology to Fogo Island, Cape Verde, an environment with complex terrain and significant socio-environmental constraints[cite: 6, 79].
* [cite_start]**Data-Driven Decision Support:** Provides planners not only with extreme solutions (e.g., lowest cost or highest safety) but also a "Best Compromise Solution" calculated using weights from the Analytic Hierarchy Process (AHP)[cite: 10, 237, 252].
* [cite_start]**Open-Source Implementation:** The workflow is implemented entirely in Python using open-source libraries such as `GeoPandas`, `Rasterio`, `OSMnx`, and `PyMoo`[cite: 108, 118].

## ⚙️ Methodology & Workflow

The project follows a structured, four-phase methodology as illustrated in the workflow diagram below. [cite_start]This approach ensures a reproducible and data-driven process from data acquisition to final site recommendation[cite: 106, 260].

![Project Workflow](work_flow.png)

1.  [cite_start]**Data Input & Candidate Generation:** A 1 km² grid is overlaid on Fogo Island, generating 524 candidate polygons for analysis[cite: 8, 144]. [cite_start]The geometric centroids of these areas are used as reference points[cite: 149, 150].
2.  [cite_start]**Criteria Analysis:** Each of the 524 candidate points is evaluated against 11 spatial criteria derived from topographical, geological, and infrastructural data[cite: 157]. [cite_start]These raw values are quantified and standardized[cite: 159, 186].
3.  [cite_start]**Multi-Objective Optimization:** The NSGA-II algorithm is executed to find a set of 14 non-dominated solutions (the Pareto Front), where each solution represents a unique combination of three distinct landfill sites[cite: 9, 213, 291].
4.  [cite_start]**Decision Making:** The Pareto Front is analyzed to understand the trade-offs[cite: 293]. [cite_start]Extreme solutions are identified, and a Best Compromise Solution is selected using a weighted scoring model based on AHP priorities[cite: 10, 237, 252].

## 📊 Results & Visualization

### Pareto Front

[cite_start]The optimization resulted in a Pareto Front composed of 14 non-dominated solutions[cite: 9, 291]. The 3D scatter plot below illustrates the trade-off between the three conflicting objectives. [cite_start]Each point represents a valid combination of three landfill sites where no single objective can be improved without compromising another[cite: 34, 293].

![Pareto Front](parreto_plot.png)

### Spatial Allocation of Optimal Sites

[cite_start]The map below visualizes the geographic distribution of the top-performing solutions for each objective, as well as the final Best Compromise Solution[cite: 328, 342].

![Map of Optimal Solutions](map.jpg)

* [cite_start]🟦 **Low Cost:** These three sites are located near existing infrastructure and in areas with favorable topography to minimize construction costs[cite: 330, 343].
* [cite_start]🟩 **Low Impact:** These three sites are positioned to minimize ecological sensitivity, located away from protected areas and water bodies[cite: 332, 344].
* [cite_start]🟥 **High Security:** These three sites are farthest from population centers and high-risk zones like the aerodrome, maximizing social safety[cite: 334, 345].
* [cite_start]🟧 **Best Compromise:** The site selected via weighted scoring[cite: 252, 346]. [cite_start]Interestingly, it shares the same spatial quadrant as one of the high-safety sites, highlighting its robust, balanced profile[cite: 337, 348].

## 📂 Repository Structure
NSGA2-GIS-Landfill-Optimization-Cape-Verde/
│
├── data/
│   ├── geologia_fogo.shp           # Shapefile for island geology and boundaries
│   └── ...                         # Other geospatial data (DEMs, protected areas, etc.)
│
├── results/
│   ├── map.jpg                     # Final map of optimal site allocations
│   ├── parreto_plot.png            # 3D plot of the Pareto Front
│   ├── work_flow.png               # Workflow diagram
│   └── ...                         # Other plots and outputs
│
├── Areas-candidatas_e_criacao_de_mapas.ipynb  # Notebook to generate candidate areas and create maps with Folium.
├── Pega_dados.ipynb                           # Notebook to extract, process, standardize, and run the NSGA-II optimization.
├── Trabalho_final___Glauco___Julio_Azancort.pdf # The scientific paper detailing the study.
└── README.md                                  # This file.


## 🚀 Getting Started

To replicate this study, follow the steps below:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/NSGA2-GIS-Landfill-Optimization-Cape-Verde.git](https://github.com/your-username/NSGA2-GIS-Landfill-Optimization-Cape-Verde.git)
    cd NSGA2-GIS-Landfill-Optimization-Cape-Verde
    ```

2.  **Create a Python environment and install dependencies:**
    It is recommended to use a virtual environment (`venv` or `conda`).
    ```bash
    pip install pandas geopandas rasterio osmnx pymoo folium matplotlib numpy
    ```

3.  **Data Structure:**
    Place all input geospatial data files (shapefiles, GeoTIFFs, etc.) in the `data/` directory.

4.  **Run the Notebooks:**
    * Execute the `Pega_dados.ipynb` notebook to process the raw data, calculate criteria for each candidate site, and run the NSGA-II optimization algorithm.
    * Execute the `Areas-candidatas_e_criacao_de_mapas.ipynb` notebook to generate the spatial visualizations, including interactive and static maps of the results.

## 📄 Citation

This work is based on the attached scientific paper. If you use this project in your research, please cite it appropriately.

[cite_start]**Title:** *Multi-Objective Optimization for Landfill Site Allocation Using GIS and NSGA-II: A Case Study on Fogo Island, Cape Verde* [cite: 1]
[cite_start]**Author:** Julio Leite Azancort Neto [cite: 2]

## 📜 License

This project is licensed under the [Apache-2.0 license](LICENSE).
