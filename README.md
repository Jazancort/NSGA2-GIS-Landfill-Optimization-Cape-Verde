# NSGA-II GIS Landfill Optimization: Fogo Island, Cape Verde

🔍 A multi-objective optimization framework for sustainable landfill site selection on Fogo Island, Cape Verde, using GIS and NSGA-II.

This repository contains the code, methodology, and results for a case study applying a multi-objective optimization genetic algorithm (NSGA-II) to solve the complex problem of landfill site allocation. The project integrates Geographic Information Systems (GIS) analysis to evaluate and identify optimal locations on Fogo Island, Cape Verde, by balancing three conflicting objectives: cost minimization, environmental protection, and social safety.

## 🌟 Key Features

* **Multi-Objective Optimization:** Utilizes the NSGA-II algorithm to find a set of optimal solutions (a Pareto Front) rather than a single answer, enabling flexible and informed decision-making.
* **GIS Integration:** Processes and analyzes 11 distinct geospatial criteria, including digital elevation models (DEM), geology, land use (OSM), and infrastructure to evaluate 524 candidate sites.
* **Real-World Case Study:** Applies the methodology to Fogo Island, Cape Verde, an environment with complex terrain and significant socio-environmental constraints.
* **Data-Driven Decision Support:** Provides planners not only with extreme solutions (e.g., lowest cost or highest safety) but also a "Best Compromise Solution" calculated using weights from the Analytic Hierarchy Process (AHP).
* **Open-Source Implementation:** The workflow is implemented entirely in Python using open-source libraries such as `GeoPandas`, `Rasterio`, `OSMnx`, and `PyMoo`.

## ⚙️ Methodology & Workflow

The project follows a structured, four-phase methodology as illustrated in the workflow diagram below. This approach ensures a reproducible and data-driven process from data acquisition to final site recommendation.

![Project Workflow](work_flow.png)

1.  **Data Input & Candidate Generation:** A 1 km² grid is overlaid on Fogo Island, generating 524 candidate polygons for analysis. The geometric centroids of these areas are used as reference points.
2.  **Criteria Analysis:** Each of the 524 candidate points is evaluated against 11 spatial criteria derived from topographical, geological, and infrastructural data. These raw values are quantified and standardized.
3.  **Multi-Objective Optimization:** The NSGA-II algorithm is executed to find a set of 14 non-dominated solutions (the Pareto Front), where each solution represents a unique combination of three distinct landfill sites.
4.  **Decision Making:** The Pareto Front is analyzed to understand the trade-offs. Extreme solutions are identified, and a Best Compromise Solution is selected using a weighted scoring model based on AHP priorities.

## 📊 Results & Visualization

### Pareto Front

The optimization resulted in a Pareto Front composed of 14 non-dominated solutions. The 3D scatter plot below illustrates the trade-off between the three conflicting objectives. Each point represents a valid combination of three landfill sites where no single objective can be improved without compromising another.

![Pareto Front](parreto_plot.png)

### Spatial Allocation of Optimal Sites

The map below visualizes the geographic distribution of the top-performing solutions for each objective, as well as the final Best Compromise Solution.

![Map of Optimal Solutions](map.jpg)

* 🟦 **Low Cost:** These three sites are located near existing infrastructure and in areas with favorable topography to minimize construction costs.
* 🟩 **Low Impact:** These three sites are positioned to minimize ecological sensitivity, located away from protected areas and water bodies.
* 🟥 **High Security:** These three sites are farthest from population centers and high-risk zones like the aerodrome, maximizing social safety.
* 🟧 **Best Compromise:** The site selected via weighted scoring. Interestingly, it shares the same spatial quadrant as one of the high-safety sites, highlighting its robust, balanced profile.

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

If you use this work in your research, please cite:

@software{azancort_neto_2024,
  author = {Azancort Neto, Julio Leite},
  title = {NSGA2-GIS-Landfill-Optimization-Cape-Verde},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/NSGA2-GIS-Landfill-Optimization-Cape-Verde}}
}

## 📜 License

This project is licensed under the [Apache-2.0 license](LICENSE).
