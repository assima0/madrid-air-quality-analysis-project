# Madrid Air Quality Analysis Project

## Overview
This project represents a comprehensive data engineering, network analysis and predictive modelling pipeline for Madrid air quality dataset (METRAQ), which contains the pollutant, weather, and traffic-related measurements collected across different sensors in the city of Madrid. 
The pipeline loads and cleans the dataset, uses different imputation methods and compares them, analyzes temporal patterns, builds spatial and correlation networks, tests parallelization and trains and tests forecasting ML models. 

## Project Structure
- `main.py` — main runner used to execute the Tasks 1-10
- `tasks/`
  - `task1_load_inspect.py` — loads the sample/full parquet dataset (upon choice), cleans the main columns, removes duplicates, adds time features, classifies variables into three , and saves the cleaned dataset and initial summaries
  - `task2_missingness.py` — reconstructs missingness by using "is_interpolated" column, summarizes missing-data patterns, detects consecutive missing periods, and checks selected data-quality issues
  - `task3_imputation.py` — applies five imputation methods(Grouped mean, Grouped median, Forward fill, Backward fill, Last-Next average), compares them with METRAQ’s provided interpolation, and saves error summaries and distribution-based comparisons
  - `task4_temporal.py` — analyzes pollutant trends, hourly and seasonal cycles, yearly behavior, station-level trend stability, and longer-term patterns such as Mann-Kendall trends.
  - `task5_spatial_network.py` — builds geographic sensor networks from UTM distances, computes graph metrics and centrality, detects spatial communities, and selects the representative connected network
  - `task6_correlation_network.py` — builds sensor networks based on similarity in pollutant time series, computes correlation-based communities and graph metrics, and compares behavioral networks with spatial networks constructed in Task5
  - `task8_parallelization.py` — compares sequential and multiprocessing execution for repeated correlation calculations, measures speedup, and summarizes stable correlation patterns across year-sensor tasks
  - `task9_forecasting.py`(one of the optional tasks) - trains optional regression models to predict pollutant concentrations from weather, traffic, and time features, then evaluates model performance and feature importance.
  - `task10_final_visualization.py` — collects saved outputs from earlier tasks and produces final presentation-ready figures and a summary dashboard.
- `dashboard.py` — interactive Streamlit dashboard
- `utils/`
  - `config.py` — project paths and task output directories
  - `helpers.py` — utility functions, specifically the timer decorator
- `requirements.txt` — Python package dependencies
- `outputs/` — specific tables, figures, graphs, and processed data generated as tasks outputs
- `files/` — raw parquet dataset directory expected by the code. In the submitted folder, the raw parquet files are omitted due to it's large size

## Input Data
The pipeline works with (can be chosen):
- a sample CSV file for development and testing
- the full parquet dataset for complete analysis.

## Dataset Availability
!The raw parquet files and sample files are not included in this repository due to their large size.  
To run the full pipeline, place the original parquet dataset inside the expected `files/` directory before executing `main.py`.
The provided `sample.csv` can still be used for testing.

## Main Outputs
The project generates:
- cleaned processed parquet data,
- missingness and data-quality summary tables,
- imputation comparison tables and figures,
- temporal trend and seasonality plots,
- spatial and correlation network summaries,
- GraphML network files,
- parallelization runtime comparisons and speedup plots,
- forecasting evaluation outputs,
- a final visualization dashboard.

### How to run

1. Install the required packages:
```sh
   pip install -r requirements.txt 
```
2. Make sure the raw parquet dataset is placed in the expected files/ folder
3. Run the Main Runner Script
```sh
   python main.py 
```
4. Generated outputs will be saved under the outputs/ directory; tables figures, graphs, and processed file outputs are separated by folders for each of them:
- `tables/` — CSV summaries and interpretation tables
- `figures/` — saved plots and dashboards
- `graphs/` — GraphML files for network tasks
- `models/` — reserved modelling outputs where applicable
5. To launch the optional interactive Streamlit dashboard:
```sh
   streamlit run dashboard.py
```
## Note: On lower-memory machines, it may be necessary to run the tasks sequentially rather than executing the full pipeline at once. During development, one team member with 8 GB RAM laptop was able to run the project successfully only by running tasks one after another.

## Implementation Notes
- The pipeline is completely modular, meaning that each task can be executed independently once its required inputs are available.
- Interpolation done by METRAQ is used as a comparison reference in Task 3(as indicated in the project task description), not as absolute ground truth.
- For Forecasting task, chronological train/test splitting is used to avoid future-to-past leakage.
