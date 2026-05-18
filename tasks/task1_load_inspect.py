"""Task 1 pipeline for loading, cleaning and doing a summary inspection of the Madrid air-quality dataset.
This file:
- loads the sample.csv file or the full METRAQ-Air-Quality parquet files,
- cleans and standardize the core columns, 
- adds time features for temporal analysis, 
- classifies the magnitudes as pollutant, weather, traffic or unknown variables
- saves the cleaned, analysis-ready dataset as parquet file,
- produces summary, descriptive statistics and initial inspection plots.

The resulting output provides the foundation for later tasks
"""

# used for optional loading of the multiple parquet files in parallel
from multiprocessing import Pool, cpu_count 

# used for data loading and creating figures
import pandas as pd
import matplotlib.pyplot as plt

# project-specific paths and output directories for Task 1
from utils.config import (
    FILES_DIR,
    SAMPLE_PATH,
    PROCESSED_DIR,
    get_task_dirs,
)

# output folders
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task1_load_inspect")

# decorator used to report runtime for some pipeline steps
from utils.helpers import timer

def read_parquet_file(file_path):
    """ Read a parquet file and returns it as a pandas DataFrame"""
    return pd.read_parquet(file_path)

class MadridDataLoader:
    """ Load, clean and summarize Madrid air-quality dataset"""
    def __init__(self, use_sample=True, parallel=False, nrows=None):
        """ Initialize the loading options for the dataset
        Args:
            use_sample: to choose between the sample CSV file or the full dataset(parquet)
            parallel: to use the multiprocessing to read parquet files
            nrows: optional row limit for quick testing
        """
        self.use_sample = use_sample
        self.parallel = parallel
        self.nrows = nrows
        self.interpolation_flag_quality = {}

    @timer
    def load_data(self):
        """ Load the data from CSV or parquet
        Returns a pandas DataFrame with normalized column names 
        """
        # checks if the sample file exists, if not raises an error
        if self.use_sample:
            if not SAMPLE_PATH.exists():
                raise FileNotFoundError(f"Sample file not found: {SAMPLE_PATH}")

            dtypes = {
                "sensor_id": "int32",
                "sensor_name": "category",
                "utm_x": "float32",
                "utm_y": "float32",
                "magnitude_id": "int16",
                "magnitude_name": "category",
                "value": "float32",
                "is_interpolated": "boolean",
            }

            df = pd.read_csv(
                SAMPLE_PATH,
                dtype=dtypes,
                parse_dates=["entry_date"],
                nrows=self.nrows,
                low_memory=False,
            )

        else:
            parquet_files = sorted(FILES_DIR.glob("*.parquet"))
            
            #checks if the parquet files available to load
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {FILES_DIR}")
            
            #if parallel processing is opted:
            if self.parallel:
                with Pool(processes=cpu_count()) as pool:
                    dfs = pool.map(read_parquet_file, parquet_files)
            else:
                dfs = [read_parquet_file(file) for file in parquet_files]

            df = pd.concat(dfs, ignore_index=True)

            if self.nrows is not None:
                df = df.head(self.nrows)

        df.columns = df.columns.str.strip().str.lower()

        return df

    def clean_data(self, df): 
        """ Clean the raw air-quality data, remove unusable records.

        Coverts the columns to appropriate types, records interpolation-flag
        quality, removes rows missing essential fields, and drops duplicates.

        Args: 
            df: raw input DataFrame

        Returns the cleaned DataFrame
        """

        df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
        # errors="coerce" converts invalid dates to NaT.

        df["sensor_id"] = pd.to_numeric(df["sensor_id"], errors="coerce")
        df["magnitude_id"] = pd.to_numeric(df["magnitude_id"], errors="coerce")
        df["utm_x"] = pd.to_numeric(df["utm_x"], errors="coerce")
        df["utm_y"] = pd.to_numeric(df["utm_y"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        # errors="coerce" converts invalid numeric values to NaN.

        missing_flags = df["is_interpolated"].isna().sum()
        missing_rate = df["is_interpolated"].isna().mean()

        self.interpolation_flag_quality = {
        "missing_is_interpolated_before_cleaning": missing_flags,
        "missing_rate_before_cleaning": missing_rate,
        "total_rows_before_cleaning": len(df),
        }

        print(f"Missing is_interpolated flags: {missing_flags:,}")
        print(f"Missing rate: {missing_rate:.4%}")

        if missing_rate == 0:
            df["is_interpolated"] = df["is_interpolated"].astype(bool)
        elif missing_rate < 0.001:
            print("Very small number of missing flags. Filling as False.")
            df["is_interpolated"] = df["is_interpolated"].fillna(False).astype(bool)

        else:
            print("Many missing flags. Keeping them as unknown.")
            df["is_interpolated"] = df["is_interpolated"].astype("boolean")
        
        # drops the empty rows for the key columns
        df = df.dropna(subset=[
        "entry_date",
        "sensor_id",
        "magnitude_name",
        "value",
        ])

        # drops the duplicates
        duplicate_rows = df.duplicated().sum()
        print(f"Duplicate rows before removal: {duplicate_rows:,}")

        df = df.drop_duplicates()

        
        return df

    @staticmethod
    def add_time_features(df):
        """ Add year, month, hour, and weekday columns from entry_date column"""

        df["year"] = df["entry_date"].dt.year.astype("int16")
        df["month"] = df["entry_date"].dt.month.astype("int8")
        df["hour"] = df["entry_date"].dt.hour.astype("int8")
        df["weekday"] = df["entry_date"].dt.dayofweek.astype("int8")

        return df

    @staticmethod
    def classify_variable(name):
        """ Classify a magnitude as pollutant, weather, traffic, or unknown"""
        
        name = str(name).strip().upper()
        pollutants = {
            "SO2", "CO", "NO", "NO2", "PM2.5", "PM10", "<PM2.5", "<PM10",
            "NOX", "O3", "TOLUENO", "BENCENO", "ETILBENCENO",
            "HIDROCARBS_TOTALES", "METANO", "HIDROCARBS_NO_METANICOS"
        }

        weather = {
            "TEMP", "HR", "PRE", "RS", "VV", "DV", "PRECIPITACION"
        }

        if name in pollutants:
            return "pollutant"

        if name in weather:
            return "weather"

        if name.startswith(("TI_", "SP_", "OC_")):
            return "traffic"

        return "unknown"

    def add_variable_type(self, df):
        """Add a variable_type column classifying each magnitude"""

        df["variable_type"] = (
            df["magnitude_name"]
            .apply(self.classify_variable)
            .astype("category")
        )

        return df

    @staticmethod
    def save_cleaned_data(df):
        """ Save the cleaned dataset as a parquet file"""
 
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned_path = PROCESSED_DIR / "cleaned_air_quality.parquet"
        df.to_parquet(cleaned_path, index=False)

        print(f"Cleaned dataset saved to: {cleaned_path}")

    #FIGURES AND TABLES

    def save_outputs(self, df):
        """ Save summary tables and inspection plots for the cleaned dataset"""

        # checks if the output folders exist before producing the plots and tables
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # summarizes the dataset schema and column-level missingness
        schema = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(x) for x in df.dtypes],
            "missing_values": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
        })
        schema.to_csv(TABLES_DIR / "schema_summary.csv", index=False)

        # saves broad descriptive statistics and frequency tables
        summary = df.describe(include="all")
        summary.to_csv(TABLES_DIR / "descriptive_statistics.csv")

        counts = df["variable_type"].value_counts(dropna=False)
        counts.to_csv(TABLES_DIR / "variable_type_counts.csv")

        magnitude_counts = df["magnitude_name"].value_counts(dropna=False)
        magnitude_counts.to_csv(TABLES_DIR / "magnitude_counts.csv")

        sensor_counts = df["sensor_name"].value_counts(dropna=False)
        sensor_counts.to_csv(TABLES_DIR / "sensor_counts.csv")

        # records the dataset's time coverage and basic dimensionality
        date_summary = pd.DataFrame({
            "min_date": [df["entry_date"].min()],
            "max_date": [df["entry_date"].max()],
            "n_rows": [len(df)],
            "n_sensors": [df["sensor_id"].nunique()],
            "n_sensor_names": [df["sensor_name"].nunique()],
            "n_magnitudes": [df["magnitude_name"].nunique()],
        })
        date_summary.to_csv(TABLES_DIR / "date_range_summary.csv", index=False)
    
        # quantifies how often values were interpolated across variable groups
        interpolation_by_type = (
            df.groupby("variable_type", observed=True)["is_interpolated"]
            .agg(
                interpolation_rate="mean",
                interpolated_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
        )
        interpolation_by_type.to_csv(TABLES_DIR / "interpolation_by_variable_type.csv")
        
        # computes the same interpolation summary for each individual magnitude
        interpolation_by_magnitude = (
            df.groupby("magnitude_name", observed=True)["is_interpolated"]
            .agg(
                interpolation_rate="mean",
                interpolated_rows="sum",
                known_flag_rows="count",
                total_rows="size",
                )
            .sort_values("interpolation_rate", ascending=False)
        )

        interpolation_by_magnitude.to_csv(TABLES_DIR / "interpolation_by_magnitude.csv")
        
        # preserve cleaning-stage information about interpolation-flag completeness
        interpolation_flag_quality = pd.DataFrame({
            "missing_is_interpolated_before_cleaning": [
                self.interpolation_flag_quality.get(
                    "missing_is_interpolated_before_cleaning")],
            "missing_rate_before_cleaning": [
                self.interpolation_flag_quality.get(
                    "missing_rate_before_cleaning")],
            "total_rows_before_cleaning": [
                self.interpolation_flag_quality.get(
                    "total_rows_before_cleaning"
                )],
            "total_rows_after_cleaning": [len(df)],
        })

        interpolation_flag_quality.to_csv(
                TABLES_DIR / "interpolation_flag_quality.csv", index=False)
        
        # plots the number of observations in each broad variable category
        plt.figure(figsize=(8, 5))
        counts.plot(kind="bar")
        plt.title("Rows by Variable Type")
        plt.ylabel("Count")
        plt.xlabel("Variable Type")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "variable_type_counts.png", dpi=300)
        plt.close()

        # plots the interpolation rate for pollutant, weather, and traffic groups
        plt.figure(figsize=(10, 5))
        interpolation_by_type["interpolation_rate"].plot(kind="bar")
        plt.title("Interpolation Rate by Variable Type")
        plt.ylabel("Interpolation Rate")
        plt.xlabel("Variable Type")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "interpolation_rate_by_variable_type.png", dpi=300)
        plt.close()

        # distribution and time-series
        # we automatically select the pollutant with the most rows.
        pollutant_names = {
            "SO2", "CO", "NO", "NO2", "PM2.5", "PM10", "<PM2.5", "<PM10",
            "NOX", "O3", "TOLUENO", "BENCENO", "ETILBENCENO",
            "HIDROCARBS_TOTALES", "METANO", "HIDROCARBS_NO_METANICOS"
        }

        pollutant_df = df[
            df["magnitude_name"].astype(str).str.upper().isin(pollutant_names)
        ].copy()

        if not pollutant_df.empty:
            main_pollutant = pollutant_df["magnitude_name"].value_counts().idxmax()

            selected = pollutant_df[
                pollutant_df["magnitude_name"] == main_pollutant
            ].copy()

            # saves which pollutant was selected
            selected_summary = pd.DataFrame({
                "selected_pollutant_for_initial_plots": [main_pollutant],
                "n_rows": [len(selected)],
                "min_date": [selected["entry_date"].min()],
                "max_date": [selected["entry_date"].max()],
                "mean_value": [selected["value"].mean()],
            })

            selected_summary.to_csv(
                TABLES_DIR / "selected_pollutant_for_initial_plots.csv",
                index=False
            )

            # distribution plot: shows how values are spread
            plt.figure(figsize=(8, 5))
            selected["value"].plot(kind="hist", bins=50)
            plt.title(f"Distribution of {main_pollutant} Values")
            plt.xlabel(f"{main_pollutant} value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"{main_pollutant}_value_distribution.png",
                dpi=300
            )
            plt.close()

            # time-series plot: x-axis is time, y-axis is average pollutant value
            monthly_series = (
                selected.set_index("entry_date")
                .resample("M")["value"]
                .mean()
            )

            plt.figure(figsize=(10, 5))
            monthly_series.plot()
            plt.title(f"Monthly Average {main_pollutant}")
            plt.xlabel("Date")
            plt.ylabel(f"Average {main_pollutant} value")
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"{main_pollutant}_monthly_timeseries.png",
                dpi=300
            )
            plt.close()


def run_task1(use_sample=True, parallel=False, nrows=None):
    """Run the full Task 1 pipeline from data loading through output generation.

    Args:
        use_sample: if True, loads the smaller sample CSV file, otherwise, loads
            and combines the full parquet dataset
        parallel: if True and `use_sample` is False, reads parquet files in
            parallel using multiprocessing
        nrows: Optional limit on the number of rows to keep. If None, all available rows are used.
    Returns:
        A cleaned and enriched pandas DataFrame containing the Task 1 outputs"""

    print("\n--- Task 1: Load and Inspect Data ---")
    loader = MadridDataLoader(
        use_sample=use_sample,
        parallel=parallel,
        nrows=nrows,
    )

    df = loader.load_data()
    print("Loaded data")

    df = loader.clean_data(df)
    print("Cleaned data")

    df = loader.add_time_features(df)
    print("Added time features")

    df = loader.add_variable_type(df)
    print("Added variable types")

    loader.save_cleaned_data(df)
    print("Saved cleaned data")

    loader.save_outputs(df)
    print("Saved outputs")

    print(f"Rows: {len(df):,}")
    print(f"Sensors: {df['sensor_id'].nunique():,}")
    print(f"Magnitudes: {df['magnitude_name'].nunique():,}")
    print("Task 1 completed")

    return df