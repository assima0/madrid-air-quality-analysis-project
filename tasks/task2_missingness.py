""" Task 2 pipeline for missingness and data-quality analysis.
This file: 
- reconstructs the dataset's original missingness using the is_interpolated flag
- examines where missing values occur across columns, variable groups, pollutants, sensors, and time
- identifies consecutive missing periods
- checks for selected invalid or inconsistent values
- saves summary tables and figures"""

# used for numerical and tabular analysis
import pandas as pd
import numpy as np

# used for plotting/visualizations
import matplotlib.pyplot as plt

# project-specific output directories and timing decorator
from utils.config import get_task_dirs
from utils.helpers import timer

# output folders
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task2_missingness")

class MissingnessAnalyzer:
    """ Analyze reconstructed missingness, data-quality issues, and temporal gaps"""

    def __init__(self, df):
        self.df = df

    def reconstruct_missingness(self):
        """ Reconstruct original missingness using "is_interpolated" column

        In METRAQ:
        True  = originally missing and reconstructed through interpolation
        False = original measurement
        <NA>  = unknown flag, if present"""
        # converts the dataset's interpolation flag into an explicit missingness indicator
        self.df["was_missing"] = self.df["is_interpolated"].astype("boolean")
        self.df["missing_flag_unknown"] = self.df["was_missing"].isna()
        return self.df

    def missingness_by_column(self):
        """ Standard missing values in the cleaned dataframe columns"""

        return pd.DataFrame({
            "column": self.df.columns,
            "missing_values": self.df.isna().sum().values,
            "missing_rate": self.df.isna().mean().values,
            "dtype": [str(dtype) for dtype in self.df.dtypes],
        }).sort_values("missing_rate", ascending=False)

    def original_missingness_by_variable_type(self):
        """Summarize reconstructed original missingness by pollutant, weather, and traffic categories"""

        return (
            self.df.groupby("variable_type", observed=True)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows=lambda s: s.fillna(False).sum(),
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_values("original_missing_rate", ascending=False)
        )

    def original_missingness_by_pollutant(self):
        """ Summarize reconstructed original missingness for pollutant type variables only"""

        pollutants = self.df[self.df["variable_type"] == "pollutant"]
        return (
            pollutants.groupby("magnitude_name", observed=True)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_values("original_missing_rate", ascending=False)
        )

    def original_missingness_by_magnitude(self):
        """ Summarize reconstructed original missingness for magnitudes"""
        return (
            self.df.groupby(["variable_type", "magnitude_name"], observed=True)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_values("original_missing_rate", ascending=False)
        )

    def original_missingness_by_sensor(self):
        """ Summarize reconstructed original missingness at the monitoring-station level"""

        return (
            self.df.groupby(["sensor_id", "sensor_name"], observed=True)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_values("original_missing_rate", ascending=False)
        )

    def original_missingness_by_sensor_and_variable(self):
        """ Summarize reconstructed original missingness for every sensor-variable combination
        - more specific and granular than sensor-level or variable-level alone"""
        return (
            self.df.groupby(
                ["sensor_id", "sensor_name", "magnitude_name"],
                observed=True
            )["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_values("original_missing_rate", ascending=False)
        )

    def temporal_missingness_daily(self):
        """ Aggregate original missingness by day to show short-term temporal variation"""

        return (
            self.df.groupby(self.df["entry_date"].dt.date)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_index()
        )

    def temporal_missingness_monthly(self):
        """ Aggregate original missingness by month"""
        return (
            self.df.groupby(self.df["entry_date"].dt.to_period("M").astype(str))["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_index()
        )

    def temporal_missingness_by_hour(self):
        """ Measure whether missingness is more frequent at particular hours of the day"""

        return (
            self.df.groupby("hour")["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .sort_index()
        )

    def temporal_missingness_by_year_and_variable_type(self):
        """ Track missingness trends over years separately for pollutant, weather, and traffic groups"""

        return (
            self.df.groupby(["year", "variable_type"], observed=True)["was_missing"]
            .agg(
                original_missing_rate="mean",
                originally_missing_rows="sum",
                known_flag_rows="count",
                total_rows="size",
            )
            .reset_index()
            .sort_values(["year", "variable_type"])
        )

    def consecutive_missing_periods(self):
        """ Detect consecutive interpolated periods by sensor and variable.

        Identifies temporal gaps: runs of was_missing=True rows for the same
        sensor-variable pair """

        temp = self.df[
            [
                "sensor_id",
                "sensor_name",
                "magnitude_name",
                "entry_date",
                "was_missing",
            ]
        ].copy()

        # sorts each sensor-variable series in chronological order
        temp = temp.sort_values(["sensor_id", "magnitude_name", "entry_date"])
        group_cols = ["sensor_id", "magnitude_name"]
        
        # start a new run whenever missingness changes from True to False or vice versa
        temp["missing_status_change"] = (
            temp.groupby(group_cols, observed=True)["was_missing"]
            .transform(lambda s: s.ne(s.shift()).cumsum())
        )
        
        # keep only the runs corresponding to original missingness
        temp = temp[temp["was_missing"].fillna(False) == True]

        gaps = (
            temp.groupby(
                ["sensor_id", "sensor_name", "magnitude_name", "missing_status_change"],
                observed=True
            )
            .agg(
                gap_start=("entry_date", "min"),
                gap_end=("entry_date", "max"),
                n_consecutive_missing_rows=("entry_date", "size"),
            )
            .reset_index()
            .drop(columns=["missing_status_change"])
            .sort_values("n_consecutive_missing_rows", ascending=False)
        )

        return gaps

    def invalid_values_summary(self):
        """ Summarize basic invalid or inconsistent values in the dataset"""
   
        magnitude_upper = self.df["magnitude_name"].astype(str).str.strip().str.upper()
        
        # variables whose values shouldn't normally be negative
        non_negative_variables = {
            "SO2", "CO", "NO", "NO2", "NOX", "PM2.5", "PM10", "<PM2.5", "<PM10",
            "O3", "TOLUENO", "BENCENO", "ETILBENCENO",
            "HIDROCARBS_TOTALES", "METANO", "HIDROCARBS_NO_METANICOS",
            "VV", "HR", "PRE", "RS", "PRECIPITACION",
        }
        non_negative_prefixes = ("TI_", "SP_", "OC_")

        is_non_negative_var = (
            magnitude_upper.isin(non_negative_variables)
            | magnitude_upper.str.startswith(non_negative_prefixes)
        )
        
        # builds masks or subsets for each quality issue to be summarized below
        # flags rows (where variables should be non-negative) if it contains negative values
        negative_invalid  = self.df[is_non_negative_var & (self.df["value"] < 0)]

        # checks for invalid relative humidity values (in percentages)
        humidity_invalid  = self.df[(magnitude_upper == "HR") & ((self.df["value"] < 0) | (self.df["value"] > 100))]
        
        # checks for invalid wind direction values (0-360)
        wind_dir_invalid  = self.df[(magnitude_upper == "DV") & ((self.df["value"] < 0) | (self.df["value"] > 360))]
        
        # checks for missing coordinate values
        coordinate_missing = self.df[self.df["utm_x"].isna() | self.df["utm_y"].isna()]
        
        # marks all repeated sensor-variable-timestamp combinations, since each 
        # combination should normally correspond to one observation
        duplicate_mask  = self.df.duplicated(subset=["sensor_id", "magnitude_name", "entry_date"], keep=False)

        summary = pd.DataFrame({
            "issue": [
                "negative_values_for_non_negative_variables",
                "relative_humidity_outside_0_100",
                "wind_direction_outside_0_360",
                "missing_coordinates",
                "duplicate_sensor_variable_timestamp_rows",
                "unknown_interpolation_flags",
            ],
            "n_rows": [
                len(negative_invalid),
                len(humidity_invalid),
                len(wind_dir_invalid),
                len(coordinate_missing),
                int(duplicate_mask.sum()),
                int(self.df["missing_flag_unknown"].sum()),
            ],
            "rate": [
                len(negative_invalid) / len(self.df),
                len(humidity_invalid) / len(self.df),
                len(wind_dir_invalid) / len(self.df),
                len(coordinate_missing) / len(self.df),
                duplicate_mask.mean(),
                self.df["missing_flag_unknown"].mean(),
            ],
        })

        return summary

    
    # FIGURES

    def _plot_sensor_variable_heatmap(self, by_sensor_variable):
        """ Sensor and pollutant missingness heatmap """

        # keeps the pollutants only
        df_heat = by_sensor_variable.reset_index()
        pollutant_types = {"pollutant"}

        if "variable_type" in df_heat.columns:
            df_heat = df_heat[df_heat["variable_type"].isin(pollutant_types)]

        if df_heat.empty:
            return

        # pivot: rows = sensor_name, columns = magnitude_name
        pivot = df_heat.pivot_table(
            index="sensor_name",
            columns="magnitude_name",
            values="original_missing_rate",
            aggfunc="mean",
        ).fillna(0).astype(float)

        # sorts sensors by average missingness (worst at top)
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.9),
                                        max(6, pivot.shape[0] * 0.45)))

        im = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)

        # annotates cells with rate values where > 0
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if val > 0.01:
                    text_color = "white" if val > 0.6 else "black"
                    ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                            fontsize=6.5, color=text_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Original missingness rate", fontsize=9)

        ax.set_title(
            "Original Missingness Rate — Sensor × Pollutant\n"
            "Each cell shows the fraction of originally missing measurements",
            fontsize=11,
        )
        ax.set_xlabel("Pollutant / Variable", fontsize=9)
        ax.set_ylabel("Sensor (station)", fontsize=9)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missingness_heatmap_sensor_variable.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: missingness_heatmap_sensor_variable.png")

    def _plot_missingness_over_time_stacked(self, by_year_type):
        """ Stacked area chart: missingness by variable type over years"""
        pivot = by_year_type.pivot_table(
            index="year",
            columns="variable_type",
            values="original_missing_rate",
            aggfunc="mean",
        ).fillna(0).astype(float)

        colors = {
            "pollutant": "#D62728",
            "weather":   "#2CA02C",
            "traffic":   "#1F77B4",
            "unknown":   "#7F7F7F",
        }

        fig, ax = plt.subplots(figsize=(12, 5))

        col_order = [c for c in ["pollutant", "weather", "traffic", "unknown"]
                     if c in pivot.columns]

        ax.stackplot(
            pivot.index.astype(int),
            [pivot[c].values.astype(float) for c in col_order],
            labels=col_order,
            colors=[colors.get(c, "#AAAAAA") for c in col_order],
            alpha=0.75,
        )

        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("Average missingness rate", fontsize=10)
        ax.set_title(
            "Original Missingness Rate by Variable Type Over Time\n"
            "Shows when each variable category was introduced and how coverage improved",
            fontsize=11,
        )
        ax.legend(loc="upper right", fontsize=9, title="Variable type")
        ax.set_xlim(int(pivot.index.min()), int(pivot.index.max()))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "missingness_by_year_type_stacked.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: missingness_by_year_type_stacked.png")

    def _plot_gap_length_distribution(self, gap_periods):
        """ Distribution of consecutive missing period lengths by variable type.
        
        - Shows whether gaps are typically short (a few hours as in sensor glitch) or 
        long (days/weeks). Short gaps are straightforward to impute; long gaps usually require a different strategy. 
        - Log-scale x-axis makes both short and long gaps visible at once.
        - Reference lines at 24h and 1 week provide practical benchmarks."""

        if gap_periods.empty:
            return

        # merge variable type onto gaps
        mag_type = (
            self.df[["magnitude_name", "variable_type"]]
            .drop_duplicates()
            .copy()
        )
        gaps = gap_periods.merge(mag_type, on="magnitude_name", how="left")

        type_colors = {
            "pollutant": "#D62728",
            "weather":   "#2CA02C",
            "traffic":   "#1F77B4",
            "unknown":   "#7F7F7F",
        }

        vtypes = [v for v in ["pollutant", "weather", "traffic", "unknown"]
                  if v in gaps["variable_type"].dropna().unique()]

        if not vtypes:
            return

        fig, axes = plt.subplots(1, len(vtypes),
                                 figsize=(5 * len(vtypes), 5),
                                 sharey=False)
        if len(vtypes) == 1:
            axes = [axes]

        for ax, vtype in zip(axes, vtypes):
            sub = gaps[gaps["variable_type"] == vtype]["n_consecutive_missing_rows"]
            if sub.empty:
                continue

            color = type_colors.get(vtype, "#888888")
            ax.hist(sub, bins=50, color=color, alpha=0.82, edgecolor="white",
                    linewidth=0.4)
            ax.set_xscale("log")
            ax.set_xlabel("Gap length (hours, log scale)", fontsize=9)
            ax.set_ylabel("Number of gaps", fontsize=9)
            ax.set_title(f"{vtype.capitalize()}\n"
                         f"median={sub.median():.0f} h, "
                         f"max={sub.max():.0f} h",
                         fontsize=9)
            ax.axvline(24, color="black", linestyle="--", linewidth=0.9,
                       label="24 h")
            ax.axvline(168, color="gray", linestyle=":", linewidth=0.9,
                       label="1 week")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.2)

        fig.suptitle(
            "Distribution of Consecutive Missing Periods by Variable Type\n"
            "Short gaps (< 24 h) are easy to impute; long gaps are more uncertain",
            fontsize=11, fontweight="medium",
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "gap_length_distribution.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: gap_length_distribution.png")

    def _plot_data_availability_matrix(self, by_year_type):
        """ Annual data availability matrix """

        pivot = by_year_type.pivot_table(
            index="year",
            columns="variable_type",
            values="original_missing_rate",
            aggfunc="mean",
        ).fillna(1.0).astype(float)

        # availability = 1 - missingness
        avail = (1 - pivot).astype(float)

        fig, ax = plt.subplots(figsize=(max(6, avail.shape[1] * 1.6),
                                        max(5, avail.shape[0] * 0.35)))

        im = ax.imshow(
            avail.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        ax.set_xticks(range(avail.shape[1]))
        ax.set_xticklabels(avail.columns, fontsize=10)
        ax.set_yticks(range(avail.shape[0]))
        ax.set_yticklabels(avail.index.astype(int), fontsize=9)

        for i in range(avail.shape[0]):
            for j in range(avail.shape[1]):
                val = avail.values[i, j]
                text_color = "white" if val < 0.35 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="medium")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Data availability (1 − missingness rate)", fontsize=9)

        ax.set_title(
            "Data Availability by Year and Variable Type\n"
            "Green = good coverage, Red = mostly interpolated",
            fontsize=11,
        )
        ax.set_xlabel("Variable type", fontsize=10)
        ax.set_ylabel("Year", fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "data_availability_matrix.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: data_availability_matrix.png")

    def _plot_data_quality_dashboard(self, invalid_summary, by_type, by_month, by_hour):
        """ Data quality dashboard (2x2 panel)

        Combines four complementary quality views into one figure:
          top-left:  Invalid value counts by issue type (horizontal bars)
          top-right: Missingness rate by variable type (horizontal bars)
          bottom-left: Monthly missingness trend with rolling mean
          bottom-right: Missingness by hour of day"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # top-left: invalid value counts
        ax = axes[0, 0]
        inv = invalid_summary[invalid_summary["n_rows"] > 0].copy()
        short_labels = {
            "negative_values_for_non_negative_variables": "Negative values",
            "relative_humidity_outside_0_100": "Humidity out of range",
            "wind_direction_outside_0_360": "Wind dir out of range",
            "missing_coordinates": "Missing coordinates",
            "duplicate_sensor_variable_timestamp_rows": "Duplicate rows",
            "unknown_interpolation_flags": "Unknown interp. flags",
        }
        inv["label"] = inv["issue"].map(short_labels).fillna(inv["issue"])
        colors_inv = ["#D62728" if r > 0 else "#AAAAAA" for r in inv["n_rows"]]
        bars = ax.barh(inv["label"], inv["n_rows"], color=colors_inv, alpha=0.85)
        ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=8)
        ax.set_xlabel("Number of rows", fontsize=9)
        ax.set_title("Invalid / inconsistent values", fontsize=10, fontweight="medium")
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        # top-right: missingness rate by variable type
        ax = axes[0, 1]
        by_type_reset = by_type.reset_index()
        type_colors = {
            "pollutant": "#D62728", "weather": "#2CA02C",
            "traffic": "#1F77B4", "unknown": "#7F7F7F",
        }
        bar_colors = [type_colors.get(str(vt), "#AAAAAA")
                      for vt in by_type_reset["variable_type"]]
        bars2 = ax.barh(
            by_type_reset["variable_type"].astype(str),
            by_type_reset["original_missing_rate"],
            color=bar_colors, alpha=0.85,
        )
        ax.bar_label(bars2, fmt="%.1%", padding=4, fontsize=9)
        ax.set_xlabel("Original missingness rate", fontsize=9)
        ax.set_title("Missingness rate by variable type", fontsize=10, fontweight="medium")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="x", alpha=0.2)
        ax.invert_yaxis()

        # bottom-left: monthly trend 
        ax = axes[1, 0]
        monthly_plot = by_month.copy()
        monthly_plot.index = pd.to_datetime(monthly_plot.index)
        raw = monthly_plot["original_missing_rate"].astype(float)
        rolling = raw.rolling(6, center=True).mean()

        x = raw.index.to_numpy()
        ax.fill_between(x, raw.values.astype(float), alpha=0.2, color="#4C72B0")
        ax.plot(x, raw.values.astype(float), color="#4C72B0", linewidth=0.6,
                alpha=0.5, label="Monthly")
        ax.plot(x, rolling.values.astype(float), color="#1B4F72", linewidth=1.8,
                label="6-month rolling mean")
        ax.set_ylabel("Missingness rate", fontsize=9)
        ax.set_xlabel("Date", fontsize=9)
        ax.set_title("Missingness trend over time", fontsize=10, fontweight="medium")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # bottom-right: by hour
        ax = axes[1, 1]
        hour_data = by_hour["original_missing_rate"].astype(float)
        hour_colors = ["#D62728" if v == hour_data.max() else "#4C72B0"
                       for v in hour_data.values]
        ax.bar(hour_data.index.astype(int), hour_data.values.astype(float),
               color=hour_colors, alpha=0.82)
        ax.set_xlabel("Hour of day", fontsize=9)
        ax.set_ylabel("Missingness rate", fontsize=9)
        ax.set_title("Missingness by hour of day", fontsize=10, fontweight="medium")
        ax.set_xticks(range(0, 24, 2))
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="y", alpha=0.2)
        worst_hour = hour_data.idxmax()
        ax.annotate(
            f"Peak: hour {worst_hour}",
            xy=(worst_hour, hour_data.max()),
            xytext=(worst_hour + 1.5, hour_data.max() * 1.05),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            fontsize=8, color="#D62728",
        )

        fig.suptitle(
            "Data Quality Overview — Madrid Air Quality Dataset",
            fontsize=13, fontweight="medium", y=1.01,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "data_quality_dashboard.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: data_quality_dashboard.png")

    # main output method

    def save_outputs(self):
        """ Compute, export, and visualize Task 2 results.

        Generates tabular summaries of reconstructed missingness, 
        temporal gap statistics, and invalid-value checks, then saves the main
        diagnostic figures"""

        
        print("Computing missingness by column...")
        by_column = self.missingness_by_column()

        print("Computing missingness by variable type...")
        by_type = self.original_missingness_by_variable_type()

        print("Computing missingness by pollutant...")
        by_pollutant = self.original_missingness_by_pollutant()

        print("Computing missingness by magnitude...")
        by_magnitude = self.original_missingness_by_magnitude()

        print("Computing missingness by sensor...")
        by_sensor = self.original_missingness_by_sensor()

        print("Computing missingness by sensor and variable...")
        by_sensor_variable = self.original_missingness_by_sensor_and_variable()

        print("Computing daily missingness...")
        by_day = self.temporal_missingness_daily()

        print("Computing monthly missingness...")
        by_month = self.temporal_missingness_monthly()

        print("Computing hourly missingness...")
        by_hour = self.temporal_missingness_by_hour()

        print("Computing yearly missingness by variable type...")
        by_year_type = self.temporal_missingness_by_year_and_variable_type()

        print("Computing consecutive missing periods...")
        gap_periods = self.consecutive_missing_periods()

        print("Checking invalid values...")
        invalid_summary = self.invalid_values_summary()

        print("Saving tables...")
        by_column.to_csv(TABLES_DIR / "missingness_by_column.csv", index=False)
        by_type.to_csv(TABLES_DIR / "original_missingness_by_variable_type.csv")
        by_pollutant.to_csv(TABLES_DIR / "original_missingness_by_pollutant.csv")
        by_magnitude.to_csv(TABLES_DIR / "original_missingness_by_magnitude.csv")
        by_sensor.to_csv(TABLES_DIR / "original_missingness_by_sensor.csv")
        by_sensor_variable.to_csv(TABLES_DIR / "original_missingness_by_sensor_and_variable.csv")
        by_day.to_csv(TABLES_DIR / "temporal_missingness_daily.csv")
        by_month.to_csv(TABLES_DIR / "temporal_missingness_monthly.csv")
        by_hour.to_csv(TABLES_DIR / "temporal_missingness_by_hour.csv")
        by_year_type.to_csv(TABLES_DIR / "temporal_missingness_by_year_and_variable_type.csv", index=False)
        gap_periods.to_csv(TABLES_DIR / "consecutive_missing_periods.csv", index=False)
        invalid_summary.to_csv(TABLES_DIR / "invalid_values_summary.csv", index=False)

        print("Saving figures...")
        self._plot_sensor_variable_heatmap(by_sensor_variable)
        self._plot_missingness_over_time_stacked(by_year_type)
        self._plot_gap_length_distribution(gap_periods)
        self._plot_data_availability_matrix(by_year_type)
        self._plot_data_quality_dashboard(invalid_summary, by_type, by_month, by_hour)


@timer
def run_task2(df):
    """ Run the Task 2 missingness analysis pipeline
    Args:
        df: cleaned DataFrame from Task 1/2
    Returns the DataFrame with two columns added"""

    print("\n--- Task 2: Missingness and Data Quality ---")

    analyzer = MissingnessAnalyzer(df)
    df = analyzer.reconstruct_missingness()
    analyzer.save_outputs()

    print("Task 2 completed")
    return df
