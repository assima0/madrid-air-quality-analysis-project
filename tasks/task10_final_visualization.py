""" Task 10 pipeline for final project visualization.

This file:
- reads saved CSV tables and figures from Tasks 1-9,
- produces a concise set of summary visualizations covering dataset overview,
  temporal patterns, sensor map, network comparison, imputation comparison,
  and forecasting results,
- assembles all panels into one summary dashboard figure.

Task 10 does not recompute any analysis, it just simply reads upstream outputs
and communicates results clearly in a presentation-ready format"""

# Used for warnings control, numerical operations, and tabular processing
import warnings
import numpy as np
import pandas as pd

# used for final project figures and multi-panel layouts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path

# project-specific output directories and processed-data paths
from utils.config import get_task_dirs, PROCESSED_DIR
from utils.helpers import timer

# hides non-critical library warnings to keep the visualization console output readable
warnings.filterwarnings("ignore", category=UserWarning)

# output folders dedicated to Task 10 summary tables and presentation figures
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task10_final_visualization")

# paths to upstream task outputs 
OUTPUTS_ROOT = TABLES_DIR.parent.parent   


def _task_tables(task_name: str) -> Path:
    return OUTPUTS_ROOT / task_name / "tables"


def _task_figures(task_name: str) -> Path:
    return OUTPUTS_ROOT / task_name / "figures"


def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    """ Read CSV if it exists, else return empty DataFrame"""
    if path.exists():
        return pd.read_csv(path, **kwargs)
    print(f"  [skip] not found: {path.name}")
    return pd.DataFrame()

class FinalVisualizer:
    """ Read upstream task outputs and produce a final set of summary figures.
    
    All plot methods read from previously saved CSV tables and PNG figures.
    If an upstream output is missing, the corresponding panel is skipped
    gracefully rather than raising an error."""

    def __init__(self, df: pd.DataFrame | None = None):
        """Initialize the final visualizer.
        
        Args: df: optional cleaned DataFrame from Task 1, used for the sensor map
              and as a fallback when upstream CSV tables are missing.
              If None, loaded from the processed parquet on disk"""
        # prefers an already-loaded DataFrame to avoid an unnecessary disk read.
        if df is not None:
            self.df = df
        else:
            parquet = PROCESSED_DIR / "cleaned_air_quality.parquet"
            print(f"  Loading cleaned data from {parquet}…")
            self.df = pd.read_parquet(parquet)
        
        # creates the output folders up front so all plot methods can save safely
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        TABLES_DIR.mkdir(parents=True, exist_ok=True)

    def plot_dataset_overview(self):
        """Save a two-panel overview of dataset composition and interpolation rates.
        
        The left panel shows row counts by variable type. The right panel shows the
        magnitudes with the highest interpolation rates, using Task 1 tables when
        available and falling back to the cleaned DataFrame otherwise"""
        # prefers task 1 summary tables so task 10 reuses earlier analysis outputs
        var_counts = _safe_read(
            _task_tables("task1_load_inspect") / "variable_type_counts.csv"
        )
        interp_mag = _safe_read(
            _task_tables("task1_load_inspect") / "interpolation_by_magnitude.csv"
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # left: variable type counts
        if not var_counts.empty:
            var_counts.columns = var_counts.columns.str.strip()
            # handles both index-saved and column-saved formats
            if "variable_type" in var_counts.columns:
                ax = axes[0]
                ax.bar(
                    var_counts["variable_type"].astype(str),
                    var_counts["count"] if "count" in var_counts.columns
                    else var_counts.iloc[:, 1],
                    color=["#4C72B0", "#55A868", "#C44E52"],
                )
                ax.set_title("Rows by Variable Type")
                ax.set_xlabel("Variable Type")
                ax.set_ylabel("Row Count")
            else:
                # fallback: compute from raw df
                counts = self.df["variable_type"].value_counts()
                axes[0].bar(counts.index.astype(str), counts.values,
                            color=["#4C72B0", "#55A868", "#C44E52"])
                axes[0].set_title("Rows by Variable Type")
                axes[0].set_ylabel("Row Count")
        else:
            counts = self.df["variable_type"].value_counts()
            axes[0].bar(counts.index.astype(str), counts.values)
            axes[0].set_title("Rows by Variable Type")
            axes[0].set_ylabel("Row Count")

        # right: top-10 missingness by magnitude
        if not interp_mag.empty:
            interp_mag.columns = interp_mag.columns.str.strip()
            rate_col = next(
                (c for c in interp_mag.columns if "rate" in c.lower()), None
            )
            name_col = interp_mag.columns[0]
            if rate_col:
                top10 = interp_mag.nlargest(10, rate_col)
                axes[1].barh(
                    top10[name_col].astype(str),
                    top10[rate_col],
                    color="#4C72B0",
                )
                axes[1].set_title("Top 10 Variables by Interpolation Rate")
                axes[1].set_xlabel("Interpolation Rate")
                axes[1].invert_yaxis()
        else:
            interp = (
                self.df.groupby("magnitude_name", observed=True)["is_interpolated"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            axes[1].barh(interp.index.astype(str), interp.values, color="#4C72B0")
            axes[1].set_title("Top 10 Variables by Interpolation Rate")
            axes[1].set_xlabel("Interpolation Rate")
            axes[1].invert_yaxis()

        fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "01_dataset_overview.png", dpi=300)
        plt.close()
        print("  Saved: 01_dataset_overview.png")

    def plot_temporal_patterns(self):
        """ Save a multi-panel summary of long-run, hourly, and seasonal pollution patterns.
        
        The figure reuses task 4 outputs to show:
        - yearly trends,
        - average hourly cycles,
        - average calendar-month cycles"""
        yearly = _safe_read(
            _task_tables("task4_temporal") / "yearly_trends_by_sensor.csv"
        )
        hourly = _safe_read(
            _task_tables("task4_temporal") / "hourly_cycle_by_sensor.csv"
        )
        seasonal = _safe_read(
            _task_tables("task4_temporal") / "seasonal_month_cycle_by_sensor.csv"
        )
        # if none of the task 4 tables exist, skips this figure rather than producing empty panels
        if yearly.empty and hourly.empty and seasonal.empty:
            print("  [skip] No Task 4 outputs found.")
            return

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
        ax_top = fig.add_subplot(gs[0, :])
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_br = fig.add_subplot(gs[1, 1])

        # yearly trend
        if not yearly.empty:
            yearly.columns = yearly.columns.str.strip()
            mag_col = next(
                (c for c in yearly.columns if "magnitude" in c.lower()), None
            )
            year_col = next(
                (c for c in yearly.columns if "year" in c.lower()), None
            )
            val_col = next(
                (c for c in yearly.columns if "mean" in c.lower() or "value" in c.lower()), None
            )
            if mag_col and year_col and val_col:
                pollutants = yearly[mag_col].value_counts().head(4).index
                annual = (
                    yearly.groupby([year_col, mag_col], observed=True)[val_col]
                    .mean()
                    .reset_index()
                )
                for p in pollutants:
                    sub = annual[annual[mag_col] == p]
                    ax_top.plot(sub[year_col], sub[val_col], marker="o", label=str(p))
                ax_top.set_title("Long-Run Yearly Pollution Trend")
                ax_top.set_xlabel("Year")
                ax_top.set_ylabel("Mean Value")
                ax_top.legend(fontsize=8)

        # hourly cycle
        if not hourly.empty:
            hourly.columns = hourly.columns.str.strip()
            mag_col = next((c for c in hourly.columns if "magnitude" in c.lower()), None)
            hour_col = next((c for c in hourly.columns if "hour" in c.lower()), None)
            val_col = next((c for c in hourly.columns if "mean" in c.lower() or "value" in c.lower()), None)
            if mag_col and hour_col and val_col:
                pollutants = hourly[mag_col].value_counts().head(3).index
                hourly_avg = (
                    hourly.groupby([hour_col, mag_col], observed=True)[val_col]
                    .mean().reset_index()
                )
                for p in pollutants:
                    sub = hourly_avg[hourly_avg[mag_col] == p]
                    ax_bl.plot(sub[hour_col], sub[val_col], marker="o", label=str(p))
                ax_bl.set_title("Average Hourly Cycle")
                ax_bl.set_xlabel("Hour of Day")
                ax_bl.set_ylabel("Mean Value")
                ax_bl.set_xticks(range(0, 24, 3))
                ax_bl.legend(fontsize=7)

        # seasonal cycle
        if not seasonal.empty:
            seasonal.columns = seasonal.columns.str.strip()
            mag_col = next((c for c in seasonal.columns if "magnitude" in c.lower()), None)
            month_col = next((c for c in seasonal.columns if "month" in c.lower()), None)
            val_col = next((c for c in seasonal.columns if "mean" in c.lower() or "value" in c.lower()), None)
            if mag_col and month_col and val_col:
                pollutants = seasonal[mag_col].value_counts().head(3).index
                seas_avg = (
                    seasonal.groupby([month_col, mag_col], observed=True)[val_col]
                    .mean().reset_index()
                )
                for p in pollutants:
                    sub = seas_avg[seas_avg[mag_col] == p]
                    ax_br.plot(sub[month_col], sub[val_col], marker="o", label=str(p))
                ax_br.set_title("Seasonal (Monthly) Cycle")
                ax_br.set_xlabel("Month")
                ax_br.set_ylabel("Mean Value")
                ax_br.set_xticks(range(1, 13))
                ax_br.legend(fontsize=7)

        fig.suptitle("Temporal Patterns", fontsize=14, fontweight="bold")
        plt.savefig(FIGURES_DIR / "02_temporal_patterns.png", dpi=300)
        plt.close()
        print("  Saved: 02_temporal_patterns.png")

    def plot_sensor_map(self):
        """ Save a geographic sensor map colored by mean NO2 concentration.
        
        Sensor coordinates are reduced to one average UTM location per sensor.
        The color scale uses mean NO2 concentration where available"""
        coords = (
            self.df
            .dropna(subset=["sensor_id", "sensor_name", "utm_x", "utm_y"])
            .groupby(["sensor_id", "sensor_name"], observed=True)[["utm_x", "utm_y"]]
            .mean()
            .reset_index()
        )

        # computes mean NO2 per sensor for colour coding
        no2_mask = self.df["magnitude_name"].astype(str).str.upper() == "NO2"
        no2_mean = (
            self.df[no2_mask]
            .groupby("sensor_id", observed=True)["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "mean_no2"})
        )
        coords = coords.merge(no2_mean, on="sensor_id", how="left")

        fig, ax = plt.subplots(figsize=(8, 7))

        sc = ax.scatter(
            coords["utm_x"],
            coords["utm_y"],
            c=coords["mean_no2"] if "mean_no2" in coords.columns else "steelblue",
            cmap="YlOrRd",
            s=120,
            edgecolors="grey",
            linewidths=0.5,
            zorder=3,
        )

        if "mean_no2" in coords.columns and coords["mean_no2"].notna().any():
            plt.colorbar(sc, ax=ax, label="Mean NO₂ (μg/m³)")

        # label sensors
        for _, row in coords.iterrows():
            ax.annotate(
                str(row["sensor_name"]),
                (row["utm_x"], row["utm_y"]),
                fontsize=5,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

        ax.set_title("Madrid Air Quality Sensor Locations\n(coloured by mean NO₂)", fontsize=12)
        ax.set_xlabel("UTM X (m)")
        ax.set_ylabel("UTM Y (m)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "03_sensor_map.png", dpi=300)
        plt.close()
        print("  Saved: 03_sensor_map.png")


    def plot_network_comparison(self):
        """Compare selected spatial and correlation-network structures side by side.
        
        The figure reads the selected task 5 spatial graph summary and one task 6
        correlation graph summary, then contrasts core metrics such as nodes,
        edges, density, and average degree"""
        # uses the selected task 5 spatial graph as the geographic-network reference
        spatial = _safe_read(
            _task_tables("task5_spatial_network") / "selected_spatial_graph_summary.csv"
        )
        # loads one Task 6 correlation-summary file if available
        corr_path = next(
            _task_tables("task6_correlation_network").glob("correlation_graph_summary_*.csv"),
            None,
        )
        corr = _safe_read(corr_path) if corr_path else pd.DataFrame()

        if spatial.empty and corr.empty:
            print("  [skip] No Task 5/6 network summaries found.")
            return

        metrics = ["n_nodes", "n_edges", "density", "average_degree"]
        labels = ["Nodes", "Edges", "Density", "Avg Degree"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5))

        for ax, metric, label in zip(axes, metrics, labels):
            values = []
            names = []

            if not spatial.empty and metric in spatial.columns:
                values.append(float(spatial[metric].iloc[0]))
                names.append("Spatial\n(kNN)")

            if not corr.empty and metric in corr.columns:
                # picks the row closest to threshold 0.7 as a representative mid-range similarity cut-off
                if "threshold" in corr.columns:
                    idx = (corr["threshold"] - 0.7).abs().idxmin()
                    values.append(float(corr.loc[idx, metric]))
                else:
                    values.append(float(corr[metric].iloc[0]))
                names.append("Correlation\n(threshold)")

            if values:
                colors = ["#4C72B0", "#55A868"][: len(values)]
                ax.bar(names, values, color=colors, width=0.4)
                ax.set_title(label)
                ax.set_ylabel(label)

        fig.suptitle("Spatial vs Correlation Network — Structural Properties",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "04_network_comparison.png", dpi=300)
        plt.close()
        print("  Saved: 04_network_comparison.png")

    def plot_imputation_comparison(self):
        """ Summarize task 3 distribution statistics across imputation methods.
        
        This figure reads the task 3 distribution-summary table and compares
        summary statistics across observed values, METRAQ interpolation, and the
        alternative imputation methods"""
        dist_path =_task_tables("task3_imputation") / "imputation_distribution_summary.csv"
        if not dist_path.exists():
            print("  [skip] No Task 3 imputation distribution output found.")
            return

        dist = pd.read_csv(dist_path)
        if dist.empty:
            return

        # finds method and value columns heuristically
        method_col = next(
            (c for c in dist.columns if "series" in c.lower() or "imputation" in c.lower()), None
        )
        val_col = next(
            (c for c in dist.columns if "mean" in c.lower() or "value" in c.lower()), None
        )

        if method_col is None or val_col is None:
            print("  [skip] Could not identify method/value columns in Task 3 output.")
            return

        methods = dist[method_col].unique()
        fig, ax = plt.subplots(figsize=(9, 5))

        data_to_plot = [
            dist[dist[method_col] == m][val_col].dropna().values
            for m in methods
        ]
        data_to_plot = [d for d in data_to_plot if len(d) > 0]

        if data_to_plot:
            ax.boxplot(data_to_plot, labels=methods[: len(data_to_plot)], patch_artist=True)
            ax.set_title("Imputation Method Distribution Comparison")
            ax.set_xlabel("Method")
            ax.set_ylabel("Imputed Value")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "05_imputation_comparison.png", dpi=300)
            plt.close()
            print("  Saved: 05_imputation_comparison.png")

    def plot_forecasting_summary(self):
        """Save a two-panel summary of Task 9 forecasting outputs.
        
        The left panel compares held-out test-set R² scores by model and target.
        The right panel shows the most important predictive features aggregated
        across available tree-based models and targets"""
        metrics = _safe_read(
            _task_tables("task9_forecasting") / "model_test_metrics.csv"
        )
        importances = _safe_read(
            _task_tables("task9_forecasting") / "feature_importances.csv"
        )
        # skips the forecasting summary entirely if task 9 metrics were not generated
        if metrics.empty:
            print("  [skip] No Task 9 metrics found.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # compares the model test performance using the saved task 9 evaluation table
        # R² comparison
        ax = axes[0]
        if "r2" in metrics.columns and "model" in metrics.columns and "target" in metrics.columns:
            pivot = metrics.pivot(index="target", columns="model", values="r2")
            pivot.plot(kind="bar", ax=ax, colormap="tab10", legend=True)
            ax.set_title("Model R² by Target Pollutant")
            ax.set_xlabel("Target")
            ax.set_ylabel("R²")
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
            ax.legend(fontsize=8)

        # feature importances — tree models only
        ax = axes[1]
        if not importances.empty and "importance" in importances.columns:
            tree_imp = importances[importances["type"] == "tree_importance"]
            if tree_imp.empty:
                tree_imp = importances  # fallback

            top = (
                tree_imp.groupby("feature")["importance"]
                .mean()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
            )
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in top["importance"]]
            ax.barh(top["feature"], top["importance"], color=colors)
            ax.set_title("Top Feature Importances\n(mean across models & targets)")
            ax.set_xlabel("Mean Importance")
            ax.invert_yaxis()

        fig.suptitle("Task 9 — Forecasting Model Results", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "06_forecasting_summary.png", dpi=300)
        plt.close()
        print("  Saved: 06_forecasting_summary.png")

    def plot_dashboard(self):
        """ Assemble available task 10 panels into a single summary dashboard.
        
        The dashboard acts as a presentation-ready overview of the overall project
        Only panels that were successfully generated are included """
        panel_paths = [
            FIGURES_DIR / "01_dataset_overview.png",
            FIGURES_DIR / "02_temporal_patterns.png",
            FIGURES_DIR / "03_sensor_map.png",
            FIGURES_DIR / "04_network_comparison.png",
            FIGURES_DIR / "06_forecasting_summary.png",
        ]
        
        available = [p for p in panel_paths if p.exists()]
        if not available:
            print("  [skip] No panels found for dashboard.")
            return

        n = len(available)
        ncols = 2
        nrows = (n + 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 6))
        axes = axes.flatten()

        for ax, path in zip(axes, available):
            img = plt.imread(str(path))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(path.stem.replace("_", " ").title(), fontsize=9)

        # hide unused axes
        for ax in axes[len(available):]:
            ax.axis("off")

        fig.suptitle(
            "Madrid Air Quality — Project Summary Dashboard",
            fontsize=15,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "00_summary_dashboard.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("  Saved: 00_summary_dashboard.png")

    def run(self):
        print("Plotting dataset overview…")
        self.plot_dataset_overview()

        print("Plotting temporal patterns…")
        self.plot_temporal_patterns()

        print("Plotting sensor map…")
        self.plot_sensor_map()

        print("Plotting network comparison…")
        self.plot_network_comparison()

        print("Plotting imputation comparison…")
        self.plot_imputation_comparison()

        print("Plotting forecasting summary…")
        self.plot_forecasting_summary()

        print("Plotting summary dashboard…")
        self.plot_dashboard()

        print(f"\n  All figures saved to: {FIGURES_DIR}")

@timer
def run_task10(df: pd.DataFrame | None = None):
    """ Run the full Task 10 final-visualization pipeline.

    Args:
        df: Optional cleaned DataFrame from Task 1. If None, the cleaned parquet
            dataset is loaded from disk.
    Returns:None: the task saves final summary figures to the task 10 figures folder"""
    print("\n--- Task 10: Final Visualization ---")

    viz = FinalVisualizer(df=df)
    viz.run()

    print("Task 10 completed")
