"""Task 3 pipeline for reconstructing missing values and comparing imputation methods.

This file:
- reconstructs originally missing observations from the is_interpolated flag,
- temporarily replaces those positions with NaN,
- applies several imputation strategies,
- compares the resulting values with METRAQ's provided interpolation,
- evaluates how well the imputed distributions resemble observed data,
- saves summary tables and figures for interpretation.

The comparison against METRAQ is relative rather than absolute: METRAQ's
interpolated values are not ground truth, but they provide a useful reference
for understanding how different imputation strategies behave.
"""

# used for numerical and tabular analysis
import numpy as np
import pandas as pd

# used for plotting/visualizations
import matplotlib.pyplot as plt

# project-specific output directories and timing decorator
from utils.config import get_task_dirs
from utils.helpers import timer

TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task3_imputation")

# All five methods we used and compare: 
ALL_METHODS = [
    "imputed_grouped_mean",
    "imputed_grouped_median",
    "imputed_ffill",
    "imputed_bfill",
    "imputed_last_next",
]

METHOD_LABELS = {
    "imputed_grouped_mean":   "Grouped mean",
    "imputed_grouped_median": "Grouped median",
    "imputed_ffill":          "Forward fill",
    "imputed_bfill":          "Backward fill",
    "imputed_last_next":      "Last-next avg",
}

METHOD_COLORS = {
    "imputed_grouped_mean":   "#4C72B0",
    "imputed_grouped_median": "#55A868",
    "imputed_ffill":          "#DD8452",
    "imputed_bfill":          "#C44E52",
    "imputed_last_next":      "#8172B2",
    "provided_interpolation": "#BCB8B1",
}


class ImputationAnalyzer:
    """ Reconstruct missing observations, apply imputation methods, and compare results"""

    def __init__(self, df, variable_type=None, magnitudes=None):
        """ Initialize the imputation analysis, optionally it can be performed only on subset of chosen variables

    Args:
        df: cleaned dataset produced by earlier (Task 1) pipeline stages
        variable_type: optional broad category filter: "pollutant", "weather" or "traffic"
        magnitudes: optional list of specific variable names to retain and perform analysis on"""

        self.df = self.prepare_subset(df, variable_type, magnitudes)

    def prepare_subset(self, df, variable_type=None, magnitudes=None):
        """ Create the working subset used for the imputation analysis.
            The method: 
            - creates and works on a copy of the dataset, preserving the full one
            - reconstructs the "was_missing" flag if Task 2 has not already done it
            - (optionally) restricts the analysis to selected variable types or magnitudes"""
        df = df.copy()
        
        # recreates the original-missingness indicator column if Task 2 output was not passed in
        if "was_missing" not in df.columns:
            df["was_missing"] = df["is_interpolated"].astype("boolean")
        
        # optionally restricts the analysis on variable_type
        if variable_type is not None:
            df = df[
                df["variable_type"].astype(str).str.lower()
                == variable_type.lower()
            ].copy()

        # optionally restricts the analysis on magnitude_type
        if magnitudes is not None:
            selected = {m.upper() for m in magnitudes}
            magnitude_upper = df["magnitude_name"].astype(str).str.upper()
            df = df[magnitude_upper.isin(selected)].copy()

        return df

    def reconstruct_original_missingness(self):
        """ Recreate the pre-interpolation missing-values. 
            Originally, reconstructed values are stored in the `value` column , 
            it's the `is_interpolated` column that identifies values that were originally absent
            This method:
            - preserves the provided interpolation for comparison
            - creates a version, where originally missing positions are reset to NaN before applying new imputation methods"""

        df = self.df.copy()
        df = df.sort_values(["sensor_id", "magnitude_name", "entry_date"])

        # Keeps the METRAQ interpolation for distribution comparison
        df["provided_interpolation"] = df["value"]

        # Sets the originally-missing positions to NaN so we can impute them using other methods
        df["value_original_missing"] = df["value"]
        missing_mask = df["was_missing"].fillna(False)
        df.loc[missing_mask, "value_original_missing"] = np.nan

        return df

    # Imputation methods: 

    def grouped_mean_imputation(self, df):
        """ Fill missing values using progressively broader mean estimates.

        - missing values are first filled with the mean for the same sensor-variable pair
        - if that group has no observed values, it'll start using the mean for the variable across all sensors, 
        - and goes further to the overall mean across all subset"""
        df = df.copy()
        group_cols = ["sensor_id", "magnitude_name"]

        # first preferred estimate: average observed mean value for the same variable and sensor
        sensor_variable_mean = df.groupby(group_cols, observed=True)["value_original_missing"].transform("mean")

        # next preferred (fallback 1): average for the same variable across all sensors
        variable_mean        = df.groupby("magnitude_name", observed=True)["value_original_missing"].transform("mean")
        
        # next preferred (fallback 2): overall mean across the full working subset
        global_mean          = df["value_original_missing"].mean()
        df["imputed_grouped_mean"] = (
            df["value_original_missing"]
            .fillna(sensor_variable_mean)
            .fillna(variable_mean)
            .fillna(global_mean)
        )
        return df

    def grouped_median_imputation(self, df):
        """ Fill missing values using progressively broader median estimates.
        Same logic as in the previous method with mean imputation, but median is more robust if extreme values are present: 
        - missing values are first filled with the median for the same sensor-variable pair
        - if that group has no observed values, it'll start using the median for the variable across all sensors, 
        - and goes further to the overall median across all subset"""

        df = df.copy()
        group_cols = ["sensor_id", "magnitude_name"]

        # first preferred estimate: observed median value for the same variable and sensor
        sensor_variable_median = df.groupby(group_cols, observed=True)["value_original_missing"].transform("median")

        # next preferred (fallback 1): median for the same variable across all sensors
        variable_median        = df.groupby("magnitude_name", observed=True)["value_original_missing"].transform("median")
        
        # next preferred (fallback 2): overall median across the full working subset
        global_median          = df["value_original_missing"].median()
        df["imputed_grouped_median"] = (
            df["value_original_missing"]
            .fillna(sensor_variable_median)
            .fillna(variable_median)
            .fillna(global_median)
        )
        return df

    def ffill_imputation(self, df):
        """ Forward fill: fill each missing value with the most recent
        observed value for the same sensor-variable pair.

        It preserves the immediate temporal context before the gap (good for short gaps where conditions change slowly)
        but if a gap is long, it carries the last known value forward indefinitely, increasing the uncertainty.
        Fallback case: any remaining NaN (gap at the very start of entries) is filled with the sensor-variable median.
        """
        df = df.copy()
        df = df.sort_values(["sensor_id", "magnitude_name", "entry_date"])
        group_cols = ["sensor_id", "magnitude_name"]

        df["imputed_ffill"] = (
            df.groupby(group_cols, observed=True)["value_original_missing"]
            .transform(lambda s: s.ffill())
        )

        # fills the values that cannot be recovered by forward fill, such as missing 
        # entries at the beginning of a sensor-variable time series.
        fallback = df.groupby(group_cols, observed=True)["value_original_missing"].transform("median")
        global_median = df["value_original_missing"].median()
        df["imputed_ffill"] = (
            df["imputed_ffill"]
            .fillna(fallback)
            .fillna(global_median)
        )
        return df

    def bfill_imputation(self, df):
        """ Backward fill: fill each missing value with the next observed
        value for the same sensor-variable pair.

        It mirrors ffill but looks forward: useful when the value 
        after the gap is more relevant than the value before it.
        However, same as the ffill for long gaps at the end of a series.
        Fallback case: any remaining NaN (gap at the very end of a series) is filled with the sensor-variable median, as in the previous case"""
        df = df.copy()
        df = df.sort_values(["sensor_id", "magnitude_name", "entry_date"])
        group_cols = ["sensor_id", "magnitude_name"]

        df["imputed_bfill"] = (
            df.groupby(group_cols, observed=True)["value_original_missing"]
            .transform(lambda s: s.bfill())
        )

        # fills values that cannot be recovered by backward fill, such as missing
        # entries at the end of a sensor-variable time series.
        fallback = df.groupby(group_cols, observed=True)["value_original_missing"].transform("median")
        global_median = df["value_original_missing"].median()
        df["imputed_bfill"] = (
            df["imputed_bfill"]
            .fillna(fallback)
            .fillna(global_median)
        )
        return df

    @staticmethod
    def _last_next_average(series):
        """Estimate missing values from the nearest observed values on both sides"""

        # propagates the most recent observed value forward
        previous_value = series.ffill()

        # propagates the most recent observed value backward
        next_value = series.bfill()

        # Combines (returns the average of) the backward-looking and forward-looking estimates
        # if only one side exists, that available value is returned
        return pd.concat([previous_value, next_value], axis=1).mean(axis=1)

    def last_next_imputation(self, df):
        """Impute missing values using the mean of neighboring past and future values.
        
        For each sensor-variable time series, the method averages the nearest
        observed value before a gap and the nearest observed value after it. Remaining
        unresolved values fall back to grouped medians (like in previous methods)
        Fallback case: applies  median-based fallbacks when no local temporal estimate is possible"""
    
        df = df.copy()

        # preserves the row order so the final DataFrame can be restored after chronological sorting
        df["_original_order"] = np.arange(len(df))

        # sorts for chronological order, because method requires both backward and forward look in time
        df = df.sort_values(["sensor_id", "magnitude_name", "entry_date"])
        group_cols = ["sensor_id", "magnitude_name"]
        
        # applies estimation from nearest observed values from both sides
        df["imputed_last_next"] = (
            df.groupby(group_cols, observed=True)["value_original_missing"]
            .transform(self._last_next_average)
        )
        
        # fallback: imputing median values if local temporal values are unavailable
        sensor_variable_median = df.groupby(group_cols, observed=True)["value_original_missing"].transform("median")
        variable_median        = df.groupby("magnitude_name", observed=True)["value_original_missing"].transform("median")
        global_median          = df["value_original_missing"].median()

        # fills any unresolved values using progressively broader fallbacks:
        # 1) median for the same sensor-variable pair
        # 2) median for the same variable across all sensors
        # 3) global median across the working subset/dataset
        
        df["imputed_last_next"] = (
            df["imputed_last_next"]
            .fillna(sensor_variable_median)
            .fillna(variable_median)
            .fillna(global_median)
        )
        
        # restores the original DataFrame order after the chronological computation
        df = df.sort_values("_original_order").drop(columns=["_original_order"])

        return df

    # Comparing the different methods 

    def compare_methods(self, df):
        """ Build a comparison table restricted to originally-missing rows"""
        
        # setting up the boolean mask to compare the methods used 
        # against to rows where imputation was actually used, restricting the comparison table to missing rows only
        missing_mask = df["was_missing"].fillna(False)
        
        # list of columns for comparison table
        cols = (
            ["sensor_id", "sensor_name", "magnitude_name", "variable_type",
             "entry_date", "provided_interpolation", "was_missing", "is_interpolated"]
            + ALL_METHODS
        )
        available_cols = [c for c in cols if c in df.columns]
        comparison = df.loc[missing_mask, available_cols].copy()

        for method in ALL_METHODS:
            if method not in comparison.columns:
                continue
            # computes errors
            comparison[f"error_{method}"]         = comparison[method] - comparison["provided_interpolation"]
            comparison[f"abs_error_{method}"]     = comparison[f"error_{method}"].abs()
            comparison[f"squared_error_{method}"] = comparison[f"error_{method}"] ** 2

        return comparison

    def error_summary(self, comparison):
        """ Relative error vs METRAQ interpolation.
        Lower = closer to METRAQ"""

        rows = []
        for method in ALL_METHODS:
            if f"abs_error_{method}" not in comparison.columns:
                continue
            error         = comparison[f"error_{method}"].dropna()
            abs_error     = comparison[f"abs_error_{method}"].dropna()
            squared_error = comparison[f"squared_error_{method}"].dropna()
            rows.append({
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "n_compared_rows": len(error),
                "mean_error": error.mean(),
                "median_error": error.median(),
                "mae_vs_metraq": abs_error.mean(),
                "median_absolute_error": abs_error.median(),
                "rmse_vs_metraq": np.sqrt(squared_error.mean()),
                "max_absolute_error": abs_error.max(),
                "note": "deviation from METRAQ interpolation, NOT absolute accuracy",
            })
        return pd.DataFrame(rows).sort_values("mae_vs_metraq")

    def error_summary_by_variable_type(self, comparison):
        """ Summarize relative deviation from METRAQ separately by variable category"""

        rows = []
        for variable_type, group in comparison.groupby("variable_type", observed=True):
            for method in ALL_METHODS:
                if f"abs_error_{method}" not in group.columns:
                    continue
                error         = group[f"error_{method}"].dropna()
                abs_error     = group[f"abs_error_{method}"].dropna()
                squared_error = group[f"squared_error_{method}"].dropna()
                if len(error) == 0:
                    continue
                rows.append({
                    "variable_type": variable_type,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "n_compared_rows": len(error),
                    "mean_error": error.mean(),
                    "mae_vs_metraq": abs_error.mean(),
                    "median_absolute_error": abs_error.median(),
                    "rmse_vs_metraq": np.sqrt(squared_error.mean()),
                })
        return pd.DataFrame(rows).sort_values(["variable_type", "mae_vs_metraq"])

    def error_summary_by_magnitude(self, comparison):
        """ Summarize relative deviation from METRAQ for each measured variable"""
        rows = []
        for magnitude, group in comparison.groupby("magnitude_name", observed=True):
            for method in ALL_METHODS:
                if f"abs_error_{method}" not in group.columns:
                    continue
                error         = group[f"error_{method}"].dropna()
                abs_error     = group[f"abs_error_{method}"].dropna()
                squared_error = group[f"squared_error_{method}"].dropna()
                if len(error) == 0:
                    continue
                rows.append({
                    "magnitude_name": magnitude,
                    "variable_type": group["variable_type"].iloc[0],
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "n_compared_rows": len(error),
                    "mean_error": error.mean(),
                    "mae_vs_metraq": abs_error.mean(),
                    "median_absolute_error": abs_error.median(),
                    "rmse_vs_metraq": np.sqrt(squared_error.mean()),
                })
        return pd.DataFrame(rows).sort_values(["variable_type", "magnitude_name", "mae_vs_metraq"])

    def distribution_summary(self, df, comparison):
        """ Summarize statistical value distributions for observed and imputed series.
        This is used to judge whether each method produces
        values with a plausible statistical shape relative to real observed data
        and METRAQ's interpolation"""

        real_observed = df.loc[~df["was_missing"].fillna(False), "value_original_missing"]

        series_dict = {"real_observed_values": real_observed,
                       "provided_interpolation": comparison["provided_interpolation"]}
        for m in ALL_METHODS:
            if m in comparison.columns:
                series_dict[METHOD_LABELS[m]] = comparison[m]

        rows = []
        for name, series in series_dict.items():
            s = series.dropna()
            if len(s) == 0:
                continue
            q = s.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            rows.append({
                "series": name,
                "n": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q01": q[0.01], "q05": q[0.05], "q25": q[0.25],
                "median": q[0.50],
                "q75": q[0.75], "q95": q[0.95], "q99": q[0.99],
                "max": s.max(),
            })
        return pd.DataFrame(rows)


    # FIGURES

    def _plot_method_comparison_panel(self, comparison, summary):
        """ Plot a Method comparison dashboard (2x2 panel).
        Describes the agreement with METRAQ's interpolation.

        - top-left:  MAE vs METRAQ per method (relative, not absolute accuracy)
        - top-right: Bias (mean signed error) per method — shows systematic
           over- or under-estimation vs METRAQ
        - bottom-left:  Distribution of errors for each method (violin / box)
        - bottom-right: MAE breakdown by variable type"""

        if comparison.empty:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Imputation Method Comparison\n"
            "(errors measured vs METRAQ spatial interpolation — lower ≠ more accurate, "
            "just closer to their approach)",
            fontsize=11, fontweight="medium",
        )

        # keeps only methods whose comparison columns were successfully created
        available_methods = [m for m in ALL_METHODS if f"abs_error_{m}" in comparison.columns]
        labels  = [METHOD_LABELS[m] for m in available_methods]

        # the summary table is already sorted by MAE, so the first method is the 
        # one closest to METRAQ under this relative metric
        best_method = summary.iloc[0]["method"] if not summary.empty else None
        colors = ["#55A868" if m == best_method else "#BBBBBB" for m in available_methods]

        # top-left: MAE bar chart 
        ax = axes[0, 0]
        mae_vals = [comparison[f"abs_error_{m}"].mean() for m in available_methods]
        bars = ax.barh(labels, mae_vals, color=colors, alpha=0.82)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xlabel("MAE vs METRAQ interpolation", fontsize=9)
        ax.set_title("Mean absolute deviation from METRAQ", fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

        # top-right: bias (mean signed error)
        ax = axes[0, 1]
        bias_vals = [comparison[f"error_{m}"].mean() for m in available_methods]
        bias_colors = ["#55A868" if m == best_method else "#BBBBBB" for m in available_methods]
        bars2 = ax.barh(labels, bias_vals, color=bias_colors, alpha=0.82)
        ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean signed error vs METRAQ  (+ = over-estimates)", fontsize=9)
        ax.set_title("Bias relative to METRAQ", fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

        # bottom-left: error distribution boxplot
        ax = axes[1, 0]
        sample = comparison.sample(n=min(50_000, len(comparison)), random_state=42)
        error_data = [
            sample[f"abs_error_{m}"].dropna().values
            for m in available_methods
            if f"abs_error_{m}" in sample.columns
        ]
        bp = ax.boxplot(
            error_data,
            labels=labels,
            patch_artist=True,
            showfliers=False,
            vert=False,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xlabel("Absolute error vs METRAQ", fontsize=9)
        ax.set_title("Error distribution (no outliers shown)", fontsize=10)
        ax.grid(axis="x", alpha=0.25)

        # bottom-right: MAE by variable type
        ax = axes[1, 1]
        if "variable_type" in comparison.columns:
            vtypes = comparison["variable_type"].dropna().unique()
            x = np.arange(len(vtypes))
            width = 0.8 / max(len(available_methods), 1)
            for i, method in enumerate(available_methods):
                col = f"abs_error_{method}"
                if col not in comparison.columns:
                    continue
                mae_by_type = [
                    comparison.loc[comparison["variable_type"] == vt, col].mean()
                    for vt in vtypes
                ]
                ax.bar(x + i * width, mae_by_type, width,
                       label=METHOD_LABELS[method], color="#55A868" if method == best_method else "#BBBBBB",
                       alpha=0.82)
            ax.set_xticks(x + width * (len(available_methods) - 1) / 2)
            ax.set_xticklabels(vtypes, rotation=15)
            ax.set_ylabel("MAE vs METRAQ", fontsize=9)
            ax.set_title("MAE by variable type", fontsize=10)
            ax.legend(fontsize=7, ncol=2)
            ax.grid(axis="y", alpha=0.25)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "imputation_method_comparison_panel.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: imputation_method_comparison_panel.png")

    def _plot_distribution_overlay(self, df, comparison):
        """ Plot a Distribution of observed and imputed values for selected magnitudes
        
        For each selected magnitude, the figure overlays density curves for:
        - real observed values
        - METRAQ's provided interpolation
        - each alternative imputation method
        
        It should assess whether imputed values preserve a plausible
        distributional shape"""
        if comparison.empty:
            return

        magnitudes = comparison["magnitude_name"].dropna().unique()
        
        # restricting for the main pollutants
        FOCUS_POLLUTANTS = {"NO2", "NOX", "NO", "O3", "CO", "SO2", "<PM10", "<PM2.5"}
        magnitudes = [m for m in magnitudes if str(m).upper() in FOCUS_POLLUTANTS]

        for magnitude in magnitudes:
            sub = comparison[comparison["magnitude_name"] == magnitude]

            # skips pollutants with too few comparison rows for a meaningful KDE
            if sub.empty or len(sub) < 50:
                continue
            
            # extracts the real observed values for the same pollutant
            # these should provide the empirical reference distribution
            real_sub = df.loc[
                (~df["was_missing"].fillna(False)) &
                (df["magnitude_name"] == magnitude),
                "value_original_missing",
            ].dropna()

            # uses a readable, filesystem-safe variable name when saving the figure
            safe_mag = (
                str(magnitude)
                .replace("/", "_").replace("<", "").replace(">", "").replace(" ", "_")
            )

            # samples the comparison rows to keep KDE plotting manageable on large data
            # while retaining a reproducible approximation of the distribution
            sample = sub.sample(n=min(30_000, len(sub)), random_state=42)

            fig, ax = plt.subplots(figsize=(10, 5))

            def safe_kde(series, label, color, linewidth=1.5, linestyle="-", alpha=0.85):
                """ Plot a KDE only when the input series is suitable
                - KDE estimation fails for small/constant series
                - This helper removes missing values, checks that enough variation
                exists, and skips problematic cases instead of interrupting the task"""
                vals = series.dropna()

                # KDE requires both enough observations and at least some variation
                if len(vals) < 10 or vals.nunique() < 2:
                    return

                try:
                    vals.plot.kde(
                        ax=ax,
                        color=color,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        alpha=alpha,
                        label=label,
                    )
                except Exception as exc:
                    print(
                        f"  Skipped KDE for {label} in {magnitude} "
                        f" because the data was constant/singular: {exc}"
                    )

            # plots the real observed values in thick black line as reference
            safe_kde(
                real_sub,
                label="Real observed (non-interpolated)",
                color="black",
                linewidth=2.2,
            )

            # plots the METRAQ imputed values in thick grey color 
            if "provided_interpolation" in sample.columns:
                prov = sample["provided_interpolation"].dropna()
                safe_kde(
                    prov,
                    label="METRAQ interpolation",
                    color="#BCB8B1",
                    linewidth=2.0,
                    linestyle="--",
                )

            # methods used by us
            for method in ALL_METHODS:
                if method not in sample.columns:
                    continue
                vals = sample[method].dropna()
                safe_kde(
                    vals,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                )

            ax.set_xlabel(f"{magnitude} value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(
                f"Imputed Value Distribution: {magnitude}\n"
                "A good method matches the real observed distribution (black line)",
                fontsize=10,
            )
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

            # restricts the horizontal range to the central 98% of observed values
            # => prevents extreme outliers from compressing the main density shape
            if len(real_sub) > 10:
                lo, hi = real_sub.quantile(0.01), real_sub.quantile(0.99)
                ax.set_xlim(lo - abs(lo) * 0.1, hi + abs(hi) * 0.1)
            
            # creates output file name based on the var.type
            plt.tight_layout()
            vtype = sub["variable_type"].iloc[0] if "variable_type" in sub.columns else "unknown"
            plt.savefig(
                FIGURES_DIR / f"distribution_overlay_{vtype}_{safe_mag}.png",
                dpi=300, bbox_inches="tight",
            )
            plt.close()

    def _plot_method_ranking_summary(self, summary):
        """ Plot an Overall method ranking
        
        Methods are ordered by MAE relative to METRAQ's provided interpolation.
        This is a relative agreement metric only: lower values mean closer behavior
        to METRAQ"""

        if summary.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        
        # highlights the first row 
        colors = ["#55A868" if i == 0 else "#BBBBBB" for i in range(len(summary))]
        bars = ax.barh(
            summary["method_label"],
            summary["mae_vs_metraq"],
            color=colors,
            alpha=0.85,
        )
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("MAE vs METRAQ spatial interpolation", fontsize=10)
        ax.set_title(
            "Imputation Method Ranking — Deviation from METRAQ\n"
            "Lower = closer to METRAQ's approach",
            fontsize=10,
        )
        ax.grid(axis="x", alpha=0.25)


        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "imputation_method_ranking.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: imputation_method_ranking.png")

    def save_plots(self, df, comparison, summary):
        """ Generate the main Task 3 figures from the imputation comparison outputs"""
        
        # if no originally missing rows were found, there is nothing meaningful to plot
        if comparison.empty:
            print("No originally missing rows found. No Task 3 plots created.")
            return

        self._plot_method_comparison_panel(comparison, summary)
        self._plot_method_ranking_summary(summary)
        self._plot_distribution_overlay(df, comparison)

    # MAIN RUN

    def run(self):
        """Execute the full Task 3 imputation and evaluation workflow"""
        
        # restores the original missing-value pattern before testing our own methods
        print("Reconstructing original missingness...")
        reconstructed = self.reconstruct_original_missingness()

        # applies each candidate imputation method and store its output in a separate column
        print("Applying grouped mean imputation...")
        imputed = self.grouped_mean_imputation(reconstructed)
        
        print("Applying grouped median imputation...")
        imputed = self.grouped_median_imputation(imputed)

        print("Applying forward fill imputation...")
        imputed = self.ffill_imputation(imputed)

        print("Applying backward fill imputation...")
        imputed = self.bfill_imputation(imputed)

        print("Applying last-next average imputation...")
        imputed = self.last_next_imputation(imputed)

        print("Comparing imputation methods...")
        comparison = self.compare_methods(imputed)

        print("Creating error summaries...")
        summary             = self.error_summary(comparison)
        summary_by_type     = self.error_summary_by_variable_type(comparison)
        summary_by_magnitude= self.error_summary_by_magnitude(comparison)
        dist_summary        = self.distribution_summary(imputed, comparison)

        print("Saving Task 3 tables...")
        comparison.head(100_000).to_csv(
            TABLES_DIR / "imputation_comparison_sample.csv", index=False)
        summary.to_csv(
            TABLES_DIR / "imputation_error_summary.csv", index=False)
        summary_by_type.to_csv(
            TABLES_DIR / "imputation_error_summary_by_variable_type.csv", index=False)
        summary_by_magnitude.to_csv(
            TABLES_DIR / "imputation_error_summary_by_magnitude.csv", index=False)
        dist_summary.to_csv(
            TABLES_DIR / "imputation_distribution_summary.csv", index=False)

        print("Saving Task 3 figures...")
        self.save_plots(imputed, comparison, summary)

        self._print_summary(summary)

        return imputed, comparison, summary

    @staticmethod
    def _print_summary(summary):
        """Print a compact console summary of the Task 3 results"""

        print("TASK 3 — Imputation summary")
        print("\nMethod ranking (by deviation from METRAQ):")
        print(summary[["method_label", "mae_vs_metraq", "rmse_vs_metraq",
                        "median_error"]].to_string(index=False))
        print(
            "\nNote: errors are relative to METRAQ's spatial interpolation,\n"
            "Primary evaluation is\n"
            "distributional — see distribution_overlay figures.\n"
        )
        print("Method highlighted for interpolation: last-next average")
        print("  → Uses both the last observed and next observed value,\n"
              "    preserving local temporal context from both directions.")
        print("  → Its performance should be interpreted together with the\n"
              "    saved error summaries and distribution-overlay figures.")

@timer
def run_task3(df, variable_type=None, magnitudes=None):
    """ Run the Task 3 imputation pipeline on the cleaned dataset.
    
    Args:
        df: cleaned DataFrame from Task 1/2
        variable_type: optional filter to restrict analysis to one variable category
        magnitudes: optional list of specific variables to analyze
    
    Returns the imputed DataFrame with all method columns added"""
    print("\n--- Task 3: Imputation ---")

    analyzer = ImputationAnalyzer(
        df,
        variable_type=variable_type,
        magnitudes=magnitudes,
    )

    imputed_df, comparison, summary = analyzer.run()

    print("Task 3 completed")
    print(summary[["method_label", "mae_vs_metraq"]].to_string(index=False))

    return imputed_df
