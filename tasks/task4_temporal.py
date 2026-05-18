"""Task 4 pipeline for temporal analysis of Madrid air quality data

This file:
- aggregates pollutant measurements by sensor, hour, weekday, month, and year,
- detects daily, weekly, and seasonal pollution cycles,
- computes Mann-Kendall trend tests to identify long-term improvements or deterioration,
- highlights the NO2/O3 anti-correlation and the COVID-19 lockdown as a natural experiment,
- saves summary tables and figures for interpretation.
"""
# used for numerical and tabular analysis
import pandas as pd
import numpy as np

# used for plotting/visualizations
import matplotlib.pyplot as plt

# project-specific output directories and timing decorator
from utils.config import get_task_dirs
from utils.helpers import timer

# output folders dedicated to Task 4 temporal-analysis tables and figures
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task4_temporal")

class TemporalAnalyzer:
    """Analyze pollutant trends, cycles, and seasonality over time"""
    def __init__(self, df, magnitudes=None, top_n_pollutants=3, include_interpolated=True):
        """ Task 4: Temporal analysis.
        Args:
            df : cleaned dataframe from previous tasks
            magnitudes : optional list of pollutants to analyze, e.g. ["NOX", "NO2", "O3"]; if None, the code selects the most frequent pollutants.
            top_n_pollutants : (int) number of pollutants to select automatically if magnitudes=None.
            include_interpolated : (bool) if True, use all values, including dataset-provided interpolation; if False, use only originally observed values"""
        
        self.include_interpolated = include_interpolated
        self.value_col = "value"
        self.df = self.prepare_subset(df, magnitudes, top_n_pollutants)

    def prepare_subset(self, df, magnitudes=None, top_n_pollutants=3):
        """ Select pollutant variables for temporal analysis.
        Task 4 focuses on pollution cycles, so we analyze pollutants """

        # keeps only columns needed for the temporal summaries and plots
        required_cols = [
            "entry_date",
            "sensor_id",
            "sensor_name",
            "magnitude_name",
            "variable_type",
            "value",
            "value_imputed",
            "year",
            "month",
            "hour",
            "weekday",
            "is_interpolated",
        ]

        available_cols = [c for c in required_cols if c in df.columns]

        temp = df[available_cols].copy()

        # task 4 focuses on pollution patterns, so weather and traffic variables are excluded
        temp = temp[temp["variable_type"].astype(str).str.lower() == "pollutant"]
        
        #optionally removes METRAQ-interpolated observations to analyze only originally observed values
        if not self.include_interpolated:
            temp = temp[temp["is_interpolated"].fillna(False) == False]
        
        # if pollutants are not explicitly provided, selects the most represented ones
        if magnitudes is None:
            selected = (
                temp["magnitude_name"]
                .value_counts()
                .head(top_n_pollutants)
                .index
                .tolist()
            )
        else:
            selected_upper = {m.upper() for m in magnitudes}
            magnitude_upper = temp["magnitude_name"].astype(str).str.upper()
            selected = temp.loc[magnitude_upper.isin(selected_upper), "magnitude_name"].unique().tolist()

        temp = temp[temp["magnitude_name"].isin(selected)].copy()
        
        # saves the pollutant-selection decision for reproducibility and reporting
        selected_table = pd.DataFrame({
            "selected_magnitude": selected,
            "selection_reason": [
                "user_selected" if magnitudes is not None else "top_pollutant_by_row_count"
                for _ in selected
            ],
        })

        selected_table.to_csv(
            TABLES_DIR / "selected_pollutants_for_temporal_analysis.csv",
            index=False,
        )

        print(f"Selected pollutants for Task 4: {selected}")

        return temp

    def monthly_by_sensor(self):
        """Aggregate pollutant values monthly for each sensor-station pair.
        
        This provides the station-level time series used for trend analysis,
        cross-station comparisons, and later summary plots"""

        monthly = (
            self.df
            .groupby(
                [
                    "magnitude_name",
                    "sensor_id",
                    "sensor_name",
                    pd.Grouper(key="entry_date", freq="M"),
                ],
                observed=True,
            )[self.value_col]
            .agg(
                mean_value="mean",
                median_value="median",
                std_value="std",
                n_observations="count",
            )
            .reset_index()
            .rename(columns={"entry_date": "month"})
        )

        return monthly

    def monthly_overall(self, monthly):
        """Aggregate monthly station-level values into city-wide pollutant series"""

        return (
            monthly
            .groupby(["magnitude_name", "month"], observed=True)["mean_value"]
            .agg(
                mean_across_sensors="mean",
                median_across_sensors="median",
                std_across_sensors="std",
                n_sensors="count",
            )
            .reset_index()
        )

    def hourly_cycle(self):
        """ Average value by hour of day. This detects daily cycles, e.g. morning/evening traffic peaks"""
        return (
            self.df
            .groupby(["magnitude_name", "sensor_id", "sensor_name", "hour"], observed=True)[self.value_col]
            .agg(
                mean_value="mean",
                median_value="median",
                n_observations="count",
            )
            .reset_index()
        )

    def weekday_cycle(self):
        """ Average value by day of week; Monday = 0, Sunday = 6"""
        return (
            self.df
            .groupby(["magnitude_name", "sensor_id", "sensor_name", "weekday"], observed=True)[self.value_col]
            .agg(
                mean_value="mean",
                median_value="median",
                n_observations="count",
            )
            .reset_index()
        )

    def seasonal_month_cycle(self):
        """ Average value by calendar month; detects seasonal cycles across the year"""
        return (
            self.df
            .groupby(["magnitude_name", "sensor_id", "sensor_name", "month"], observed=True)[self.value_col]
            .agg(
                mean_value="mean",
                median_value="median",
                n_observations="count",
            )
            .reset_index()
        )

    def yearly_trends(self):
        """Aggregate pollutant values yearly for each sensor-station pair"""

        return (
            self.df
            .groupby(["magnitude_name", "sensor_id", "sensor_name", "year"], observed=True)[self.value_col]
            .agg(
                mean_value="mean",
                median_value="median",
                n_observations="count",
            )
            .reset_index()
        )

    @staticmethod
    def _trend_slope(group):
        """Estimate a simple linear monthly trend slope for one sensor series.

        The slope is computed from monthly mean values ordered over time.
        Positive values indicate increasing pollutant levels, while negative
        values indicate decreasing levels"""

        # ensures that the numeric trend is fitted in chronological month order
        group = group.sort_values("month")

        # at least two monthly observations are required to fit a line
        if len(group) < 2:
            return np.nan

        x = np.arange(len(group))
        y = group["mean_value"].to_numpy()

        if np.all(pd.isna(y)):
            return np.nan
        # fits a first-degree polynomial; the first coefficient is the monthly slope
        return np.polyfit(x, y, 1)[0]

    def station_trend_summary(self, monthly):
        """Compare pollutant trend slopes across monitoring stations.
        
        This helps to determine whether long-term trends are broadly consistent across the city or vary by station"""  

        # compute one monthly trend slope for each pollutant-station series
        slopes = (
            monthly
            .dropna(subset=["mean_value"])
            .groupby(["magnitude_name", "sensor_id", "sensor_name"], observed=True)
            .apply(self._trend_slope)
            .reset_index(name="monthly_trend_slope")
        )
        
        # summarize how stable those station-level slopes are per pollutant
        summary = (
            slopes
            .groupby("magnitude_name", observed=True)["monthly_trend_slope"]
            .agg(
                mean_station_slope="mean",
                median_station_slope="median",
                std_station_slope="std",
                min_station_slope="min",
                max_station_slope="max",
                n_sensors="count",
            )
            .reset_index()
        )

        return slopes, summary

    def mann_kendall_trend(self, monthly_overall):
        """Estimate monotonic long-term trends with the Mann-Kendall test.
        
        Mann-Kendall test is applied to each pollutant's city-wide monthly time series.
        It detects whether values tend to increase/decrease consistently over time without assuming normality of the observations.
        
        Outputs include:
        - Mann-Kendall S statistic,
        - Kendall-like Tau direction measure,
        - Z statistic and approximate two-sided p-value,
        - trend direction label,
        - Theil-Sen median slope per month
        The test statistic is computed directly and SciPy is used only for the normal-CDF p-value"""

        from scipy import stats as scipy_stats

        rows = []
        # works pollutant by pollutant using the chronologically ordered monthly city-wide series
        for pollutant, group in monthly_overall.groupby("magnitude_name", observed=True):
            series = (
                group.sort_values("month")["mean_across_sensors"]
                .dropna()
                .values
            )
            # skips relatively short series because trend estimates would be unstable
            if len(series) < 8:
                continue

            n = len(series)
            # computes the Mann-Kendall S statistic from all pairwise temporal comparisons
            s = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    diff = series[j] - series[i]
                    if diff > 0:
                        s += 1
                    elif diff < 0:
                        s -= 1

            var_s = n * (n - 1) * (2 * n + 5) / 18
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0.0
            # converts S into a standardized Z statistic and two-sided approximate p-value
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            tau = s / (0.5 * n * (n - 1))

            # Theil-Sen slope: median pairwise slope, robust to outliers
            slopes = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    slopes.append((series[j] - series[i]) / (j - i))
            theil_sen_slope = float(np.median(slopes))

            rows.append({
                "magnitude_name": pollutant,
                "mk_s": int(s),
                "mk_tau": round(tau, 4),
                "mk_z": round(z, 4),
                "mk_p_value": round(p_value, 4),
                "mk_significant": p_value < 0.05,
                "trend_direction": "decreasing" if tau < 0 else ("increasing" if tau > 0 else "no trend"),
                "theil_sen_slope_per_month": round(theil_sen_slope, 6),
                "n_months": n,
            })

        return pd.DataFrame(rows).sort_values("mk_tau")

    def cycle_strength_summary(self, hourly, seasonal):
        """ Measure the amplitude of daily and seasonal pollutant cycles.
        
        Daily cycle strength is the difference between the highest and lowest
        average hourly values. Seasonal cycle strength is the difference between
        the highest and lowest average calendar-month values"""
        
        # summarizes seasonal patterns across all stations into one overall cycle per pollutant
        hourly_overall = (
            hourly
            .groupby(["magnitude_name", "hour"], observed=True)["mean_value"]
            .mean()
            .reset_index()
        )
        
        # summarizes station-level seasonal patterns into one calendar-month profile per pollutant
        seasonal_overall = (
            seasonal
            .groupby(["magnitude_name", "month"], observed=True)["mean_value"]
            .mean()
            .reset_index()
        )
        
        hourly_strength = (
            hourly_overall
            .groupby("magnitude_name", observed=True)["mean_value"]
            .agg(hourly_min="min", hourly_max="max")
            .reset_index()
        )

        # cycle amplitude measures how much the average value changes across the cycle
        hourly_strength["hourly_amplitude"] = (
            hourly_strength["hourly_max"] - hourly_strength["hourly_min"]
        )

        seasonal_strength = (
            seasonal_overall
            .groupby("magnitude_name", observed=True)["mean_value"]
            .agg(seasonal_min="min", seasonal_max="max")
            .reset_index()
        )

        seasonal_strength["seasonal_amplitude"] = (
            seasonal_strength["seasonal_max"] - seasonal_strength["seasonal_min"]
        )

        return hourly_strength.merge(seasonal_strength, on="magnitude_name", how="outer")

    def autocorrelation_at_lag12(self, monthly_overall):
        """Measure annual seasonality using monthly autocorrelation.
        
        Lag-12 autocorrelation compares each month with the same month one year earlier: higher positive values means stronger recurring yearly patterns.
        Lag-6 and lag-1 are also reported for additional context""" 

        rows = []

        for pollutant, group in monthly_overall.groupby("magnitude_name", observed=True):
            series = (
                group.sort_values("month")["mean_across_sensors"]
                .dropna()
                .reset_index(drop=True)
            )
            # at least 13 months are needed to compute a lag-12 autocorrelation
            if len(series) < 13:
                continue

            rows.append({
                "magnitude_name": pollutant,
                "acf_lag12": series.autocorr(lag=12),  # yearly cycle
                "acf_lag6":  series.autocorr(lag=6),   # 6-month cycle
                "acf_lag1":  series.autocorr(lag=1),   # month-to-month persistence
                "n_months":  len(series),
            })

        return pd.DataFrame(rows).sort_values("acf_lag12", ascending=False)

    def station_year_heatmap(self, monthly_by_sensor):
        """Plot a station-by-year heatmap for the first selected pollutant.
        
        The figure focuses on the 10 stations with the highest average levels
        of that pollutant and shows how annual concentrations vary across stations; columns are years """

        # uses the first selected pollutant as a compact station-comparison example
        first_magnitude = self.df["magnitude_name"].unique()[0]

        station_subset = monthly_by_sensor[
            monthly_by_sensor["magnitude_name"] == first_magnitude
        ].copy()
        
        # focuses on the ten stations with the highest average chosen pollutant level
        top_sensors = (
            station_subset.groupby("sensor_name", observed=True)["mean_value"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        station_subset = station_subset[station_subset["sensor_name"].isin(top_sensors)]
        station_subset["year"] = station_subset["month"].dt.year
        
        #creates the station and year matrix for heatmap plot
        heatmap_data = station_subset.pivot_table(
            index="sensor_name",
            columns="year",
            values="mean_value",
            aggfunc="mean"
        )

        # sort the stations by overall average pollution
        heatmap_data = heatmap_data.loc[
            heatmap_data.mean(axis=1).sort_values(ascending=False).index
        ]

        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap_data, aspect="auto")
        plt.colorbar(label=f"Average {first_magnitude}")
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)

        plt.title(f"Station-Year Heatmap for {first_magnitude}")
        plt.xlabel("Year")
        plt.ylabel("Station")

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"station_year_heatmap_{first_magnitude}.png",
            dpi=300
        )
        plt.close()

    def plot_no2_o3_anticorrelation(self, monthly_overall):
        """ Compare NO2 and O3 seasonal cycles on a dual-axis chart.
        
        The figure highlights their contrasting seasonal profiles and reports the
        Pearson correlation between the two 12-month seasonal vectors.
        This function intends to show the interpretive visualization of their opposite seasonal behavior.
        The figure is skipped if pollutants are unavailable"""

        upper = monthly_overall["magnitude_name"].astype(str).str.upper()
        no2_data = monthly_overall[upper == "NO2"].copy()
        o3_data  = monthly_overall[upper == "O3"].copy()

        if no2_data.empty or o3_data.empty:
            print("  NO2/O3 anti-correlation plot skipped (one or both pollutants absent).")
            return

        # computes calendar-month averages for each pollutant (Jan=1 … Dec=12)
        no2_data["cal_month"] = pd.to_datetime(no2_data["month"]).dt.month
        o3_data["cal_month"]  = pd.to_datetime(o3_data["month"]).dt.month

        no2_seasonal = no2_data.groupby("cal_month")["mean_across_sensors"].mean()
        o3_seasonal  = o3_data.groupby("cal_month")["mean_across_sensors"].mean()

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        color_no2 = "#D62728"
        color_o3  = "#1F77B4"

        ax1.plot(no2_seasonal.index, no2_seasonal.values,
                 color=color_no2, marker="o", linewidth=2.2, label="NO₂")
        ax1.fill_between(no2_seasonal.index, no2_seasonal.values,
                         alpha=0.12, color=color_no2)
        ax1.set_xlabel("Month")
        ax1.set_ylabel("NO₂  (µg/m³)", color=color_no2)
        ax1.tick_params(axis="y", labelcolor=color_no2)

        ax2.plot(o3_seasonal.index, o3_seasonal.values,
                 color=color_o3, marker="s", linewidth=2.2, linestyle="--", label="O₃")
        ax2.fill_between(o3_seasonal.index, o3_seasonal.values,
                         alpha=0.10, color=color_o3)
        ax2.set_ylabel("O₃  (µg/m³)", color=color_o3)
        ax2.tick_params(axis="y", labelcolor=color_o3)

        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(month_labels)

        # computes the Pearson correlation between the two seasonal vectors, if enough shared months exist
        common_idx = no2_seasonal.index.intersection(o3_seasonal.index)
        if len(common_idx) >= 3:
            r = np.corrcoef(no2_seasonal[common_idx].values,
                            o3_seasonal[common_idx].values)[0, 1]
            ax1.set_title(
                f"NO₂ vs O₃ Seasonal Cycle — Madrid\n"
                f"Pearson r = {r:.2f}  |  Winter NO₂ peak vs Summer O₃ peak",
                fontsize=11,
            )
        else:
            ax1.set_title("NO₂ vs O₃ Seasonal Cycle — Madrid")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.annotate(
            "Winter: low UV → slower\nO₃ production, NO₂ builds up",
            xy=(1, no2_seasonal.get(1, no2_seasonal.iloc[0])),
            xytext=(2.5, no2_seasonal.max() * 0.88),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            fontsize=8, color="gray",
        )
        ax1.annotate(
            "Summer: high UV →\nphotochemical O₃ formation",
            xy=(7, no2_seasonal.get(7, no2_seasonal.iloc[6])),
            xytext=(8, no2_seasonal.min() * 1.15),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            fontsize=8, color="gray",
        )

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "no2_o3_anticorrelation_seasonal.png", dpi=300,
                    bbox_inches="tight")
        plt.close()
        print("  Saved: no2_o3_anticorrelation_seasonal.png")

    def plot_longterm_trend_with_covid(self, monthly_overall, mk_df):
        """ Plot yearly pollutant trends with trend-test and COVID-period annotations.
        
        For each selected pollutant:
        - converts monthly city-wide values into yearly averages,
        - shades the approximate COVID-era time interval shown in the chart,
        - adds Mann-Kendall direction and Theil-Sen slope information to the title,
        - annotates the change between pre-lockdown and lockdown-period averages when those periods are available """ 

        selected = self.df["magnitude_name"].unique().tolist()
        n = len(selected)
        if n == 0:
            return

        fig, axes = plt.subplots(n, 1, figsize=(13, 3.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        covid_start = pd.Timestamp("2020-03-01")
        covid_end   = pd.Timestamp("2021-06-30")

        for ax, mag in zip(axes, selected):
            sub = monthly_overall[
                monthly_overall["magnitude_name"] == mag
            ].copy()
            sub["month"] = pd.to_datetime(sub["month"])
            sub["year"] = sub["month"].dt.year
            # converts monthly values to yearly means to produce a cleaner long-run trend line
            sub = sub.groupby("year")["mean_across_sensors"].mean().reset_index()

            ax.plot(sub["year"], sub["mean_across_sensors"],
                color="#2C5F8A", linewidth=2.0, marker="o", markersize=4, label="Yearly mean")

            # shades the period used as the COVID-era visual reference in this figure
            ymin, ymax = ax.get_ylim()
            ax.axvspan(2020, 2022, color="#F5A623", alpha=0.15, label="COVID lockdown period")

            # Mann-Kendall annotation
            mk_row = mk_df[mk_df["magnitude_name"] == mag]
            if not mk_row.empty:
                r = mk_row.iloc[0]
                direction = r["trend_direction"]
                tau       = r["mk_tau"]
                p         = r["mk_p_value"]
                sig_star  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                slope_str = f"{r['theil_sen_slope_per_month']:+.3f} µg/m³/month"
                title_str = (
                    f"{mag} — Mann-Kendall: {direction} "
                    f"(τ={tau:.3f}{sig_star}, p={p:.3f})  |  "
                    f"Theil-Sen slope: {slope_str}"
                )
            else:
                title_str = mag

            ax.set_title(title_str, fontsize=10)
            ax.set_ylabel("Concentration (µg/m³)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.2)

            # annotates COVID drop if we have data around that period
            covid_pre  = sub[sub["year"].between(2017, 2019)]
            covid_lock = sub[sub["year"].between(2020, 2021)]
            if not covid_pre.empty and not covid_lock.empty:
                pre_mean  = covid_pre["mean_across_sensors"].mean()
                lock_mean = covid_lock["mean_across_sensors"].mean()
                pct_change = (lock_mean - pre_mean) / pre_mean * 100
                ax.annotate(
                    f"Lockdown:\n{pct_change:+.1f}%",
                    xy=(2020, lock_mean),
                    xytext=(2016, pre_mean * 1.05),
                    arrowprops=dict(arrowstyle="->", color="#B05E00", lw=0.9),
                    fontsize=8, color="#B05E00",
                )

        axes[-1].set_xlabel("Date")
        fig.suptitle(
            "Long-term Pollution Trends — Madrid  (shaded = COVID-19 lockdown period)",
            fontsize=13, fontweight="medium",
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "longterm_trend_covid_annotated.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
        print("  Saved: longterm_trend_covid_annotated.png")

    def plot_mann_kendall(self, mk_df):
        """ Plot Mann-Kendall Tau values for all analyzed pollutants.
        
        Negative Tau values indicate decreasing long-term trends, positive values
        shows increasing trends, and significant results are marked with an asterisk"""

        if mk_df.empty:
            return
        
        # sorts from the most negative to the most positive trend direction
        mk_sorted = mk_df.sort_values("mk_tau")
        colors = ["#2196F3" if t < 0 else "#D62728" for t in mk_sorted["mk_tau"]]
        
        # adds an asterisk to pollutant labels with statistically significant trends
        labels = []
        for _, row in mk_sorted.iterrows():
            sig = " *" if row["mk_significant"] else ""
            labels.append(f"{row['magnitude_name']}{sig}")

        fig, ax = plt.subplots(figsize=(8, max(4, len(mk_sorted) * 0.45)))
        bars = ax.barh(labels, mk_sorted["mk_tau"].values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mann-Kendall Tau  (negative = decreasing trend)")
        ax.set_title(
            "Long-term Trend Direction per Pollutant — Mann-Kendall Test\n"
            "* = significant at p < 0.05",
            fontsize=11,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "mann_kendall_trend_by_pollutant.png", dpi=300)
        plt.close()
        print("  Saved: mann_kendall_trend_by_pollutant.png")

    def save_plots(self, monthly_overall, hourly, weekday, seasonal, monthly_by_sensor):
        """Save the core temporal-analysis figures for Task 4.
        The outputs summarize yearly trends, hourly cycles, weekday cycles,
        seasonal month profiles, and a station-year heatmap"""

        selected = self.df["magnitude_name"].unique().tolist()

        # plot 1: converts monthly city-wide series into yearly pollutant trends
        yearly_overall = (
            monthly_overall.copy()
        )
        yearly_overall["year"] = pd.to_datetime(yearly_overall["month"]).dt.year
        yearly_overall = (
            yearly_overall
            .groupby(["magnitude_name", "year"], observed=True)["mean_across_sensors"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(11, 6))
        for magnitude in selected:
            subset = yearly_overall[yearly_overall["magnitude_name"] == magnitude]
            plt.plot(subset["year"], subset["mean_across_sensors"], marker="o", label=str(magnitude))

        plt.title("Yearly Pollution Trends")
        plt.xlabel("Year")
        plt.ylabel("Average Value Across Sensors")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "yearly_pollution_trends.png", dpi=300)
        plt.close()

        # plot 2: averages station-level hourly summaries into pollutant-level daily cycles
        hourly_overall = (
            hourly
            .groupby(["magnitude_name", "hour"], observed=True)["mean_value"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 5))
        for magnitude in selected:
            subset = hourly_overall[hourly_overall["magnitude_name"] == magnitude]
            plt.plot(subset["hour"], subset["mean_value"], marker="o", label=str(magnitude))

        plt.title("Average Hourly Pollution Cycle")
        plt.xlabel("Hour of Day")
        plt.ylabel("Average Value")
        plt.xticks(range(0, 24))
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "hourly_pollution_cycle.png", dpi=300)
        plt.close()

        # plot 3: averages station-level weekday summaries into pollutant-level weekly cycles
        weekday_overall = (
            weekday
            .groupby(["magnitude_name", "weekday"], observed=True)["mean_value"]
            .mean()
            .reset_index()
        )

        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        plt.figure(figsize=(10, 5))
        for magnitude in selected:
            subset = weekday_overall[weekday_overall["magnitude_name"] == magnitude]
            plt.plot(subset["weekday"], subset["mean_value"], marker="o", label=str(magnitude))

        plt.title("Average Weekday Pollution Cycle")
        plt.xlabel("Day of Week")
        plt.ylabel("Average Value")
        plt.xticks(range(0, 7), day_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "weekday_pollution_cycle.png", dpi=300)
        plt.close()

        # plot 4: averages station-level month summaries into pollutant-level seasonal profiles
        seasonal_overall = (
            seasonal
            .groupby(["magnitude_name", "month"], observed=True)["mean_value"]
            .mean()
            .reset_index()
        )

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        plt.figure(figsize=(10, 5))
        for magnitude in selected:
            subset = seasonal_overall[seasonal_overall["magnitude_name"] == magnitude]
            plt.plot(subset["month"], subset["mean_value"], marker="o", label=str(magnitude))

        plt.title("Average Seasonal Pollution Cycle (calendar month)")
        plt.xlabel("Month")
        plt.ylabel("Average Value")
        plt.xticks(range(1, 13), month_labels)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "seasonal_pollution_cycle.png", dpi=300)
        plt.close()

        # plot 5: station-year heatmap for first selected pollutant
        self.station_year_heatmap(monthly_by_sensor)

    # plot 6: ACF bar chart
    def save_acf_plot(self, acf_df):
        """Plot lag-12 autocorrelation as a visual summary of annual seasonality"""

        if acf_df.empty:
            return

        acf_df_sorted = acf_df.sort_values("acf_lag12", ascending=False)
        plt.figure(figsize=(10, 5))
        bars = plt.bar(
            acf_df_sorted["magnitude_name"].astype(str),
            acf_df_sorted["acf_lag12"],
        )

        # color bars: steelblue if strong seasonality (>0.5), grey otherwise
        for bar, val in zip(bars, acf_df_sorted["acf_lag12"]):
            bar.set_color("steelblue" if val > 0.5 else "lightgrey")

        plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="threshold = 0.5")
        plt.title("Autocorrelation at Lag 12 (Annual Cycle Strength) by Pollutant")
        plt.xlabel("Pollutant")
        plt.ylabel("ACF at Lag 12")
        plt.ylim(-1, 1)
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "acf_lag12_by_pollutant.png", dpi=300)
        plt.close()

    def build_interpretation_table(self, acf_df, cycle_summary, mk_df=None):
        """ Build a compact summary table for temporal interpretation.
        
        The table combines seasonality strength, daily/seasonal amplitudes
        when available, long-term trend statistics from the Mann-Kendall analysis"""

        if acf_df.empty:
            return pd.DataFrame()
        
        # combines the annual-seasonality metrics with daily and seasonal amplitude summaries
        summary = acf_df.merge(
            cycle_summary,
            on="magnitude_name",
            how="left"
        )
        
        # adds trend-test outputs when Mann-Kendall results are available
        if mk_df is not None and not mk_df.empty:
            summary = summary.merge(
                mk_df[["magnitude_name", "mk_tau", "mk_p_value",
                        "trend_direction", "theil_sen_slope_per_month"]],
                on="magnitude_name",
                how="left",
            )
        
        # translate numeric lag-12 autocorrelation into an easier-to-read seasonality label
        def label_acf(x):
            if x >= 0.7:
                return "very strong"
            elif x >= 0.5:
                return "strong"
            elif x >= 0.3:
                return "moderate"
            else:
                return "weak"

        summary["seasonality_strength"] = summary["acf_lag12"].apply(label_acf)

        round_cols = {
            "acf_lag12": 3,
            "hourly_amplitude": 2,
            "seasonal_amplitude": 2,
        }
        if "mk_tau" in summary.columns:
            round_cols["mk_tau"] = 3
            round_cols["theil_sen_slope_per_month"] = 4

        summary = summary.round(round_cols)

        output_cols = [
            "magnitude_name",
            "acf_lag12",
            "seasonality_strength",
            "hourly_amplitude",
            "seasonal_amplitude",
        ]
        if "mk_tau" in summary.columns:
            output_cols += ["mk_tau", "mk_p_value", "trend_direction",
                            "theil_sen_slope_per_month"]
        
        # returns only the columns needed for final interpretation/report
        return summary[output_cols]

    def run(self):
        print("Aggregating monthly values by sensor...")
        monthly = self.monthly_by_sensor()

        print("Aggregating monthly values across sensors...")
        monthly_overall = self.monthly_overall(monthly)

        print("Computing hourly cycles...")
        hourly = self.hourly_cycle()

        print("Computing weekday cycles...")
        weekday = self.weekday_cycle()

        print("Computing seasonal month cycles...")
        seasonal = self.seasonal_month_cycle()

        print("Computing yearly trends...")
        yearly = self.yearly_trends()

        print("Computing station trend stability...")
        station_slopes, station_slope_summary = self.station_trend_summary(monthly)

        print("Computing cycle strength summary...")
        cycle_summary = self.cycle_strength_summary(hourly, seasonal)

        print("Computing autocorrelation at lag 12...")
        acf_df = self.autocorrelation_at_lag12(monthly_overall)

        print("Running Mann-Kendall trend test...")
        mk_df = self.mann_kendall_trend(monthly_overall)

        print("Building interpretation summary table...")
        interpretation_table = self.build_interpretation_table(acf_df, cycle_summary, mk_df)

        print("Saving Task 4 tables...")
        monthly.to_csv(TABLES_DIR / "monthly_by_sensor.csv", index=False)
        monthly_overall.to_csv(TABLES_DIR / "monthly_overall.csv", index=False)
        hourly.to_csv(TABLES_DIR / "hourly_cycle_by_sensor.csv", index=False)
        weekday.to_csv(TABLES_DIR / "weekday_cycle_by_sensor.csv", index=False)
        seasonal.to_csv(TABLES_DIR / "seasonal_month_cycle_by_sensor.csv", index=False)
        yearly.to_csv(TABLES_DIR / "yearly_trends_by_sensor.csv", index=False)
        station_slopes.to_csv(TABLES_DIR / "station_trend_slopes.csv", index=False)
        station_slope_summary.to_csv(TABLES_DIR / "station_trend_stability_summary.csv", index=False)
        cycle_summary.to_csv(TABLES_DIR / "cycle_strength_summary.csv", index=False)
        acf_df.to_csv(TABLES_DIR / "acf_lag12_summary.csv", index=False)
        mk_df.to_csv(TABLES_DIR / "mann_kendall_trend_summary.csv", index=False)
        interpretation_table.to_csv(TABLES_DIR / "temporal_interpretation_summary.csv", index=False)

        print("Saving Task 4 figures...")
        self.save_plots(monthly_overall, hourly, weekday, seasonal, monthly)
        self.save_acf_plot(acf_df)
        self.plot_mann_kendall(mk_df)                              # NEW
        self.plot_no2_o3_anticorrelation(monthly_overall)         # NEW
        self.plot_longterm_trend_with_covid(monthly_overall, mk_df)  # NEW

        self._print_summary(mk_df, acf_df)

        return {
            "monthly_by_sensor": monthly,
            "monthly_overall": monthly_overall,
            "hourly_cycle": hourly,
            "weekday_cycle": weekday,
            "seasonal_cycle": seasonal,
            "yearly_trends": yearly,
            "station_slopes": station_slopes,
            "cycle_summary": cycle_summary,
            "acf_df": acf_df,
            "mk_df": mk_df,
            "interpretation_table": interpretation_table,
        }

    @staticmethod
    def _print_summary(mk_df, acf_df):
        """ Print a compact console summary of Task 4 trend and seasonality results"""
        print("TASK 4 — Temporal analysis summary")

        if not mk_df.empty:
            dec = mk_df[mk_df["trend_direction"] == "decreasing"]
            inc = mk_df[mk_df["trend_direction"] == "increasing"]
            sig = mk_df[mk_df["mk_significant"]]
            print(f"\nMann-Kendall trend test ({len(mk_df)} pollutants):")
            print(f"  Decreasing (improving): {len(dec)} — "
                  f"{', '.join(dec['magnitude_name'].tolist())}")
            print(f"  Increasing (worsening): {len(inc)} — "
                  f"{', '.join(inc['magnitude_name'].tolist())}")
            print(f"  Statistically significant (p<0.05): {len(sig)}")

        if not acf_df.empty:
            strong = acf_df[acf_df["acf_lag12"] >= 0.5]
            print(f"\nStrong annual seasonality (ACF lag-12 ≥ 0.5): "
                  f"{', '.join(strong['magnitude_name'].tolist())}")

        print("\nKey insight: NO₂ and O₃ anti-correlate seasonally.")
        print("  NO₂ peaks in winter (cold, low UV, traffic).")
        print("  O₃ peaks in summer (photochemical production from NOx).")
        print("  COVID-19 lockdown (Mar 2020–Jun 2021) provides a natural")
        print("  experiment: traffic drop → visible NOx reduction in trend plot.")
        print("=" * 60)


@timer
def run_task4(df, magnitudes=None, top_n_pollutants=3, include_interpolated=True):
    """Run Task 4 temporal-analysis pipeline.

    Args:
        df: Cleaned DataFrame from earlier pipeline stages.
        magnitudes: Optional list of pollutants to analyze directly.
        top_n_pollutants: Number of pollutants to select automatically when `magnitudes` is not provided.
        include_interpolated: If True, include METRAQ-provided interpolated values. If False, restrict the analysis to originally observed values. 
        This is just super optional parameter we added.
    Returns:
        A dictionary containing the main Task 4 summary tables, including
        temporal aggregates, cycle metrics, trend-test results, and the final
        interpretation table"""
    
    print("\n--- Task 4: Temporal Analysis ---")

    analyzer = TemporalAnalyzer(
        df,
        magnitudes=magnitudes,
        top_n_pollutants=top_n_pollutants,
        include_interpolated=include_interpolated,
    )

    results = analyzer.run()

    print("Task 4 completed")
    return results
