"""Task 8 pipeline for parallelized hourly correlation analysis.

This file:
- computes hourly correlation matrices for every (year, sensor) pair,
- runs the computation sequentially and in parallel using multiprocessing.Pool,
- measures wall-clock time and produces an empirical speedup curve,
- identifies variable pairs with stable strong correlations across time and space,
- identifies which variables are associated with increases or reductions in pollution,
- saves summary tables, runtime comparisons, and figures for interpretation.

Each parallel worker independently reads only its relevant subset from disk
using parquet column and row-group filters, minimizing I/O overhead.
"""
# used for timing and task construction
import time
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count

# used for numerical operations, tabular summaries, and plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# project-specific output directories and timing decorator
from utils.config import PROCESSED_DIR, get_task_dirs
from utils.helpers import timer

# output folders dedicated to Task 8 tables, figures, and supporting artifacts
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs(
    "task8_parallelization"
)
# cleaned dataset produced by Task 1 and reused by worker processes when needed
CLEANED_PARQUET = PROCESSED_DIR / "cleaned_air_quality.parquet"

def _worker(args: tuple) -> dict:
    """ Compute one hourly correlation matrix for a single year-sensor task.

    Each multiprocessing worker:
    - reads the requested `(year, sensor_id)` subset from parquet,
    - reshapes hourly measurements into a variable-by-time table,
    - checks whether enough rows and variables exist,
    - computes a Pearson correlation matrix when valid.

    Args:
        args: Tuple containing `(year, sensor_id, parquet_path, min_hours)`
    Returns:
        A dictionary containing task identifiers, the correlation matrix when
        available, the number of hourly rows used, included variables, and
        worker runtime"""
    
    year, sensor_id, parquet_path, min_hours = args
    # starts a worker-level timer so task runtimes can be inspected separately
    t0 = time.perf_counter()

    # each worker reads the dataset independently (as required by the task spec).
    # only the columns we need are loaded, this cuts I/O significantly on large
    # datasets (Parquet column pruning reads only the relevant byte ranges)
    df = pd.read_parquet(
        parquet_path,
        filters=[
            ("year", "==", year),
            ("sensor_id", "==", sensor_id),
        ],
        columns=["entry_date", "magnitude_name", "value"],
    )
    # prepare a standard empty result so failed or insufficient tasks still return
    # the same schema as successful ones
    result = {
        "year": year,
        "sensor_id": int(sensor_id),
        "corr_matrix": None,
        "n_hours": 0,
        "variables": [],
        "elapsed": 0.0,
    }

    if df.empty:
        result["elapsed"] = time.perf_counter() - t0
        return result

    # reshapes long-format measurements into:
    # rows = hourly timestamps, columns = variables, values = observed values
    pivot = df.pivot_table(
        index="entry_date",
        columns="magnitude_name",
        values="value",
        aggfunc="mean",
    )

    # drops columns that are entirely NaN
    pivot = pivot.dropna(axis=1, how="all")

    # we neeed at least 2 variables and enough hourly rows
    if pivot.shape[1] < 2 or len(pivot) < min_hours:
        result["elapsed"] = time.perf_counter() - t0
        return result

    corr = pivot.corr(method="pearson")

    result["corr_matrix"] = corr
    result["n_hours"] = len(pivot)
    result["variables"] = list(pivot.columns)
    result["elapsed"] = time.perf_counter() - t0

    return result


class HourlyCorrelationAnalyzer:
    """ Compute and compare hourly correlation matrices across all sensor-year pairs.
    
    - Runs the same computation sequentially and in parallel, 
    - measures the resulting speedup,
    - summarizes which variable relationships are strong and stable across years and sensors"""

    def __init__(
        self,
        df: pd.DataFrame,
        threshold: float = 0.6,
        min_hours: int = 24,
        n_workers: int | None = None,
        force_partitions: bool = False,
        max_years: int | None = None,
        save_matrices: bool = False,
        stability_frac: float = 0.80,
    ):
        """ Task 8 parallel-correlation workflow.

    Args:
        df: cleaned DataFrame
        threshold: absolute correlation threshold used to label pairs as strong.
        min_hours: minimum hourly observations required for one year-sensor task.
        n_workers: number of worker processes for multiprocessing. Defaults to
                `.cpu_count() - 1`, with a minimum of one worker.
        force_partitions: If True, write partitioned parquet files by year and sensor before worker execution.
        max_years: optional cap on the number of years processed, useful for benchmarking or quick runs.
        save_matrices: If True, save every individual correlation matrix to CSV.
        stability_frac: Minimum fraction of year-sensor cells in which a pair 
                       must exceed the threshold to be considered stable"""
    
        # keeps the local copy so this task can add helper columns without mutating input
        self.df = df.copy()
        self.threshold = threshold
        self.min_hours = min_hours
        self.n_workers = n_workers if n_workers is not None else max(1, cpu_count() - 1)
        self.force_partitions = force_partitions
        self.max_years = max_years
        self.save_matrices = save_matrices
        self.stability_frac = stability_frac

        # ensures we have the year column
        if "year" not in self.df.columns:
            self.df["year"] = self.df["entry_date"].dt.year.astype("int16")

    def _get_tasks(self):
        """ Create the complete list of `(year, sensor_id)` tasks to process"""        
        # optionally restrict years to reduce runtime during benchmarking.
        years = sorted(self.df["year"].dropna().unique())
        if self.max_years is not None:
            years = years[: self.max_years]

        sensors = sorted(self.df["sensor_id"].dropna().unique())
        # each task corresponds to one correlation matrix for one sensor in one year
        tasks = list(itertools.product(years, sensors))
        return tasks

    def _ensure_parquet(self) -> Path:
        """ Return the parquet path used by worker processes.
        
        If partitioning is requested, a year/sensor-partitioned parquet dataset is created once and reused. 
        Otherwise, workers read from the cleaned Task 1 parquet file, which is created if it does not already exist"""
        # partitioning can reduce repeated scan work when many workers read small subsets
        if self.force_partitions:
            partition_root = PROCESSED_DIR / "partitioned"
            if not partition_root.exists():
                print("  Writing partitioned parquet (year / sensor_id)…")
                self.df.to_parquet(
                    partition_root,
                    partition_cols=["year", "sensor_id"],
                    index=False,
                )
            return partition_root

        if not CLEANED_PARQUET.exists():
            print("  Saving cleaned parquet for worker reads…")
            self.df.to_parquet(CLEANED_PARQUET, index=False)

        return CLEANED_PARQUET

    def run_sequential(self, tasks: list, parquet_path: Path) -> tuple[list, float]:
        """ process all year-sensor tasks one after another and measure wall time"""
        print(f"  Running sequentially ({len(tasks)} tasks)…")
        t_start = time.perf_counter()

        # reuses the same worker logic as the parallel path so runtime comparison is fair
        results = []
        for args in tasks:
            res = _worker((args[0], args[1], parquet_path, self.min_hours))
            results.append(res)

        elapsed = time.perf_counter() - t_start
        print(f"  Sequential finished in {elapsed:.2f}s")
        return results, elapsed

    def run_parallel(self, tasks: list, parquet_path: Path,
                     n_workers: int | None = None) -> tuple[list, float]:
        """ Process all year-sensor tasks with a multiprocessing worker pool"""

        workers = n_workers if n_workers is not None else self.n_workers
        print(f"  Running in parallel ({workers} workers, {len(tasks)} tasks)…")
        worker_args = [
            (year, sensor_id, parquet_path, self.min_hours)
            for year, sensor_id in tasks
        ]

        t_start = time.perf_counter()
        
        # pool.map preserves task order while distributing work across processes.
        with Pool(processes=workers) as pool:
            results = pool.map(_worker, worker_args)

        elapsed = time.perf_counter() - t_start
        print(f"  Parallel finished in {elapsed:.2f}s")
        return results, elapsed

    def run_speedup_curve(self, tasks: list, parquet_path: Path,
                          time_seq: float) -> pd.DataFrame:
        """ Measure wall time for 1, 2, and n_workers parallel workers
        to produce an empirical speedup curve.

        The curve illustrates Amdahl's Law: speedup plateaus as the
        serial fraction of the work (I/O, inter-process communication)
        becomes the bottleneck.
        We always include n_workers=1 using the parallel path (not
        sequential) to isolate Pool overhead from serial logic """

        worker_counts = sorted(set([1, 2, self.n_workers]))
        rows = [{"n_workers": 0, "wall_time_s": time_seq, "speedup": 1.0,
                 "mode": "sequential"}]

        for w in worker_counts:
            if w < 1:
                continue
            _, elapsed = self.run_parallel(tasks, parquet_path, n_workers=w)
            speedup = time_seq / max(elapsed, 1e-9)
            rows.append({
                "n_workers": w,
                "wall_time_s": round(elapsed, 4),
                "speedup": round(speedup, 4),
                "mode": "parallel",
            })
            print(f"    {w} workers: {elapsed:.2f}s  (speedup {speedup:.2f}×)")

        return pd.DataFrame(rows)

    def _extract_pairs(self, results: list) -> pd.DataFrame:
        """ Flatten worker correlation matrices into one variable-pair table.
        
        Each row represents one variable pair in one `(year, sensor)` cell,
        preserving the signed and absolute Pearson correlation values"""
        rows = []
        for res in results:
            corr = res["corr_matrix"]
            # skips tasks where no valid correlation matrix could be computed
            if corr is None:
                continue
            variables = list(corr.columns)
            # visit only the upper triangle of the symmetric correlation matrix, so to avoid duplicates
            for i, var_a in enumerate(variables):
                for var_b in variables[i + 1:]:
                    val = corr.loc[var_a, var_b]
                    if pd.isna(val):
                        continue
                    rows.append(
                        {
                            "year": res["year"],
                            "sensor_id": res["sensor_id"],
                            "var_a": var_a,
                            "var_b": var_b,
                            "pair": tuple(sorted([var_a, var_b])),
                            "correlation": float(val),
                            "abs_correlation": float(abs(val)),
                        }
                    )
        return pd.DataFrame(rows)

    def _build_stable_pairs_table(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Identify variable pairs that are strongly correlated across many cells.
        
        A pair is considered stable when `|r|` exceeds the chosen threshold in at
        least `stability_frac` of all year-sensor cells where that pair is observed.
        
        These stable pairs highlight robust co-varying relationships that persist
        across both time and monitoring locations """
        if pairs_df.empty:
            return pd.DataFrame()

        # total number of (year, sensor) cells that produced a result for each pair
        pair_stats = (
            pairs_df.groupby("pair")
            .agg(
                n_cells=("abs_correlation", "count"),
                n_above=("abs_correlation", lambda x: (x >= self.threshold).sum()),
                median_abs_corr=("abs_correlation", "median"),
                mean_abs_corr=("abs_correlation", "mean"),
                std_abs_corr=("abs_correlation", "std"),
                n_years=("year", "nunique"),
                n_sensors=("sensor_id", "nunique"),
                mean_signed_corr=("correlation", "mean"),
            )
            .reset_index()
        )

        pair_stats["stability_frac"] = pair_stats["n_above"] / pair_stats["n_cells"]
        pair_stats["pair_str"] = pair_stats["pair"].apply(lambda p: f"{p[0]} ↔ {p[1]}")

        # a pair is "stable" if it's above threshold in ≥ stability_frac of cells
        stable = pair_stats[
            pair_stats["stability_frac"] >= self.stability_frac
        ].sort_values(["stability_frac", "median_abs_corr"], ascending=False)

        return stable

    def summarise(self, pairs_df: pd.DataFrame) -> dict:
        """ Summarize the main analytical findings from all correlation pairs.
        
        This method computes:
        - variable pairs with consistently high median absolute correlation,
        - a stricter stable-pairs table based on the fraction of strong cells,
        - the yearly percentage of pairs above the threshold,
        - signed associations between non-pollutant variables and pollutant variables""" 

        if pairs_df.empty:
            return {}

        # builds a broad stable strong correlations (original + new stable_pairs table)
        stable = (
            pairs_df.groupby("pair")
            .agg(
                median_abs_corr=("abs_correlation", "median"),
                mean_abs_corr=("abs_correlation", "mean"),
                std_abs_corr=("abs_correlation", "std"),
                n_observations=("abs_correlation", "count"),
                n_years=("year", "nunique"),
            )
            .reset_index()
        )
        stable["pair_str"] = stable["pair"].apply(lambda p: f"{p[0]} ↔ {p[1]}")
        stable_strong = stable[stable["median_abs_corr"] >= self.threshold].sort_values(
            "median_abs_corr", ascending=False
        )

        # stable pairs table 
        stable_pairs = self._build_stable_pairs_table(pairs_df)

        # percentage above threshold per year 
        yearly_pct = (
            pairs_df.groupby("year")["abs_correlation"]
            .apply(lambda g: (g >= self.threshold).mean() * 100)
            .reset_index(name="pct_above_threshold")
        )

        # variable–pollution association 
        pollutants = {
            "SO2", "CO", "NO", "NO2", "PM2.5", "PM10", "<PM2.5", "<PM10",
            "NOX", "O3", "TOLUENO", "BENCENO", "ETILBENCENO",
            "HIDROCARBS_TOTALES", "METANO", "HIDROCARBS_NO_METANICOS",
        }

        # compute SIGNED mean correlation (not just abs) for directionality
        def _mean_signed_corr_with_pollutants(df_sub, var_col, other_col):
            mask = df_sub[other_col].str.upper().isin(
                {p.upper() for p in pollutants}
            )
            if mask.sum() == 0:
                return pd.Series(dtype=float)
            return (
                df_sub[mask]
                .groupby(var_col)["correlation"]
                .mean()
            )

        corr_a_signed = _mean_signed_corr_with_pollutants(pairs_df, "var_a", "var_b")
        corr_b_signed = _mean_signed_corr_with_pollutants(pairs_df, "var_b", "var_a")

        pollution_assoc = (
            pd.concat([corr_a_signed, corr_b_signed])
            .groupby(level=0)
            .mean()
            .reset_index()
        )
        pollution_assoc.columns = ["variable", "mean_signed_corr_with_pollutants"]
        pollution_assoc["mean_abs_corr_with_pollutants"] = (
            pollution_assoc["mean_signed_corr_with_pollutants"].abs()
        )
        pollution_assoc = pollution_assoc[
            ~pollution_assoc["variable"].str.upper().isin(
                {p.upper() for p in pollutants}
            )
        ].sort_values("mean_abs_corr_with_pollutants", ascending=False)

        return {
            "stable_strong": stable_strong,
            "stable_pairs": stable_pairs,        
            "yearly_pct": yearly_pct,
            "pollution_assoc": pollution_assoc,
            "all_pairs": pairs_df,
            "stable_all": stable,
        }

    def _save_matrices(self, results: list):
        """ optionally save every valid year-sensor correlation matrix as a CSV file"""

        matrices_dir = TABLES_DIR / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            if res["corr_matrix"] is None:
                continue
            fname = f"corr_year{res['year']}_sensor{res['sensor_id']}.csv"
            res["corr_matrix"].to_csv(matrices_dir / fname)

    def save_tables(self, summary: dict, runtime: dict,
                    speedup_df: pd.DataFrame | None = None):
        """ Save Task 8 analytical summaries and runtime diagnostics to CSV files"""
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        
        # saves each summary only when it contains rows, keeping outputs tidy.
        if "stable_strong" in summary and not summary["stable_strong"].empty:
            summary["stable_strong"].drop(columns="pair").to_csv(
                TABLES_DIR / "stable_strong_correlations.csv", index=False
            )

        # stable pairs table
        if "stable_pairs" in summary and not summary["stable_pairs"].empty:
            summary["stable_pairs"].drop(columns="pair").to_csv(
                TABLES_DIR / "stable_pairs_table.csv", index=False
            )

        if "yearly_pct" in summary and not summary["yearly_pct"].empty:
            summary["yearly_pct"].to_csv(
                TABLES_DIR / "yearly_pct_above_threshold.csv", index=False
            )

        if "pollution_assoc" in summary and not summary["pollution_assoc"].empty:
            summary["pollution_assoc"].to_csv(
                TABLES_DIR / "variable_pollution_association.csv", index=False
            )

        if "all_pairs" in summary and not summary["all_pairs"].empty:
            summary["all_pairs"].drop(columns="pair").to_csv(
                TABLES_DIR / "all_correlation_pairs.csv", index=False
            )

        runtime_df = pd.DataFrame([runtime])
        runtime_df.to_csv(TABLES_DIR / "runtime_comparison.csv", index=False)

        # always saves the runtime information, as runtime comparison is the most important comparison feature
        if speedup_df is not None:
            speedup_df.to_csv(TABLES_DIR / "speedup_curve.csv", index=False)

        print("  Tables saved.")

    # figures

    @staticmethod
    def _classify_variable(name: str) -> tuple[str, str]:
        """ Classify a variable for chart coloring.
    
        Returns: A tuple containing category label and display color used in plots """

        POLLUTANTS = {
            "SO2","CO","NO","NO2","PM2.5","PM10","<PM2.5","<PM10",
            "NOX","O3","TOLUENO","BENCENO","ETILBENCENO",
            "HIDROCARBS_TOTALES","METANO","HIDROCARBS_NO_METANICOS",
        }
        WEATHER = {"TEMP","HR","PRE","RS","VV","DV","PRECIPITACION"}
        TRAFFIC_PREFIXES = ("TI_","SP_","OC_")

        u = name.strip().upper()
        if u in POLLUTANTS:
            return "Pollutant", "#D62728"
        if u in WEATHER:
            return "Weather", "#2CA02C"
        if any(u.startswith(p) for p in TRAFFIC_PREFIXES):
            return "Traffic", "#1F77B4"
        return "Other", "#7F7F7F"

    def save_no2_driver_chart(self, pairs_df: pd.DataFrame,
                               target: str = "NO2") -> None:
        """ Save a signed driver-style correlation chart for a target pollutant.
        
        The figure summarizes which variables are most positively or negatively
        associated with a chosen pollutant, using:
        - mean signed correlation,
        - standard-deviation error bars across year-sensor cells,
        - variable-category coloring,
        - counts of contributing cells,
        - a side summary of top positive and negative associations.
        
        The output describes correlation patterns, not causal effects"""
        if pairs_df.empty:
            return

        target_upper = target.strip().upper()

        # collects per-cell correlations with the target 
        mask_a = pairs_df["var_b"].str.upper() == target_upper
        mask_b = pairs_df["var_a"].str.upper() == target_upper

        rows_a = pairs_df[mask_a][["var_a", "correlation", "year", "sensor_id"]].rename(
            columns={"var_a": "variable"}
        )
        rows_b = pairs_df[mask_b][["var_b", "correlation", "year", "sensor_id"]].rename(
            columns={"var_b": "variable"}
        )
        combined = pd.concat([rows_a, rows_b], ignore_index=True)

        if combined.empty:
            print(f"  Driver chart skipped: no pairs found for {target}.")
            return

        # aggregates signed correlations so positive and negative directions remain visible
        agg = (
            combined.groupby("variable")["correlation"]
            .agg(mean_r="mean", std_r="std", n_cells="count")
            .reset_index()
        )
        # drops the target itself (self-correlation would be 1.0)
        agg = agg[agg["variable"].str.upper() != target_upper].copy()

        # classifies each variable for consistent pollutant / weather / traffic coloring
        agg["category"], agg["color"] = zip(
            *agg["variable"].map(self._classify_variable)
        )

        # sorts from most negative at the bottom to most positive at the top
        agg = agg.sort_values("mean_r", ascending=True).reset_index(drop=True)

        # layout features 
        n = len(agg)
        bar_height = 0.55
        fig_height = max(6, n * 0.38 + 2.5)

        fig = plt.figure(figsize=(13, fig_height))

        # main axes occupies left ~70 % of the figure
        ax = fig.add_axes([0.28, 0.08, 0.48, 0.84])

        # horizontal bars 
        y_pos = np.arange(n)

        bars = ax.barh(
            y_pos,
            agg["mean_r"],
            height=bar_height,
            color=agg["color"],
            alpha=0.85,
            zorder=3,
        )

        # error bars (±1 std)
        ax.errorbar(
            agg["mean_r"],
            y_pos,
            xerr=agg["std_r"].fillna(0),
            fmt="none",
            color="black",
            linewidth=0.8,
            capsize=2.5,
            alpha=0.55,
            zorder=4,
        )

        # zero line
        ax.axvline(0, color="black", linewidth=0.9, zorder=5)

        # y-axis labels (variable names) 
        # shortens the  long traffic names for readability
        def _shorten(name):
            replacements = {
                "RBF_MULTICUADRIC": "RBF_MQ",
                "RBF_GAUSSIAN": "RBF_G",
                "RBF_LINEAR": "RBF_L",
                "PRECIPITACION": "PRECIP",
                "HIDROCARBS_TOTALES": "HC_TOT",
                "HIDROCARBS_NO_METANICOS": "HC_NM",
                "ETILBENCENO": "ETILBENZ",
            }
            for long, short in replacements.items():
                name = name.replace(long, short)
            return name

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [_shorten(v) for v in agg["variable"]],
            fontsize=8.5,
        )

        # colour the y-tick labels to match bar category
        for tick, color in zip(ax.get_yticklabels(), agg["color"]):
            tick.set_color(color)


        ax.set_xlabel(
            f"Mean correlation with {target}  "
            f"(error bars = ±1 std across years × sensors)",
            fontsize=9,
        )
        ax.set_title(
            f"What drives {target} pollution in Madrid?",
            fontsize=11,
            fontweight="medium",
            pad=10,
        )
        ax.grid(axis="x", alpha=0.25, zorder=0)
        ax.set_axisbelow(True)

        # secondary x-axis label at the top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel(
            f"← reduces {target}     |     raises {target} →",
            fontsize=8, color="#666666",
        )
        ax2.tick_params(bottom=False, top=False, labelbottom=False, labeltop=False)

        # legend for variable types 
        from matplotlib.patches import Patch
        category_colors = {
            "Pollutant": "#D62728",
            "Traffic":   "#1F77B4",
            "Weather":   "#2CA02C",
            "Other":     "#7F7F7F",
        }
        handles = [
            Patch(facecolor=c, label=cat, alpha=0.85)
            for cat, c in category_colors.items()
            if cat in agg["category"].values
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
            title="Variable type",
            title_fontsize=8,
        )

        # summary callout panel (right side) 
        summary_ax = fig.add_axes([0.78, 0.08, 0.20, 0.84])
        summary_ax.axis("off")

        def _top_by_category(category, sign, k=3):
            """Return top-k variable names for a given category and correlation sign."""
            sub = agg[
                (agg["category"] == category) &
                (np.sign(agg["mean_r"]) == sign)
            ].sort_values("mean_r", ascending=(sign < 0))
            return sub["variable"].head(k).tolist()

        def _shorten_list(names):
            return [_shorten(n) for n in names]

        sections = [
            ("TOP POSITIVE", +1, [
                ("Pollutant", "#D62728"),
                ("Traffic",   "#1F77B4"),
                ("Weather",   "#2CA02C"),
            ]),
            ("TOP NEGATIVE", -1, [
                ("Weather",   "#2CA02C"),
                ("Traffic",   "#1F77B4"),
                ("Pollutant", "#D62728"),
            ]),
        ]

        y_cursor = 0.97
        line_h   = 0.045
        gap      = 0.035

        summary_ax.text(
            0.0, y_cursor, "SUMMARY",
            fontsize=9, fontweight="bold", color="#222222",
            transform=summary_ax.transAxes, va="top",
        )
        y_cursor -= line_h * 1.2

        for section_title, sign, cats in sections:
            # section header with coloured background bar
            summary_ax.add_patch(plt.Rectangle(
                (0, y_cursor - line_h * 0.15), 1.0, line_h * 1.05,
                transform=summary_ax.transAxes,
                color="#1B4F72" if sign > 0 else "#641E16",
                clip_on=False, zorder=2,
            ))
            summary_ax.text(
                0.05, y_cursor + line_h * 0.35,
                section_title,
                fontsize=7.5, fontweight="bold", color="white",
                transform=summary_ax.transAxes, va="center", zorder=3,
            )
            y_cursor -= line_h * 1.3

            for cat, cat_color in cats:
                names = _top_by_category(cat, sign, k=3)
                if not names:
                    continue
                # category label
                summary_ax.text(
                    0.04, y_cursor,
                    cat,
                    fontsize=7, fontweight="bold", color=cat_color,
                    transform=summary_ax.transAxes, va="top",
                )
                y_cursor -= line_h * 0.85
                for nm in _shorten_list(names):
                    summary_ax.text(
                        0.08, y_cursor,
                        f"• {nm}",
                        fontsize=7, color="#333333",
                        transform=summary_ax.transAxes, va="top",
                    )
                    y_cursor -= line_h * 1.0
                y_cursor -= gap * 0.4

            y_cursor -= gap

        # saves
        fname = FIGURES_DIR / f"what_drives_{target.lower().replace('<','pm')}_pollution.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname.name}")

    def save_figures(self, summary: dict, runtime: dict,
                     speedup_df: pd.DataFrame | None = None):
        """ Save runtime, stability, association, and scaling figures for Task 8"""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        # 1. runtime bar chart (sequential vs parallel) 
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Sequential", f"Parallel\n({self.n_workers} workers)"]
        times = [runtime["sequential_s"], runtime["parallel_s"]]
        colors = ["#4C72B0", "#55A868"]
        bars = ax.bar(labels, times, color=colors, width=0.5)
        ax.bar_label(bars, fmt="%.1f s", padding=4, fontsize=9)
        ax.set_ylabel("Wall-clock time (s)")
        ax.set_title("Task 8 — Sequential vs Parallel Runtime")
        speedup = runtime["sequential_s"] / max(runtime["parallel_s"], 1e-6)
        ax.text(
            0.97, 0.97,
            f"Speed-up: {speedup:.2f}×",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="grey"),
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "runtime_comparison.png", dpi=300)
        plt.close()

        # 2. speedup curve shows whether more workers continue to help or hit diminishing returns
        if speedup_df is not None and not speedup_df.empty:
            par_rows = speedup_df[speedup_df["mode"] != "sequential"].copy()
            if not par_rows.empty:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))

                # left: wall time vs n_workers
                axes[0].plot(par_rows["n_workers"], par_rows["wall_time_s"],
                             marker="o", color="#4C72B0", linewidth=2)
                axes[0].axhline(runtime["sequential_s"], color="gray", linestyle="--",
                                linewidth=1, label="Sequential baseline")
                axes[0].set_xlabel("Number of workers")
                axes[0].set_ylabel("Wall-clock time (s)")
                axes[0].set_title("Wall Time vs Workers")
                axes[0].legend(fontsize=8)
                axes[0].grid(alpha=0.25)

                # right: speedup vs n_workers with ideal line
                axes[1].plot(par_rows["n_workers"], par_rows["speedup"],
                             marker="o", color="#55A868", linewidth=2, label="Observed speedup")
                max_w = par_rows["n_workers"].max()
                axes[1].plot([1, max_w], [1, max_w], "k--", linewidth=1,
                             alpha=0.4, label="Ideal (linear) speedup")
                axes[1].set_xlabel("Number of workers")
                axes[1].set_ylabel("Speedup (×)")
                axes[1].set_title("Speedup Curve  (Amdahl's Law)")
                axes[1].legend(fontsize=8)
                axes[1].grid(alpha=0.25)

                fig.suptitle(
                    "Task 8 — Parallelisation Scaling Analysis\n"
                    "Speedup plateau = serial I/O fraction becomes the bottleneck",
                    fontsize=11,
                )
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / "speedup_curve.png", dpi=300,
                            bbox_inches="tight")
                plt.close()
                print("  Saved speedup_curve.png")

        # 3. % of pairs above threshold per year
        yearly_pct = summary.get("yearly_pct", pd.DataFrame())
        if not yearly_pct.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(
                yearly_pct["year"].astype(str),
                yearly_pct["pct_above_threshold"],
                color="#4C72B0",
            )
            ax.axhline(
                yearly_pct["pct_above_threshold"].mean(),
                color="red", linestyle="--", linewidth=1,
                label=f"Mean = {yearly_pct['pct_above_threshold'].mean():.1f}%",
            )
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
            ax.set_xlabel("Year")
            ax.set_ylabel(f"% pairs |corr| ≥ {self.threshold}")
            ax.set_title(
                f"Percentage of Variable Pairs with |Correlation| ≥ {self.threshold} by Year"
            )
            plt.xticks(rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "yearly_pct_above_threshold.png", dpi=300)
            plt.close()

        # 4. top stable strong correlations (horizontal bar)
        stable_strong = summary.get("stable_strong", pd.DataFrame())
        if not stable_strong.empty:
            top = stable_strong.head(20).copy()
            fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
            ax.barh(top["pair_str"], top["median_abs_corr"], color="#55A868")
            ax.axvline(self.threshold, color="red", linestyle="--", linewidth=1,
                       label=f"Threshold = {self.threshold}")
            ax.set_xlabel("Median |Pearson r|")
            ax.set_title(f"Top Stable Strong Variable Pairs (threshold = {self.threshold})")
            ax.invert_yaxis()
            ax.legend()
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "stable_strong_correlations.png", dpi=300)
            plt.close()

        # 5. stable pairs table heatmap (top pairs × years present): 
        # highlights relationships that persist across both years and sensors
        stable_pairs = summary.get("stable_pairs", pd.DataFrame())
        if not stable_pairs.empty:
            top_sp = stable_pairs.head(15).copy()
            fig, ax = plt.subplots(figsize=(9, max(4, len(top_sp) * 0.45)))
            scatter_colors = [
                "#D62728" if s > 0 else "#1F77B4"
                for s in top_sp["mean_signed_corr"]
            ]
            bars = ax.barh(top_sp["pair_str"], top_sp["stability_frac"],
                           color=scatter_colors, alpha=0.82)
            ax.axvline(self.stability_frac, color="orange", linestyle="--",
                       linewidth=1.2, label=f"Stability threshold = {self.stability_frac:.0%}")
            ax.set_xlabel(f"Fraction of (year, sensor) cells with |r| ≥ {self.threshold}")
            ax.set_title(
                "Stable Variable Pairs — Reliable Across Time & Space\n"
                "Red = positively correlated with pollutants, Blue = negative",
                fontsize=10,
            )
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            ax.set_xlim(0, 1.05)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "stable_pairs_table.png", dpi=300,
                        bbox_inches="tight")
            plt.close()
            print("  Saved stable_pairs_table.png")

        # 6. improved variable-pollution association chart with sign: separates positive from negative relationships
        pollution_assoc = summary.get("pollution_assoc", pd.DataFrame())
        if not pollution_assoc.empty:
            top_pa = pollution_assoc.head(15).copy()
            fig, ax = plt.subplots(figsize=(9, max(4, len(top_pa) * 0.45)))

            # use signed correlation for colour: red = increases pollution, blue = reduces
            bar_colors = [
                "#D62728" if v > 0 else "#1F77B4"
                for v in top_pa["mean_signed_corr_with_pollutants"]
            ]
            bars = ax.barh(
                top_pa["variable"],
                top_pa["mean_signed_corr_with_pollutants"],
                color=bar_colors,
                alpha=0.85,
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Mean signed Pearson r with pollutants")
            ax.set_title(
                "Variables Associated with Pollution Levels\n"
                "Red = positively correlated (increases pollution), "
                "Blue = negatively correlated (reduces/offsets)",
                fontsize=10,
            )
            ax.invert_yaxis()
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            ax.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "variable_pollution_association.png", dpi=300,
                        bbox_inches="tight")
            plt.close()

        # 7.detailed driver chart for primary pollutants
        all_pairs = summary.get("all_pairs", pd.DataFrame())
        if not all_pairs.empty:
            for target_pol in ["NO2", "NOX"]:
                self.save_no2_driver_chart(all_pairs, target=target_pol)

        print("  Figures saved.")

    def run(self) -> dict:
        """ Execute the full Task 8 parallelization and correlation-analysis workflow 
            Returns a dict containing summary tables, runtime metrics, and the speedup DataFrame """
        print("Preparing data and tasks…")
        # prepares disk-backed worker input and the full task list
        parquet_path = self._ensure_parquet()
        tasks = self._get_tasks()
        print(f"  {len(tasks)} (year, sensor) combinations to process.")

        # sequential run 
        results_seq, time_seq = self.run_sequential(tasks, parquet_path)

        # parallel run 
        results_par, time_par = self.run_parallel(tasks, parquet_path)

        # speedup curve 
        print("  Building speedup curve…")
        speedup_df = self.run_speedup_curve(tasks, parquet_path, time_seq)

        # uses the parallel result set for downstream summaries, it should match
        # the sequential result set but is the main execution path of interest.
        pairs_df = self._extract_pairs(results_par)

        print("Summarising observations…")
        summary = self.summarise(pairs_df)

        runtime = {
            "n_tasks": len(tasks),
            "n_workers": self.n_workers,
            "threshold": self.threshold,
            "min_hours": self.min_hours,
            "stability_frac": self.stability_frac,
            "sequential_s": round(time_seq, 4),
            "parallel_s": round(time_par, 4),
            "speedup": round(time_seq / max(time_par, 1e-9), 4),
        }

        print("Saving tables and figures…")
        self.save_tables(summary, runtime, speedup_df)
        self.save_figures(summary, runtime, speedup_df)

        if self.save_matrices:
            print("Saving individual correlation matrices…")
            self._save_matrices(results_par)

        self._print_summary(summary, runtime, speedup_df)

        return {"summary": summary, "runtime": runtime, "speedup_df": speedup_df}

    @staticmethod
    def _print_summary(summary: dict, runtime: dict,
                       speedup_df: pd.DataFrame | None = None):
        """ Print a concise console summary of runtime and correlation findings."""
        print("TASK 8 — summary")

        print(f"\nRuntime comparison ({runtime['n_tasks']} tasks):")
        print(f"  Sequential : {runtime['sequential_s']:.2f}s")
        print(f"  Parallel   : {runtime['parallel_s']:.2f}s  "
              f"({runtime['n_workers']} workers)")
        print(f"  Speed-up   : {runtime['speedup']:.2f}×")

        if speedup_df is not None and not speedup_df.empty:
            par_rows = speedup_df[speedup_df["mode"] == "parallel"]
            if not par_rows.empty:
                best = par_rows.loc[par_rows["speedup"].idxmax()]
                print(f"  Best speedup in curve: {best['speedup']:.2f}× "
                      f"at {int(best['n_workers'])} workers")

        stable = summary.get("stable_strong", pd.DataFrame())
        if not stable.empty:
            print(
                f"\nStable strong correlations (median |r| ≥ {runtime['threshold']}): "
                f"{len(stable)} pairs"
            )
            print(stable[["pair_str", "median_abs_corr", "n_years"]].head(10).to_string(index=False))

        # stable pairs
        sp = summary.get("stable_pairs", pd.DataFrame())
        if not sp.empty:
            print(f"\nStable pairs (|r| ≥ {runtime['threshold']} in ≥"
                  f"{runtime['stability_frac']:.0%} of cells): {len(sp)} pairs")
            cols = ["pair_str", "stability_frac", "median_abs_corr",
                    "mean_signed_corr", "n_years", "n_sensors"]
            available_cols = [c for c in cols if c in sp.columns]
            print(sp[available_cols].head(10).to_string(index=False))
            print(
                "\n  Positive mean_signed_corr → variable tends to move with pollution\n"
                "  Negative mean_signed_corr → variable tends to move in the opposite direction\n"
                "  (e.g. wind speed or solar radiation reducing ground-level concentrations)"
            )

        yearly = summary.get("yearly_pct", pd.DataFrame())
        if not yearly.empty:
            mean_pct = yearly["pct_above_threshold"].mean()
            print(f"\nMean % of pairs above threshold across years: {mean_pct:.1f}%")

        pa = summary.get("pollution_assoc", pd.DataFrame())
        if not pa.empty:
            print("\nTop variables associated with pollutant levels (signed):")
            print(pa.head(10).to_string(index=False))

        print("\nScalability and I/O bottleneck notes:")
        print("  • Parallelised across (year, sensor) pairs using multiprocessing.Pool.")
        print("  • Each worker reads only its subset via parquet row-group filters.")
        print("  • Column pruning (columns=['entry_date','magnitude_name','value'])")
        print("    reduces I/O by loading only 3 of the ~15 parquet columns.")
        print("  • Speedup plateau is caused by the serial I/O fraction:")
        print("    all workers compete for the same disk; adding more workers")
        print("    beyond ~4 gives diminishing returns on a single-disk machine.")
        print("  • To scale further: use Dask, pre-partition by year/sensor, or")
        print("    distribute across multiple nodes with different disk spindles.")


@timer
def run_task8(
    df: pd.DataFrame | None = None,
    threshold: float = 0.6, #common cutoff threshold
    min_hours: int = 24,
    n_workers: int | None = None,
    force_partitions: bool = False,
    max_years: int | None = None,
    save_matrices: bool = False,
    stability_frac: float = 0.80,) -> dict: #pair must be strongly correlated in at least 80% of all year x sensor cells to be called "stable"
    """ Run Task 8 parallelization and correlation analysis pipeline.

    Args:
        df: cleaned DataFrame; if None, loaded automatically
            from the cleaned parquet file on disk.
        threshold: absolute correlation value considered strong (default 0.6).
        min_hours: minimum hourly observations required per (year, sensor)
            pair to compute a meaningful correlation (default 24).
        n_workers: number of parallel worker processes, defaults to
            cpu_count() - 1 if not specified, so the machine doesn't become completely unresponsive during the parallel run
        force_partitions: if True, pre-partitions the dataset by year and
            sensor_id on disk before running workers. Uses more memory
            but can reduce per-worker I/O time.
        max_years: optional limit on the number of years processed, useful
            for benchmarking or memory-constrained runs.
        save_matrices: if True, saves each individual correlation matrix
            as a separate CSV file.
        stability_frac: fraction of (year, sensor) cells where |r| must
            exceed threshold for a pair to be declared stable (default 0.80).

    Returns: A dictionary containing summary tables, runtime metrics, and the speedup DataFrame"""
    print("\n--- Task 8: Parallelization ---")

    if df is None:
        if not CLEANED_PARQUET.exists():
            raise FileNotFoundError(
                f"Cleaned parquet not found at {CLEANED_PARQUET}. "
                "Run Task 1 first, or pass a DataFrame directly."
            )
        print(f"  Loading cleaned data from {CLEANED_PARQUET}…")
        df = pd.read_parquet(CLEANED_PARQUET)

    analyzer = HourlyCorrelationAnalyzer(
        df=df,
        threshold=threshold,
        min_hours=min_hours,
        n_workers=n_workers,
        force_partitions=force_partitions,
        max_years=max_years,
        save_matrices=save_matrices,
        stability_frac=stability_frac,
    )

    results = analyzer.run()
    print("Task 8 completed")
    return results
