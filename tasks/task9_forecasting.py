"""Task 9 pipeline for forecasting pollutant concentrations (optional task).

This file:
- builds a wide panel from the long-format dataset by pivoting to one column per variable,
- aggregates to daily averages per sensor to reduce noise and memory use,
- trains Random Forest, Gradient Boosting, and XGBoost models per target pollutant,
- evaluates performance using chronological train/test split and TimeSeriesSplit cross-validation,
- extracts feature importances to interpret which variables drive pollution,
- saves summary tables and figures for interpretation.

Features are restricted to meteorological and traffic variables only — pollutants
are excluded from features to avoid data leakage into the prediction target"""

# used for warnings control, numerical operations, and tabular processing
import warnings
import numpy as np
import pandas as pd

# used for forecasting visualizations and custom plot layouts
import matplotlib.pyplot as plt

# machine-learning models, preprocessing, validation, and metrics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# project-specific output directories and timing decorator
from utils.config import get_task_dirs, PROCESSED_DIR
from utils.helpers import timer

# hides non-critical library warnings to keep Task 9 console output readable
warnings.filterwarnings("ignore", category=UserWarning)

# output folders dedicated to Task 9 tables, figures, and supporting artifacts
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task9_forecasting")

# variable catalogues used to separate pollutant targets from allowable predictors
POLLUTANTS = {
    "SO2", "CO", "NO", "NO2", "PM2.5", "PM10", "<PM2.5", "<PM10",
    "NOX", "O3", "TOLUENO", "BENCENO", "ETILBENCENO",
    "HIDROCARBS_TOTALES", "METANO", "HIDROCARBS_NO_METANICOS",
}

# weather variables allowed as forecasting features
WEATHER_VARS = {"TEMP", "HR", "PRE", "RS", "VV", "DV", "PRECIPITACION"}

# traffic-related predictor variables are identified by these prefixes
TRAFFIC_PREFIX = ("TI_", "SP_", "OC_")


def _is_confounder(name: str) -> bool:
    """ Return True when a variable is allowed as a non-pollutant predictor.

    Eligible predictors are restricted to:
    - meteorological variables,
    - traffic variables identified by their naming prefixes"""

    # normalizes spelling/case before checking catalogue membership
    name_up = name.strip().upper()
    return (
        name_up in WEATHER_VARS
        or any(name_up.startswith(p) for p in TRAFFIC_PREFIX)
    )

class ForecastingAnalyzer:
    """ Build and evaluate regression models to estimate the impact of
    meteorological and traffic variables on pollutant concentrations.
    
    Models are trained on weather and traffic features only, with pollutants
    used exclusively as prediction targets. Chronological train/test split
    is used to respect the temporal structure of the data"""

    def __init__(
        self,
        df: pd.DataFrame,
        targets: list[str] | None = None,
        aggregate: str = "daily",
        test_frac: float = 0.20,
        n_cv_splits: int = 5,
        rf_n_estimators: int = 200,
        rf_max_depth: int = 10,
        random_state: int = 42,
        year_start: int | None = None,
        year_end: int | None = None,
        max_sensors: int | None = None,):
        """ Initialize the forecasting workflow.
        
        Args:
        df: cleaned DataFrame
        targets: optional list of pollutant names to predict; if None, the
            three most frequent pollutants are selected automatically.
        aggregate: temporal aggregation level, either `"daily"` or `"hourly"`.
            daily aggregation is the recommended default for reducing noise and memory use.
        test_frac: fraction of the ordered timeline reserved for final testing.
        n_cv_splits: number of folds used in TimeSeriesSplit cross-validation.
        rf_n_estimators: number of trees in the Random Forest model.
        rf_max_depth: maximum Random Forest tree depth.
        random_state: reproducibility seed.
        year_start: optional inclusive lower bound on the year range.
        year_end: optional inclusive upper bound on the year range.
        max_sensors: optional cap on the number of most represented sensors kept before model-building, useful for memory control"""
        self.aggregate = aggregate
        self.test_frac = test_frac
        self.n_cv_splits = n_cv_splits
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.random_state = random_state
        
        # restricts the dataset before expensive pivoting and model training
        self.df_raw = self._slice(df, year_start, year_end, max_sensors)
        # resolve either user-provided targets or automatically selected pollutants
        self.targets = self._resolve_targets(targets)
        print(f"  Prediction targets: {self.targets}")

    def _slice(self, df, year_start, year_end, max_sensors):
        """ Apply optional year and sensor restrictions before modeling.
        
        These filters reduce memory use and can focus the forecasting exercise on
        periods or sensors with better overlap between pollutants and predictors"""
        df = df.copy()
        
        # ensures a year column exists before applying optional temporal filters
        if "year" not in df.columns:
            df["year"] = df["entry_date"].dt.year.astype("int16")
        
        # limit the modeling period when the user requests a specific year window
        if year_start is not None:
            df = df[df["year"] >= year_start]
            print(f"  Filtering year >= {year_start}")

        if year_end is not None:
            df = df[df["year"] <= year_end]
            print(f"  Filtering year <= {year_end}")
        
        # keep only the most represented sensors when memory reduction is needed
        if max_sensors is not None:
            top_sensors = (
                df["sensor_id"].value_counts().head(max_sensors).index
            )
            df = df[df["sensor_id"].isin(top_sensors)]
            print(f"  Keeping top {max_sensors} sensors ({len(top_sensors)} found)")

        print(f"  Working dataset: {len(df):,} rows after slicing")
        # resets the index after filtering so downstream row counts and splits are clean
        return df.reset_index(drop=True)

    # data preparation step

    def _resolve_targets(self, targets):
        """ Determine which pollutant targets should be modeled"""
        # respects explicit target choices when supplied by the user.
        if targets is not None:
            return [t.upper() for t in targets]
        # auto-selects the top pollutants by row count
        pollutant_mask = (
            self.df_raw["magnitude_name"]
            .astype(str)
            .str.upper()
            .isin(POLLUTANTS)
        )
        top = (
            self.df_raw[pollutant_mask]["magnitude_name"]
            .value_counts()
            .head(3)
            .index
            .tolist()
        )
        return [str(t).upper() for t in top]

    def build_wide_panel(self) -> pd.DataFrame:
        """ Construct the sensor-time modeling table used by forecasting models.
        
        - optionally simplifies traffic inputs to one representative variant
        - aggregates long-format observations to daily or hourly sensor-level means,
        - pivots variables into separate columns,
        - adds time-derived features such as hour, weekday, month, and year.
        Returns:
        A wide DataFrame where each row represents one sensor at one time point """
        df = self.df_raw.copy()

        # traffic variables may contain several interpolation variants 
        # so retains only one representative family to reduce duplicated predictors.
        traffic_mask = df["magnitude_name"].astype(str).str.upper().str.startswith(
            ("TI_", "SP_", "OC_")
        )
        traffic_df = df[traffic_mask].copy()

        # picks one traffic method per base metric
        if not traffic_df.empty:
            preferred = traffic_df[
                traffic_df["magnitude_name"].astype(str).str.upper().str.startswith("TI_")
            ]
            non_traffic = df[~traffic_mask]
            df = pd.concat([non_traffic, preferred], ignore_index=True)

        # temporal grouping key
        if self.aggregate == "daily":
            df["time_key"] = df["entry_date"].dt.normalize()
        else:
            df["time_key"] = df["entry_date"].dt.floor("H")

        # aggregates: mean per (time_key, sensor_id, magnitude_name)
        agg = (
            df.groupby(
                ["time_key", "sensor_id", "magnitude_name"],
                observed=True,
            )["value"]
            .mean()
            .reset_index()
        )

        # pivot from long format to model-ready wide format:
        # one row per sensor-time point, one column per variable.
        wide = agg.pivot_table(
            index=["time_key", "sensor_id"],
            columns="magnitude_name",
            values="value",
            aggfunc="mean",
        )
        wide.columns = [str(c).strip() for c in wide.columns]
        wide = wide.reset_index()
        wide = wide.sort_values("time_key").reset_index(drop=True)

        # adds time features as confounders (so models can capture cyclical patterns)
        wide["hour"] = wide["time_key"].dt.hour.astype("int8")
        wide["weekday"] = wide["time_key"].dt.dayofweek.astype("int8")
        wide["month"] = wide["time_key"].dt.month.astype("int8")
        wide["year"] = wide["time_key"].dt.year.astype("int16")

        return wide

    def _get_features(self, wide: pd.DataFrame, target: str) -> list[str]:
        """ Return leakage-safe predictor columns for one pollutant target.
        
        Features are:
        - weather variables,
        - traffic variables,
        - basic time features.
        All pollutant columns are excluded so the model does not predict one
        pollutant using another pollutant directly"""

        time_features = ["hour", "weekday", "month", "year"]

        confounder_cols = [
            c for c in wide.columns
            if c not in {"time_key", "sensor_id"}
            and c not in time_features
            and c.upper() not in POLLUTANTS
            and _is_confounder(c)
        ]

        return confounder_cols + time_features

    def _train_test_split(self, wide: pd.DataFrame, target: str):
        """ Create a chronological train/test split for one target pollutant.
        
        The earliest portion of the timeline is used for training, while the latest
        `test_frac` portion is reserved for final evaluation. This preserves time
        order and avoids leaking future information into the training set"""
        
        # locates the target column case-insensitively in the wide modeling table
        target_col = next(
            (c for c in wide.columns if c.upper() == target.upper()), None
        )
        if target_col is None:
            return None, None, None, None, None

        features = self._get_features(wide, target)
        available_features = [f for f in features if f in wide.columns]

        if not available_features:
            return None, None, None, None, None
        
        # drops rows missing either the target or any selected predictor
        sub = wide[["time_key"] + available_features + [target_col]].dropna()
        # keep only target datasets large enough for a meaningful modeling exercise
        if len(sub) < 50:
            print(f"    Not enough data for {target} ({len(sub)} rows). Skipping.")
            return None, None, None, None, None
        
        # splits chronologically rather than randomly to preserve forecasting realism
        split_idx = int(len(sub) * (1 - self.test_frac))
        train = sub.iloc[:split_idx]
        test = sub.iloc[split_idx:]

        X_train = train[available_features].values
        y_train = train[target_col].values
        X_test = test[available_features].values
        y_test = test[target_col].values

        return X_train, y_train, X_test, y_test, available_features

    # modelling step 

    def _build_models(self):
        """Create the regression-model pipelines evaluated in current Task 9"""        
        # each model is wrapped in a pipeline so preprocessing and fitting are handled
        # consistently across cross-validation and final testing
        return {
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=self.rf_n_estimators,
                    max_depth=self.rf_max_depth,
                    n_jobs=-1,
                    random_state=self.random_state,
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=self.random_state,
                )),
            ]),
            "XGBoost": Pipeline([
                ("scaler", StandardScaler()),
                ("model", XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbosity=0,
                )),
            ]),
        }

    def _evaluate(self, y_true, y_pred, model_name, target, split) -> dict:
        """ Compute predictive-performance metrics for one model evaluation"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            "target": target,
            "model": model_name,
            "split": split,
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "n_samples": len(y_true),
        }

    def _cross_validate(self, pipeline, X_train, y_train, model_name, target) -> list:
        """ Evaluate one model with time-aware cross-validation on the training period"""
        # TimeSeriesSplit preserves temporal ordering across validation folds
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        rows = []
        # refits the pipeline inside each fold using only the earlier training slice.
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred = pipeline.predict(X_train[val_idx])
            row = self._evaluate(y_train[val_idx], y_pred, model_name, target, f"cv_fold_{fold}")
            rows.append(row)
        return rows

    def _feature_importance(self, pipeline, feature_names, model_name, target) -> pd.DataFrame:
        """Extract model-based feature-importance outputs after fitting.
        
        Tree models report impurity-based feature importances. The coefficient branch
        is retained for compatibility with possible future linear models"""        
        model = pipeline.named_steps["model"]
        rows = []
        # current tree-based models expose `feature_importances_`.
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for name, imp in zip(feature_names, importances):
                rows.append({
                    "target": target,
                    "model": model_name,
                    "feature": name,
                    "importance": float(imp),
                    "type": "tree_importance",
                })
        elif hasattr(model, "coef_"): 
            scaler = pipeline.named_steps["scaler"]
            # un-scale coefficients for interpretability
            coefs = model.coef_ / scaler.scale_
            for name, coef in zip(feature_names, coefs):
                rows.append({
                    "target": target,
                    "model": model_name,
                    "feature": name,
                    "importance": float(coef),
                    "type": "coefficient",
                })

        return pd.DataFrame(rows)

    # main run 
    def run(self) -> dict:
        """Execute the full Task 9 forecasting workflow.

    The workflow (using the functions specified above):
    - builds the wide modeling panel,
    - prepares each pollutant target,
    - trains and evaluates all candidate models,
    - performs time-series cross-validation,
    - extracts feature-importance summaries,
    - saves tables and figures,
    - prints a concise console report.

    Returns: a dictionary containing model metrics, cross-validation metrics, 
    feature importances, the wide panel, and stored predictions."""
        print("Building wide panel…")
        # builds the model-ready sensor-time panel once and reuse it for every target
        wide = self.build_wide_panel()
        print(f"  Wide panel: {len(wide):,} rows × {wide.shape[1]} columns")

        all_metrics = []
        all_cv_metrics = []
        all_importances = []
        predictions = {}

        # trains a separate model family for each target pollutant
        for target in self.targets:
            print(f"\n  ── Target: {target} ──")

            X_train, y_train, X_test, y_test, feature_names = self._train_test_split(
                wide, target
            )
            # skips targets that lack sufficient feature overlap or sample size
            if X_train is None:
                print(f"    Skipping {target}: insufficient data.")
                continue

            print(f"    Train: {len(y_train):,} | Test: {len(y_test):,} | Features: {len(feature_names)}")

            models = self._build_models()
            target_preds = {"y_test": y_test}

            for model_name, pipeline in models.items():
                print(f"    Fitting {model_name}…")

                # cross-validation on train set
                cv_rows = self._cross_validate(
                    pipeline, X_train, y_train, model_name, target
                )
                all_cv_metrics.extend(cv_rows)

                # final fit on full train set
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                # test metrics
                test_row = self._evaluate(y_test, y_pred, model_name, target, "test")
                all_metrics.append(test_row)

                print(
                    f"      R²={test_row['r2']:.3f}  "
                    f"RMSE={test_row['rmse']:.3f}  "
                    f"MAE={test_row['mae']:.3f}"
                )

                # feature importances / coefficients
                imp_df = self._feature_importance(pipeline, feature_names, model_name, target)
                all_importances.append(imp_df)

                target_preds[model_name] = y_pred

            predictions[target] = target_preds

        metrics_df = pd.DataFrame(all_metrics)
        cv_df = pd.DataFrame(all_cv_metrics)
        importances_df = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()
        #saves predictions so the best-model scatter plots can be created later
        print("\nSaving tables…")
        self.save_tables(metrics_df, cv_df, importances_df, wide)

        print("Saving figures…")
        self.save_figures(metrics_df, importances_df, predictions)

        self._print_summary(metrics_df, importances_df)

        return {
            "metrics": metrics_df,
            "cv_metrics": cv_df,
            "importances": importances_df,
            "wide_panel": wide,
            "predictions": predictions,
        }

    def save_tables(self, metrics_df, cv_df, importances_df, wide):
        """ Save forecasting metrics, feature importances, and panel-coverage tables"""

        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        # saves final held-out test-set metrics
        if not metrics_df.empty:
            metrics_df.to_csv(TABLES_DIR / "model_test_metrics.csv", index=False)
        # summarize cross-validation performance across folds
        if not cv_df.empty:
            cv_summary = (
                cv_df.groupby(["target", "model"])
                .agg(
                    mean_r2=("r2", "mean"),
                    std_r2=("r2", "std"),
                    mean_rmse=("rmse", "mean"),
                    mean_mae=("mae", "mean"),
                )
                .reset_index()
            )
            cv_summary.to_csv(TABLES_DIR / "model_cv_summary.csv", index=False)
        # save all model-specific importance values for later interpretation and analysis
        if not importances_df.empty:
            importances_df.to_csv(TABLES_DIR / "feature_importances.csv", index=False)

        # data availability summary
        availability = pd.DataFrame({
            "column": wide.columns.tolist(),
            "non_null": wide.notna().sum().values,
            "null_rate": wide.isna().mean().round(4).values,
        })
        availability.to_csv(TABLES_DIR / "wide_panel_availability.csv", index=False)

    def save_figures(self, metrics_df, importances_df, predictions):
        """ Save model-performance, feature-importance, and fit-quality figures"""
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        if metrics_df.empty:
            return

        # 1) compares the model test performance across R², RMSE, and MAE
        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
        for ax, metric in zip(axes, ["r2", "rmse", "mae"]):
            pivot = metrics_df.pivot(index="target", columns="model", values=metric)
            pivot.plot(kind="bar", ax=ax, legend=(metric == "r2"))
            ax.set_title(metric.upper())
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=30)
            if metric == "r2":
                ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        fig.suptitle("Task 9 — Model Performance on Test Set", fontsize=13)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300)
        plt.close()

        # 2) plots the most influential features for each target-model combination
        if not importances_df.empty:
            for target in importances_df["target"].unique():
                for model_name in importances_df["model"].unique():
                    sub = importances_df[
                        (importances_df["target"] == target)
                        & (importances_df["model"] == model_name)
                    ].copy()
                    if sub.empty:
                        continue

                    sub = sub.sort_values("importance", key=abs, ascending=True).tail(15)
                    colors = ["#d62728" if v > 0 else "#1f77b4" for v in sub["importance"]]

                    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.4)))
                    ax.barh(sub["feature"], sub["importance"], color=colors)
                    ax.axvline(0, color="grey", linewidth=0.8)
                    imp_type = sub["type"].iloc[0]
                    label = "Coefficient (scaled)" if imp_type == "coefficient" else "Feature Importance"
                    ax.set_xlabel(label)
                    ax.set_title(f"{model_name} — {target}\nModel Feature Importance")
                    plt.tight_layout()
                    safe_model = model_name.lower().replace(" ", "_")
                    plt.savefig(
                        FIGURES_DIR / f"importance_{target.lower()}_{safe_model}.png",
                        dpi=300,
                    )
                    plt.close()

        # 3) for each target shows predicted vs actual values for the best R² model
        best_models = (
            metrics_df.sort_values("r2", ascending=False)
            .groupby("target")
            .first()
            .reset_index()
        )

        for _, row in best_models.iterrows():
            target = row["target"]
            model_name = row["model"]

            if target not in predictions or model_name not in predictions[target]:
                continue

            y_test = predictions[target]["y_test"]
            y_pred = predictions[target][model_name]

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4C72B0")
            lims = [
                min(y_test.min(), y_pred.min()),
                max(y_test.max(), y_pred.max()),
            ]
            ax.plot(lims, lims, "r--", linewidth=1, label="Perfect fit")
            ax.set_xlabel(f"Actual {target}")
            ax.set_ylabel(f"Predicted {target}")
            ax.set_title(
                f"{target} — {model_name}\n"
                f"R²={row['r2']:.3f}  RMSE={row['rmse']:.3f}"
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"predicted_vs_actual_{target.lower()}.png",
                dpi=300,
            )
            plt.close()

    @staticmethod
    def _print_summary(metrics_df, importances_df):
        """ Print a concise console summary of forecasting results and limitations"""
        print("TASK 9 — summary")

        if metrics_df.empty:
            print("No results — insufficient overlapping data.")
            return

        print("\nTest-set performance:")
        print(
            metrics_df[["target", "model", "r2", "rmse", "mae"]]
            .sort_values(["target", "r2"], ascending=[True, False])
            .to_string(index=False)
        )
        if not importances_df.empty:
            print("\nTop features (by |importance|) across all models:")
            top = (
                importances_df.groupby("feature")["importance"]
                .apply(lambda x: x.abs().mean())
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
            )
            top.columns = ["feature", "mean_abs_importance"]
            print(top.to_string(index=False))

        print("\nLimitations / assumptions:")
        print("  • Daily aggregation smooths intra-day variation.")
        print("  • Feature-importance outputs describe model usage, not causal effects.")
        print("  • Random Forest importances do not indicate directionality.")
        print("  • Traffic data coverage is sparser than pollutant data.")
        print("  • Chronological split avoids data leakage but limits CV folds.")

@timer
def run_task9(
    df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
    aggregate: str = "daily",
    test_frac: float = 0.20,
    n_cv_splits: int = 5,
    rf_n_estimators: int = 200,
    rf_max_depth: int = 10,
    random_state: int = 42,
    year_start: int | None = None,
    year_end: int | None = None,
    max_sensors: int | None = None,
) -> dict:
    """ Run the full Task 9 forecasting pipeline.
    
    Args:
        df: cleaned DataFrame, if None, loaded from disk.
        targets: pollutants to predict. Auto-selected if None.
        aggregate: temporal aggregation level, "daily" or "hourly".
        test_frac: fraction of timeline reserved for test evaluation.
        n_cv_splits: number of TimeSeriesSplit folds for cross-validation.
        rf_n_estimators: number of trees in the Random Forest.
        rf_max_depth: maximum tree depth, lower values reduce overfitting and speed up training.
        random_state: reproducibility seed.
        year_start: restrict data to this year onward, useful to align
            with the period where weather and traffic data are available.
        year_end: restrict data up to this year inclusive.
        max_sensors: keep only the N sensors with the most rows,
            use this to reduce memory on large datasets.
    
    Returns a dict containing test metrics, CV metrics, feature importances, the wide panel, and per-target predictions"""
    print("\n--- Task 9: Forecasting Model ---")
    if df is None:
        cleaned = PROCESSED_DIR / "cleaned_air_quality.parquet"
        if not cleaned.exists():
            raise FileNotFoundError(
                f"Cleaned parquet not found at {cleaned}. Run Task 1 first."
            )
        print(f"  Loading cleaned data from {cleaned}…")
        df = pd.read_parquet(cleaned)

    analyzer = ForecastingAnalyzer(
        df=df,
        targets=targets,
        aggregate=aggregate,
        test_frac=test_frac,
        n_cv_splits=n_cv_splits,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        random_state=random_state,
        year_start=year_start,
        year_end=year_end,
        max_sensors=max_sensors,
    )

    results = analyzer.run()
    print("Task 9 completed")
    return results
