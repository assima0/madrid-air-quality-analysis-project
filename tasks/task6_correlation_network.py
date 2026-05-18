""" Task 6 pipeline for building and analysing a correlation-based sensor network.

This file:
- aggregates pollutant measurements to monthly averages per sensor,
- computes pairwise Pearson correlations between sensor time series,
- builds correlation-threshold graphs at multiple cut-offs (0.50 → 0.90),
- computes node-level centrality metrics and detects behavioural communities,
- compares the resulting network against Task 5's spatial distance network,
- saves summary tables, graph files, and figures for interpretation.

Difference from Task 5 is that in Task 6, edges represent time-series similarity rather
than geographic proximity: two sensors are connected because they behave
similarly, not because they are physically close """

# used for numerical and tabular analysis
import numpy as np
import pandas as pd

# used for network creation and analysis
import networkx as nx

# used for plotting/visualizations
import matplotlib.pyplot as plt

# used for custom legend handles and readable node labels
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# project-specific output directories and timing decorator
from utils.config import get_task_dirs
from utils.helpers import timer

# output folders dedicated to Task 6 correlation-network tables, figures, and graph files
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs(
    "task6_correlation_network"
)

class CorrelationNetworkAnalyzer:
    """ Build and analyze pollutant-specific sensor correlation networks.
    
    For one selected pollutant, the analyzer:
    - aggregates monthly sensor time series,
    - computes pairwise temporal correlations,
    - builds graphs under multiple correlation thresholds,
    - evaluates graph structure, centrality, and communities,
    - saves tables, plots, and graph files for interpretation"""

    def __init__(self, df, magnitude="NO2", min_months=12):
        """ Initialize the correlation-network analysis.
        
        Args:
            df: cleaned air-quality DataFrame.
            magnitude: pollutant name used to build the sensor similarity network.
            min_months: minimum number of monthly observations required for a
            sensor to remain in the correlation analysis"""
        self.df = df.copy()
        self.magnitude = magnitude
        self.min_months = min_months

    # data preparation step

    def prepare_monthly_sensor_series(self):
        """ Aggregate one pollutant to monthly mean values per sensor.

        The correlation network compares sensor time series, so raw hourly records
        are first reduced to a common monthly resolution. This follows the project
        suggestion to simplify pairwise similarity analysis through temporal
        aggregation"""

        df = self.df.copy()
        
        # restricts the analysis to the pollutant selected for this analyzer instance
        magnitude_upper = df["magnitude_name"].astype(str).str.upper()
        df = df[magnitude_upper == self.magnitude.upper()].copy()

        if df.empty:
            raise ValueError(f"No rows found for magnitude: {self.magnitude}")

        value_col = "value"
        
        # converts each timestamp to the first day of its month 
        # so records can be grouped into a consistent monthly time series
        df["month_period"] = df["entry_date"].dt.to_period("M").dt.to_timestamp()
        
        # averages repeated hourly observations within each sensor-month pair
        monthly = (
            df.groupby(
                ["month_period", "sensor_id", "sensor_name"],
                observed=True,
            )[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: "monthly_mean_value"})
        )

        monthly.to_csv(
            TABLES_DIR / f"monthly_sensor_series_{self.magnitude}.csv",
            index=False,
        )

        return monthly

    def build_sensor_matrix(self, monthly):
        """ Convert monthly sensor values into a correlation-ready matrix.
        
        Rows represent months and columns represent sensors. Sensors with fewer than
        `min_months` available monthly observations are removed so correlations are
        based on a minimally informative time series"""

        # builds readable labels so saved edge tables can show both sensor name and ID.
        sensor_labels = (
            monthly[["sensor_id", "sensor_name"]]
            .drop_duplicates()
            .copy()
        )
        sensor_labels["sensor_label"] = (
            sensor_labels["sensor_name"].astype(str)
            + " ("
            + sensor_labels["sensor_id"].astype(str)
            + ")"
        )
        label_map = dict(
            zip(sensor_labels["sensor_id"], sensor_labels["sensor_label"])
        )
        
        # pivots long-format monthly data into a wide matrix suitable for correlation.
        matrix = monthly.pivot_table(
            index="month_period",
            columns="sensor_id",
            values="monthly_mean_value",
            aggfunc="mean",
        )
        matrix = matrix.sort_index()
        
        valid_counts = matrix.notna().sum()
        keep_sensors = valid_counts[valid_counts >= self.min_months].index
        matrix = matrix[keep_sensors]

        matrix.to_csv(TABLES_DIR / f"sensor_month_matrix_{self.magnitude}.csv")
        
        # retains only sensors with enough monthly coverage to support stable comparison
        coverage = pd.DataFrame({
            "sensor_id": valid_counts.index,
            "n_months_available": valid_counts.values,
            "kept_for_correlation": valid_counts.index.isin(keep_sensors),
        })
        coverage["sensor_name"] = coverage["sensor_id"].map(
            monthly.drop_duplicates("sensor_id").set_index("sensor_id")["sensor_name"]
        )
        # saves the coverage diagnostics to show which sensors were retained or excluded.
        coverage.to_csv(
            TABLES_DIR / f"sensor_month_coverage_{self.magnitude}.csv",
            index=False,
        )

        return matrix, label_map

    def compute_correlation_matrix(self, matrix):
        """ Compute pairwise Pearson sensor correlations.
        
        Correlations are calculated only when at least `min_months` overlapping
        monthly observations are available for each sensor pair"""

        corr = matrix.corr(method="pearson", min_periods=self.min_months)
        corr.to_csv(
            TABLES_DIR / f"sensor_correlation_matrix_{self.magnitude}.csv"
        )
        return corr

    def correlation_pairs(self, corr, label_map):
        """ Flatten the sensor correlation matrix into a sorted pair table.
        
        Each unordered sensor pair appears once, together with its signed Pearson 
        correlation and absolute correlation"""

        rows = []
        sensors = corr.columns.tolist()
        # visits only the upper triangle of the symmetric matrix, so we can avoid duplicate pairs.
        for i, s1 in enumerate(sensors):
            for s2 in sensors[i + 1:]:
                val = corr.loc[s1, s2]
                # skips sensor pairs whose correlation could not be computed.
                if pd.isna(val):
                    continue
                rows.append({
                    "sensor_id_1": int(s1),
                    "sensor_name_1": label_map.get(s1, str(s1)),
                    "sensor_id_2": int(s2),
                    "sensor_name_2": label_map.get(s2, str(s2)),
                    "correlation": float(val),
                    "absolute_correlation": float(abs(val)),
                })
        pairs = pd.DataFrame(rows).sort_values("correlation", ascending=False)
        pairs.to_csv(
            TABLES_DIR / f"sensor_correlation_pairs_{self.magnitude}.csv",
            index=False,
        )
        return pairs

    # graph construction step

    def _add_nodes(self, G, label_map):
        """ Add sensor nodes and attach readable sensor labels"""
        for sensor_id, label in label_map.items():
            G.add_node(
                int(sensor_id),
                sensor_label=str(label),
                sensor_name=str(label).split(" (")[0],
            )
        return G

    def build_threshold_graph(self, label_map, pairs, threshold):
        """Build a graph linking sensors above a correlation threshold.
        
        Sensors become connected when their pollutant time series have a Pearson
        correlation greater than or equal to the specified threshold"""        
        
        G = nx.Graph()
        self._add_nodes(G, label_map)

        # keeps sensor pairs whose behavioral similarity satisfies the threshold
        selected = pairs[pairs["correlation"] >= threshold]
        # stores correlation as edge strength and 1-correlation as a path distance
        for _, row in selected.iterrows():
            corr_val = float(row["correlation"])
            G.add_edge(
                int(row["sensor_id_1"]),
                int(row["sensor_id_2"]),
                correlation=corr_val,
                weight=corr_val,
                distance=float(1 - corr_val),
            )

        G.graph["method"] = "correlation_threshold"
        G.graph["magnitude"] = self.magnitude
        G.graph["threshold"] = threshold
        return G

    # graph metrics 

    def graph_summary(self, G, graph_name):
        """ Compute graph-level statistics for one correlation network.
        
        The summary statistics includes:
        - node and edge counts,
        - density and degree statistics,
        - connected-component structure,
        - weighted clustering,
        - average shortest-path distance on the largest connected component"""

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # handles empty graphs, so downstream tables remain well-formed
        if n_nodes == 0:
            return {
                "graph_name": graph_name,
                "magnitude": self.magnitude,
                "threshold": G.graph.get("threshold", np.nan),
                "n_nodes": 0, "n_edges": 0,
            }

        degrees = dict(G.degree())
        components = list(nx.connected_components(G))
        largest = max(components, key=len) if components else set()
        largest_sub = G.subgraph(largest).copy()

        asp = np.nan
        if largest_sub.number_of_nodes() > 1 and largest_sub.number_of_edges() > 0:
            asp = nx.average_shortest_path_length(largest_sub, weight="distance")

        return {
            "graph_name": graph_name,
            "magnitude": self.magnitude,
            "threshold": G.graph.get("threshold", np.nan),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": nx.density(G),
            "average_degree": np.mean(list(degrees.values())) if degrees else np.nan,
            "min_degree": min(degrees.values()) if degrees else np.nan,
            "max_degree": max(degrees.values()) if degrees else np.nan,
            "is_connected": nx.is_connected(G),
            "n_connected_components": len(components),
            "largest_component_size": len(largest),
            "average_clustering": nx.average_clustering(G, weight="weight") if n_edges > 0 else 0,
            "average_shortest_path_largest_component": asp,
        }

    def node_metrics(self, G, graph_name):
        """ Compute node-level centrality metrics for a correlation graph.
        Metrics include:
        - degree,
        - weighted degree,
        - degree centrality,
        - betweenness centrality,
        - closeness centrality,
        - clustering coefficient,
        - eigenvector centrality.

        These measures identify sensors that are:
        - highly connected, 
        - act as bridges,
        - sit near the center of the behavioral network,
        - are correlated with other influential sensors""" 
        
        if G.number_of_nodes() == 0: 
            return pd.DataFrame()

        degree = dict(G.degree())
        weighted_degree = dict(G.degree(weight="weight"))
        degree_centrality = nx.degree_centrality(G)

        if G.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)
            closeness = nx.closeness_centrality(G, distance="distance")
            clustering = nx.clustering(G, weight="weight")
            try:
                eigenvector = nx.eigenvector_centrality(
                    G, weight="weight", max_iter=500, tol=1e-6
                )
            except nx.PowerIterationFailedConvergence:
                eigenvector = {n: 0.0 for n in G.nodes()}
        else:
            betweenness  = {n: 0 for n in G.nodes()}
            closeness    = {n: 0 for n in G.nodes()}
            clustering   = {n: 0 for n in G.nodes()}
            eigenvector  = {n: 0 for n in G.nodes()}

        rows = []
        for node, attrs in G.nodes(data=True):
            rows.append({
                "graph_name": graph_name,
                "sensor_id": node,
                "sensor_name": attrs.get("sensor_name"),
                "degree": degree.get(node),
                "weighted_degree": weighted_degree.get(node),
                "degree_centrality": degree_centrality.get(node),
                "betweenness_centrality": betweenness.get(node),
                "closeness_centrality": closeness.get(node),
                "clustering_coefficient": clustering.get(node),
                "eigenvector_centrality": eigenvector.get(node),   
            })
        return pd.DataFrame(rows)

    def detect_communities(self, G, graph_name):
        """ Detect communities of sensors with similar temporal behavior.
        
        Louvain modularity communities are used when available, with greedy
        modularity as a fallback. Communities represent sensors that are more
        strongly correlated with one another than with the rest of the network"""

        # a graph without edges has no meaningful multi-node communities, so each
        # sensor is assigned to its own singleton community

        if G.number_of_edges() == 0:
            communities = [{node} for node in G.nodes()]
            modularity = np.nan
            algorithm = "none"
        else:
            try:
                communities = nx.community.louvain_communities(
                    G, weight="weight", seed=42
                )
                algorithm = "louvain"
            except AttributeError:
                communities = list(
                    nx.community.greedy_modularity_communities(G, weight="weight")
                )
                algorithm = "greedy_modularity"

            modularity = nx.community.modularity(G, communities, weight="weight")

        rows = []
        for cid, community in enumerate(communities):
            for node in community:
                rows.append({
                    "graph_name": graph_name,
                    "sensor_id": node,
                    "community_id": cid,
                    "community_size": len(community),
                    "modularity": modularity,
                    "community_algorithm": algorithm,
                })
        return pd.DataFrame(rows), communities, modularity

    # figures  

    def save_graph_plot(self, G, graph_name, communities_df=None, node_metrics_df=None):
        """ Save a force-directed correlation-network visualization.

        Visual encoding:
        - node color: detected behavioral community,
        - node size: betweenness centrality,
        - edge width and opacity: correlation strength"""

        if G.number_of_nodes() == 0:
            return
        
        # uses a force-directed layout 
        pos = (
            nx.spring_layout(G, seed=42, weight="weight", k=1.1)
            if G.number_of_edges() > 0
            else nx.circular_layout(G)
        )

        if communities_df is not None and not communities_df.empty:
            cmap_map = dict(zip(
                communities_df["sensor_id"].astype(int),
                communities_df["community_id"].astype(int),
            ))
            node_communities = [cmap_map.get(int(n), -1) for n in G.nodes()]
            unique_communities = sorted(set(node_communities))
        else:
            node_communities = [0] * G.number_of_nodes()
            unique_communities = [0]

        cmap = plt.cm.Set2
        n_communities = max(1, len(unique_communities))

        # scales node size by betweenness centrality to emphasize bridge-like sensors
        if node_metrics_df is not None and not node_metrics_df.empty:
            bc_map = dict(zip(
                node_metrics_df["sensor_id"].astype(int),
                node_metrics_df["betweenness_centrality"].astype(float),
            ))
            centrality = np.array(
                [bc_map.get(int(n), 0.0) for n in G.nodes()], dtype=float
            )
            if centrality.max() > centrality.min():
                node_sizes = 500 + 1700 * (
                    (centrality - centrality.min())
                    / (centrality.max() - centrality.min())
                )
            else:
                node_sizes = np.full(len(centrality), 850.0)
        else:
            node_sizes = np.full(G.number_of_nodes(), 850.0)

        correlations = np.array(
            [d.get("correlation", 0.0) for _, _, d in G.edges(data=True)], dtype=float
        )
        if len(correlations) > 0 and correlations.max() > correlations.min():
            scale = (correlations - correlations.min()) / (correlations.max() - correlations.min())
            edge_widths = 0.6 + 3.0 * scale
            edge_alphas = 0.25 + 0.45 * scale
        else:
            edge_widths = np.full(G.number_of_edges(), 1.4)
            edge_alphas = np.full(G.number_of_edges(), 0.35)

        fig, ax = plt.subplots(figsize=(13, 10))

        for (u, v, data), width, alpha in zip(G.edges(data=True), edge_widths, edge_alphas):
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)], ax=ax,
                edge_color="gray", width=float(width), alpha=float(alpha),
            )

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_communities,
            cmap=cmap,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.9,
        )
        # shortens long station names to reduce overlap in the final figure
        _label_replacements = {
            "Barajas Pueblo": "Barajas",
            "Urbanización Embajada": "Embajada",
            "Ensanche de Vallecas": "Ensanche",
            "Plaza de España": "Plaza España",
            "Cuatro Caminos": "Cuatro Cam.",
            "Plaza de Castilla": "Plaza Cast.",
            "Escuelas Aguirre": "Escuelas A.",
        }
        labels = {
            node: _label_replacements.get(
                str(attrs.get("sensor_name", node)),
                str(attrs.get("sensor_name", node)),
            )
            for node, attrs in G.nodes(data=True)
        }
        text_items = nx.draw_networkx_labels(
            G, pos, labels=labels, font_size=8, font_weight="medium", ax=ax
        )
        # adds a white outline, so text labels are readable over nodes and edges
        for text in text_items.values():
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

        community_handles = [
            Line2D([0], [0], marker="o", color="w",
                   label="Unknown" if cid == -1 else f"Community {cid}",
                   markerfacecolor=cmap(cid / max(1, n_communities - 1)),
                   markeredgecolor="black", markersize=10)
            for cid in unique_communities
        ]
        community_legend = ax.legend(
            handles=community_handles,
            title="Louvain communities",
            loc="upper right",
            frameon=True,
        )
        ax.add_artist(community_legend)

        size_handles = [
            Line2D([0], [0], marker="o", color="w", label="Higher betweenness",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=14),
            Line2D([0], [0], marker="o", color="w", label="Lower betweenness",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=7),
        ]
        ax.legend(handles=size_handles, title="Node size", loc="lower right", frameon=True)

        ax.set_title(
            f"Correlation Network: {self.magnitude}  |  {graph_name}\n"
            "Node colour = community · Node size = betweenness · Edge width = correlation",
            fontsize=13, pad=12,
        )
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"correlation_network_{self.magnitude}_{graph_name}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

    def save_centrality_ranking(self, metrics_df, graph_name):
        """ Save ranking plots for the main behavioral-network centrality measures.
        
        The figure compares sensors by:
         - betweenness centrality,
         - closeness centrality,
         - eigenvector centrality.
         
         This is a visualization to show which sensors are structurally important in the pollutant-similarity network"""
        
        sub = metrics_df[metrics_df["graph_name"] == graph_name].copy()
        if sub.empty or sub["betweenness_centrality"].isna().all():
            return

        metric_cols = [
            ("betweenness_centrality", "Betweenness", "#D62728"),
            ("closeness_centrality",   "Closeness",   "#1F77B4"),
            ("eigenvector_centrality", "Eigenvector", "#2CA02C"),
        ]

        _rep = {
            "Urbanización Embajada": "Embajada",
            "Ensanche de Vallecas": "Ensanche",
            "Plaza de España": "Plaza España",
            "Cuatro Caminos": "Cuatro Cam.",
        }
        sub["sensor_name"] = sub["sensor_name"].replace(_rep)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        for ax, (col, label, color) in zip(axes, metric_cols):
            ranked = sub.nlargest(len(sub), col)[["sensor_name", col]]
            bars = ax.barh(ranked["sensor_name"], ranked[col], color=color, alpha=0.82)
            ax.invert_yaxis()
            ax.set_xlabel(f"{label} centrality")
            ax.set_title(f"{label}\n({graph_name})", fontsize=10)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)
            ax.grid(axis="x", alpha=0.25)

        fig.suptitle(
            f"Sensor Centrality Rankings — Correlation Network ({self.magnitude})\n"
            "Which stations dominate the behavioural similarity network?",
            fontsize=12, fontweight="medium",
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"centrality_ranking_{self.magnitude}_{graph_name}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved centrality ranking: {self.magnitude} {graph_name}")

    def save_degree_distribution(self, G, graph_name):
        """ Save the degree distribution for one correlation graph as CSV and plot"""
        degrees = [d for _, d in G.degree()]
        if not degrees:
            return

        degree_counts = (
            pd.Series(degrees).value_counts().sort_index().reset_index()
        )
        degree_counts.columns = ["degree", "n_sensors"]
        degree_counts.to_csv(
            TABLES_DIR / f"degree_distribution_{self.magnitude}_{graph_name}.csv",
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.bar(degree_counts["degree"].astype(str), degree_counts["n_sensors"])
        plt.title(f"Degree Distribution: {self.magnitude}, {graph_name}")
        plt.xlabel("Degree")
        plt.ylabel("Number of Sensors")
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"degree_distribution_{self.magnitude}_{graph_name}.png",
            dpi=300,
        )
        plt.close()

    def save_threshold_comparison_plots(self, summary_df):
        """ Compare graph structure across correlation thresholds.
        
        The saved plots show how density, number of connected components, and
        average degree change as the correlation threshold becomes stricter"""

        if summary_df.empty:
            return

        labels = summary_df["graph_name"]

        for metric, ylabel, title_suffix in [
            ("density", "Density", "Density"),
            ("n_connected_components", "# Components", "Connected Components"),
            ("average_degree", "Average Degree", "Average Degree"),
        ]:
            if metric not in summary_df.columns:
                continue
            plt.figure(figsize=(10, 5))
            plt.bar(labels, summary_df[metric])
            plt.title(
                f"Correlation Network {title_suffix} Across Thresholds ({self.magnitude})"
            )
            plt.ylabel(ylabel)
            plt.xlabel("Correlation Threshold")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                FIGURES_DIR / f"threshold_{metric}_{self.magnitude}.png",
                dpi=300,
            )
            plt.close()

    def compare_with_spatial_network(self, summary_df):
        """ Compare the selected correlation graph with Task 5's spatial graph.

        The method reads Task 5's selected spatial-network summary and places it
        beside one representative Task 6 correlation graph, allowing direct
        comparison of network size, density, connectivity, and component structure"""
        task5_path = (
            TABLES_DIR.parent.parent
            / "task5_spatial_network"
            / "tables"
            / "selected_spatial_graph_summary.csv"
        )
        if not task5_path.exists():
            print("  Task 5 selected graph summary not found — skipping spatial comparison.")
            return pd.DataFrame()

        spatial_summary = pd.read_csv(task5_path)

        # uses the strongest connected correlation graph when available
        # otherwise uses the graph with the largest connected component
        connected = summary_df[summary_df["is_connected"] == True]
        if not connected.empty:
            selected_corr = connected.sort_values("threshold", ascending=False).head(1)
        else:
            selected_corr = summary_df.sort_values(
                "largest_component_size", ascending=False
            ).head(1)

        s = spatial_summary.iloc[0].to_dict()
        c = selected_corr.iloc[0].to_dict()

        comparison = pd.DataFrame([
            {
                "network_type": "spatial_distance",
                "selected_graph": s.get("graph_name"),
                "n_nodes": s.get("n_nodes"),
                "n_edges": s.get("n_edges"),
                "density": s.get("density"),
                "average_degree": s.get("average_degree"),
                "n_connected_components": s.get("n_connected_components"),
                "largest_component_size": s.get("largest_component_size"),
            },
            {
                "network_type": "correlation_behaviour",
                "selected_graph": c.get("graph_name"),
                "n_nodes": c.get("n_nodes"),
                "n_edges": c.get("n_edges"),
                "density": c.get("density"),
                "average_degree": c.get("average_degree"),
                "n_connected_components": c.get("n_connected_components"),
                "largest_component_size": c.get("largest_component_size"),
            },
        ])
        comparison.to_csv(
            TABLES_DIR / f"spatial_vs_correlation_comparison_{self.magnitude}.csv",
            index=False,
        )
        return comparison

    def run(self, thresholds=None):
        """Execute the full correlation-network workflow for one pollutant.
        
        Returns a dict containing monthly series, correlation matrix, graph
        summaries, node metrics, community assignments, and spatial comparison"""
        
        # uses a default range of increasingly strict correlation cut-offs unless provided.
        if thresholds is None:
            thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

        print(f"  Preparing monthly series for {self.magnitude}…")
        monthly = self.prepare_monthly_sensor_series()

        print("  Building sensor-month matrix…")
        matrix, label_map = self.build_sensor_matrix(monthly)

        # at least two sensors are required to compute any pairwise correlation
        if matrix.shape[1] < 2:
            print(f"  Not enough sensors for {self.magnitude}. Skipping.")
            return {}

        print("  Computing correlation matrix…")
        corr = self.compute_correlation_matrix(matrix)

        print("  Building pairwise edge table…")
        pairs = self.correlation_pairs(corr, label_map)

        all_summaries, all_metrics, all_communities = [], [], []

        print(f"  Building graphs for thresholds: {thresholds}")
        for threshold in thresholds:
            graph_name = f"corr_{threshold:.2f}"
            G = self.build_threshold_graph(label_map, pairs, threshold)

            summary = self.graph_summary(G, graph_name)
            all_summaries.append(summary)

            metrics = self.node_metrics(G, graph_name)
            all_metrics.append(metrics)

            comm_df, _, _ = self.detect_communities(G, graph_name)
            all_communities.append(comm_df)

            self.save_graph_plot(G, graph_name, comm_df, metrics)
            self.save_degree_distribution(G, graph_name)
            nx.write_graphml(
                G,
                GRAPHS_DIR / f"correlation_network_{self.magnitude}_{graph_name}.graphml",
            )

        summary_df = pd.DataFrame(all_summaries)
        metrics_df = pd.concat(all_metrics, ignore_index=True)
        communities_df = pd.concat(all_communities, ignore_index=True)

        summary_df.to_csv(
            TABLES_DIR / f"correlation_graph_summary_{self.magnitude}.csv", index=False
        )
        metrics_df.to_csv(
            TABLES_DIR / f"correlation_node_metrics_{self.magnitude}.csv", index=False
        )
        communities_df.to_csv(
            TABLES_DIR / f"correlation_communities_{self.magnitude}.csv", index=False
        )

        self.save_threshold_comparison_plots(summary_df)

        # centrality ranking for the most connected graph (lowest threshold)
        best_graph_name = summary_df.sort_values("threshold").iloc[0]["graph_name"]
        self.save_centrality_ranking(metrics_df, best_graph_name)     

        print("  Comparing with Task 5 spatial network…")
        spatial_comparison = self.compare_with_spatial_network(summary_df)

        return {
            "monthly": monthly,
            "matrix": matrix,
            "correlation_matrix": corr,
            "correlation_pairs": pairs,
            "graph_summary": summary_df,
            "node_metrics": metrics_df,
            "communities": communities_df,
            "spatial_comparison": spatial_comparison,
        }


# plots to compare across pollutants

def _save_cross_pollutant_plots(all_summaries: dict):
    """ Compare representative correlation-network structure across pollutants.
    
    For each pollutant, the function uses the lowest tested threshold and saves
    cross-pollutant plots of density, average degree, and total edge count """

    rows = []
    for pollutant, results in all_summaries.items():
        summary_df = results.get("graph_summary", pd.DataFrame())
        if summary_df.empty:
            continue
        # uses the lowest threshold for each pollutant so all networks are compared at
        # their least restrictive tested similarity level
        row = summary_df.sort_values("threshold").iloc[0].to_dict()
        row["pollutant"] = pollutant
        rows.append(row)

    if not rows:
        return

    combined = pd.DataFrame(rows)
    combined.to_csv(
        TABLES_DIR / "correlation_network_comparison_all_pollutants.csv",
        index=False,
    )

    for metric, ylabel, title in [
        ("density", "Density", "Correlation Network Density by Pollutant"),
        ("average_degree", "Average Degree", "Average Degree by Pollutant"),
        ("n_edges", "Number of Edges", "Number of Correlation Edges by Pollutant"),
    ]:
        if metric not in combined.columns:
            continue
        plt.figure(figsize=(9, 5))
        plt.bar(combined["pollutant"], combined[metric], color="#4C72B0")
        plt.title(title)
        plt.xlabel("Pollutant")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"cross_pollutant_{metric}.png", dpi=300
        )
        plt.close()


def _save_joint_network_overlay(all_results: dict, spatial_communities_path=None):
    """ Draw both the spatial (Task 5 kNN) and correlation networks on the same
    UTM coordinate canvas for one representative pollutant (NO2 is preferred).

    Edge colours:
      - Blue dashed  = spatial only (geographically close, behaviourally different)
      - Red solid    = correlation only (behaviourally similar, geographically distant)
      - Green solid  = shared (both geographically close AND behaviourally similar)

    This is the key comparative visualisation: edges that appear in only one
    network reveal where geography and behaviour disagree, which hints at
    shared pollution sources or wind corridors that cross spatial boundaries.

    Requires: - Task 5 selected kNN graphml saved at outputs/task5_spatial_network/graphs/
              - Sensor UTM coordinates in Task 5 tables"""
    import os
    from pathlib import Path
    
    # loads the selected Task 5 spatial graph used as the geographic reference.
    task5_graphs_dir = TABLES_DIR.parent.parent / "task5_spatial_network" / "graphs"
    task5_tables_dir = TABLES_DIR.parent.parent / "task5_spatial_network" / "tables"

    # finds selected spatial graph
    selected_files = list(task5_graphs_dir.glob("selected_*.graphml"))
    if not selected_files:
        print("  Joint overlay skipped: Task 5 graphml not found")
        return

    try:
        G_spatial = nx.read_graphml(str(selected_files[0]))
    except Exception as e:
        print(f"  Joint overlay skipped: could not read spatial graph ({e}).")
        return

    # prefers NO2 correlation graph at threshold 0.60
    corr_pollutant = None
    corr_G = None
    for pref_pol in ["NO2", "NOX", "NO"]:
        results = all_results.get(pref_pol, {})
        summary_df = results.get("graph_summary", pd.DataFrame())
        if summary_df.empty:
            continue
        # picks threshold closest to 0.60
        row = summary_df.iloc[(summary_df["threshold"] - 0.60).abs().argsort()[:1]]
        graph_name = row.iloc[0]["graph_name"]
        graphml_path = (
            GRAPHS_DIR / f"correlation_network_{pref_pol}_{graph_name}.graphml"
        )
        if graphml_path.exists():
            corr_G = nx.read_graphml(str(graphml_path))
            corr_pollutant = pref_pol
            corr_threshold = row.iloc[0]["threshold"]
            break

    if corr_G is None:
        print("  Joint overlay skipped: no correlation graphml found.")
        return

    # loads UTM coordinates
    coords_path = task5_tables_dir / "sensor_coordinates.csv"
    if not coords_path.exists():
        print("  Joint overlay skipped: sensor_coordinates.csv not found.")
        return

    coords_df = pd.read_csv(coords_path)
    coords_df["sensor_id"] = coords_df["sensor_id"].astype(str)

    pos = {
        str(row["sensor_id"]): (row["utm_x"], row["utm_y"])
        for _, row in coords_df.iterrows()
    }

    # builds edge sets (string node IDs for cross-graph comparison)
    def edge_set(G):
        return {tuple(sorted([str(u), str(v)])) for u, v in G.edges()}

    spatial_edges = edge_set(G_spatial)
    corr_edges    = edge_set(corr_G)

    shared_edges  = spatial_edges & corr_edges
    spatial_only  = spatial_edges - corr_edges
    corr_only     = corr_edges - spatial_edges

    fig, ax = plt.subplots(figsize=(13, 10))

    def draw_edges(edge_set_pairs, color, style, width, alpha, label):
        for u, v in edge_set_pairs:
            if u in pos and v in pos:
                x_vals = [pos[u][0], pos[v][0]]
                y_vals = [pos[u][1], pos[v][1]]
                ax.plot(x_vals, y_vals, color=color, linestyle=style,
                        linewidth=width, alpha=alpha)
        # dummy handle for legend
        return plt.Line2D([0], [0], color=color, linestyle=style,
                          linewidth=width, label=label)

    h1 = draw_edges(spatial_only, "#1F77B4", "--", 1.2, 0.45,
                    f"Spatial only ({len(spatial_only)})")
    h2 = draw_edges(corr_only,    "#D62728", "-",  1.5, 0.55,
                    f"Correlation only ({len(corr_only)})")
    h3 = draw_edges(shared_edges, "#2CA02C", "-",  2.2, 0.80,
                    f"Shared ({len(shared_edges)})")

    # draws nodes (all sensors present in either graph)
    all_nodes = set(G_spatial.nodes()) | set(corr_G.nodes())
    node_pos_list = [(str(n), pos[str(n)]) for n in all_nodes if str(n) in pos]

    xs = [p[0] for _, p in node_pos_list]
    ys = [p[1] for _, p in node_pos_list]
    ax.scatter(xs, ys, s=200, color="#555555", edgecolors="black",
               linewidths=0.8, zorder=5)

    # labels
    for sensor_id, (x, y) in node_pos_list:
        # get sensor name from either graph
        name = str(sensor_id)
        if sensor_id in G_spatial.nodes(data=True):
            name = G_spatial.nodes[sensor_id].get("sensor_name", sensor_id)
        short_names = {
            "Barajas Pueblo": "Barajas", "Urbanización Embajada": "Embajada",
            "Ensanche de Vallecas": "Ensanche", "Plaza de España": "Plaza España",
            "Cuatro Caminos": "Cuatro Cam.", "Plaza de Castilla": "Plaza Cast.",
            "Escuelas Aguirre": "Escuelas A.",
        }
        label = short_names.get(str(name), str(name))
        text = ax.text(x, y + 150, label, fontsize=7.5, ha="center", va="bottom")
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ])

    ax.legend(handles=[h1, h2, h3], fontsize=10, loc="upper right", frameon=True,
              title="Edge type")

    ax.set_title(
        f"Spatial vs Correlation Network — Shared and Exclusive Edges\n"
        f"Spatial: {selected_files[0].stem}  |  "
        f"Correlation: {corr_pollutant} @ r ≥ {corr_threshold:.2f}",
        fontsize=12,
    )
    ax.set_xlabel("UTM X")
    ax.set_ylabel("UTM Y")
    ax.axis("equal")
    ax.grid(alpha=0.12)
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "joint_spatial_correlation_overlay.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close()
    print(
        f"  Saved joint overlay: {len(shared_edges)} shared, "
        f"{len(spatial_only)} spatial-only, {len(corr_only)} correlation-only edges."
    )

    # saves summary table
    overlay_summary = pd.DataFrame([{
        "spatial_graph": selected_files[0].stem,
        "correlation_graph": f"{corr_pollutant}_corr_{corr_threshold:.2f}",
        "n_spatial_edges": len(spatial_edges),
        "n_correlation_edges": len(corr_edges),
        "n_shared_edges": len(shared_edges),
        "n_spatial_only": len(spatial_only),
        "n_correlation_only": len(corr_only),
        "jaccard_edge_similarity": len(shared_edges) / max(1, len(spatial_edges | corr_edges)),
    }])
    overlay_summary.to_csv(TABLES_DIR / "joint_network_overlay_summary.csv", index=False)
    print(f"  Jaccard edge similarity: {overlay_summary['jaccard_edge_similarity'].iloc[0]:.3f}")


def _compute_jaccard_community_similarity(all_results: dict):
    """ Compare Task 5 spatial Louvain communities with Task 6 correlation
    communities using the Jaccard similarity index.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    We compute pairwise Jaccard between every spatial community and every
    correlation community, then report the best-match average as a single
    score. A high score means geographic clusters = behavioural clusters"""

    task5_tables_dir = TABLES_DIR.parent.parent / "task5_spatial_network" / "tables"
    spatial_comm_path = task5_tables_dir / "selected_spatial_communities.csv"

    if not spatial_comm_path.exists():
        print("  Jaccard community similarity skipped: Task 5 communities not found.")
        return

    spatial_comm_df = pd.read_csv(spatial_comm_path)
    spatial_comm_df["sensor_id"] = spatial_comm_df["sensor_id"].astype(str)

    rows = []

    for pollutant, results in all_results.items():
        comm_df = results.get("communities", pd.DataFrame())
        if comm_df.empty:
            continue

        # uses the most connected graph (lowest threshold)
        summary_df = results.get("graph_summary", pd.DataFrame())
        if summary_df.empty:
            continue
        best_graph = summary_df.sort_values("threshold").iloc[0]["graph_name"]
        corr_comm = comm_df[comm_df["graph_name"] == best_graph].copy()
        corr_comm["sensor_id"] = corr_comm["sensor_id"].astype(str)

        # builds community sets
        spatial_sets = {
            cid: set(grp["sensor_id"])
            for cid, grp in spatial_comm_df.groupby("community_id")
        }
        corr_sets = {
            cid: set(grp["sensor_id"])
            for cid, grp in corr_comm.groupby("community_id")
        }

        if not spatial_sets or not corr_sets:
            continue

        # best-match Jaccard: for each spatial community find best-matching corr community
        best_matches = []
        for s_id, s_set in spatial_sets.items():
            best_j = max(
                len(s_set & c_set) / max(1, len(s_set | c_set))
                for c_set in corr_sets.values()
            )
            best_matches.append(best_j)

        mean_jaccard = float(np.mean(best_matches))
        rows.append({
            "pollutant": pollutant,
            "threshold": best_graph,
            "n_spatial_communities": len(spatial_sets),
            "n_corr_communities": len(corr_sets),
            "mean_best_match_jaccard": round(mean_jaccard, 4),
            "interpretation": (
                "high agreement" if mean_jaccard >= 0.5
                else "moderate agreement" if mean_jaccard >= 0.3
                else "low agreement"
            ),
        })

    if not rows:
        return

    jaccard_df = pd.DataFrame(rows)
    jaccard_df.to_csv(TABLES_DIR / "community_jaccard_similarity.csv", index=False)
    print("\nCommunity Jaccard Similarity (spatial vs correlation):")
    print(jaccard_df.to_string(index=False))
    print(
        "\n  A high Jaccard score means spatially close sensors also behave\n"
        "  similarly. A low score suggests shared pollution sources or\n"
        "  wind corridors that cross geographic boundaries.\n"
    )

    # bar chart
    if len(jaccard_df) > 1:
        plt.figure(figsize=(8, 4))
        colors = [
            "#2CA02C" if j >= 0.5 else ("#FF7F0E" if j >= 0.3 else "#D62728")
            for j in jaccard_df["mean_best_match_jaccard"]
        ]
        bars = plt.bar(jaccard_df["pollutant"], jaccard_df["mean_best_match_jaccard"],
                       color=colors)
        plt.axhline(0.5, color="green", linestyle="--", linewidth=1, label="High agreement (0.5)")
        plt.axhline(0.3, color="orange", linestyle="--", linewidth=1, label="Moderate (0.3)")
        plt.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        plt.ylim(0, 1.05)
        plt.xlabel("Pollutant")
        plt.ylabel("Mean best-match Jaccard")
        plt.title(
            "Community Agreement: Spatial vs Correlation Network\n"
            "Do geographically close sensors also behave similarly?"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "community_jaccard_similarity.png", dpi=300)
        plt.close()


@timer
def run_task6(
    df,
    pollutants=None,
    thresholds=None,
    min_months=12,):
    """Run the full Task 6 correlation-network analysis pipeline.

    Args:
        df: cleaned DataFrame
        pollutants: optional list of pollutant names to analyze. If None,
            defaults to ["NO2", "NO", "NOX", "O3", "<PM10"]
        thresholds: optional list of correlation thresholds used to build graph
            variants. If None, defaults to [0.50, 0.60, 0.70, 0.80, 0.90]
        min_months: minimum number of monthly observations required per sensor
            and per sensor-pair correlation.
    Returns:
        A dictionary keyed by pollutant name. Each entry contains monthly series,
        correlation matrices, graph summaries, community outputs, and the
        spatial-comparison table for that pollutant """
    
    print("\n--- Task 6: Correlation Network ---")

    if pollutants is None:
        pollutants = ["NO2", "NO", "NOX", "O3", "<PM10"]

    if thresholds is None:
        thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    all_results = {}

    for pollutant in pollutants:
        print(f"\nBuilding correlation network for {pollutant}…")
        try:
            analyzer = CorrelationNetworkAnalyzer(
                df, magnitude=pollutant, min_months=min_months
            )
            results = analyzer.run(thresholds=thresholds)
            if results:
                all_results[pollutant] = results
        except ValueError as e:
            print(f"  Skipping {pollutant}: {e}")
            continue

    print("\nSaving cross-pollutant comparison plots…")
    _save_cross_pollutant_plots(all_results)

    print("\nSaving joint spatial + correlation overlay…")       
    _save_joint_network_overlay(all_results)

    print("\nComputing Jaccard community similarity…")           
    _compute_jaccard_community_similarity(all_results)

    print("Task 6 completed")
    return all_results
