"""Task 5 pipeline for building and analysing a spatial sensor network.

This file:
- extracts sensor UTM coordinates and computes pairwise distances,
- builds distance-threshold and k-nearest-neighbor graphs using NetworkX,
- computes node-level centrality metrics (degree, betweenness, closeness, eigenvector),
- detects spatial communities using the Louvain algorithm,
- selects the smallest connected kNN graph as the recommended network,
- saves summary tables, graph files, and figures for interpretation.
Sensors are nodes, spatial proximity defines edges, and UTM distance is used as an edge weight for centrality and shortest-path computations."""

# used for numerical and tabular analysis
import numpy as np
import pandas as pd

# used for plotting/visualizations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# used for custom legend handles and readable node labels
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# used for network creation and analysis
import networkx as nx

# project-specific output directories and timing decorator
from utils.config import get_task_dirs
from utils.helpers import timer

# output folders dedicated to Task 5 spatial-network tables, figures, and graph files
TABLES_DIR, FIGURES_DIR, GRAPHS_DIR, MODELS_DIR = get_task_dirs("task5_spatial_network")

class SpatialNetworkAnalyzer:
    """ Build and analyze spatial networks of Madrid monitoring sensors.
    
    Sensors are represented as graph nodes, while edges are created from geographic proximity using either:
    - distance thresholds,
    - k-nearest-neighbor rules.

    The analyzer compares graph structures, computes node centrality,
    detects spatial communities, selects a representative connected graph,
    and produces figures for interpretation and reporting"""

    def __init__(self, df):
        """ Initialize the spatial-network analysis.
        
        Args:
           df: Cleaned DataFrame containing sensor identifiers, coordinates, pollutant names, and observed values"""
        self.df = df

        # pre-computes one average pollution value per sensor so selected network figures can visually compare spatial position and pollution intensity
        self._sensor_avg_pollution = self._compute_sensor_avg_pollution()

    def _compute_sensor_avg_pollution(self):
        """ Compute one representative average pollutant value per sensor.
        
        The method prioritizes NO2, then NOX, then NO. The first pollutant available
        in the dataset is used to color nodes in the pollution-based network figure.
        Returns:
        A dictionary mapping sensor IDs to mean pollutant values"""

        # uses pollutants that are central to the project's air-quality analysis
        pollutant_candidates = ["NO2", "NOX", "NO"]
        df = self.df.copy()
        # normalizes pollutant names once so matching is case-insensitive
        mag_upper = df["magnitude_name"].astype(str).str.upper()
        
        # uses the first available pollutant from the priority list above
        for pol in pollutant_candidates:
            sub = df[mag_upper == pol]
            if sub.empty:
                continue
            val_col = "value"
            result = (
                sub.groupby("sensor_id", observed=True)[val_col]
                .mean()
                .to_dict()
            )
            # stores the pollutant label so the plot colorbar can explain what is being shown
            self._pollution_label = pol
            return {int(k): float(v) for k, v in result.items()}

        self._pollution_label = "pollutant"
        return {}

    def get_sensor_coordinates(self):
        """ Extract one average coordinate pair for each monitoring sensor.
        
        Sensors appear repeatedly in the long-format dataset because they have many
        observations over time and across variables. This method collapses those
        repeated rows into one spatial record per sensor.

        Returns: DataFrame with sensor ID, sensor name, and average UTM coordinates"""
        # removes rows that cannot contribute to a spatial network
        coords = (
            self.df
            .dropna(subset=["sensor_id", "sensor_name", "utm_x", "utm_y"])
            .groupby(["sensor_id", "sensor_name"], observed=True)[["utm_x", "utm_y"]]
            .mean()
            .reset_index()
        )

        coords["sensor_id"] = coords["sensor_id"].astype(int)
        # saves the one-row-per-sensor coordinate table for reuse and auditing
        coords.to_csv(TABLES_DIR / "sensor_coordinates.csv", index=False)
        return coords

    @staticmethod
    def euclidean_distance(row_a, row_b):
        """ Compute straight-line distance between two sensors in UTM space.
        
        UTM coordinates are projected coordinates, so the resulting Euclidean
        distance is approximately measured in meters"""
        
        dx = row_a["utm_x"] - row_b["utm_x"]
        dy = row_a["utm_y"] - row_b["utm_y"]
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def pairwise_distances(self, coords):
        """ Compute all unique pairwise spatial distances between sensors.
        
        Each unordered sensor pair is stored only once, with`sensor_id_1 < sensor_id_2`.
        Args:
            coords: one-row-per-sensor coordinate table.
        Returns:
            A DataFrame of pairwise sensor distances sorted from nearest to farthest"""
        rows = []

        for _, a in coords.iterrows():
            for _, b in coords.iterrows():
                # skips self-pairs and duplicate pair orderings such as (A, B) and (B, A) and so on
                if int(a["sensor_id"]) >= int(b["sensor_id"]):
                    continue

                distance = self.euclidean_distance(a, b)

                rows.append({
                    "sensor_id_1": int(a["sensor_id"]),
                    "sensor_name_1": str(a["sensor_name"]),
                    "sensor_id_2": int(b["sensor_id"]),
                    "sensor_name_2": str(b["sensor_name"]),
                    "distance_m": distance,
                })

        distances = pd.DataFrame(rows).sort_values("distance_m")
        # saves the distance table for both threshold and kNN graph construction
        distances.to_csv(TABLES_DIR / "pairwise_sensor_distances.csv", index=False)

        return distances

    def add_nodes(self, G, coords):
        """ Add sensor nodes and attach spatial and pollution attributes."""
        for _, row in coords.iterrows():
            sid = int(row["sensor_id"])
            # stores attributes directly on each node so later metrics/plots can reuse them
            G.add_node(
                sid,
                sensor_name=str(row["sensor_name"]),
                utm_x=float(row["utm_x"]),
                utm_y=float(row["utm_y"]),
                avg_pollution=self._sensor_avg_pollution.get(sid, float("nan")),
            )

        return G

    def add_edge_with_distance(self, G, u, v, distance):
        """ Add an undirected edge with distance and proximity weights.
        
        The edges:
        - `distance`: geographic separation in meters, used as a shortest-path cost,
        - `similarity`: inverse-distance weight, used where stronger nearby links
        should receive larger importance, such as community detection"""
        G.add_edge(
            int(u),
            int(v),
            distance=float(distance),
            similarity=float(1 / (1 + distance)),
        )

        return G

    def build_threshold_graph(self, coords, distances, threshold_m):
        """Build a graph connecting sensors within a fixed distance threshold.
        
        Args: coords: one-row-per-sensor coordinate table.
              distances: pairwise sensor distance table.
              threshold_m: maximum allowed edge distance in meters.
        Returns: NetworkX undirected graph """

        G = nx.Graph()
        G = self.add_nodes(G, coords)
        
        # keeps only the sensor pairs close enough to satisfy the threshold rule
        selected_edges = distances[distances["distance_m"] <= threshold_m]

        for _, row in selected_edges.iterrows():
            self.add_edge_with_distance(
                G,
                row["sensor_id_1"],
                row["sensor_id_2"],
                row["distance_m"],
            )

        G.graph["method"] = "distance_threshold"
        G.graph["threshold_m"] = threshold_m

        return G

    def build_knn_graph(self, coords, distances, k):
        """ Build an undirected k-nearest-neighbor spatial graph. 
        
        Each sensor is connected to its k closest neighbors by distance.
        Since the graph is undirected, if sensor A lists B as a neighbor, the edge 
        is shared regardless of whether B lists A

        Args: coords: one-row-per-sensor coordinate table.
              distances: pairwise sensor distance table.
              k: number of nearest neighbors requested per sensor.
        Returns: NetworkX undirected graph"""

        G = nx.Graph()
        G = self.add_nodes(G, coords)
        # dictionary, sorted sensor pairs used as keys, to avoid duplicates
        edges_to_add = {}
        sensors_with_fewer_neighbors = []

        for sensor_id in coords["sensor_id"]:
            # retrieves all candidate distances involving this sensor
            sensor_edges = distances[
                (distances["sensor_id_1"] == sensor_id)
                | (distances["sensor_id_2"] == sensor_id)
            ].copy()
            
            # keeps only the "k" geographically closest candidate neighbors
            sensor_edges = sensor_edges.sort_values("distance_m")

            if len(sensor_edges) < k:
                sensors_with_fewer_neighbors.append({
                    "sensor_id": sensor_id,
                    "available_neighbors": len(sensor_edges),
                    "requested_k": k,
                })

            sensor_edges = sensor_edges.head(k)

            for _, row in sensor_edges.iterrows():
                u = int(row["sensor_id_1"])
                v = int(row["sensor_id_2"])
                edge_key = tuple(sorted((u, v)))
                edges_to_add[edge_key] = row["distance_m"]

        for (u, v), distance in edges_to_add.items():
            self.add_edge_with_distance(G, u, v, distance)

        G.graph["method"] = "knn"
        G.graph["k"] = k
        
        # track rare cases: where a sensor has fewer available neighbors than needed
        if sensors_with_fewer_neighbors:
            print(
                f"  Warning: {len(sensors_with_fewer_neighbors)} sensor(s) "
                f"have fewer than k={k} neighbors available."
            )
            pd.DataFrame(sensors_with_fewer_neighbors).to_csv(
                TABLES_DIR / f"knn_k{k}_underfull_sensors.csv",
                index=False,
            )

        return G

    def graph_summary(self, G, graph_name):
        """ Compute graph-level statistics summary for one spatial network.
        
        The summary includes:
         - number of nodes and edges,
         - density and degree statistics
         - connected-component structure,
         - clustering coefficient,
         - average shortest-path distance on the largest connected component"""
        
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # handles empty graphs, so later tables remain well-formed
        if n_nodes == 0:
            return {
                "graph_name": graph_name,
                "method": G.graph.get("method"),
                "threshold_m": G.graph.get("threshold_m", np.nan),
                "k": G.graph.get("k", np.nan),
                "n_nodes": 0,
                "n_edges": 0,
            }

        degrees = dict(G.degree())
        components = list(nx.connected_components(G))

        if components:
            largest_component = max(components, key=len)
            largest_subgraph = G.subgraph(largest_component).copy()
        else:
            largest_component = set()
            largest_subgraph = G.copy()

        is_connected = nx.is_connected(G) if n_nodes > 0 else False

        # shortest-path statistics are only meaningful on a connected component containing at least two sensors
        if largest_subgraph.number_of_nodes() > 1:
            # the weighted shortest path uses geographic distance as travel cost
            average_shortest_path = nx.average_shortest_path_length(
                largest_subgraph,
                weight="distance",
            )
        else:
            average_shortest_path = np.nan

        summary = {
            "graph_name": graph_name,
            "method": G.graph.get("method"),
            "threshold_m": G.graph.get("threshold_m", np.nan),
            "k": G.graph.get("k", np.nan),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": nx.density(G),
            "average_degree": np.mean(list(degrees.values())) if degrees else np.nan,
            "min_degree": min(degrees.values()) if degrees else np.nan,
            "max_degree": max(degrees.values()) if degrees else np.nan,
            "is_connected": is_connected,
            "n_connected_components": len(components),
            "largest_component_size": len(largest_component),
            "average_clustering": nx.average_clustering(G) if n_edges > 0 else 0,
            "average_shortest_path_largest_component_m": average_shortest_path,
        }

        return summary

    def node_metrics(self, G, graph_name):
        """ Compute node-level importance measures for one spatial graph.
        
        Metrics are:
        - degree,
        - normalized degree centrality,
        - betweenness centrality,
        - closeness centrality,
        - eigenvector centrality.

        These measures describe whether a station is 
        - locally connected, acts as a bridge, 
        - is spatially accessible,
        - is linked to other important nodes"""
        
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        degree = dict(G.degree())
        degree_centrality = nx.degree_centrality(G)
        # uses distance as a path cost for betweenness and closeness calculations
        if G.number_of_edges() > 0:
            betweenness = nx.betweenness_centrality(
                G,
                weight="distance",
                normalized=True,
            )

            closeness = nx.closeness_centrality(
                G,
                distance="distance",
            )

            # Eigenvector centrality: uses similarity as weight so nearby sensors (high similarity) reinforce each other's score
            try:
                eigenvector = nx.eigenvector_centrality(
                    G,
                    weight="similarity",
                    max_iter=500,
                    tol=1e-6,
                )
            # falls back to PageRank if the iterative eigenvector calculation does not converge
            except nx.PowerIterationFailedConvergence:
                print("  Warning: Eigenvector centrality did not converge; falling back to PageRank.")
                eigenvector = nx.pagerank(G, weight="similarity")
        else:
            betweenness = {node: 0 for node in G.nodes()}
            closeness   = {node: 0 for node in G.nodes()}
            eigenvector = {node: 0 for node in G.nodes()}

        rows = []

        for node, attrs in G.nodes(data=True):
            rows.append({
                "graph_name": graph_name,
                "sensor_id": node,
                "sensor_name": attrs.get("sensor_name"),
                "utm_x": attrs.get("utm_x"),
                "utm_y": attrs.get("utm_y"),
                "avg_pollution": attrs.get("avg_pollution"),
                "degree": degree.get(node),
                "degree_centrality": degree_centrality.get(node),
                "betweenness_centrality": betweenness.get(node),
                "closeness_centrality": closeness.get(node),
                "eigenvector_centrality": eigenvector.get(node),  
            })

        return pd.DataFrame(rows)

    def detect_communities(self, G, graph_name):
        """ Detect spatial sensor communities using the Louvain algorithm.
        
        Communities group sensors that are more densely connected to each other
        than to the rest of the network — likely representing the same urban
        micro-zone. 
        Falls back to greedy modularity if Louvain is unavailable.
        Modularity > 0.3 indicates meaningful cluster structure"""

        # if a graph with has no edges, it has no meaningful multi-node communities
        # consequently, each sensor becomes its own singleton group
        if G.number_of_edges() == 0:
            communities = [{node} for node in G.nodes()]
            modularity = np.nan
            community_algorithm = "none"
        else:
            try:
                communities = nx.community.louvain_communities(
                    G,
                    weight="similarity",
                    seed=42,
                )
                community_algorithm = "louvain"
            except AttributeError:
                communities = list(
                    nx.community.greedy_modularity_communities(
                        G,
                        weight="similarity",
                    )
                )
                community_algorithm = "greedy_modularity"

            modularity = nx.community.modularity(
                G,
                communities,
                weight="similarity",
            )

        rows = []

        for community_id, community in enumerate(communities):
            for node in community:
                rows.append({
                    "graph_name": graph_name,
                    "sensor_id": node,
                    "community_id": community_id,
                    "community_size": len(community),
                    "modularity": modularity,
                    "community_algorithm": community_algorithm,
                })

        return pd.DataFrame(rows), communities, modularity

    def save_graph_plot(self, G, graph_name, communities_df=None, node_metrics_df=None):
        """ Save a spatial network plot colored by different communities
        
        Visual encoding:
        - node color: detected spatial community,
        - node size: betweenness centrality,
        - edge width/opacity: proximity strength derived from distance"""

        if G.number_of_nodes() == 0:
            return
        
        # places each node at its real UTM coordinate so graph geometry matches geography
        pos = {
            node: (attrs["utm_x"], attrs["utm_y"])
            for node, attrs in G.nodes(data=True)
        }

        if communities_df is not None and not communities_df.empty:
            community_map = dict(
                zip(
                    communities_df["sensor_id"].astype(int),
                    communities_df["community_id"].astype(int),
                )
            )
            node_communities = [
                community_map.get(int(node), -1)
                for node in G.nodes()
            ]
            unique_communities = sorted(set(node_communities))
        else:
            node_communities = [0 for _ in G.nodes()]
            unique_communities = [0]

        cmap = plt.cm.Set2
        n_communities = max(1, len(unique_communities))

        # scales node size by betweenness centrality to highlight bridge-like sensors
        if node_metrics_df is not None and not node_metrics_df.empty:
            metric_map = dict(
                zip(
                    node_metrics_df["sensor_id"].astype(int),
                    node_metrics_df["betweenness_centrality"].astype(float),
                )
            )

            centrality_values = np.array(
                [metric_map.get(int(node), 0.0) for node in G.nodes()],
                dtype=float,
            )

            if centrality_values.max() > centrality_values.min():
                node_sizes = (
                    500
                    + 1800
                    * (centrality_values - centrality_values.min())
                    / (centrality_values.max() - centrality_values.min())
                )
            else:
                node_sizes = np.full(len(centrality_values), 900.0)
        else:
            node_sizes = np.full(G.number_of_nodes(), 900.0)

        distances = np.array(
            [data.get("distance", 1.0) for _, _, data in G.edges(data=True)],
            dtype=float,
        )
        
        # makes shorter geographic links appear visually stronger than longer ones
        if len(distances) > 0 and distances.max() > distances.min():
            edge_strength = 1 - (
                (distances - distances.min())
                / (distances.max() - distances.min())
            )
            edge_widths = 0.5 + 1.8 * edge_strength
            edge_alphas = 0.18 + 0.30 * edge_strength
        else:
            edge_widths = np.full(G.number_of_edges(), 1.2)
            edge_alphas = np.full(G.number_of_edges(), 0.35)

        fig, ax = plt.subplots(figsize=(13, 10))

        for (u, v, data), width, alpha in zip(
            G.edges(data=True),
            edge_widths,
            edge_alphas,
        ):
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                ax=ax,
                edge_color="gray",
                width=float(width),
                alpha=float(alpha),
            )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=node_communities,
            cmap=cmap,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=0.9,
        )
        
        # shortens long station names to reduce label overlap in the result figure
        replacements = {
            "Barajas Pueblo": "Barajas",
            "Urbanización Embajada": "Embajada",
            "Ensanche de Vallecas": "Ensanche",
            "Plaza de España": "Plaza España",
            "Ramón y Cajal": "Ramón y Cajal",
            "Cuatro Caminos": "Cuatro Cam.",
            "Plaza de Castilla": "Plaza Cast.",
            "Escuelas Aguirre": "Escuelas A.",
        }

        labels = {}
        for node, attrs in G.nodes(data=True):
            name = str(attrs.get("sensor_name", node))
            labels[node] = replacements.get(name, name)

        # offsets labels slightly upward so they do not sit directly on the node markers
        label_pos = {
            node: (x, y + 150)
            for node, (x, y) in pos.items()
        }
        
        text_items = nx.draw_networkx_labels(
            G,
            label_pos,
            labels=labels,
            font_size=8,
            font_weight="medium",
            ax=ax,
        )
        
        # adds a white text outline to keep labels readable on top of edges and nodes
        for text in text_items.values():
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

        legend_elements = []

        for cid in unique_communities:
            label = "Unknown" if cid == -1 else f"Community {cid}"

            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=cmap(cid / max(1, n_communities - 1)),
                    markeredgecolor="black",
                    markersize=10,
                )
            )

        community_legend = ax.legend(
            handles=legend_elements,
            title="Louvain communities",
            loc="upper right",
            frameon=True,
        )
        ax.add_artist(community_legend)

        size_legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Higher betweenness",
                markerfacecolor="lightgray",
                markeredgecolor="black",
                markersize=14,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Lower betweenness",
                markerfacecolor="lightgray",
                markeredgecolor="black",
                markersize=7,
            ),
        ]

        ax.legend(
            handles=size_legend,
            title="Node size",
            loc="lower right",
            frameon=True,
        )

        ax.set_title(
            f"Spatial Sensor Network: {graph_name}\n"
            "Node color = community, node size = betweenness centrality",
            fontsize=16,
            pad=15,
        )
        ax.set_xlabel("UTM X", fontsize=12)
        ax.set_ylabel("UTM Y", fontsize=12)
        ax.axis("equal")
        ax.grid(alpha=0.15)

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"spatial_network_{graph_name}_community_centrality.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def save_pollution_coloured_plot(self, G, graph_name, node_metrics_df=None):
        """ Save an alternative spatial network plot colored by pollution intensity
        
        Visual encoding:
        - node color: average pollutant concentration for the chosen pollutant,
        - node size: eigenvector centrality,
        - edge width: geographic proximity strength.

        Current figure links network structure with the project's question of which areas appear most affected by pollution"""
        if G.number_of_nodes() == 0:
            return

        pos = {
            node: (attrs["utm_x"], attrs["utm_y"])
            for node, attrs in G.nodes(data=True)
        }

        pollution_values = np.array(
            [G.nodes[n].get("avg_pollution", np.nan) for n in G.nodes()],
            dtype=float,
        )
        valid_mask = ~np.isnan(pollution_values)

        if valid_mask.sum() < 2:
            return  # not enough data to colour by pollution

        # normalises for colourmap
        vmin = np.nanpercentile(pollution_values, 5)
        vmax = np.nanpercentile(pollution_values, 95)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdYlBu_r  # red = high pollution, blue = low

        node_colors = [
            cmap(norm(pollution_values[i])) if valid_mask[i] else "lightgray"
            for i in range(len(pollution_values))
        ]

        # node size = eigenvector centrality
        if node_metrics_df is not None and not node_metrics_df.empty:
            ev_map = dict(zip(
                node_metrics_df["sensor_id"].astype(int),
                node_metrics_df["eigenvector_centrality"].astype(float),
            ))
            ev_vals = np.array(
                [ev_map.get(int(n), 0.0) for n in G.nodes()], dtype=float
            )
            if ev_vals.max() > ev_vals.min():
                node_sizes = 400 + 1600 * (ev_vals - ev_vals.min()) / (ev_vals.max() - ev_vals.min())
            else:
                node_sizes = np.full(len(ev_vals), 700.0)
        else:
            node_sizes = np.full(G.number_of_nodes(), 700.0)

        distances = np.array(
            [data.get("distance", 1.0) for _, _, data in G.edges(data=True)],
            dtype=float,
        )
        edge_widths = 0.8 + 1.4 * (
            1 - (distances - distances.min()) / max(distances.max() - distances.min(), 1)
        ) if len(distances) > 0 else []

        fig, ax = plt.subplots(figsize=(13, 10))

        for (u, v, data), width in zip(G.edges(data=True), edge_widths):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                                   edge_color="gray", width=float(width), alpha=0.25)

        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors,
                               node_size=node_sizes,
                               edgecolors="black", linewidths=0.9)

        replacements = {
            "Barajas Pueblo": "Barajas", "Urbanización Embajada": "Embajada",
            "Ensanche de Vallecas": "Ensanche", "Plaza de España": "Plaza España",
            "Cuatro Caminos": "Cuatro Cam.", "Plaza de Castilla": "Plaza Cast.",
            "Escuelas Aguirre": "Escuelas A.",
        }
        labels = {n: replacements.get(str(attrs.get("sensor_name", n)), str(attrs.get("sensor_name", n)))
                  for n, attrs in G.nodes(data=True)}
        label_pos = {n: (x, y + 150) for n, (x, y) in pos.items()}
        text_items = nx.draw_networkx_labels(G, label_pos, labels=labels,
                                              font_size=8, font_weight="medium", ax=ax)
        for text in text_items.values():
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground="white"),
                path_effects.Normal(),
            ])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(f"Average {self._pollution_label} (µg/m³)", fontsize=10)

        size_legend = [
            Line2D([0],[0], marker="o", color="w", label="Higher eigenvector centrality",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=14),
            Line2D([0],[0], marker="o", color="w", label="Lower eigenvector centrality",
                   markerfacecolor="lightgray", markeredgecolor="black", markersize=7),
        ]
        ax.legend(handles=size_legend, title="Node size", loc="lower right", frameon=True)

        ax.set_title(
            f"Spatial Network: {graph_name}\n"
            f"Node colour = avg {self._pollution_label} · Node size = eigenvector centrality",
            fontsize=14, pad=12,
        )
        ax.set_xlabel("UTM X")
        ax.set_ylabel("UTM Y")
        ax.axis("equal")
        ax.grid(alpha=0.15)
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"spatial_network_{graph_name}_pollution_eigenvector.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved pollution-coloured network: {graph_name}")

    def save_centrality_ranking(self, node_metrics_df, graph_name):
        """ Save ranking plots for the main node-centrality measures.
        
        The figure compares sensors by:
            - betweenness centrality,
            - closeness centrality,
            - eigenvector centrality.
            
         Provides a direct visual summary of which stations are structurally 
         important under different network interpretations"""
        if node_metrics_df.empty:
            return
        # restricts the ranking table to the selected graph currently being plotted
        sub = node_metrics_df[
            node_metrics_df["graph_name"] == graph_name
        ].copy()

        if sub.empty:
            return

        metrics = [
            ("betweenness_centrality", "Betweenness", "#D62728"),
            ("closeness_centrality",   "Closeness",   "#1F77B4"),
            ("eigenvector_centrality", "Eigenvector", "#2CA02C"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))

        # show all sensors, ordered from highest to lowest centrality for each metric
        # shorten the longest station names so labels fit cleanly.
        for ax, (col, label, color) in zip(axes, metrics):
            ranked = sub.nlargest(len(sub), col)[["sensor_name", col]].copy()
            ranked["sensor_name"] = ranked["sensor_name"].str.replace(
                "Urbanización Embajada", "Embajada", regex=False
            ).str.replace("Ensanche de Vallecas", "Ensanche", regex=False)

            bars = ax.barh(ranked["sensor_name"], ranked[col], color=color, alpha=0.82)
            ax.invert_yaxis()
            ax.set_xlabel(f"{label} centrality")
            ax.set_title(f"{label} centrality\n({graph_name})", fontsize=10)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)
            ax.grid(axis="x", alpha=0.25)

        fig.suptitle(
            " Sensor Centrality Rankings — Spatial Network\n"
            " Which Madrid stations are most critical for air-quality monitoring?",
            fontsize=12, fontweight="medium",
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"centrality_ranking_{graph_name}.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()
        print(f"  Saved centrality ranking chart: {graph_name}")

    def save_degree_distribution(self, G, graph_name):
        """ Save the degree distribution of a graph as a bar chart and CSV.
        
        Each bar shows how many sensors have that number of spatial connections, 
        revealing whether the network is uniform or has highly connected hubs"""     

        degrees = [degree for _, degree in G.degree()]
        if not degrees:
            return

        degree_counts = (
            pd.Series(degrees)
            .value_counts()
            .sort_index()
            .reset_index()
        )
        degree_counts.columns = ["degree", "n_sensors"]
        degree_counts.to_csv(
            TABLES_DIR / f"degree_distribution_{graph_name}.csv",
            index=False,
        )

        plt.figure(figsize=(8, 5))
        plt.bar(
            degree_counts["degree"].astype(str),
            degree_counts["n_sensors"],
        )

        plt.title(f"Degree Distribution: {graph_name}")
        plt.xlabel("Degree")
        plt.ylabel("Number of Sensors")
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"degree_distribution_{graph_name}.png",
            dpi=300,
        )
        plt.close()

    def save_comparison_plot(self, summary_df):
        """ Plot density, connected components, and average degree across all graph variants.
        
        Gives direct visual comparison of how the network structure changes
        under different distance thresholds and kNN values"""        
        if summary_df.empty:
            return
        
        # missing summary values are replaced with 0, only for plotting convenience 
        summary_df = summary_df.fillna(0)
        labels = summary_df["graph_name"]

        # generate one comparison plot per graph-level metric.
        for metric, ylabel, title in [
            ("density", "Density", "Network Density Across Spatial Edge Rules"),
            ("n_connected_components", "Number of Components", "Connected Components Across Spatial Edge Rules"),
            ("average_degree", "Average Degree", "Average Degree Across Spatial Edge Rules"),
        ]:
            plt.figure(figsize=(10, 5))
            plt.bar(labels, summary_df[metric])
            plt.title(title)
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"network_{metric}_comparison.png", dpi=300)
            plt.close()

    def run(self, thresholds_m=None, k_values=None):
        """ Execute the full Task 5 spatial-network analysis.
    Args:
        thresholds_m: Optional list of distance thresholds in meters. If None, thresholds are derived from pairwise-distance quantiles.
        k_values: Optional list of k values for k-nearest-neighbor graphs; defaults to [2, 3, 4, 5].

    Returns:
        A dictionary containing coordinates, pairwise distances, graph summaries,
        node metrics, community assignments, and the selected graph object"""
        
        # uses a small range of k values unless the caller supplies alternatives
        if k_values is None:
            k_values = [2, 3, 4, 5]
        
        print("Extracting sensor coordinates...")
        coords = self.get_sensor_coordinates()

        print("Computing pairwise sensor distances...")
        distances = self.pairwise_distances(coords)
    
        distance_quantiles = (
            distances["distance_m"]
            .quantile([0.10, 0.25, 0.50, 0.75])
            .reset_index()
        )

        distance_quantiles.columns = ["quantile", "distance_m"]

        distance_quantiles.to_csv(
            TABLES_DIR / "pairwise_distance_quantiles.csv",
            index=False,
        )

        print("Pairwise distance quantiles:")
        print(distance_quantiles)

        if thresholds_m is None:
            thresholds_m = (
                distance_quantiles["distance_m"]
                .round()
                .astype(int)
                .drop_duplicates()
                .tolist()
            )

        print(f"Using threshold values: {thresholds_m}")

        all_graph_summaries = []
        all_node_metrics = []
        all_community_rows = []

        graphs = {}

        print("Building threshold graphs...")
        for threshold in thresholds_m:
            graph_name = f"threshold_{threshold}m"
            G = self.build_threshold_graph(coords, distances, threshold)
            graphs[graph_name] = G

        print("Building k-nearest-neighbor graphs...")
        for k in k_values:
            graph_name = f"knn_k{k}"
            G = self.build_knn_graph(coords, distances, k)
            graphs[graph_name] = G

        print("Analyzing graphs...")
        for graph_name, G in graphs.items():
            summary = self.graph_summary(G, graph_name)
            all_graph_summaries.append(summary)

            metrics = self.node_metrics(G, graph_name)
            all_node_metrics.append(metrics)

            communities_df, communities, modularity = self.detect_communities(
                G,
                graph_name,
            )
            all_community_rows.append(communities_df)

            self.save_graph_plot(
                G,
                graph_name,
                communities_df=communities_df,
                node_metrics_df=metrics,
            )

            self.save_degree_distribution(G, graph_name)

            nx.write_graphml(G, GRAPHS_DIR / f"{graph_name}.graphml")

        summary_df = pd.DataFrame(all_graph_summaries)

        if all_node_metrics:
            node_metrics_df = pd.concat(all_node_metrics, ignore_index=True)
        else:
            node_metrics_df = pd.DataFrame()

        if all_community_rows:
            communities_df = pd.concat(all_community_rows, ignore_index=True)
        else:
            communities_df = pd.DataFrame()

        print("Saving Task 5 tables...")
        summary_df.to_csv(TABLES_DIR / "spatial_graph_summary.csv", index=False)
        node_metrics_df.to_csv(TABLES_DIR / "spatial_node_centrality.csv", index=False)
        communities_df.to_csv(TABLES_DIR / "spatial_communities.csv", index=False)

        print("Saving comparison figures...")
        self.save_comparison_plot(summary_df)

        # select the smallest connected kNN graph as the recommended network for discussion
        knn_summaries = summary_df[summary_df["method"] == "knn"].copy()
        connected_knn = knn_summaries[knn_summaries["is_connected"] == True]
        
        # falls back to knn_k3 or the first available graph if no connected kNN exists
        if not connected_knn.empty:
            selected_graph_name = connected_knn.sort_values("k").iloc[0]["graph_name"]
        else:
            selected_graph_name = "knn_k3" if "knn_k3" in graphs else list(graphs.keys())[0]

        selected_G = graphs[selected_graph_name]

        selected_summary = summary_df[
            summary_df["graph_name"] == selected_graph_name
        ]

        selected_summary.to_csv(
            TABLES_DIR / "selected_spatial_graph_summary.csv",
            index=False,
        )

        selected_node_metrics = node_metrics_df[
            node_metrics_df["graph_name"] == selected_graph_name
        ].copy()

        selected_node_metrics.to_csv(
            TABLES_DIR / "selected_spatial_node_centrality.csv",
            index=False,
        )

        selected_communities = communities_df[
            communities_df["graph_name"] == selected_graph_name
        ].copy()

        selected_communities.to_csv(
            TABLES_DIR / "selected_spatial_communities.csv",
            index=False,
        )

        nx.write_graphml(
            selected_G,
            GRAPHS_DIR / f"selected_{selected_graph_name}.graphml",
        )

        print("Saving enhanced figures for selected graph...")
        self.save_pollution_coloured_plot(
            selected_G, selected_graph_name, selected_node_metrics
        )
        self.save_centrality_ranking(node_metrics_df, selected_graph_name)

        print(f"Selected graph for discussion: {selected_graph_name}")

        self._print_policy_interpretation(selected_node_metrics, selected_communities)

        return {
            "sensor_coordinates": coords,
            "pairwise_distances": distances,
            "graph_summary": summary_df,
            "node_metrics": node_metrics_df,
            "communities": communities_df,
            "selected_graph_name": selected_graph_name,
            "selected_graph": selected_G,
        }

    @staticmethod
    def _print_policy_interpretation(node_metrics_df, communities_df):
        """ Print plain-language interpretations of selected network findings.
        
        The console summary highlights:
        - bridge-like sensors,
        - spatially accessible sensors,
        - hub-like sensors,
        - highly polluted sensors when available,
        - the number and strength of spatial communities"""

        if node_metrics_df.empty:
            return

        print("TASK 5 — Spatial Network Interpretation")

        top_between = node_metrics_df.nlargest(3, "betweenness_centrality")[
            ["sensor_name", "betweenness_centrality"]
        ]
        top_close = node_metrics_df.nlargest(3, "closeness_centrality")[
            ["sensor_name", "closeness_centrality"]
        ]
        top_eigen = node_metrics_df.nlargest(3, "eigenvector_centrality")[
            ["sensor_name", "eigenvector_centrality"]
        ]
        top_polluted = node_metrics_df.dropna(subset=["avg_pollution"]).nlargest(
            3, "avg_pollution"
        )[["sensor_name", "avg_pollution"]]

        print("\nTop 3 sensors by betweenness (bridge/bottleneck nodes):")
        print(top_between.to_string(index=False))
        print(
            "  → These stations lie on the most shortest paths between other\n"
            "    sensors. Removing or failing to monitor them would most\n"
            "    disconnect the spatial monitoring network.\n"
        )

        print("Top 3 sensors by closeness (most accessible):")
        print(top_close.to_string(index=False))
        print(
            "  → These stations reach all other stations via the shortest\n"
            "    total distance. They are the best candidates for city-wide\n"
            "    representative measurements.\n"
        )

        print("Top 3 sensors by eigenvector centrality (hub-of-hubs):")
        print(top_eigen.to_string(index=False))
        print(
            "  → These stations are important AND connected to other important\n"
            "    stations — the most structurally influential nodes.\n"
        )

        if not top_polluted.empty:
            print(f"Top 3 sensors by average pollution level:")
            print(top_polluted.to_string(index=False))
            print(
                "  → These areas experience the highest concentrations and\n"
                "    should be priority zones for policy intervention.\n"
            )

        if not communities_df.empty:
            n_communities = communities_df["community_id"].nunique()
            mod = communities_df["modularity"].iloc[0]
            print(f"Louvain communities detected: {n_communities}  (modularity = {mod:.3f})")
            print(
                "  → Higher modularity suggests that the graph separates into clearer spatial\n"
                "    clusters, which may reflect geographically coherent sensor groupings.\n"
                "    Sensors in the same community share more spatial connections\n"
                "    and likely represent the same urban micro-zone.\n"
            )

@timer
def run_task5(df, thresholds_m=None, k_values=None):
    """ Run the full Task 5 spatial network analysis pipeline.
    Args:
        df: cleaned DataFrame from Task 1/2
        thresholds_m: list of distance thresholds in meters for threshold graphs. If None, derived automatically from distance quantiles.
        k_values: list of k values for kNN graphs (default: [2, 3, 4, 5])

    Returns a dict of coordinates, distances, graph summaries, and the selected graph"""

    print("\n--- Task 5: Spatial Network ---")

    analyzer = SpatialNetworkAnalyzer(df)

    results = analyzer.run(
        thresholds_m=thresholds_m,
        k_values=k_values,
    )

    print("Task 5 completed")
    return results
