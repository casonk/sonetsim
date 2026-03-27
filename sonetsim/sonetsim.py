"""
This module contains the `GraphSimulator` and `GraphEvaluator` classes in the `sonetsim` package.

The `GraphSimulator` class provides functionality for simulating a graph with specified parameters.
The `GraphEvaluator` class provides functionality for evaluating the simulated graph.

Classes:
    - GraphSimulator: Class for simulating a graph with specified parameters.
    - GraphEvaluator: Class for evaluating the simulated graph.

"""

# IMPORTS
## INTERNAL IMPORTS
import importlib
import random

## EXTERNAL IMPORTS
import networkx as nx
import numpy as np
import pandas as pd


class GraphSimulator:  # pylint: disable=too-many-instance-attributes
    """
    Class for simulating a graph with specified parameters.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_edges (int): Number of edges in the graph.
        num_communities (int): Number of communities in the graph.
        homophily (float or array-like): Homophily parameter(s) for each community.
        isolation (float or array-like): Isolation parameter(s) for each community.
        insulation (float or array-like): Insulation parameter(s) for each community.
        affinity (float or array-like): Affinity parameter(s) for each community.
        seed (int): Seed for random number generation.

    Raises:
        TypeError: If integer or probability parameter types are invalid.
        ValueError: If graph sizes are non-positive or probabilities fall outside [0, 1].

    Attributes:
        num_nodes (int): Number of nodes in the graph.
        num_edges (int): Number of edges in the graph.
        num_communities (int): Number of communities in the graph.
        homophily (ndarray): Homophily parameter(s) for each community.
        isolation (ndarray): Isolation parameter(s) for each community.
        insulation (ndarray): Insulation parameter(s) for each community.
        affinity (ndarray): Affinity parameter(s) for each community.
        seed (int): Seed for random number generation.
        nodes (ndarray): Array of node IDs.
        communities (ndarray): Array of community IDs for each node.
        labels (ndarray): Array of labels (e.g., ideologies) for each node.
        source_nodes (ndarray): Array of source node IDs for each edge.
        source_communities (ndarray): Array of source community IDs for each edge.
        destination_communities (ndarray): Array of destination community IDs for each edge.
        destination_nodes (ndarray): Array of destination node IDs for each edge.
        edge_sentiments (ndarray): Array of sentiment values for each edge.
        positive_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
        neutral_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
        negative_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
        count_graph (nx.DiGraph): Graph with all edges weighted to 1.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        num_nodes=50,
        num_edges=250,
        num_communities=5,
        homophily=0.5,
        isolation=0.5,
        insulation=0.5,
        affinity=0.5,
        seed=0,
    ) -> None:
        """
        Initialize the GraphSimulator object with the specified parameters.
        """

        # Set graph static parameters
        self.num_nodes = self.__validate_positive_integer__(
            value=num_nodes, name="num_nodes"
        )
        self.num_edges = self.__validate_positive_integer__(
            value=num_edges, name="num_edges"
        )
        self.num_communities = self.__validate_positive_integer__(
            value=num_communities, name="num_communities"
        )

        # Set graph simulation parameters
        self.homophily = self.__validate_probability_param__(
            param=homophily, name="homophily"
        )
        self.isolation = self.__validate_probability_param__(
            param=isolation, name="isolation"
        )
        self.insulation = self.__validate_probability_param__(
            param=insulation, name="insulation"
        )
        self.affinity = self.__validate_probability_param__(
            param=affinity, name="affinity"
        )

        self.seed = seed
        self.nodes = None
        self.communities = None
        self.labels = None
        self.source_nodes = None
        self.source_communities = None
        self.destination_communities = None
        self.destination_nodes = None
        self.edge_sentiments = None
        self.positive_sentiment_graph = None
        self.neutral_sentiment_graph = None
        self.negative_sentiment_graph = None
        self.count_graph = None

    @staticmethod
    def __validate_positive_integer__(value, name):
        """
        Validate that a configuration value is a positive integer.
        """
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be greater than 0.")

        return value

    def __validate_probability_param__(self, param, name):
        """
        Validate and normalize a probability parameter.
        """
        if isinstance(param, (int, float, np.integer, np.floating)) and not isinstance(
            param, bool
        ):
            param_value = float(param)
            if not 0 <= param_value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

            return np.full(self.num_communities, param_value, dtype=float)

        try:
            param_array = np.asarray(param, dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be a float or a one-dimensional array-like of floats."
            ) from exc

        if param_array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

        if len(param_array) != self.num_communities:
            raise ValueError(
                f"{name} must contain exactly {self.num_communities} values."
            )

        if ((param_array < 0) | (param_array > 1)).any():
            raise ValueError(f"{name} must contain only values between 0 and 1.")

        return param_array

    def __initialize_seed__(self):
        """
        Initialize the random number generator seed.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

    def set_seed(self, seed):
        """
        Set the seed for random number generation.

        Args:
            seed (int): The seed value to set.
        """
        self.seed = seed
        self.__initialize_seed__()

    def __initialize_graph_data__(self):
        """
        Initialize the graph data (nodes, communities, labels, edges, sentiments).
        """
        # Initialize graph nodes
        self.nodes = np.arange(self.num_nodes)

        # Initialize node communities
        a = list(range(self.num_communities))
        self.communities = np.random.choice(
            a=a, size=self.num_nodes, replace=True
        )  ## Uniform distribution across communities

        # Initialize node labels (e.g. ideologies)
        labels = []
        for c in self.communities:
            h = self.homophily[c]
            a = [c] + [_c for _c in pd.Series(self.communities).unique() if _c != c]
            p = [h] + [(1 - h) / (len(a) - 1)] * (len(a) - 1)
            labels.append(
                np.random.choice(a=a, size=1, p=p)[0]
            )  ## Probabilities based on community homophily
        self.labels = np.array(labels)

        # Initialize edge source nodes
        self.source_nodes = np.random.choice(
            a=self.nodes, size=self.num_edges, replace=True
        )  ## Uniform distribution across nodes
        # Resolve source communities from source nodes
        self.source_communities = np.array(
            [self.communities[n] for n in self.source_nodes]
        )

        # Initialize edge destination communities
        destination_communities = []
        for c in self.source_communities:
            i = self.isolation[c]
            a = [c] + [_c for _c in pd.Series(self.communities).unique() if _c != c]
            p = [i] + [(1 - i) / (len(a) - 1)] * (len(a) - 1)
            destination_communities.append(
                np.random.choice(a=a, size=1, p=p)[0]
            )  ## Probabilities based on community isolation
        self.destination_communities = np.array(destination_communities)

        # Resolve destination nodes from destination communities
        destination_nodes = []
        for c in self.destination_communities:
            a = self.nodes[np.where(self.communities == c)]
            destination_nodes.append(
                np.random.choice(a=a, size=1)[0]
            )  ## Uniform distribution across nodes in destination community
        self.destination_nodes = np.array(destination_nodes)

        # Initialize edge sentiments
        edge_sentiments = []
        for c, _c in zip(self.source_communities, self.destination_communities):
            i = self.insulation[c]
            f = self.affinity[c]
            a = [1, 0, -1]  ## Positive, Neutral, Negative
            p_f = [f, (1 - f) / 2, (1 - f) / 2]
            p_e = [(1 - i), i / 2, i / 2]
            if c == _c:  ## internal edge
                edge_sentiments.append(
                    np.random.choice(a=a, size=1, p=p_f)[0]
                )  ## Probabilities based on community affinity
            else:  ## external edge
                edge_sentiments.append(
                    np.random.choice(a=a, size=1, p=p_e)[0]
                )  ## Probabilities based on community insulation

        self.edge_sentiments = np.array(edge_sentiments)

    def __initialize_graphs__(self):
        """
        Initialize the sentiment and count graphs.
        """
        # Initialize graphs
        ## Sentiment Graph (edges are weighted to sentiment)
        self.positive_sentiment_graph = nx.DiGraph()
        ## Sentiment Graph (edges are weighted to sentiment)
        self.neutral_sentiment_graph = nx.DiGraph()
        ## Sentiment Graph (edges are weighted to sentiment)
        self.negative_sentiment_graph = nx.DiGraph()
        ## Count Graph (all edges are weighted to 1)
        self.count_graph = nx.DiGraph()

        # Add nodes to graphs
        for n, c, l in zip(self.nodes, self.communities, self.labels):
            self.positive_sentiment_graph.add_node(n, community=c, label=l)
            self.neutral_sentiment_graph.add_node(n, community=c, label=l)
            self.negative_sentiment_graph.add_node(n, community=c, label=l)
            self.count_graph.add_node(n, community=c, label=l)

        # Add edges to graphs
        for u, v, s in zip(
            self.source_nodes, self.destination_nodes, self.edge_sentiments
        ):
            self.positive_sentiment_graph.add_edge(u, v, weight=2 + s)
            self.neutral_sentiment_graph.add_edge(u, v, weight=2 - np.abs(s))
            self.negative_sentiment_graph.add_edge(u, v, weight=2 - s)
            self.count_graph.add_edge(u, v, weight=1)

    def simulate(self):
        """
        Simulate the graph based on the specified parameters.

        Returns:
            tuple: A tuple containing the positive sentiment graph, neutral sentiment graph,
                   negative sentiment graph, and count graph.
        """
        self.__initialize_seed__()
        self.__initialize_graph_data__()
        self.__initialize_graphs__()

        return (
            self.positive_sentiment_graph,
            self.neutral_sentiment_graph,
            self.negative_sentiment_graph,
            self.count_graph,
        )


class GraphEvaluator:  # pylint: disable=too-many-instance-attributes
    """
    Class for evaluating a graph using a specified algorithm.

    Args:
        simulator (GraphSimulator): The GraphSimulator object to evaluate.
        seed (int): Seed for random number generation.
        algorithm (str): Algorithm to use for community detection.
        resolution (float): Resolution parameter for the algorithm.

    Attributes:
        simulator (GraphSimulator): The GraphSimulator object to evaluate.
        seed (int): Seed for random number generation.
        algorithm (str): Algorithm to use for community detection.
        resolution (float): Resolution parameter for the algorithm.
        graph (nx.DiGraph): The graph to evaluate.
        communities (list): List of detected communities.
        node_df (pd.DataFrame): DataFrame containing node information.
        edge_df (pd.DataFrame): DataFrame containing edge information.
        metrics_df (pd.DataFrame): DataFrame containing evaluation metrics.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, simulator, seed=0, algorithm="louvain", resolution=1.0, alpha=0.5
    ) -> None:
        """
        Initialize the GraphEvaluator object with the specified parameters.
        """
        self.simulator = simulator
        self.seed = seed
        self.algorithm = None
        self.resolution = resolution
        self.alpha = alpha
        self.graph = simulator.count_graph
        self.communities = None
        self.node_df = None
        self.edge_df = None
        self.metrics_df = None
        self.graph_map = {"count": 0, "positive": 1, "neutral": 2, "negative": 3}
        self.graph_attribute_map = {
            "count": "count_graph",
            0: "count_graph",
            "positive": "positive_sentiment_graph",
            1: "positive_sentiment_graph",
            "neutral": "neutral_sentiment_graph",
            2: "neutral_sentiment_graph",
            "negative": "negative_sentiment_graph",
            3: "negative_sentiment_graph",
        }
        self.supported_algorithms = {"louvain", "leiden", "eva", "infomap"}
        self.set_algorithm(algorithm)

    def __initialize_seed__(self):
        """
        Initialize the random number generator seed.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

    def set_seed(self, seed):
        """
        Set the seed for random number generation.

        Args:
            seed (int): The seed value to set.
        """
        self.seed = seed
        self.__initialize_seed__()

    @staticmethod
    def __get_cdlib_algorithms__():
        """
        Import cdlib algorithms lazily to avoid import-time side effects.
        """
        return importlib.import_module("cdlib.algorithms")

    def __require_simulated_graphs__(self):
        """
        Ensure the simulator has been run before evaluation.
        """
        required_attributes = (
            "nodes",
            "communities",
            "labels",
            "source_nodes",
            "source_communities",
            "destination_communities",
            "destination_nodes",
            "edge_sentiments",
            "positive_sentiment_graph",
            "neutral_sentiment_graph",
            "negative_sentiment_graph",
            "count_graph",
        )

        if any(getattr(self.simulator, attr) is None for attr in required_attributes):
            raise RuntimeError(
                "GraphEvaluator requires a simulated graph. Call simulator.simulate() first."
            )

    def set_graph(self, graph="count"):
        """
        Set the graph to evaluate.

        Args:
            graph (str) or (int): The graph to set.
            Options are "count" or 0, "positive" or 1, "neutral" or 2, "negative" or 3.
        """
        self.__require_simulated_graphs__()

        if graph not in self.graph_attribute_map:
            raise ValueError(
                'graph must be one of "count", "positive", "neutral", "negative", '
                "0, 1, 2, or 3."
            )

        self.graph = getattr(self.simulator, self.graph_attribute_map[graph])
        return self.graph

    def set_algorithm(self, algorithm):
        """
        Set the algorithm to use for community detection.

        Args:
            algorithm (str): The algorithm to set.
        """
        if algorithm not in self.supported_algorithms:
            raise ValueError(
                f'Algorithm "{algorithm}" not supported. Must be one of '
                f"{sorted(self.supported_algorithms)}."
            )

        self.algorithm = algorithm

    def __initialize_dataframes__(self):
        """
        Initialize the dataframes for node, edge, and metrics information.
        """
        node_df = pd.DataFrame(
            data={
                "node": self.simulator.nodes,
                "set_community": self.simulator.communities,
                "set_label": self.simulator.labels,
            }
        )
        community_mapper = {}
        for enum in enumerate(self.communities):
            for node in enum[1]:
                community_mapper[node] = enum[0]

        node_df["detected_community"] = node_df["node"].map(community_mapper)

        edge_df = pd.DataFrame(
            data={
                "source_node": self.simulator.source_nodes,
                "destination_node": self.simulator.destination_nodes,
                "edge_sentiments": self.simulator.edge_sentiments,
            }
        )
        edge_df["set_source_community"] = edge_df["source_node"].map(
            node_df["set_community"]
        )
        edge_df["set_source_label"] = edge_df["source_node"].map(node_df["set_label"])
        edge_df["detected_source_community"] = edge_df["source_node"].map(
            node_df["detected_community"]
        )
        edge_df["detected_destination_community"] = edge_df["destination_node"].map(
            node_df["detected_community"]
        )

        self.node_df = node_df
        self.edge_df = edge_df

    def detect_communities(self, graph=None, algorithm=None):
        """
        Detects communities in the graph using the specified algorithm.

        Args:
            graph (str): The graph to set. Options are "count", "positive", "neutral", "negative".
            algorithm (str): The algorithm to use for detection.

        Returns:
            list: A list of sets, where each set represents a community.

        Raises:
            RuntimeError: If the simulator has not been run yet.
            ValueError: If the specified graph or algorithm is not supported.
        """
        self.__require_simulated_graphs__()

        if graph is not None:
            self.set_graph(graph)

        if algorithm is not None:
            self.set_algorithm(algorithm)

        self.__initialize_seed__()

        if self.algorithm == "louvain":
            self.communities = nx.algorithms.community.louvain_communities(
                G=self.graph,
                weight="weight",
                resolution=self.resolution,
                seed=self.seed,
            )
        elif self.algorithm == "leiden":
            cdlib_algorithms = self.__get_cdlib_algorithms__()
            self.communities = [
                set(community)
                for community in cdlib_algorithms.leiden(
                    g_original=self.graph,
                    weights=[e[2] for e in self.graph.edges(data="weight")],
                ).communities
            ]
        elif self.algorithm == "eva":
            cdlib_algorithms = self.__get_cdlib_algorithms__()
            self.communities = [
                set(community)
                for community in cdlib_algorithms.eva(
                    g_original=self.graph.to_undirected(),
                    labels={
                        n: {"label": self.graph.nodes[n]["label"]}
                        for n in self.graph.nodes
                    },
                    weight="weight",
                    resolution=self.resolution,
                    alpha=self.alpha,
                ).communities
            ]
        elif self.algorithm == "infomap":
            cdlib_algorithms = self.__get_cdlib_algorithms__()
            self.communities = [
                set(community)
                for community in cdlib_algorithms.infomap(
                    g_original=self.graph,
                    flags="-d",
                ).communities
            ]
        self.__initialize_dataframes__()

        return self.communities

    def evaluate_single_community(self, community):  # pylint: disable=too-many-locals
        """
        Evaluate a single community based on various metrics.

        Args:
            community (int): The community to evaluate.

        Returns:
            A tuple containing the following metrics:
            - detected_homophily: The percentage of the most frequent labels in the community.
            - detected_isolation: The percentage of internal edges in the community.
            - detected_insulation: The percentage of negative external edges in the community.
            - detected_affinity: The percentage of positive internal edges in the community.
            - detected_purity: The product of the percentages of all labels in the community.
            - detected_conductance: The percentage of external edges in the community.
            - detected_equity: The percentage of neutral external edges in the community.
            - detected_altruism: The percentage of positive external edges in the community.
            - detected_balance: The percentage of neutral internal edges in the community.
            - detected_hostility: The percentage of negative internal edges in the community.
            - num_nodes: The number of nodes in the community.
            - num_internal_edges: The number of internal edges in the community.
            - num_external_edges: The number of external edges in the community.
        """
        # Generate helper dataframes
        comm_specific_node_df = self.node_df[
            self.node_df.detected_community == community
        ]
        comm_specific_edge_df = self.edge_df[
            self.edge_df.detected_source_community == community
        ]
        comm_specific_external_edge_df = comm_specific_edge_df[
            comm_specific_edge_df.detected_destination_community != community
        ]
        comm_specific_internal_edge_df = comm_specific_edge_df[
            comm_specific_edge_df.detected_destination_community == community
        ]

        # Evaluate community on generic metrics
        try:
            detected_homophily = (
                comm_specific_node_df.set_label.value_counts().sort_index().iloc[0]
                / len(comm_specific_node_df)
            )  # % of the most frequent labels
        except (KeyError, ZeroDivisionError):
            detected_homophily = 0  # THERE ARE NO NODES
        try:
            detected_isolation = len(comm_specific_internal_edge_df) / len(
                comm_specific_edge_df
            )  # % of internal edges (1 - condunctance)
        except ZeroDivisionError:
            detected_isolation = 0  # THERE ARE NO EDGES
        try:
            # negative external edges are encoded as -1
            detected_insulation = (
                comm_specific_external_edge_df.edge_sentiments.value_counts().get(-1, 0)
                / len(comm_specific_external_edge_df)
            )  # % negative external edges
        except ZeroDivisionError:
            detected_insulation = 0  # THERE ARE NO EXTERNAL EDGES
        try:
            # positive internal edges are encoded as 1
            detected_affinity = (
                comm_specific_internal_edge_df.edge_sentiments.value_counts().get(1, 0)
                / len(comm_specific_internal_edge_df)
            )  # % of positive internal edges
        except ZeroDivisionError:
            detected_affinity = 0  # THERE ARE NO INTERNAL EDGES
        if len(comm_specific_node_df) == 0:
            detected_purity = 0  # THERE ARE NO NODES
        else:
            detected_purity = (
                comm_specific_node_df.set_label.value_counts()
                / len(comm_specific_node_df)
            ).prod()
            # product of % of all labels (Similar to `detected_homophily` but looks at all labels)
        try:
            detected_conductance = len(comm_specific_external_edge_df) / len(
                comm_specific_edge_df
            )  # % of external edges (1 - isolation)
        except ZeroDivisionError:
            detected_conductance = 0  # THERE ARE NO EDGES
        try:
            # neutral external edges are encoded as 0
            detected_equity = (
                comm_specific_external_edge_df.edge_sentiments.value_counts().get(0, 0)
                / len(comm_specific_external_edge_df)
            )  # % neutral external edges
        except ZeroDivisionError:
            detected_equity = 0  # THERE ARE NO EXTERNAL EDGES
        try:
            # positive external edges are encoded as 1
            detected_altruism = (
                comm_specific_external_edge_df.edge_sentiments.value_counts().get(1, 0)
                / len(comm_specific_external_edge_df)
            )  # % positive external edges
        except ZeroDivisionError:
            detected_altruism = 0  # THERE ARE NO EXTERNAL EDGES
        try:
            # neutral internal edges are encoded as 0
            detected_balance = (
                comm_specific_internal_edge_df.edge_sentiments.value_counts().get(0, 0)
                / len(comm_specific_internal_edge_df)
            )  # % of neutral internal edges
        except ZeroDivisionError:
            detected_balance = 0  # THERE ARE NO INTERNAL EDGES
        try:
            # negative internal edges are encoded as -1
            detected_hostility = (
                comm_specific_internal_edge_df.edge_sentiments.value_counts().get(-1, 0)
                / len(comm_specific_internal_edge_df)
            )  # % of negative internal edges
        except ZeroDivisionError:
            detected_hostility = 0  # THERE ARE NO INTERNAL EDGES

        return (
            detected_homophily,
            detected_isolation,
            detected_insulation,
            detected_affinity,
            detected_purity,
            detected_conductance,
            detected_equity,
            detected_altruism,
            detected_balance,
            detected_hostility,
            len(comm_specific_node_df),
            len(comm_specific_internal_edge_df),
            len(comm_specific_external_edge_df),
        )

    def evaluate_all_communities(self):
        """
        Evaluate all communities in the network and return a DataFrame with metrics.

        Returns:
            metrics_df (pd.DataFrame):
            DataFrame containing the evaluation metrics for each community.
                The columns of the DataFrame include:
                - homophily
                - isolation
                - insulation
                - affinity
                - purity
                - conductance
                - equity
                - altruism
                - balance
                - hostility
                - num_nodes
                - num_internal_edges
                - num_external_edges
        """
        metrics = []
        for community in self.node_df["detected_community"].unique():
            community_metrics = self.evaluate_single_community(community)
            metrics.append(community_metrics)

        self.metrics_df = pd.DataFrame(
            data=metrics,
            columns=[
                "homophily",
                "isolation",
                "insulation",
                "affinity",
                "purity",
                "conductance",
                "equity",
                "altruism",
                "balance",
                "hostility",
                "num_nodes",
                "num_internal_edges",
                "num_external_edges",
            ],
        )

        return self.metrics_df

    def evaluate_single_graph(self, graph=None, algorithm=None):
        """
        Evaluates the communities in the graph.

        Parameters:
        - graph (str|int|None): The graph identifier to evaluate.
        - algorithm (str|None): The algorithm to use for detection.

        Returns:
        - metrics_df (DataFrame): The metrics dataframe containing the evaluation results.
        """
        self.__require_simulated_graphs__()
        self.detect_communities(graph=graph, algorithm=algorithm)
        self.evaluate_all_communities()
        return self.metrics_df

    def evaluate(self, algorithm=None):
        """
        Evaluates the communities in all graphs.

        Parameters:
        - algorithm (str|None): The algorithm to use for detection.

        Returns:
        - metrics_df (DataFrame): The metrics dataframe containing the evaluation results.
        """
        self.__require_simulated_graphs__()
        cnt_df = self.evaluate_single_graph(graph=0, algorithm=algorithm)
        pos_df = self.evaluate_single_graph(graph=1, algorithm=algorithm)
        neu_df = self.evaluate_single_graph(graph=2, algorithm=algorithm)
        neg_df = self.evaluate_single_graph(graph=3, algorithm=algorithm)
        cnt_df["weight_method"] = 0
        pos_df["weight_method"] = 1
        neu_df["weight_method"] = 2
        neg_df["weight_method"] = 3
        self.metrics_df = pd.concat([cnt_df, pos_df, neu_df, neg_df])
        return self.metrics_df

    def evaluate_algorithms(self, algorithms=None):
        """
        Evaluates the communities in all graphs.

        Parameters:
        - algorithms (list(str)): The algorithms to use for detection.

        Returns:
        - metrics_df (DataFrame): The metrics dataframe containing the evaluation results.
        """
        self.__require_simulated_graphs__()

        if algorithms is None:
            algorithms = ["louvain", "leiden", "eva", "infomap"]

        alg_dfs = []
        for algorithm in algorithms:
            alg_df = self.evaluate(algorithm=algorithm)
            alg_df["algorithm"] = algorithm
            alg_dfs.append(alg_df)
        self.metrics_df = pd.concat(alg_dfs)
        return self.metrics_df


if __name__ == "__main__":
    pass
