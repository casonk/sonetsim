# IMPORTS

# INTERNAL IMPORTS
import random

# EXTERNAL IMPORTS
from cdlib import algorithms
import networkx as nx
import pandas as pd
import numpy as np


class GraphSimulator:
    def __init__(
        self,
        num_nodes=10,
        num_edges=50,
        num_communities=2,
        homophily=0.5,
        isolation=0.5,
        insulation=0.5,
        affinity=0.5,
        seed=0
    ) -> None:

        # Set graph static parameters
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_communities = num_communities

        # Check graph static parameters
        assert self.num_nodes > 0, "Number of nodes must be greater than 0"
        assert self.num_edges > 0, "Number of edges must be greater than 0"
        assert self.num_communities > 0, "Number of communities must be greater than 0"

        # Function to check & update simulation parameters
        def __check_param__(self, param):
            if type(param) in [int, float]:
                if param < 0 or param > 1:
                    raise ValueError(f"Parameter {param} must be between 0 and 1.")
                param = np.array([param for _ in range(num_communities)])
            else:
                try:
                    param = np.array(param)
                except TypeError as exc:
                    raise TypeError(
                        f"Parameter {param} must be a float or array-like of floats."
                    ) from exc
            return param
        
        # Set graph simulation parameters
        self.homophily  = __check_param__(self, homophily)
        self.isolation  = __check_param__(self, homophily)
        self.insulation = __check_param__(self, homophily)
        self.affinity   = __check_param__(self, homophily)

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

    def __initialize_seed__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def __initialize_graph_data__(self):
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
            p = [h] + [((1 - h) / (len(a) - 1))] * (len(a) - 1)
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
            p = [i] + [((1 - i) / (len(a) - 1))] * (len(a) - 1)
            destination_communities.append(
                np.random.choice(a=a, size=1, p=p)[0]
            )  ## Probabilities based on community isolation
        self.destination_communities = np.array(destination_communities)

        # Resolve destination nodes from destination communities
        destination_nodes = []
        try:
            for c in self.destination_communities:
                a = self.nodes[np.where(self.communities == c)]
                destination_nodes.append(
                    np.random.choice(a=a, size=1)[0]
                )  ## Uniform distribution across nodes in destination community
            self.destination_nodes = np.array(destination_nodes)
        except Exception as e:
            print("a", a)
            print("c", c)
            print("nodes", self.nodes)
            print("communities", self.communities)
            print("destination_communities", self.destination_communities)
            raise e

        # Initialize edge sentiments
        edge_sentiments = []
        for c, _c in zip(self.source_communities, self.destination_communities):
            i = self.insulation[c]
            f = self.affinity[c]
            a = [3, 2, 1]  ## Positive, Neutral, Negative
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
            self.positive_sentiment_graph.add_edge(u, v, weight=s)
            self.neutral_sentiment_graph.add_edge(u, v, weight=2 - np.abs(2 - s))
            self.negative_sentiment_graph.add_edge(u, v, weight=4 - s)
            self.count_graph.add_edge(u, v, weight=1)

    def simulate(self):
        self.__initialize_seed__()
        self.__initialize_graph_data__()
        self.__initialize_graphs__()

        return self.positive_sentiment_graph, self.neutral_sentiment_graph, self.negative_sentiment_graph, self.count_graph


if __name__ == "__main__":
    pass
