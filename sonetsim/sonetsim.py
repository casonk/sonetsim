# IMPORTS
# from cdlib import algorithms
# import networkx as nx
# import pandas as pd
# import numpy as np
# import random

class GraphSimulator():
    def __init__(self, num_nodes=10, num_edges=50, num_communities=2, homophily=0.5, isolation=0.5, insulation=0.5, affinity=0.5) -> None:
        import numpy as np

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_communities = num_communities
        self.homophily = homophily
        self.isolation = isolation
        self.insulation = insulation
        self.affinity = affinity

        assert self.num_nodes > 0, "Number of nodes must be greater than 0"
        assert self.num_edges > 0, "Number of edges must be greater than 0"
        assert self.num_communities > 0, "Number of communities must be greater than 0"

        for param in [self.homophily, self.isolation, self.insulation, self.affinity]:
            if type(param) in [int, float]:
                if param < 0 or param > 1:
                    raise ValueError(f"Parameter {param} must be between 0 and 1.")
                param = np.array([param for _ in range(num_communities)])
            else:
                try:
                    param = np.array(param)
                except TypeError:
                    raise TypeError(f"Parameter {param} must be a float or array-like of floats.")
                
    def simulate(self):
        """
        Initializes the graph data for simulation.

        Parameters:
        - num_nodes (int): The number of nodes in the graph.
        - num_edges (int): The number of edges in the graph.
        - num_communities (int): The number of communities in the graph.
        - homophily (list): A list of homophily values for each community.
        - isolation (list): A list of isolation values for each community.
        - insulation (list): A list of insulation values for each community.
        - affinity (list): A list of affinity values for each community.

        Returns:
        - nodes (numpy.ndarray): An array of node IDs.
        - communities (numpy.ndarray): An array of community IDs for each node.
        - labels (numpy.ndarray): An array of labels (ideologies) for each node.
        - source_nodes (numpy.ndarray): An array of source node IDs for each edge.
        - destination_nodes (numpy.ndarray): An array of destination node IDs for each edge.
        - edge_sentiments (numpy.ndarray): An array of sentiments for each edge.
        """

        import networkx as nx
        import pandas as pd
        import numpy as np

        # Initialize graph nodes
        self.nodes = np.arange(self.num_nodes)

        # Initialize node communities
        a                = list(range(self.num_communities))
        self.communities = np.random.choice( # Uniform distribution across communities
            a            = a, 
            size         = self.num_nodes,
            replace      = True
            )
        
        # Initialize node labels (e.g. ideologies)
        labels = []
        for c in self.communities:
            h = self.homophily[c]
            a = [c] + [_c for _c in pd.Series(self.communities).unique() if _c != c]
            p = [h] + [((1 - h) / (len(a) - 1))]*(len(a) - 1)
            labels.append(np.random.choice( # Probabilities based on community homophily
                a    = a, 
                size = 1, 
                p    = p
                )[0])
        self.labels = np.array(labels)

        # Initialize edge source nodes
        self.source_nodes = np.random.choice( # Uniform distribution across nodes
            a             = self.nodes, 
            size          = self.num_edges,
            replace       = True
            )
        # Resolve source communities from source nodes
        self.source_communities = np.array([self.communities[n] for n in self.source_nodes])

        # Initialize edge destination communities
        destination_communities = []
        for c in self.source_communities:
            i = self.isolation[c]
            a = [c] + [_c for _c in pd.Series(self.communities).unique() if _c != c]
            p = [i] + [((1 - i) / (len(a) - 1))]*(len(a) - 1)
            destination_communities.append(np.random.choice( # Probabilities based on community isolation
                a    = a, 
                size = 1, 
                p    = p
                )[0])
        self.destination_communities = np.array(destination_communities)

        # Resolve destination nodes from destination communities
        destination_nodes = []
        try:
            for c in self.destination_communities:
                a = self.nodes[np.where(self.communities == c)]
                destination_nodes.append(np.random.choice( # Uniform distribution across nodes in destination community
                    a    = a, 
                    size = 1
                    )[0])
            self.destination_nodes = np.array(destination_nodes)
        except Exception as e:
            print('a', a)
            print('c', c)
            print('nodes', self.nodes)
            print('communities', self.communities)
            print('destination_communities', self.destination_communities)
            raise e

        # Initialize edge sentiments
        edge_sentiments = []
        for c, _c in zip(self.source_communities, self.destination_communities):
            i   = self.insulation[c]
            f   = self.affinity[c]
            a   = [3, 2, 1] # Positive, Neutral, Negative
            p_f = [f, (1 - f) / 2, (1 - f) / 2]
            p_e = [(1 - i), i / 2, i / 2]
            if c == _c: # internal edge
                edge_sentiments.append(np.random.choice( # Probabilities based on community affinity
                    a    = a, 
                    size = 1, 
                    p    = p_f
                    )[0])
            else: # external edge
                edge_sentiments.append(np.random.choice( # Probabilities based on community insulation
                    a    = a, 
                    size = 1, 
                    p    = p_e
                    )[0])
                
        self.edge_sentiments = np.array(edge_sentiments)

        # Initialize graphs
        self.positive_sentiment_graph = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
        self.neutral_sentiment_graph  = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
        self.negative_sentiment_graph = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
        self.count_graph              = nx.DiGraph() # Count Graph (all edges are weighted to 1)

        # Add nodes to graphs
        for n, c, l in zip(self.nodes, self.communities, self.labels):
            self.positive_sentiment_graph.add_node(n, community=c, label=l)
            self.neutral_sentiment_graph.add_node(n, community=c, label=l)
            self.negative_sentiment_graph.add_node(n, community=c, label=l)
            self.count_graph.add_node(n, community=c, label=l)
            
        # Add edges to graphs
        for u, v, s in zip(self.source_nodes, self.destination_nodes, self.edge_sentiments):
            self.positive_sentiment_graph.add_edge(u, v, weight=s)
            self.neutral_sentiment_graph.add_edge(u, v, weight=(2 - np.abs(2 - s)))
            self.negative_sentiment_graph.add_edge(u, v, weight=(4 - s))
            self.count_graph.add_edge(u, v, weight=1)


if __name__ == "__main__":
    pass