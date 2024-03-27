# IMPORTS
from cdlib import algorithms
import networkx as nx
import pandas as pd
import numpy as np
import random

class GraphSimulator():
    def __init__(self, num_nodes=10, num_edges=50, num_communities=2, homophily=0.5, isolation=0.5, insulation=0.5, affinity=0.5) -> None:
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



if __name__ == "__main__":
    pass