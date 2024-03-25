# IMPORTS
from cdlib import algorithms
import networkx as nx
# import cugraph as cx
import pandas as pd
import numpy as np
import random

# STATIC VARIABLES
using_gpu_base = False

# FUNCTIONS
def initialize_graph_data(num_nodes, num_edges, num_communities, homophily, isolation, insulation, affinity):
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

    # Initialize graph nodes
    nodes = np.arange(num_nodes)

    # Initialize node communities
    a           = list(range(num_communities))
    communities = np.random.choice( # Uniform distribution across communities
        a       = a, 
        size    = num_nodes,
        replace = True
        )
    
    # Initialize node labels (e.g. ideologies)
    labels = []
    for c in communities:
        h = homophily[c]
        a = [c] + [_c for _c in pd.Series(communities).unique() if _c != c]
        p = [h] + [((1 - h) / (len(a) - 1))]*(len(a) - 1)
        labels.append(np.random.choice( # Probabilities based on community homophily
            a    = a, 
            size = 1, 
            p    = p
            )[0])
    labels = np.array(labels)

    # Initialize edge source nodes
    source_nodes = np.random.choice( # Uniform distribution across nodes
        a       = nodes, 
        size    = num_edges,
        replace = True
        )
    # Resolve source communities from source nodes
    source_communities = np.array([communities[n] for n in source_nodes])

    # Initialize edge destination communities
    destination_communities = []
    for c in source_communities:
        i = isolation[c]
        a = [c] + [_c for _c in pd.Series(communities).unique() if _c != c]
        p = [i] + [((1 - i) / (len(a) - 1))]*(len(a) - 1)
        destination_communities.append(np.random.choice( # Probabilities based on community isolation
            a    = a, 
            size = 1, 
            p    = p
            )[0])
    destination_communities = np.array(destination_communities)

    # Resolve destination nodes from destination communities
    destination_nodes = []
    try:
        for c in destination_communities:
            a = nodes[np.where(communities == c)]
            destination_nodes.append(np.random.choice( # Uniform distribution across nodes in destination community
                a    = a, 
                size = 1
                )[0])
        destination_nodes = np.array(destination_nodes)
    except Exception as e:
        print('a', a)
        print('c', c)
        print('nodes', nodes)
        print('communities', communities)
        print('destination_communities', destination_communities)
        raise e

    # Initialize edge sentiments
    edge_sentiments = []
    for c, _c in zip(source_communities, destination_communities):
        i   = insulation[c]
        f   = affinity[c]
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
    edge_sentiments = np.array(edge_sentiments)
    
    return nodes, communities, labels, source_nodes, destination_nodes, edge_sentiments

def initialize_graphs(nodes, communities, labels, source_nodes, destination_nodes, edge_sentiments):
    """
    Initializes and returns four different graphs based on the given parameters.

    Parameters:
    - nodes (list): List of nodes in the graph.
    - communities (list): List of community labels for each node.
    - labels (list): List of labels for each node.
    - source_nodes (list): List of source nodes for each edge.
    - destination_nodes (list): List of destination nodes for each edge.
    - edge_sentiments (list): List of sentiment values for each edge.

    Returns:
    - positive_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
    - neutral_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
    - negative_sentiment_graph (nx.DiGraph): Graph with edges weighted to sentiment.
    - count_graph (nx.DiGraph): Graph with all edges weighted to 1.
    """

    # Initialize graphs
    positive_sentiment_graph = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
    neutral_sentiment_graph  = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
    negative_sentiment_graph = nx.DiGraph() # Sentiment Graph (edges are weighted to sentiment)
    count_graph              = nx.DiGraph() # Count Graph (all edges are weighted to 1)

    # Add nodes to graphs
    for n, c, l in zip(nodes, communities, labels):
        positive_sentiment_graph.add_node(n, community=c, label=l)
        neutral_sentiment_graph.add_node(n, community=c, label=l)
        negative_sentiment_graph.add_node(n, community=c, label=l)
        count_graph.add_node(n, community=c, label=l)
        
    # Add edges to graphs
    for u, v, s in zip(source_nodes, destination_nodes, edge_sentiments):
        positive_sentiment_graph.add_edge(u, v, weight=s)
        neutral_sentiment_graph.add_edge(u, v, weight=(2 - np.abs(2 - s)))
        negative_sentiment_graph.add_edge(u, v, weight=(4 - s))
        count_graph.add_edge(u, v, weight=1)

    return positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph

def simulation(homophily=0.5, isolation=0.5, insulation=0.5, affinity=0.5, num_nodes=10, num_edges=50, num_communities=2, seed=0):
    """
    Run a simulation of a social network.

    Parameters:
    - homophily (float or array-like of floats): Homophily parameter(s) for each community. Must be between 0 and 1.
    - isolation (float or array-like of floats): Isolation parameter(s) for each community. Must be between 0 and 1.
    - insulation (float or array-like of floats): Insulation parameter(s) for each community. Must be between 0 and 1.
    - affinity (float or array-like of floats): Affinity parameter(s) for each community. Must be between 0 and 1.
    - num_nodes (int): Number of nodes in the network.
    - num_edges (int): Number of edges in the network.
    - num_communities (int): Number of communities in the network.
    - seed (int): Seed for random number generation.

    Returns:
    - positive_sentiment_graph (Graph): Graph representing positive sentiment relationships.
    - neutral_sentiment_graph (Graph): Graph representing neutral sentiment relationships.
    - negative_sentiment_graph (Graph): Graph representing negative sentiment relationships.
    - count_graph (Graph): Graph representing the count of relationships.
    - nodes (list): List of nodes in the network.
    - communities (list): List of community assignments for each node.
    - labels (list): List of labels for each node.
    - source_nodes (list): List of source nodes for each edge.
    - destination_nodes (list): List of destination nodes for each edge.
    - edge_sentiments (list): List of sentiments for each edge.
    """
    
    # Set program seed
    random.seed(seed)
    np.random.seed(seed)

    # Check parameters are passed as expected
    def check_param(param):
        """
        Check and validate the parameter value.

        Parameters:
        param (int, float, or array-like): The parameter value to be checked.

        Returns:
        numpy.ndarray: The validated parameter value.

        Raises:
        ValueError: If the parameter is a number and not between 0 and 1.
        TypeError: If the parameter is not a number or array-like of numbers.
        """

        if type(param) in [int, float]:
            if param < 0 or param > 1:
                raise ValueError(f"Parameter {param} must be between 0 and 1.")
            param = np.array([param for _ in range(num_communities)])
        else:
            try:
                param = np.array(param)
            except TypeError:
                raise TypeError(f"Parameter {param} must be a float or array-like of floats.")
        return param

    # Execute paramater checks
    homophily = check_param(homophily)
    isolation = check_param(isolation)
    insulation = check_param(insulation)
    affinity = check_param(affinity)
            
    # Initialize graph data & graphs
    nodes, communities, labels, source_nodes, destination_nodes, edge_sentiments = initialize_graph_data(num_nodes, num_edges, num_communities, homophily, isolation, insulation, affinity)
    positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph = initialize_graphs(nodes, communities, labels, source_nodes, destination_nodes, edge_sentiments)

    return positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph

def initialize_evaluation_lists():
    """
    Initializes and returns evaluation lists used for storing evaluation metrics.

    Returns:
    algs (list): List to store algorithm names.
    comms (list): List to store community names.
    homophilies (list): List to store homophily values.
    isolations (list): List to store isolation values.
    insulations (list): List to store insulation values.
    affinities (list): List to store affinity values.
    purities (list): List to store purity values.
    coundunctances (list): List to store conductance values.
    equities (list): List to store equity values.
    altruisims (list): List to store altruism values.
    balances (list): List to store balance values.
    hostilities (list): List to store hostility values.
    weight_method (list): List to store weight method names.
    """

    # Create evaluation lists
    algs           = []
    comms          = []
    homophilies    = []
    isolations     = []
    insulations    = []
    affinities     = []
    purities       = []
    coundunctances = []
    equities       = []
    altruisims     = []
    balances       = []
    hostilities    = []
    weight_method  = []

    return algs, comms, homophilies, isolations, insulations, affinities, purities, coundunctances, equities, altruisims, balances, hostilities, weight_method

def initialize_evaluation_dictionaries():
    """
    Initializes the evaluation dictionaries for sentiment analysis.

    Returns:
        positive_sentiment_graph_comms (dict): A dictionary to store positive sentiment graph communities.
        neutral_sentiment_graph_comms (dict): A dictionary to store neutral sentiment graph communities.
        negative_sentiment_graph_comms (dict): A dictionary to store negative sentiment graph communities.
        count_graph_comms (dict): A dictionary to store the count of graph communities.
    """

    # Function implementation goes here
    positive_sentiment_graph_comms = {}
    neutral_sentiment_graph_comms  = {}
    negative_sentiment_graph_comms = {}
    count_graph_comms              = {}

    return positive_sentiment_graph_comms, neutral_sentiment_graph_comms, negative_sentiment_graph_comms, count_graph_comms

def evaluate_single_community(comm, node_df, edge_df, base_graph):
    """
    Evaluate the characteristics of a single community in a social network.

    Parameters:
    comm (str): The community identifier.
    node_df (DataFrame): The DataFrame containing node information.
    edge_df (DataFrame): The DataFrame containing edge information.

    Returns:
    Tuple: A tuple containing the following characteristics of the community:
        - detected_homophily (float): The percentage of the most frequent labels in the community.
        - detected_isolation (float): The percentage of internal edges in the community.
        - detected_insulation (float): The percentage of negative external edges in the community.
        - detected_affinity (float): The percentage of positive internal edges in the community.
        - detected_purity (float): The product of the percentages of all labels in the community.
        - detected_conductance (float): The percentage of external edges in the community.
        - detected_equity (float): The percentage of neutral external edges in the community.
        - detected_altruism (float): The percentage of positive external edges in the community.
        - detected_balance (float): The percentage of neutral internal edges in the community.
        - detected_hostility (float): The percentage of negative internal edges in the community.
    """

    # Generate helper dataframes
    comm_specific_node_df          = node_df[node_df[base_graph+'_detected_community'] == comm]
    comm_specific_edge_df          = edge_df[edge_df[base_graph+'_detected_source_community'] == comm]
    comm_specific_external_edge_df = comm_specific_edge_df[comm_specific_edge_df[base_graph+'_detected_destination_community'] != comm]
    comm_specific_internal_edge_df = comm_specific_edge_df[comm_specific_edge_df[base_graph+'_detected_destination_community'] == comm]

    # Evaluate community on generic metrics
    try:
        detected_homophily   = comm_specific_node_df.set_label.value_counts().sort_index().iloc[0] / len(comm_specific_node_df) # % of the most frequent labels
    except:
        detected_homophily   = 0 # THERE ARE NO NODES
    try:
        detected_isolation   = len(comm_specific_internal_edge_df) / len(comm_specific_edge_df) # % of internal edges (1 - condunctance)
    except ZeroDivisionError:
        detected_isolation   = 0 # THERE ARE NO EDGES
    try:
        detected_insulation  = comm_specific_external_edge_df.edge_sentiments.value_counts().loc[1] / len(comm_specific_external_edge_df) # % negative external edges
    except KeyError:
        detected_insulation  = 0 # THERE ARE NO EXTERNAL EDGES
    try:
        detected_affinity    = comm_specific_internal_edge_df.edge_sentiments.value_counts().loc[3] / len(comm_specific_internal_edge_df) # % of positive internal edges
    except KeyError:
        detected_affinity    = 0 # THERE ARE NO INTERNAL EDGES
    try:
        detected_purity      = (comm_specific_node_df.set_label.value_counts() / len(comm_specific_node_df)).prod() # product of % of all labels (Similar to `detected_homophily` but looks at all labels)
    except KeyError:
        detected_purity      = 0 # THERE ARE NO NODES 
    try:
        detected_conductance = len(comm_specific_external_edge_df) / len(comm_specific_edge_df) # % of external edges (1 - isolation)
    except ZeroDivisionError:
        detected_conductance = 0 # THERE ARE NO EDGES
    try:
        detected_equity      = comm_specific_external_edge_df.edge_sentiments.value_counts().loc[2] / len(comm_specific_external_edge_df) # % neutral external edges
    except KeyError:
        detected_equity      = 0 # THERE ARE NO EXTERNAL EDGES
    try:
        detected_altruism    = comm_specific_external_edge_df.edge_sentiments.value_counts().loc[3] / len(comm_specific_external_edge_df) # % positive external edges
    except KeyError:
        detected_altruism    = 0 # THERE ARE NO EXTERNAL EDGES
    try:
        detected_balance     = comm_specific_internal_edge_df.edge_sentiments.value_counts().loc[2] / len(comm_specific_internal_edge_df) # % of neutral internal edges
    except KeyError:
        detected_balance     = 0 # THERE ARE NO INTERNAL EDGES
    try:
        detected_hostility   = comm_specific_internal_edge_df.edge_sentiments.value_counts().loc[1] / len(comm_specific_internal_edge_df) # % of negative internal edges
    except KeyError:
        detected_hostility   = 0 # THERE ARE NO INTERNAL EDGES

    return detected_homophily, detected_isolation, detected_insulation, detected_affinity, detected_purity, detected_conductance, detected_equity, detected_altruism, detected_balance, detected_hostility

def augment_evaluation_df(eval_df, homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, resolution, seed):
    """
    Augments the evaluation DataFrame with additional columns.

    Parameters:
    - eval_df (DataFrame): The evaluation DataFrame to be augmented.
    """

    # Augment evaluation dataframe with simulation metadata
    eval_df['target_homophily']  = [homophily]*len(eval_df)
    eval_df['target_isolation']  = [isolation]*len(eval_df)
    eval_df['target_insulation'] = [insulation]*len(eval_df)
    eval_df['target_affinity']   = [affinity]*len(eval_df)
    eval_df['num_nodes']         = [num_nodes]*len(eval_df)
    eval_df['num_edges']         = [num_edges]*len(eval_df)
    eval_df['edge_multiplier']   = eval_df['num_edges'] / eval_df['num_nodes']
    eval_df['num_communities']   = [num_communities]*len(eval_df)
    eval_df['resolution']        = [resolution]*len(eval_df)
    eval_df['seed']              = [seed]*len(eval_df)

    return eval_df

def evaluate_communities_parallel(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed, positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph, resolution=1, algos=['eva'], eva_alpha=0.5):
    """
    Evaluates communities in parallel for multiple sentiment graphs and a count graph.

    Parameters:
    - positive_sentiment_graph (networkx.Graph): Graph representing positive sentiment connections.
    - neutral_sentiment_graph (networkx.Graph): Graph representing neutral sentiment connections.
    - negative_sentiment_graph (networkx.Graph): Graph representing negative sentiment connections.
    - count_graph (networkx.Graph): Graph representing count connections.
    - seed (int): Seed value for random number generation (default: 0).
    - resolution (float): Resolution parameter for community detection algorithms (default: 1).
    - algos (list): List of community detection algorithms to use (default: ['eva']).
    - eva_alpha (float): Alpha parameter for the EVA algorithm (default: 0.5).
    """

    # Initialize evaluation lists
    algs, comms, homophilies, isolations, insulations, affinities, purities, coundunctances, equities, altruisims, balances, hostilities, weight_method = initialize_evaluation_lists()

    # Initialize community dictionaries
    positive_sentiment_graph_comms, neutral_sentiment_graph_comms, negative_sentiment_graph_comms, count_graph_comms = initialize_evaluation_dictionaries()

    # Resolve graph independent node attributes
    nodes       = np.array(
            [
            n for n in count_graph.nodes()
            ]
        )
    labels      = np.array(
            [
            l for l in nx.get_node_attributes(count_graph, 'label').values()
            ]
        )
    communities = np.array(
            [
            l for l in nx.get_node_attributes(count_graph, 'community').values()
            ]
        )

    # Resolve graph independent edge attributes
    source_nodes      = np.array(
            [
            e[0] for e in count_graph.edges()
            ]
        )
    destination_nodes = np.array(
            [
            e[0] for e in count_graph.edges()
            ]
        )
    edge_sentiments   = np.array(
            [
            s for s in nx.get_edge_attributes(positive_sentiment_graph, 'weight').values()
            ]
        )
    
    # Resolve node labels for cdlib
    positive_sentiment_graph_labels = {
        n:{
            'label':positive_sentiment_graph.nodes[n]['label']
            } for n in positive_sentiment_graph.nodes
        }
    neutral_sentiment_graph_labels  = {
        n:{
            'label':neutral_sentiment_graph.nodes[n]['label']
            } for n in neutral_sentiment_graph.nodes
        }
    negative_sentiment_graph_labels = {
        n:{
            'label':negative_sentiment_graph.nodes[n]['label']
            } for n in negative_sentiment_graph.nodes
        }
    count_graph_labels              = {
        n:{
            'label':count_graph.nodes[n]['label']
            } for n in count_graph.nodes
        }

    # Detect Communities
    # Using leiden algorithm
    if 'leiden' in algos:
        positive_sentiment_graph_comms['leiden'] = [
            set(comm) for comm in algorithms.leiden(
                g_original=positive_sentiment_graph, 
                weights=[e[2] for e in positive_sentiment_graph.edges(data='weight')]
                ).communities
                ]
        neutral_sentiment_graph_comms['leiden']  = [
            set(comm) for comm in algorithms.leiden(
                g_original=neutral_sentiment_graph, 
                weights=[e[2] for e in neutral_sentiment_graph.edges(data='weight')]
                ).communities
                ]
        negative_sentiment_graph_comms['leiden'] = [
            set(comm) for comm in algorithms.leiden(
                g_original=negative_sentiment_graph, 
                weights=[e[2] for e in negative_sentiment_graph.edges(data='weight')]
                ).communities
                ]
        count_graph_comms['leiden']              = [
            set(comm) for comm in algorithms.leiden(
                g_original=count_graph, 
                weights=[e[2] for e in positive_sentiment_graph.edges(data='weight')]
                ).communities
                ]

    # Using louvain algorithm
    if 'louvain' in algos:
        # Using cdlib
        # positive_sentiment_graph_comms['louvain'] = [
        #     set(comm) for comm in algorithms.louvain(
        #         g_original=positive_sentiment_graph.to_undirected(), 
        #         weight='weight', 
        #         resolution=resolution
        #         ).communities
        #         ]
        # neutral_sentiment_graph_comms['louvain']  = [
        #     set(comm) for comm in algorithms.louvain(
        #         g_original=neutral_sentiment_graph.to_undirected(), 
        #         weight='weight', 
        #         resolution=resolution
        #         ).communities
        #         ]
        # negative_sentiment_graph_comms['louvain'] = [
        #     set(comm) for comm in algorithms.louvain(
        #         g_original=negative_sentiment_graph.to_undirected(), 
        #         weight='weight', 
        #         resolution=resolution
        #         ).communities
        #         ]
        # count_graph_comms['louvain']              = [
        #     set(comm) for comm in algorithms.louvain(
        #         g_original=count_graph.to_undirected(), 
        #         weight='weight', 
        #         resolution=resolution
        #         ).communities
        #         ]

        # Using networkx
        positive_sentiment_graph_comms['louvain'] = nx.algorithms.community.louvain_communities(
            positive_sentiment_graph, 
            resolution=resolution, 
            seed=seed
            )
        neutral_sentiment_graph_comms['louvain']  = nx.algorithms.community.louvain_communities(
            neutral_sentiment_graph, 
            resolution=resolution, 
            seed=seed
            )
        negative_sentiment_graph_comms['louvain'] = nx.algorithms.community.louvain_communities(
            negative_sentiment_graph, 
            resolution=resolution, 
            seed=seed
            )
        count_graph_comms['louvain']              = nx.algorithms.community.louvain_communities(
            count_graph, 
            resolution=resolution, 
            seed=seed
            )

    # Using eva algorithm
    if 'eva' in algos:
        positive_sentiment_graph_comms['eva'] = [
            set(comm) for comm in algorithms.eva(
                g_original=positive_sentiment_graph.to_undirected(), 
                labels=positive_sentiment_graph_labels, 
                weight='weight', 
                resolution=resolution, 
                alpha=eva_alpha
                ).communities
                ]
        neutral_sentiment_graph_comms['eva']  = [
            set(comm) for comm in algorithms.eva(
                g_original=neutral_sentiment_graph.to_undirected(), 
                labels=neutral_sentiment_graph_labels, 
                weight='weight', 
                resolution=resolution, 
                alpha=eva_alpha
                ).communities
                ]
        negative_sentiment_graph_comms['eva'] = [
            set(comm) for comm in algorithms.eva(
                g_original=negative_sentiment_graph.to_undirected(), 
                labels=negative_sentiment_graph_labels, 
                weight='weight', 
                resolution=resolution, 
                alpha=eva_alpha
                ).communities
                ]
        count_graph_comms['eva']              = [
            set(comm) for comm in algorithms.eva(
                g_original=count_graph.to_undirected(), 
                labels=count_graph_labels, 
                weight='weight', 
                resolution=resolution, 
                alpha=eva_alpha
                ).communities
                ]

    # Using walktrap algorithm
    if 'walktrap' in algos:
        positive_sentiment_graph_comms['walktrap'] = [
            set(comm) for comm in algorithms.walktrap(
                g_original=positive_sentiment_graph
                ).communities
                ]
        neutral_sentiment_graph_comms['walktrap']  = [
            set(comm) for comm in algorithms.walktrap(
                g_original=neutral_sentiment_graph
                ).communities
                ]
        negative_sentiment_graph_comms['walktrap'] = [
            set(comm) for comm in algorithms.walktrap(
                g_original=negative_sentiment_graph
                ).communities
                ]
        count_graph_comms['walktrap']              = [
            set(comm) for comm in algorithms.walktrap(
                g_original=count_graph
                ).communities
                ]

    # Itterate through algorithms
    for algo in algos:
        positive_sentiment_graph_detected_communities = {}
        for enum in enumerate(positive_sentiment_graph_comms[algo]):
            for node in enum[1]:
                positive_sentiment_graph_detected_communities[node] = enum[0]
        neutral_sentiment_graph_detected_communities = {}
        for enum in enumerate(neutral_sentiment_graph_comms[algo]):
            for node in enum[1]:
                neutral_sentiment_graph_detected_communities[node] = enum[0]
        negative_sentiment_graph_detected_communities = {}
        for enum in enumerate(negative_sentiment_graph_comms[algo]):
            for node in enum[1]:
                negative_sentiment_graph_detected_communities[node] = enum[0]
        count_graph_detected_communities = {}
        for enum in enumerate(count_graph_comms[algo]):
            for node in enum[1]:
                count_graph_detected_communities[node] = enum[0]

        # Create node dataframe
        node_df = pd.DataFrame({'node':nodes, 'set_community':communities, 'set_label':labels})
        node_df['positive_sentiment_graph_detected_community'] = node_df['node'].map(positive_sentiment_graph_detected_communities)
        node_df['neutral_sentiment_graph_detected_community']  = node_df['node'].map(neutral_sentiment_graph_detected_communities)
        node_df['negative_sentiment_graph_detected_community'] = node_df['node'].map(negative_sentiment_graph_detected_communities)
        node_df['count_graph_detected_community']              = node_df['node'].map(count_graph_detected_communities)

        # Create edge dataframe
        edge_df = pd.DataFrame(
                {
                'source_node':source_nodes, 
                'destination_node':destination_nodes, 
                'edge_sentiments':edge_sentiments
                }
            )
        edge_df['set_source_community']                                    = edge_df['source_node'].map(node_df['set_community'])
        edge_df['set_source_label']                                        = edge_df['source_node'].map(node_df['set_label'])
        edge_df['positive_sentiment_graph_detected_source_community']      = edge_df['source_node'].map(node_df['positive_sentiment_graph_detected_community'])
        edge_df['neutral_sentiment_graph_detected_source_community']       = edge_df['source_node'].map(node_df['neutral_sentiment_graph_detected_community'])
        edge_df['negative_sentiment_graph_detected_source_community']      = edge_df['source_node'].map(node_df['negative_sentiment_graph_detected_community'])
        edge_df['count_graph_detected_source_community']                   = edge_df['source_node'].map(node_df['count_graph_detected_community'])
        edge_df['set_destination_community']                               = edge_df['destination_node'].map(node_df['set_community'])
        edge_df['set_destination_label']                                   = edge_df['destination_node'].map(node_df['set_label'])
        edge_df['positive_sentiment_graph_detected_destination_community'] = edge_df['destination_node'].map(node_df['positive_sentiment_graph_detected_community'])
        edge_df['neutral_sentiment_graph_detected_destination_community']  = edge_df['destination_node'].map(node_df['neutral_sentiment_graph_detected_community'])
        edge_df['negative_sentiment_graph_detected_destination_community'] = edge_df['destination_node'].map(node_df['negative_sentiment_graph_detected_community'])
        edge_df['count_graph_detected_destination_community']              = edge_df['destination_node'].map(node_df['count_graph_detected_community'])

        # Helper function to evaluate communities per base graph
        def evaluate_detected_communities(base_graph):
            """
            Evaluate the detected communities based on the given base graph.

            Parameters:
            - base_graph (str): The base graph to evaluate the communities on.

            Returns:
            None
            """

            # Set weight method
            if base_graph == 'positive_sentiment_graph':
                wm = 1
            elif base_graph == 'neutral_sentiment_graph':
                wm = 2
            elif base_graph == 'negative_sentiment_graph':
                wm = 3
            elif base_graph == 'count_graph':
                wm = 0
            
            # Evaluate communities
            for comm in node_df[base_graph+'_detected_community'].unique():
                detected_homophily, detected_isolation, detected_insulation, detected_affinity, detected_purity, detected_conductance, detected_equity, detected_altruism, detected_balance, detected_hostility = evaluate_single_community(comm, node_df, edge_df, base_graph)

                algs.append(algo)
                comms.append(comm)
                homophilies.append(detected_homophily)
                isolations.append(detected_isolation)
                insulations.append(detected_insulation)
                affinities.append(detected_affinity)
                purities.append(detected_purity)
                coundunctances.append(detected_conductance)
                equities.append(detected_equity)
                altruisims.append(detected_altruism)
                balances.append(detected_balance)
                hostilities.append(detected_hostility)
                weight_method.append(wm)

        # Evaluate communities per base graph 
        for base_graph in ['positive_sentiment_graph', 'neutral_sentiment_graph', 'negative_sentiment_graph', 'count_graph']:
            evaluate_detected_communities(base_graph)

    # Store evaluations
    eval_df = pd.DataFrame(
            {
            'algorithm':algs, 
            'community':comms, 
            'homophily':homophilies, 
            'isolation':isolations, 
            'insulation':insulations, 
            'affinity':affinities, 
            'purity':purities, 
            'conductance':coundunctances, 
            'equity':equities, 
            'altruism':altruisims, 
            'balance':balances, 
            'hostility':hostilities, 
            'weight_method':weight_method
            }
        ).sort_values(by='community')

    # Augment evaulations with metadata
    eval_df = augment_evaluation_df(eval_df, homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, resolution, seed)

    return eval_df

def simulate_and_evaluate_parallel(homophily=0.5, isolation=0.5, insulation=0.5, affinity=0.5, num_nodes=10, num_edges=50, num_communities=2, resolution=1, seed=0):
    """
    Simulates a social network and evaluates the communities using multiple algorithms in parallel.

    Parameters:
    - homophily (float): The degree of similarity between connected nodes in terms of their attributes.
    - isolation (float): The tendency of nodes to form connections within their own community.
    - insulation (float): The tendency of nodes to avoid connections outside their own community.
    - affinity (float): The overall attraction between nodes in the network.
    - num_nodes (int): The number of nodes in the network.
    - num_edges (int): The number of edges in the network.
    - num_communities (int): The number of communities to be formed in the network.
    - resolution (int): The resolution parameter used by the community detection algorithms.
    - seed (int): The random seed used for generating the network.

    Returns:
    - eval_df (DataFrame): A DataFrame containing the evaluation results for each community detection algorithm.
    """

    # Generate simulated graphs
    positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph = simulation(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed)
    
    # Evaluate simulated graphs
    eval_df = evaluate_communities_parallel(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed, positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph, resolution, algos=['louvain', 'eva'])

    return eval_df

def evaluate_communities_serial(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed, positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph, resolution=1, algos=['infomap']):
    """
    Evaluate communities in a serial manner.

    Args:
        positive_sentiment_graph (networkx.Graph): Graph representing positive sentiment.
        neutral_sentiment_graph (networkx.Graph): Graph representing neutral sentiment.
        negative_sentiment_graph (networkx.Graph): Graph representing negative sentiment.
        count_graph (networkx.Graph): Graph representing count.
        seed (int, optional): Seed value for random number generator. Defaults to 0.
        resolution (int, optional): Resolution parameter for the Leiden algorithm. Defaults to 1.
        algos (list, optional): List of algorithms to use for community detection. Defaults to ['infomap'].

    Returns:
        pandas.DataFrame: DataFrame containing evaluation results for each community.
    """

    # Initialize evaluation lists
    algs, comms, homophilies, isolations, insulations, affinities, purities, coundunctances, equities, altruisims, balances, hostilities, weight_method = initialize_evaluation_lists()

    # Initialize community dictionaries
    positive_sentiment_graph_comms, neutral_sentiment_graph_comms, negative_sentiment_graph_comms, count_graph_comms = initialize_evaluation_dictionaries()

    # Resolve graph independent node attributes
    nodes = np.array([n for n in count_graph.nodes()])
    labels = np.array([l for l in nx.get_node_attributes(count_graph, 'label').values()])
    communities = np.array([l for l in nx.get_node_attributes(count_graph, 'community').values()])

    # Resolve graph independent edge attributes
    source_nodes = np.array([e[0] for e in count_graph.edges()])
    destination_nodes = np.array([e[0] for e in count_graph.edges()])
    edge_sentiments = np.array([s for s in nx.get_edge_attributes(positive_sentiment_graph, 'weight').values()])

    # Resolve node labels for cdlib
    positive_sentiment_graph_labels = {
        n: {'label': positive_sentiment_graph.nodes[n]['label']} for n in positive_sentiment_graph.nodes
    }
    neutral_sentiment_graph_labels = {
        n: {'label': neutral_sentiment_graph.nodes[n]['label']} for n in neutral_sentiment_graph.nodes
    }
    negative_sentiment_graph_labels = {
        n: {'label': negative_sentiment_graph.nodes[n]['label']} for n in negative_sentiment_graph.nodes
    }
    count_graph_labels = {
        n: {'label': count_graph.nodes[n]['label']} for n in count_graph.nodes
    }

    # Using infomap algorithm
    if 'infomap' in algos:
        positive_sentiment_graph_comms['infomap'] = [set(comm) for comm in algorithms.infomap(g_original=positive_sentiment_graph, flags="-d").communities]
        neutral_sentiment_graph_comms['infomap'] = [set(comm) for comm in algorithms.infomap(g_original=neutral_sentiment_graph, flags="-d").communities]
        negative_sentiment_graph_comms['infomap'] = [set(comm) for comm in algorithms.infomap(g_original=negative_sentiment_graph, flags="-d").communities]
        count_graph_comms['infomap'] = [set(comm) for comm in algorithms.infomap(g_original=count_graph, flags="-d").communities]

    # Using (cx) leiden algorithm
    if 'leiden' in algos:
        positive_sentiment_graph_comms['leiden'], positive_sentiment_graph_comms['leiden_modularity'] = cx.leiden(G=positive_sentiment_graph.to_undirected(), max_iter=1000, resolution=resolution, random_state=seed) ## Returns a dict?
        neutral_sentiment_graph_comms['leiden'], neutral_sentiment_graph_comms['leiden_modularity'] = cx.leiden(G=neutral_sentiment_graph.to_undirected(), max_iter=1000, resolution=resolution, random_state=seed) ## Returns a dict?
        negative_sentiment_graph_comms['leiden'], negative_sentiment_graph_comms['leiden_modularity'] = cx.leiden(G=negative_sentiment_graph.to_undirected(), max_iter=1000, resolution=resolution, random_state=seed) ## Returns a dict?
        count_graph_comms['leiden'], count_graph_comms['leiden_modularity'] = cx.leiden(G=count_graph.to_undirected(), max_iter=1000, resolution=resolution, random_state=seed) ## Returns a dict?

    # Iterate through algorithms
    for algo in algos:
        positive_sentiment_graph_detected_communities = {}
        for enum in enumerate(positive_sentiment_graph_comms[algo]):
            for node in enum[1]:
                positive_sentiment_graph_detected_communities[node] = enum[0]
        neutral_sentiment_graph_detected_communities = {}
        for enum in enumerate(neutral_sentiment_graph_comms[algo]):
            for node in enum[1]:
                neutral_sentiment_graph_detected_communities[node] = enum[0]
        negative_sentiment_graph_detected_communities = {}
        for enum in enumerate(negative_sentiment_graph_comms[algo]):
            for node in enum[1]:
                negative_sentiment_graph_detected_communities[node] = enum[0]
        count_graph_detected_communities = {}
        for enum in enumerate(count_graph_comms[algo]):
            for node in enum[1]:
                count_graph_detected_communities[node] = enum[0]

        # Create node dataframe
        node_df = pd.DataFrame({'node': nodes, 'set_community': communities, 'set_label': labels})
        node_df['positive_sentiment_graph_detected_community'] = node_df['node'].map(positive_sentiment_graph_detected_communities)
        node_df['neutral_sentiment_graph_detected_community'] = node_df['node'].map(neutral_sentiment_graph_detected_communities)
        node_df['negative_sentiment_graph_detected_community'] = node_df['node'].map(negative_sentiment_graph_detected_communities)
        node_df['count_graph_detected_community'] = node_df['node'].map(count_graph_detected_communities)

        # Create edge dataframe
        edge_df = pd.DataFrame(
            {
                'source_node': source_nodes,
                'destination_node': destination_nodes,
                'edge_sentiments': edge_sentiments
            }
        )
        edge_df['set_source_community'] = edge_df['source_node'].map(node_df['set_community'])
        edge_df['set_source_label'] = edge_df['source_node'].map(node_df['set_label'])
        edge_df['positive_sentiment_graph_detected_source_community'] = edge_df['source_node'].map(
            node_df['positive_sentiment_graph_detected_community'])
        edge_df['neutral_sentiment_graph_detected_source_community'] = edge_df['source_node'].map(
            node_df['neutral_sentiment_graph_detected_community'])
        edge_df['negative_sentiment_graph_detected_source_community'] = edge_df['source_node'].map(
            node_df['negative_sentiment_graph_detected_community'])
        edge_df['count_graph_detected_source_community'] = edge_df['source_node'].map(
            node_df['count_graph_detected_community'])
        edge_df['set_destination_community'] = edge_df['destination_node'].map(node_df['set_community'])
        edge_df['set_destination_label'] = edge_df['destination_node'].map(node_df['set_label'])
        edge_df['positive_sentiment_graph_detected_destination_community'] = edge_df['destination_node'].map(
            node_df['positive_sentiment_graph_detected_community'])
        edge_df['neutral_sentiment_graph_detected_destination_community'] = edge_df['destination_node'].map(
            node_df['neutral_sentiment_graph_detected_community'])
        edge_df['negative_sentiment_graph_detected_destination_community'] = edge_df['destination_node'].map(
            node_df['negative_sentiment_graph_detected_community'])
        edge_df['count_graph_detected_destination_community'] = edge_df['destination_node'].map(
            node_df['count_graph_detected_community'])

        # Helper function to evaluate communities per base graph
        def evaluate_detected_communities(base_graph):
            """
            Evaluate the detected communities based on the given base graph.

            Parameters:
            - base_graph (str): The base graph to evaluate the communities on.

            Returns:
            None
            """

            # Set weight method
            if base_graph == 'positive_sentiment_graph':
                wm = 1
            elif base_graph == 'neutral_sentiment_graph':
                wm = 2
            elif base_graph == 'negative_sentiment_graph':
                wm = 3
            elif base_graph == 'count_graph':
                wm = 0

            # Evaluate communities
            for comm in node_df[base_graph + '_detected_community'].unique():
                detected_homophily, detected_isolation, detected_insulation, detected_affinity, detected_purity, detected_conductance, detected_equity, detected_altruism, detected_balance, detected_hostility = evaluate_single_community(
                    comm, node_df, edge_df, base_graph)

                algs.append(algo)
                comms.append(comm)
                homophilies.append(detected_homophily)
                isolations.append(detected_isolation)
                insulations.append(detected_insulation)
                affinities.append(detected_affinity)
                purities.append(detected_purity)
                coundunctances.append(detected_conductance)
                equities.append(detected_equity)
                altruisims.append(detected_altruism)
                balances.append(detected_balance)
                hostilities.append(detected_hostility)
                weight_method.append(wm)

        # Evaluate communities per base graph
        for base_graph in ['positive_sentiment_graph', 'neutral_sentiment_graph', 'negative_sentiment_graph',
                           'count_graph']:
            evaluate_detected_communities(base_graph)

    # Store evaluations
    eval_df = pd.DataFrame(
        {
            'algorithm': algs, 
            'community': comms, 
            'homophily': homophilies, 
            'isolation': isolations,
            'insulation': insulations, 
            'affinity': affinities, 
            'purity': purities, 
            'conductance': coundunctances,
            'equity': equities, 
            'altruism': altruisims, 
            'balance': balances, 
            'hostility': hostilities,
            'weight_method': weight_method
            }
        ).sort_values(by='community')
    
    # Augment evaluations with metadata
    eval_df = augment_evaluation_df(eval_df, homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, resolution, seed)

    return eval_df

def simulate_and_evaluate_serial(homophily=0.5, isolation=0.5, insulation=0.5, affinity=0.5, num_nodes=10, num_edges=50, num_communities=2, resolution=1, seed=0):
    """
    Simulates a social network and evaluates the communities using various algorithms in a serial manner.

    Parameters:
    - homophily (float): The probability of two nodes being connected based on their similarity.
    - isolation (float): The probability of a node being isolated from the rest of the network.
    - insulation (float): The probability of a node forming connections outside its community.
    - affinity (float): The probability of a node forming connections within its community.
    - num_nodes (int): The number of nodes in the network.
    - num_edges (int): The number of edges in the network.
    - num_communities (int): The number of communities in the network.
    - resolution (int): The resolution parameter used by the community detection algorithms.
    - seed (int): The random seed used for reproducibility.

    Returns:
    - eval_df (DataFrame): A DataFrame containing the evaluation results for each community detection algorithm.
    """

    # Generate simulated graphs
    positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph = simulation(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed)
    # Evaluate simulated graphs
    eval_df = evaluate_communities_serial(homophily, isolation, insulation, affinity, num_nodes, num_edges, num_communities, seed, positive_sentiment_graph, neutral_sentiment_graph, negative_sentiment_graph, count_graph, resolution, algos=['infomap'])

    return eval_df
