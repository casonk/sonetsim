"""
This module contains unit tests for the 
`GraphSimulator` and `GraphEvaluator` classes in the `sonetsim` package.

The tests cover the default initialization of the `GraphSimulator` class, 
custom initialization of the `GraphSimulator` class, 
default seeding, custom seeding, and default metrics evaluation.

The `GraphSimulator` class is responsible for simulating graph data with different configurations, 
while the `GraphEvaluator` class is responsible for evaluating the metrics of the simulated graphs.

The unit tests ensure that the `GraphSimulator` and `GraphEvaluator` classes 
are functioning correctly by asserting the expected values of various attributes and metrics.

Note: This module requires the `pytest` and `numpy` packages to run the tests.
"""

import pytest
import numpy as np
from sonetsim.sonetsim import GraphSimulator
from sonetsim.sonetsim import GraphEvaluator


@pytest.fixture
def default_validator1():
    """
    This function returns an instance of the GraphSimulator class.

    Returns:
        GraphSimulator: An instance of the GraphSimulator class.
    """
    return GraphSimulator()


@pytest.fixture
def default_validator2():
    """
    Returns an instance of GraphSimulator.

    This function serves as a default validator for the simulator. It creates and returns
    an instance of the GraphSimulator class.

    Returns:
        GraphSimulator: An instance of the GraphSimulator class.

    """
    return GraphSimulator()


@pytest.fixture
def custom_validator1():
    """
    This function creates and returns a GraphSimulator object with custom parameters.

    Returns:
        GraphSimulator: A GraphSimulator object with the specified parameters.
    """
    return GraphSimulator(
        num_nodes=100,
        num_edges=1000,
        num_communities=50,
        homophily=0.75,
        isolation=0.65,
        insulation=0.55,
        affinity=0.45,
        seed=1,
    )


@pytest.fixture
def custom_validator2():
    """
    Returns a GraphSimulator object with custom parameters.

    This function creates a GraphSimulator object with custom parameters for simulation.
    The GraphSimulator simulates a graph with a specified number of nodes, edges, communities,
    and various other parameters.

    Returns:
        GraphSimulator: A GraphSimulator object with custom parameters.
    """
    return GraphSimulator(
        num_nodes=100,
        num_edges=1000,
        num_communities=50,
        homophily=0.75,
        isolation=0.65,
        insulation=0.55,
        affinity=0.45,
        seed=1,
    )


def test__default_init(default_validator1):
    """
    Test case for the default initialization of the `default_validator1` object.

    This test case checks that the `default_validator1` object
    is correctly initialized with the default values.

    Assertions:
    - `num_nodes` must default to 10
    - `num_edges` must default to 50
    - `num_communities` must default to 2
    - `homophily` must default to np.array([0.5, 0.5])
    - `isolation` must default to np.array([0.5, 0.5])
    - `insulation` must default to np.array([0.5, 0.5])
    - `affinity` must default to np.array([0.5, 0.5])
    - `seed` must default to 0
    - `nodes` must be None
    - `communities` must be None
    - `labels` must be None
    - `source_nodes` must be None
    - `source_communities` must be None
    - `destination_communities` must be None
    - `destination_nodes` must be None
    - `edge_sentiments` must be None
    - `positive_sentiment_graph` must be None
    - `neutral_sentiment_graph` must be None
    - `negative_sentiment_graph` must be None
    - `count_graph` must be None
    """
    assert default_validator1.num_nodes == 10, "Number of nodes must default to 10"
    assert default_validator1.num_edges == 50, "Number of edges must default to 50"
    assert (
        default_validator1.num_communities == 2
    ), "Number of communities must default to 2"
    assert (
        default_validator1.homophily == np.array([0.5, 0.5])
    ).all(), "Homophily must default to np.array([0.5, 0.5])"
    assert (
        default_validator1.isolation == np.array([0.5, 0.5])
    ).all(), "Isolation must default to np.array([0.5, 0.5])"
    assert (
        default_validator1.insulation == np.array([0.5, 0.5])
    ).all(), "Insulation must default to np.array([0.5, 0.5])"
    assert (
        default_validator1.affinity == np.array([0.5, 0.5])
    ).all(), "Affinity must default to np.array([0.5, 0.5])"
    assert default_validator1.seed == 0, "Seed must be default to 0"
    assert default_validator1.nodes is None, "Nodes must be None"
    assert default_validator1.communities is None, "Communities must be None"
    assert default_validator1.labels is None, "Labels must be None"
    assert default_validator1.source_nodes is None, "Source nodes must be None"
    assert (
        default_validator1.source_communities is None
    ), "Source communities must be None"
    assert (
        default_validator1.destination_communities is None
    ), "Destination communities must be None"
    assert (
        default_validator1.destination_nodes is None
    ), "Destination nodes must be None"
    assert default_validator1.edge_sentiments is None, "Edge sentiments must be None"
    assert (
        default_validator1.positive_sentiment_graph is None
    ), "Positive sentiment graph must be None"
    assert (
        default_validator1.neutral_sentiment_graph is None
    ), "Neutral sentiment graph must be None"
    assert (
        default_validator1.negative_sentiment_graph is None
    ), "Negative sentiment graph must be None"
    assert default_validator1.count_graph is None, "Count graph must be None"
    assert default_validator1.nodes is None, "Nodes must be None"
    assert default_validator1.communities is None, "Communities must be None"
    assert default_validator1.labels is None, "Labels must be None"
    assert default_validator1.source_nodes is None, "Source nodes must be None"
    assert (
        default_validator1.source_communities is None
    ), "Source communities must be None"
    assert (
        default_validator1.destination_communities is None
    ), "Destination communities must be None"
    assert (
        default_validator1.destination_nodes is None
    ), "Destination nodes must be None"


def test__custom_init(custom_validator1):
    """
    Test case for the custom initialization of the `custom_validator1` object.

    This test verifies that the `custom_validator1` object is
    correctly initialized with the default values for its attributes.

    The following assertions are made:
    - `num_nodes` is set to 100
    - `num_edges` is set to 1000
    - `num_communities` is set to 50
    - `homophily` is set to an array of size `num_communities` with all elements equal to 0.75
    - `isolation` is set to an array of size `num_communities` with all elements equal to 0.65
    - `insulation` is set to an array of size `num_communities` with all elements equal to 0.55
    - `affinity` is set to an array of size `num_communities` with all elements equal to 0.45
    - `seed` is set to 1
    - `nodes`, `communities`, `labels`, `source_nodes`, `source_communities`,
    `destination_communities`, `destination_nodes`, `edge_sentiments`,
    `positive_sentiment_graph`, `neutral_sentiment_graph`, `negative_sentiment_graph`, `count_graph`
    are all set to None

    If any of the assertions fail, an appropriate error message is raised.

    Parameters:
    - `custom_validator1`: The `custom_validator1` object to be tested.
    """

    assert custom_validator1.num_nodes == 100, "Number of nodes must default to 10"
    assert custom_validator1.num_edges == 1000, "Number of edges must default to 50"
    assert (
        custom_validator1.num_communities == 50
    ), "Number of communities must default to 2"
    assert (
        custom_validator1.homophily
        == np.array([0.75 for _ in range(custom_validator1.num_communities)])
    ).all(), (
        "Homophily must default to np.array of size num_communities and input value"
    )
    assert (
        custom_validator1.isolation
        == np.array([0.65 for _ in range(custom_validator1.num_communities)])
    ).all(), (
        "Isolation must default to np.array of size num_communities and input value"
    )
    assert (
        custom_validator1.insulation
        == np.array([0.55 for _ in range(custom_validator1.num_communities)])
    ).all(), (
        "Insulation must default to np.array of size num_communities and input value"
    )
    assert (
        custom_validator1.affinity
        == np.array([0.45 for _ in range(custom_validator1.num_communities)])
    ).all(), "Affinity must default to np.array of size num_communities and input value"
    assert custom_validator1.seed == 1, "Seed must be set to input seed value"
    assert custom_validator1.nodes is None, "Nodes must be None"
    assert custom_validator1.communities is None, "Communities must be None"
    assert custom_validator1.labels is None, "Labels must be None"
    assert custom_validator1.source_nodes is None, "Source nodes must be None"
    assert (
        custom_validator1.source_communities is None
    ), "Source communities must be None"
    assert (
        custom_validator1.destination_communities is None
    ), "Destination communities must be None"
    assert custom_validator1.destination_nodes is None, "Destination nodes must be None"
    assert custom_validator1.edge_sentiments is None, "Edge sentiments must be None"
    assert (
        custom_validator1.positive_sentiment_graph is None
    ), "Positive sentiment graph must be None"
    assert (
        custom_validator1.neutral_sentiment_graph is None
    ), "Neutral sentiment graph must be None"
    assert (
        custom_validator1.negative_sentiment_graph is None
    ), "Negative sentiment graph must be None"
    assert custom_validator1.count_graph is None, "Count graph must be None"
    assert custom_validator1.nodes is None, "Nodes must be None"
    assert custom_validator1.communities is None, "Communities must be None"
    assert custom_validator1.labels is None, "Labels must be None"
    assert custom_validator1.source_nodes is None, "Source nodes must be None"
    assert (
        custom_validator1.source_communities is None
    ), "Source communities must be None"
    assert (
        custom_validator1.destination_communities is None
    ), "Destination communities must be None"
    assert custom_validator1.destination_nodes is None, "Destination nodes must be None"


def test__default_seeding(default_validator1, default_validator2):
    """
    Test function to validate the default seeding of two simulators.

    Args:
        default_validator1: The first default validator object.
        default_validator2: The second default validator object.

    Returns:
        None

    Raises:
        AssertionError: If the nodes or edges of the simulated graphs are not the same.

    """
    pos_g1, neu_g1, neg_g1, cnt_g1 = default_validator1.simulate()
    pos_g2, neu_g2, neg_g2, cnt_g2 = default_validator2.simulate()
    assert (
        pos_g1.nodes() == pos_g2.nodes()
    ), "Nodes must be the same for positive sentiment graphs"
    assert (
        neu_g1.nodes() == neu_g2.nodes()
    ), "Nodes must be the same for neutral sentiment graphs"
    assert (
        neg_g1.nodes() == neg_g2.nodes()
    ), "Nodes must be the same for negative sentiment graphs"
    assert (
        cnt_g1.nodes() == cnt_g2.nodes()
    ), "Nodes must be the same for edge count graphs"
    assert (
        pos_g1.edges() == pos_g2.edges()
    ), "Edges must be the same for positive sentiment graphs"
    assert (
        neu_g1.edges() == neu_g2.edges()
    ), "Edges must be the same for neutral sentiment graphs"
    assert (
        neg_g1.edges() == neg_g2.edges()
    ), "Edges must be the same for negative sentiment graphs"
    assert (
        cnt_g1.edges() == cnt_g2.edges()
    ), "Edges must be the same for edge count graphs"


def test__custom_seeding(custom_validator1, custom_validator2):
    """
    Test function to validate the custom seeding simulation.

    Args:
        custom_validator1: The first custom validator object.
        custom_validator2: The second custom validator object.

    Raises:
        AssertionError: If the nodes or edges of the sentiment graphs
        or edge count graphs are not the same.

    Returns:
        None
    """
    pos_g1, neu_g1, neg_g1, cnt_g1 = custom_validator1.simulate()
    pos_g2, neu_g2, neg_g2, cnt_g2 = custom_validator2.simulate()
    assert (
        pos_g1.nodes() == pos_g2.nodes()
    ), "Nodes must be the same for positive sentiment graphs"
    assert (
        neu_g1.nodes() == neu_g2.nodes()
    ), "Nodes must be the same for neutral sentiment graphs"
    assert (
        neg_g1.nodes() == neg_g2.nodes()
    ), "Nodes must be the same for negative sentiment graphs"
    assert (
        cnt_g1.nodes() == cnt_g2.nodes()
    ), "Nodes must be the same for edge count graphs"
    assert (
        pos_g1.edges() == pos_g2.edges()
    ), "Edges must be the same for positive sentiment graphs"
    assert (
        neu_g1.edges() == neu_g2.edges()
    ), "Edges must be the same for neutral sentiment graphs"
    assert (
        neg_g1.edges() == neg_g2.edges()
    ), "Edges must be the same for negative sentiment graphs"
    assert (
        cnt_g1.edges() == cnt_g2.edges()
    ), "Edges must be the same for edge count graphs"


@pytest.fixture
def default_evaluator1(default_validator1):
    """
    This function takes a `default_validator1` object,
    simulates it, and returns a `GraphEvaluator` object.

    Parameters:
    default_validator1 (object): The `default_validator1` object to be simulated.

    Returns:
    GraphEvaluator: The `GraphEvaluator` object created from the simulated `default_validator1`.
    """
    default_validator1.simulate()
    return GraphEvaluator(simulator=default_validator1)


@pytest.fixture
def default_evaluator2(default_validator2):
    """
    This function takes a `default_validator2` object and performs a simulation using it.
    It then returns a `GraphEvaluator` object initialized with the `default_validator2` simulator.

    Parameters:
    - default_validator2: The `default_validator2` object to be used for simulation.

    Returns:
    - A `GraphEvaluator` object initialized with the `default_validator2` simulator.
    """
    default_validator2.simulate()
    return GraphEvaluator(simulator=default_validator2)


def test__default_metrics__count(default_evaluator1):
    """
    Test function to evaluate default metrics for counting communities.

    Args:
        default_evaluator1: An instance of the default evaluator class.

    Raises:
        AssertionError: If the validation communities do not meet the specified criteria.
    """
    default_evaluator1.evaluate(graph="count")
    node_mask = default_evaluator1.metrics_df.num_nodes > 1
    internal_edge_mask = default_evaluator1.metrics_df.num_internal_edges > 1
    external_edge_mask = default_evaluator1.metrics_df.num_external_edges > 1
    validation_communities_df = default_evaluator1.metrics_df[
        node_mask & internal_edge_mask & external_edge_mask
    ]
    assert (
        len(validation_communities_df) > 0
    ), "Validation communities must have at least 1 community"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) > 0.999
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) < 1.001
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"


def test__default_metrics__positive(default_evaluator1):
    """
    Test function to evaluate default metrics for positive communities.

    Args:
        default_evaluator1: An instance of the default evaluator class.

    Raises:
        AssertionError: If the validation communities do not meet the specified criteria.
    """
    default_evaluator1.evaluate(graph="positive")
    node_mask = default_evaluator1.metrics_df.num_nodes > 1
    internal_edge_mask = default_evaluator1.metrics_df.num_internal_edges > 1
    external_edge_mask = default_evaluator1.metrics_df.num_external_edges > 1
    validation_communities_df = default_evaluator1.metrics_df[
        node_mask & internal_edge_mask & external_edge_mask
    ]
    assert (
        len(validation_communities_df) > 0
    ), "Validation communities must have at least 1 community"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) > 0.999
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) < 1.001
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"


def test__default_metrics__neutral(default_evaluator1):
    """
    Test function to evaluate default metrics for neutral communities.

    Args:
        default_evaluator1: An instance of the default evaluator class.

    Raises:
        AssertionError: If the validation communities do not meet the specified criteria.
    """
    default_evaluator1.evaluate(graph="neutral")
    node_mask = default_evaluator1.metrics_df.num_nodes > 1
    internal_edge_mask = default_evaluator1.metrics_df.num_internal_edges > 1
    external_edge_mask = default_evaluator1.metrics_df.num_external_edges > 1
    validation_communities_df = default_evaluator1.metrics_df[
        node_mask & internal_edge_mask & external_edge_mask
    ]
    assert (
        len(validation_communities_df) > 0
    ), "Validation communities must have at least 1 community"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) > 0.999
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) < 1.001
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"


def test__default_metrics__negative(default_evaluator1):
    """
    Test function to evaluate default metrics for negative communities.

    Args:
        default_evaluator1: An instance of the default evaluator class.

    Raises:
        AssertionError: If the validation communities do not meet the specified criteria.
    """
    default_evaluator1.evaluate(graph="negative")
    node_mask = default_evaluator1.metrics_df.num_nodes > 1
    internal_edge_mask = default_evaluator1.metrics_df.num_internal_edges > 1
    external_edge_mask = default_evaluator1.metrics_df.num_external_edges > 1
    validation_communities_df = default_evaluator1.metrics_df[
        node_mask & internal_edge_mask & external_edge_mask
    ]
    assert (
        len(validation_communities_df) > 0
    ), "Validation communities must have at least 1 community"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) > 0.999
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["isolation", "conductance"]].sum(axis=1) < 1.001
    ).all(), "Sum of isolation and conductance must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["affinity", "balance", "hostility"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of affinity, balance, and hostility must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        > 0.999
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"
    assert (
        validation_communities_df[["insulation", "equity", "altruism"]].sum(axis=1)
        < 1.001
    ).all(), "Sum of insulation, equity, and altruism must be close to 1"
