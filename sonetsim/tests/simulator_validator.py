import pytest
from sonetsim.sonetsim import GraphSimulator
import numpy as np

@pytest.fixture
def default_validator1():
    return GraphSimulator()

@pytest.fixture
def default_validator2():
    return GraphSimulator()

@pytest.fixture
def custom_validator1():
    return GraphSimulator(
        num_nodes=100,
        num_edges=1000,
        num_communities=50,
        homophily=0.75,
        isolation=0.65,
        insulation=0.55,
        affinity=0.45,
        seed=1
        )

@pytest.fixture
def custom_validator2():
    return GraphSimulator(
        num_nodes=100,
        num_edges=1000,
        num_communities=50,
        homophily=0.75,
        isolation=0.65,
        insulation=0.55,
        affinity=0.45,
        seed=1
        )

def test__default_init(default_validator1):
    assert default_validator1.num_nodes == 10, "Number of nodes must default to 10"
    assert default_validator1.num_edges == 50, "Number of edges must default to 50"
    assert default_validator1.num_communities == 2, "Number of communities must default to 2"
    assert (default_validator1.homophily == np.array([0.5, 0.5])).all(), "Homophily must default to np.array([0.5, 0.5])"
    assert (default_validator1.isolation == np.array([0.5, 0.5])).all(), "Isolation must default to np.array([0.5, 0.5])"
    assert (default_validator1.insulation == np.array([0.5, 0.5])).all(), "Insulation must default to np.array([0.5, 0.5])"
    assert (default_validator1.affinity == np.array([0.5, 0.5])).all(), "Affinity must default to np.array([0.5, 0.5])"
    assert default_validator1.seed == 0, "Seed must be default to 0"
    assert default_validator1.nodes is None, "Nodes must be None"
    assert default_validator1.communities is None, "Communities must be None"
    assert default_validator1.labels is None, "Labels must be None"
    assert default_validator1.source_nodes is None, "Source nodes must be None"
    assert default_validator1.source_communities is None, "Source communities must be None"
    assert default_validator1.destination_communities is None, "Destination communities must be None"
    assert default_validator1.destination_nodes is None, "Destination nodes must be None"
    assert default_validator1.edge_sentiments is None, "Edge sentiments must be None"
    assert default_validator1.positive_sentiment_graph is None, "Positive sentiment graph must be None"
    assert default_validator1.neutral_sentiment_graph is None, "Neutral sentiment graph must be None"
    assert default_validator1.negative_sentiment_graph is None, "Negative sentiment graph must be None"
    assert default_validator1.count_graph is None, "Count graph must be None"
    assert default_validator1.nodes is None, "Nodes must be None"
    assert default_validator1.communities is None, "Communities must be None"
    assert default_validator1.labels is None, "Labels must be None"
    assert default_validator1.source_nodes is None, "Source nodes must be None"
    assert default_validator1.source_communities is None, "Source communities must be None"
    assert default_validator1.destination_communities is None, "Destination communities must be None"
    assert default_validator1.destination_nodes is None, "Destination nodes must be None"

def test__custom_init(custom_validator1):
    assert custom_validator1.num_nodes == 100, "Number of nodes must default to 10"
    assert custom_validator1.num_edges == 1000, "Number of edges must default to 50"
    assert custom_validator1.num_communities == 50, "Number of communities must default to 2"
    assert (custom_validator1.homophily == np.array([0.75 for _ in range(custom_validator1.num_communities)])).all(), "Homophily must default to np.array of size num_communities and input homophily value"
    assert (custom_validator1.isolation == np.array([0.65 for _ in range(custom_validator1.num_communities)])).all(), "Isolation must default to np.array of size num_communities and input isolation value"
    assert (custom_validator1.insulation == np.array([0.55 for _ in range(custom_validator1.num_communities)])).all(), "Insulation must default to np.array of size num_communities and input insulation value"
    assert (custom_validator1.affinity == np.array([0.45 for _ in range(custom_validator1.num_communities)])).all(), "Affinity must default to np.array of size num_communities and input affinity value"
    assert custom_validator1.seed == 1, "Seed must be set to input seed value"
    assert custom_validator1.nodes is None, "Nodes must be None"
    assert custom_validator1.communities is None, "Communities must be None"
    assert custom_validator1.labels is None, "Labels must be None"
    assert custom_validator1.source_nodes is None, "Source nodes must be None"
    assert custom_validator1.source_communities is None, "Source communities must be None"
    assert custom_validator1.destination_communities is None, "Destination communities must be None"
    assert custom_validator1.destination_nodes is None, "Destination nodes must be None"
    assert custom_validator1.edge_sentiments is None, "Edge sentiments must be None"
    assert custom_validator1.positive_sentiment_graph is None, "Positive sentiment graph must be None"
    assert custom_validator1.neutral_sentiment_graph is None, "Neutral sentiment graph must be None"
    assert custom_validator1.negative_sentiment_graph is None, "Negative sentiment graph must be None"
    assert custom_validator1.count_graph is None, "Count graph must be None"
    assert custom_validator1.nodes is None, "Nodes must be None"
    assert custom_validator1.communities is None, "Communities must be None"
    assert custom_validator1.labels is None, "Labels must be None"
    assert custom_validator1.source_nodes is None, "Source nodes must be None"
    assert custom_validator1.source_communities is None, "Source communities must be None"
    assert custom_validator1.destination_communities is None, "Destination communities must be None"
    assert custom_validator1.destination_nodes is None, "Destination nodes must be None"

def test__default_seeding(default_validator1, default_validator2):
    posG1, neuG1, negG1, cntG1 = default_validator1.simulate()
    posG2, neuG2, negG2, cntG2 = default_validator2.simulate()
    assert posG1.nodes() == posG2.nodes(), "Nodes must be the same for positive sentiment graphs"
    assert neuG1.nodes() == neuG2.nodes(), "Nodes must be the same for neutral sentiment graphs"
    assert negG1.nodes() == negG2.nodes(), "Nodes must be the same for negative sentiment graphs"
    assert cntG1.nodes() == cntG2.nodes(), "Nodes must be the same for edge count graphs"
    assert posG1.edges() == posG2.edges(), "Edges must be the same for positive sentiment graphs"
    assert neuG1.edges() == neuG2.edges(), "Edges must be the same for neutral sentiment graphs"
    assert negG1.edges() == negG2.edges(), "Edges must be the same for negative sentiment graphs"
    assert cntG1.edges() == cntG2.edges(), "Edges must be the same for edge count graphs"

def test__custom_seeding(custom_validator1, custom_validator2):
    posG1, neuG1, negG1, cntG1 = custom_validator1.simulate()
    posG2, neuG2, negG2, cntG2 = custom_validator2.simulate()
    assert posG1.nodes() == posG2.nodes(), "Nodes must be the same for positive sentiment graphs"
    assert neuG1.nodes() == neuG2.nodes(), "Nodes must be the same for neutral sentiment graphs"
    assert negG1.nodes() == negG2.nodes(), "Nodes must be the same for negative sentiment graphs"
    assert cntG1.nodes() == cntG2.nodes(), "Nodes must be the same for edge count graphs"
    assert posG1.edges() == posG2.edges(), "Edges must be the same for positive sentiment graphs"
    assert neuG1.edges() == neuG2.edges(), "Edges must be the same for neutral sentiment graphs"
    assert negG1.edges() == negG2.edges(), "Edges must be the same for negative sentiment graphs"
    assert cntG1.edges() == cntG2.edges(), "Edges must be the same for edge count graphs"