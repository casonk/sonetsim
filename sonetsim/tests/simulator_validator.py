import pytest
from sonetsim.sonetsim import GraphSimulator


@pytest.fixture
def validator():
    return GraphSimulator()

def test__num_nodes(validator):
    assert validator.num_nodes > 0, "Number of nodes must be greater than 0"

def test__num_edges(validator):
    assert validator.num_edges > 0, "Number of edges must be greater than 0"


    # assert validator.num_communities > 0, "Number of communities must be greater than 0"
    # assert validator.homophily >= 0 and validator.homophily <= 1, "Parameter homophily must be between 0 and 1"
    # assert validator.isolation >= 0 and validator.isolation <= 1, "Parameter isolation must be between 0 and 1"
    # assert validator.insulation >= 0 and validator.insulation <= 1, "Parameter insulation must be between 0 and 1"
    # assert validator.affinity >= 0 and validator.affinity <= 1, "Parameter affinity must be between 0 and 1"
    # assert validator.seed >= 0, "Seed must be greater than or equal to 0"
    # assert validator.nodes is None, "Nodes must be None"
    # assert validator.communities is None, "Communities must be None"
    # assert validator.labels is None, "Labels must be None"
    # assert validator.source_nodes is None, "Source nodes must be None"
    # assert validator.source_communities is None, "Source communities must be None"
    # assert validator.destination_communities is None, "Destination communities must be None"
    # assert validator.destination_nodes is None, "Destination nodes must be None"
    # assert validator.edge_sentiments is None, "Edge sentiments must be None"
    # assert validator.positive_sentiment_graph is None, "Positive sentiment graph must be None"
    # assert validator.neutral_sentiment_graph is None, "Neutral sentiment graph must be None"
    # assert validator.negative_sentiment_graph is None, "Negative sentiment graph must be None"
    # assert validator.count_graph is None, "Count graph must be None"
    # assert validator.nodes is None, "Nodes must be None"
    # assert validator.communities is None, "Communities must be None"
    # assert validator.labels is None, "Labels must be None"
    # assert validator.source_nodes is None, "Source nodes must be None"
    # assert validator.source_communities is None, "Source communities must be None"
    # assert validator.destination_communities is None, "Destination communities must be None"
    # assert validator.destination_nodes is None, "Destination nodes must be None"