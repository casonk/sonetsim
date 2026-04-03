"""
Unit tests for the public sonetsim simulation and evaluation APIs.
"""

# pylint: disable=redefined-outer-name,missing-function-docstring,duplicate-code

import os
import subprocess
import sys

import numpy as np
import pytest

from sonetsim import GraphEvaluator, GraphSimulator, __version__

PROBABILITY_COLUMNS = [
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
]


def _assert_probability_partition(metrics_df, columns, mask, label):
    partition_rows = metrics_df[mask]
    if partition_rows.empty:
        return

    totals = partition_rows[columns].sum(axis=1)
    assert np.isclose(totals, 1.0, atol=1e-3).all(), f"Sum of {label} must be close to 1"


def _assert_metric_invariants(metrics_df):
    assert len(metrics_df) > 0, "Metrics dataframe must contain at least one community"

    for column in PROBABILITY_COLUMNS:
        assert metrics_df[column].between(0, 1).all(), f"{column} must stay between 0 and 1"

    _assert_probability_partition(
        metrics_df=metrics_df,
        columns=["isolation", "conductance"],
        mask=(metrics_df.num_internal_edges + metrics_df.num_external_edges) > 0,
        label="isolation and conductance",
    )
    _assert_probability_partition(
        metrics_df=metrics_df,
        columns=["affinity", "balance", "hostility"],
        mask=metrics_df.num_internal_edges > 0,
        label="affinity, balance, and hostility",
    )
    _assert_probability_partition(
        metrics_df=metrics_df,
        columns=["insulation", "equity", "altruism"],
        mask=metrics_df.num_external_edges > 0,
        label="insulation, equity, and altruism",
    )


def _assert_weight_methods(metrics_df):
    assert (
        metrics_df["weight_method"].isin([0, 1, 2, 3]).all()
    ), "Weight method (e.g. graph type) must be one of the supported values"


@pytest.fixture
def default_simulator1():
    return GraphSimulator()


@pytest.fixture
def default_simulator2():
    return GraphSimulator()


@pytest.fixture
def custom_simulator1():
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
def custom_simulator2():
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
def default_evaluator1(default_simulator1):
    default_simulator1.simulate()
    return GraphEvaluator(simulator=default_simulator1)


def test__package_exports():
    assert GraphSimulator is not None, "GraphSimulator must be exported from sonetsim"
    assert GraphEvaluator is not None, "GraphEvaluator must be exported from sonetsim"
    assert isinstance(__version__, str), "__version__ must be defined as a string"


def test__package_import__is_quiet():
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    result = subprocess.run(
        [sys.executable, "-c", "import sonetsim; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok", "Package import must not emit extra stdout"
    assert result.stderr.strip() == "", "Package import must not emit stderr noise"


def test__default_init(default_simulator1):
    assert default_simulator1.num_nodes == 50, "Number of nodes must default to 50"
    assert default_simulator1.num_edges == 250, "Number of edges must default to 250"
    assert default_simulator1.num_communities == 5, "Number of communities must default to 5"
    assert (
        default_simulator1.homophily == np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ).all(), "Homophily must default to np.array([0.5, 0.5, 0.5, 0.5, 0.5])"
    assert (
        default_simulator1.isolation == np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ).all(), "Isolation must default to np.array([0.5, 0.5, 0.5, 0.5, 0.5])"
    assert (
        default_simulator1.insulation == np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ).all(), "Insulation must default to np.array([0.5, 0.5, 0.5, 0.5, 0.5])"
    assert (
        default_simulator1.affinity == np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ).all(), "Affinity must default to np.array([0.5, 0.5, 0.5, 0.5, 0.5])"
    assert default_simulator1.seed == 0, "Seed must default to 0"
    assert default_simulator1.nodes is None, "Nodes must be None"
    assert default_simulator1.communities is None, "Communities must be None"
    assert default_simulator1.labels is None, "Labels must be None"
    assert default_simulator1.source_nodes is None, "Source nodes must be None"
    assert default_simulator1.source_communities is None, "Source communities must be None"
    assert (
        default_simulator1.destination_communities is None
    ), "Destination communities must be None"
    assert default_simulator1.destination_nodes is None, "Destination nodes must be None"
    assert default_simulator1.edge_sentiments is None, "Edge sentiments must be None"
    assert (
        default_simulator1.positive_sentiment_graph is None
    ), "Positive sentiment graph must be None"
    assert (
        default_simulator1.neutral_sentiment_graph is None
    ), "Neutral sentiment graph must be None"
    assert (
        default_simulator1.negative_sentiment_graph is None
    ), "Negative sentiment graph must be None"
    assert default_simulator1.count_graph is None, "Count graph must be None"


def test__custom_init(custom_simulator1):
    assert custom_simulator1.num_nodes == 100, "Number of nodes must equal 100"
    assert custom_simulator1.num_edges == 1000, "Number of edges must equal 1000"
    assert custom_simulator1.num_communities == 50, "Number of communities must equal 50"
    assert (
        custom_simulator1.homophily
        == np.array([0.75 for _ in range(custom_simulator1.num_communities)])
    ).all(), "Homophily must default to np.array of size num_communities and input value"
    assert (
        custom_simulator1.isolation
        == np.array([0.65 for _ in range(custom_simulator1.num_communities)])
    ).all(), "Isolation must default to np.array of size num_communities and input value"
    assert (
        custom_simulator1.insulation
        == np.array([0.55 for _ in range(custom_simulator1.num_communities)])
    ).all(), "Insulation must default to np.array of size num_communities and input value"
    assert (
        custom_simulator1.affinity
        == np.array([0.45 for _ in range(custom_simulator1.num_communities)])
    ).all(), "Affinity must default to np.array of size num_communities and input value"
    assert custom_simulator1.seed == 1, "Seed must be set to the input seed value"
    assert custom_simulator1.nodes is None, "Nodes must be None"
    assert custom_simulator1.communities is None, "Communities must be None"
    assert custom_simulator1.labels is None, "Labels must be None"
    assert custom_simulator1.source_nodes is None, "Source nodes must be None"
    assert custom_simulator1.source_communities is None, "Source communities must be None"
    assert custom_simulator1.destination_communities is None, "Destination communities must be None"
    assert custom_simulator1.destination_nodes is None, "Destination nodes must be None"
    assert custom_simulator1.edge_sentiments is None, "Edge sentiments must be None"
    assert (
        custom_simulator1.positive_sentiment_graph is None
    ), "Positive sentiment graph must be None"
    assert custom_simulator1.neutral_sentiment_graph is None, "Neutral sentiment graph must be None"
    assert (
        custom_simulator1.negative_sentiment_graph is None
    ), "Negative sentiment graph must be None"
    assert custom_simulator1.count_graph is None, "Count graph must be None"


@pytest.mark.parametrize(
    ("kwargs", "expected_exception", "message"),
    [
        ({"num_nodes": 0}, ValueError, "num_nodes must be greater than 0"),
        ({"num_edges": 0}, ValueError, "num_edges must be greater than 0"),
        ({"num_communities": 0}, ValueError, "num_communities must be greater than 0"),
        (
            {"homophily": [0.2, 0.3], "num_communities": 5},
            ValueError,
            "homophily must contain exactly 5 values",
        ),
        (
            {"isolation": [0.1, 1.2, 0.2, 0.3, 0.4]},
            ValueError,
            "isolation must contain only values between 0 and 1",
        ),
        (
            {"affinity": "not-a-probability"},
            TypeError,
            "affinity must be a float or a one-dimensional array-like of floats",
        ),
    ],
)
def test__invalid_init__raises(kwargs, expected_exception, message):
    with pytest.raises(expected_exception, match=message):
        GraphSimulator(**kwargs)


def test__default_seeding(default_simulator1, default_simulator2):
    pos_g1, neu_g1, neg_g1, cnt_g1 = default_simulator1.simulate()
    pos_g2, neu_g2, neg_g2, cnt_g2 = default_simulator2.simulate()
    assert pos_g1.nodes() == pos_g2.nodes()
    assert neu_g1.nodes() == neu_g2.nodes()
    assert neg_g1.nodes() == neg_g2.nodes()
    assert cnt_g1.nodes() == cnt_g2.nodes()
    assert pos_g1.edges() == pos_g2.edges()
    assert neu_g1.edges() == neu_g2.edges()
    assert neg_g1.edges() == neg_g2.edges()
    assert cnt_g1.edges() == cnt_g2.edges()


def test__custom_seeding(custom_simulator1, custom_simulator2):
    pos_g1, neu_g1, neg_g1, cnt_g1 = custom_simulator1.simulate()
    pos_g2, neu_g2, neg_g2, cnt_g2 = custom_simulator2.simulate()
    assert pos_g1.nodes() == pos_g2.nodes()
    assert neu_g1.nodes() == neu_g2.nodes()
    assert neg_g1.nodes() == neg_g2.nodes()
    assert cnt_g1.nodes() == cnt_g2.nodes()
    assert pos_g1.edges() == pos_g2.edges()
    assert neu_g1.edges() == neu_g2.edges()
    assert neg_g1.edges() == neg_g2.edges()
    assert cnt_g1.edges() == cnt_g2.edges()


def test__default_metrics__count(default_evaluator1):
    default_evaluator1.evaluate_single_graph(graph="count")
    _assert_metric_invariants(default_evaluator1.metrics_df)


def test__default_metrics__positive(default_evaluator1):
    default_evaluator1.evaluate_single_graph(graph="positive")
    _assert_metric_invariants(default_evaluator1.metrics_df)


def test__default_metrics__neutral(default_evaluator1):
    default_evaluator1.evaluate_single_graph(graph="neutral")
    _assert_metric_invariants(default_evaluator1.metrics_df)


def test__default_metrics__negative(default_evaluator1):
    default_evaluator1.evaluate_single_graph(graph="negative")
    _assert_metric_invariants(default_evaluator1.metrics_df)


def test__default_metrics__all(default_evaluator1):
    default_evaluator1.evaluate()
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)


def test__louvain_metrics__all(default_evaluator1):
    default_evaluator1.evaluate(algorithm="louvain")
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)


def test__leiden_metrics__all(default_evaluator1):
    pytest.importorskip("leidenalg")
    default_evaluator1.evaluate(algorithm="leiden")
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)


def test__eva_metrics__all(default_evaluator1):
    default_evaluator1.evaluate(algorithm="eva")
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)


def test__infomap_metrics__all(default_evaluator1):
    pytest.importorskip("infomap")
    default_evaluator1.evaluate(algorithm="infomap")
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)


def test__all_metrics__all(default_evaluator1):
    available_algorithms = ["louvain", "eva"]

    try:
        __import__("leidenalg")
        available_algorithms.append("leiden")
    except ImportError:
        pass

    try:
        __import__("infomap")
        available_algorithms.append("infomap")
    except ImportError:
        pass

    default_evaluator1.evaluate_algorithms(algorithms=available_algorithms)
    _assert_metric_invariants(default_evaluator1.metrics_df)
    _assert_weight_methods(default_evaluator1.metrics_df)
    assert (
        default_evaluator1.metrics_df["algorithm"].isin(available_algorithms).all()
    ), "Algorithm must be one of the supported values"


def test__evaluate_without_simulate__raises(default_simulator1):
    evaluator = GraphEvaluator(simulator=default_simulator1)
    with pytest.raises(
        RuntimeError,
        match="GraphEvaluator requires a simulated graph. Call simulator.simulate\\(\\) first.",
    ):
        evaluator.evaluate()
