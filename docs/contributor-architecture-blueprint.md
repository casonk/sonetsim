# Contributor Architecture Blueprint

This document is a concise map of how `sonetsim` turns simulation parameters into weighted graph variants, detected communities, and evaluation metrics suitable for research comparison.

## High-Level Layers

1. Public package surface (`sonetsim/__init__.py`)
   - The package re-exports `GraphSimulator`, `GraphEvaluator`, and `__version__`.
   - Because the project is published to PyPI, preserve that import surface intentionally.
2. Simulation layer (`GraphSimulator` in `sonetsim/sonetsim.py`)
   - Validates graph size and probability inputs.
   - Seeds Python and NumPy RNG state.
   - Generates node communities, labels, source and destination edges, and edge sentiments.
   - Materializes four directed graph variants: count, positive sentiment, neutral sentiment, and negative sentiment.
3. Evaluation layer (`GraphEvaluator` in `sonetsim/sonetsim.py`)
   - Requires a prior simulation run.
   - Selects one graph view at a time and runs community detection with `louvain`, `leiden`, `eva`, or `infomap`.
   - Builds node and edge DataFrames and computes community-level metrics such as homophily, isolation, insulation, affinity, conductance, and hostility.
   - Can aggregate results across all graph weight methods and across multiple algorithms.
4. Validation and release layer (`sonetsim/tests/simulator_validator.py`, `pyproject.toml`, workflows)
   - The validator suite checks package exports, import quietness, deterministic seeding, invalid-input handling, and metric invariants.
   - Packaging metadata and CI workflows keep the published package, dependency contract, and release flow aligned.

## Key Flows

- Simulation flow: constructor parameters -> validation -> seeded graph-data generation -> four weighted `networkx.DiGraph` instances
- Single-graph evaluation flow: simulated graph variant -> community detection -> node and edge DataFrames -> community metrics DataFrame
- Full experiment flow: one simulator run -> all four graph variants -> one or more algorithms -> concatenated metrics DataFrame with `weight_method` and optional `algorithm`
- Validation flow: pytest fixtures -> simulator/evaluator APIs -> invariants on reproducibility, supported algorithms, and metric ranges

## Key Entry Points

- `GraphSimulator(...).simulate()`: generate the graph variants
- `GraphEvaluator(simulator=...).evaluate()`: evaluate all four weight methods for one algorithm
- `GraphEvaluator(...).evaluate_algorithms(...)`: compare multiple algorithms against the same simulated network
- `sonetsim/tests/simulator_validator.py`: public API and invariant validator
- `.github/workflows/black-pylint-pytest.yml`: lint and test enforcement
- `.github/workflows/python-publish.yml`: release publication path

## Validation

```bash
poetry install
poetry run pytest sonetsim/tests/
black --check .
pylint sonetsim/ --fail-under=10
```

When simulation or evaluation behavior changes, verify both the direct API behavior and the metric invariants enforced by the validator suite.
