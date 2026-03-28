# sonetsim

[![PyPI](https://img.shields.io/pypi/v/sonetsim)](https://pypi.org/project/sonetsim/)
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)
[![CI](https://github.com/casonk/sonetsim/actions/workflows/black-pylint-pytest.yml/badge.svg)](https://github.com/casonk/sonetsim/actions)

A library for **so**cial **net**work **sim**ulation and community detection.

## About

Authored by Cason Konzer as part of an ongoing research project with the University of Michigan — Flint.

## Installation

```bash
pip install sonetsim
```

Or from source with Poetry:
```bash
git clone https://github.com/casonk/sonetsim.git
cd sonetsim
poetry install
```

## Supported Python

3.8 · 3.9 · 3.10

## Dependencies

- [cdlib](https://cdlib.readthedocs.io/) — Community detection
- [bayanpy](https://github.com/saref/bayan) — Bayan algorithm
- [leidenalg](https://leidenalg.readthedocs.io/) — Leiden algorithm
- [infomap](https://mapequation.github.io/infomap/) — Infomap algorithm
- [wurlitzer](https://github.com/minrk/wurlitzer) — C-level output capture

## Development

```bash
poetry run pytest sonetsim/tests/    # tests
black .                               # format
pylint sonetsim/ --fail-under=10      # lint
```

## Architecture

`sonetsim` is organized around a simulation-evaluation loop:

- `GraphSimulator` validates inputs, seeds RNG state, assigns communities and labels, generates directed edges and sentiments, and emits four weighted graph views: count, positive, neutral, and negative.
- `GraphEvaluator` selects one of those graph views, runs community detection (`louvain`, `leiden`, `eva`, or `infomap`), builds node and edge dataframes, and computes community-level metrics.
- `sonetsim/tests/simulator_validator.py` enforces package-export behavior, seeded reproducibility, supported algorithm coverage, and metric invariants across the graph variants.

## License

[LGPL-2.1](LICENSE)
