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

## License

[LGPL-2.1](LICENSE)
