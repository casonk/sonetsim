# Contributor Architecture Blueprint

This document is a concise map of how `sonetsim` exposes its simulation library, validation suite, and release metadata.

## High-Level Layers

1. Package surface (`sonetsim/__init__.py`)
   - Re-exports the public library interface.
   - Public API changes should be intentional because the package is published.
2. Simulation engine (`sonetsim/sonetsim.py`)
   - Core network simulation and community-detection logic live here.
   - Algorithm changes should be paired with validation updates.
3. Test layer (`sonetsim/tests/simulator_validator.py`)
   - The validator suite is the main protection against behavioral regressions.
   - Extend the existing style instead of introducing a parallel testing pattern.
4. Packaging and release layer (`pyproject.toml`, `requirements.txt`, GitHub workflows)
   - `pyproject.toml` is the source of truth for package metadata and versions.
   - CI and publish workflows enforce formatting, linting, testing, and release automation.

## Key Entry Points

- `poetry run pytest sonetsim/tests/`
- `poetry build`
- `.github/workflows/black-pylint-pytest.yml`
- `.github/workflows/python-publish.yml`

## Validation

```bash
poetry install
poetry run pytest sonetsim/tests/
black --check .
pylint sonetsim/
```
