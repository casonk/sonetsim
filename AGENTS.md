# AGENTS.md

## Project Purpose

A Python library for **so**cial **net**work **sim**ulation and community detection research. Published to PyPI. Developed as part of an ongoing research project with the University of Michigan - Flint.

## Repository Layout

- `sonetsim/` — Main Python package
  - `__init__.py` — Package init
  - `simulator.py` — Core simulation engine
  - `tests/simulator_validator.py` — Pytest test suite (~876 lines)
- `pyproject.toml` — Poetry-based package metadata and dependency management
- `poetry.lock` — Locked dependency versions
- `requirements.txt` — Pip-compatible dependency list
- `.github/workflows/` — CI/CD pipelines
  - `black-pylint-pytest.yml` — Lint and test on push
  - `python-publish.yml` — PyPI publishing on release
- `LICENSE` — LGPL-2.1
- `README.md` — Project overview

## Setup

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

## Commands

```bash
# Run tests
poetry run pytest sonetsim/tests/
# Or: PYTHONPATH=. pytest sonetsim/tests/

# Check formatting
black --check .

# Lint
pylint sonetsim/ --fail-under=10

# Build package
poetry build

# Publish (maintainer only)
poetry publish
```

## Operating Rules

- Maintain Python 3.8–3.10 compatibility (per pyproject.toml constraint).
- Run `black`, `pylint`, and `pytest` before committing — CI enforces all three.
- Keep `pyproject.toml` as the single source of truth for version and dependencies.
- Update `requirements.txt` when changing `pyproject.toml` dependencies.
- When bumping version, update `pyproject.toml` and create a matching git tag.
- Preserve the existing test patterns in `simulator_validator.py` — use pytest fixtures.
- Do not add runtime dependencies without strong justification.

## Release Process

1. Update version in `pyproject.toml`.
2. Commit: `chore: bump version to X.Y.Z`
3. Tag: `git tag -a X.Y.Z -m "Release X.Y.Z"`
4. Push tag: `git push origin X.Y.Z`
5. GitHub Actions workflow publishes to PyPI automatically on release.

## Portfolio Standards Reference

For portfolio-wide repository standards and baseline conventions, consult the control-plane repo at `./util-repos/traction-control` from the portfolio root.

Start with:
- `./util-repos/traction-control/AGENTS.md`
- `./util-repos/traction-control/README.md`
- `./util-repos/traction-control/LESSONSLEARNED.md`

Shared implementation repos available portfolio-wide:
- `./util-repos/archility` for architecture inventory, blueprint scaffolding, and architecture-documentation drift checks
- `./util-repos/auto-pass` for KeePassXC-backed password management and secret retrieval/update flows
- `./util-repos/nordility` for NordVPN-based VPN switching and connection orchestration
- `./util-repos/shock-relay` for external messaging across supported providers such as Signal, Telegram, Twilio SMS, WhatsApp, and Gmail IMAP

When another repo needs architecture inventory/scaffolding, password management, VPN switching, or external messaging, prefer integrating with these repos instead of re-implementing the capability locally.

## Agent Memory

Use `./LESSONSLEARNED.md` as the tracked durable lessons file for this repo.
Use `./CHATHISTORY.md` as the standard local handoff file for this repo.

- `LESSONSLEARNED.md` is tracked and should capture only reusable lessons.
- `CHATHISTORY.md` is local-only, gitignored, and should capture transient handoff context.
- Read `LESSONSLEARNED.md` and `CHATHISTORY.md` after `AGENTS.md` when resuming work.
- Add durable lessons to `LESSONSLEARNED.md` when they should influence future sessions.
- Keep transient entries brief and focused on simulation changes, validation status, blockers, and next steps.
