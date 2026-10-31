# Changelog

All notable changes to `sonetsim` are documented here.

## Unreleased

- Build the PyPI distribution with the reusable `python-build.yml`; the upload stays in a caller-local `pypi` job so trusted publishing works, and the release can also be started by hand.

- Added the portfolio-standard governance, continuity, and contributor baseline files.

## 1.0.0 - 2024-03-28

Tag `1.0.0`.

- Rework the simulator as an object-oriented package: `GraphSimulator` and
  `GraphEvaluator` classes with `set_seed()`, edge-type expansion,
  `set_graph` from a string, and evaluation metrics.
- Add pytest coverage, including seeding and artifact checks, and lint, format
  and test workflows across a wider Python version matrix.
- Update the pybind11 requirement to 2.12.0.

## 0.0.1 - 2024-03-26

Tag `0.0.1.x` (GitHub release titled `0.0.1`).

- Bump the package version to 0.0.1 after the initial build and publish setup.

## 0.0.0 - 2024-03-26

Tag `0.0.x` (GitHub release titled `0.0.0`).

- Initial release: the graph simulator, examples, a fix for a destination-edge
  bug, the package rename and build, and a PyPI publish workflow.
- The GitHub pre-release titled `0.0.1.x` uses tag `0.0.1`, which points at this
  same commit. Published tags are not renamed; the mismatch is recorded here.
