# LESSONSLEARNED.md

Tracked durable lessons for `sonetsim`.
Unlike `CHATHISTORY.md`, this file should keep only reusable lessons that should change how future sessions work in this repo.

## How To Use

- Read this file after `AGENTS.md` and before `CHATHISTORY.md` when resuming work.
- Add lessons that generalize beyond a single session.
- Keep entries concise and action-oriented.
- Do not use this file for transient status updates or full session logs.

## Lessons

- Document the repository around its real execution, curation, or integration flow instead of only the top-level folder list.
- Keep local-only, private, reference-only, or generated boundaries explicit so published or runtime behavior is not confused with offline material or non-committable inputs.
- Re-run repo-appropriate validation after changing generated artifacts, diagrams, workflows, or other CI-facing files so formatting and compatibility issues are caught before push.

### 2026-04-03 — Older CI matrices must not require dyno-lab unconditionally

- `dyno-lab` currently requires Python `>=3.10`, while `sonetsim` still supports Python `3.8` through `3.10`.
- Keep `pytest_plugins` loading for `dyno_lab.fixtures` conditional in `sonetsim/tests/conftest.py` so older CI jobs continue to run when that optional package is unavailable.

### 2026-03-26 — Keep CI on the base dependency set used by the validator

- `requirements.txt` includes optional research extras and native-build packages that are not required for the public API lint/test workflow.
- For GitHub Actions lint/test jobs, install the base dependencies needed by the validator instead of the full exported requirements set unless the job is explicitly exercising those optional integrations.

### 2026-03-27 — `sonetsim` architecture is a simulation-evaluation loop, not just a package boundary

- When documenting or changing `sonetsim`, center the architecture on `GraphSimulator` producing four weighted graph variants and `GraphEvaluator` re-running community detection plus metrics over those variants.
- Treat `sonetsim/tests/simulator_validator.py` as a validator of invariants for that experiment loop, not as a generic afterthought.
