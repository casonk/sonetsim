# LESSONSLEARNED.md

Tracked durable lessons for `sonetsim`.
Unlike `CHATHISTORY.md`, this file should keep only reusable lessons that should change how future sessions work in this repo.

## How To Use

- Read this file after `AGENTS.md` and before `CHATHISTORY.md` when resuming work.
- Add lessons that generalize beyond a single session.
- Keep entries concise and action-oriented.
- Do not use this file for transient status updates or full session logs.

## Lessons

### 2026-03-26 — Keep CI on the base dependency set used by the validator

- `requirements.txt` includes optional research extras and native-build packages that are not required for the public API lint/test workflow.
- For GitHub Actions lint/test jobs, install the base dependencies needed by the validator instead of the full exported requirements set unless the job is explicitly exercising those optional integrations.
