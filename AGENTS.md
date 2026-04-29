# Agent Operating Guide

This repo is Jayden Piao's public fork for the Partcl/HRT Macro Placement Challenge 2026.

## Mission

Optimize a legal macro placer for the official challenge evaluator without modifying evaluator behavior. The winning path is:

1. Zero hard-macro overlaps on every benchmark.
2. Under 60 minutes per benchmark in the official environment.
3. Top-7 average proxy score on the 17 IBM benchmarks.
4. Credible NG45/OpenROAD behavior for WNS, TNS, and area.

As of 2026-04-29, the public proxy targets are:

- RePlAce baseline average: `1.4578`
- Top-7 public cutoff: `1.3479`
- Current public lead: `1.1172`

## Non-Negotiable Rules

- Do not modify `macro_place/objective.py`, `macro_place/evaluate.py`, or official benchmark data to improve scores.
- Do not use LLMs, VLMs, or hosted model calls inside contest runtime while issue #55 remains unresolved.
- Do not hardcode benchmark-specific final coordinates.
- Do not use external/proprietary placement tools in the submission.
- Hard-macro overlap must be exactly zero; add a positive legalization gap.
- Hard-macro orientations, if added later, are limited to `N`, `FN`, `FS`, and `S`.
- Keep commits and PRs small enough to review.

## Required Startup Context

Every new agent chat should read these files first:

1. `docs/CURRENT_STATE.md`
2. `docs/architecture.md`
3. `docs/experiment-log.md`
4. The role file in `docs/agents/`
5. Official upstream `README.md`, `SETUP.md`, and `SCORING.md`

## Local Commands

```bash
git submodule update --init external/MacroPlacement
uv sync --extra dev
uv run --extra dev pytest
uv run evaluate submissions/jaydenpiao/placer.py -b ibm01
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --benchmarks ibm01 --run-id smoke-ibm01
uv run python scripts/check_results.py results/smoke-ibm01/summary.json
uv run python scripts/compare_results.py results/baseline/summary.json results/candidate/summary.json
```

Use a Linux GPU machine for serious all-benchmark and Docker parity runs.

## Working Model

- Researcher chats propose hypotheses, read papers/issues, and update `docs/experiment-log.md`.
- Coder chats implement one narrow improvement at a time, add tests first for code changes, and record results.
- Promotion requires a saved `results/<run-id>/summary.json` plus the raw log.
