# Coder Agent Brief

You are the implementation lane for this competition repo.

## Startup

Read, in order:

1. `AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. `docs/architecture.md`
4. `docs/experiment-log.md`
5. The current issue or plan you are implementing

## Development Rules

- Keep changes narrow and reviewable.
- Add tests before behavior code.
- Do not edit official evaluator or benchmark files to improve scores.
- Keep `submissions/jaydenpiao/placer.py` importable when mounted alone with its sibling files.
- Save experiment output under `results/<run-id>/` and summarize it in `docs/experiment-log.md`.

## Verification Ladder

Use the cheapest command that proves the current change, then climb only as needed:

```bash
uv run --extra dev pytest test/test_jaydenpiao_submission.py
uv run --extra dev pytest test/test_experiment_harness.py
uv run --extra dev pytest
uv run evaluate submissions/jaydenpiao/placer.py -b ibm01
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --benchmarks ibm01 --run-id smoke-ibm01
uv run python scripts/check_results.py results/smoke-ibm01/summary.json
uv run evaluate submissions/jaydenpiao/placer.py --all
```

Use cloud Ubuntu/GPU for Docker parity and serious all-IBM runs.
