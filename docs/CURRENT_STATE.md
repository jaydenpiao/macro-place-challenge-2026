# Current State

Last updated: 2026-04-29

## Repository

- Fork: `https://github.com/jaydenpiao/macro-place-challenge-2026`
- Upstream: `https://github.com/partcleda/macro-place-challenge-2026`
- Primary working branch: `infra/bootstrap-foundation`
- Local implementation worktree: `/Users/jaydenpiao/.config/superpowers/worktrees/macro-place-challenge-2026/bootstrap-foundation`

## Verified Baseline

Local bootstrap completed on macOS arm64 with `uv 0.11.7`.

Working commands:

```bash
git submodule update --init external/MacroPlacement
uv sync --extra dev
uv run --extra dev pytest
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
```

Observed baseline:

- `uv run --extra dev pytest`: 7 tests passed.
- Greedy row placer on `ibm01`: proxy `2.0463`, overlaps `0`, valid.

Note: plain `uv run pytest` did not install the optional `dev` extra and used the ambient Anaconda pytest, so use `uv run --extra dev pytest`.

## Known Local Limits

- This machine has no Docker CLI.
- This machine has no NVIDIA GPU.
- Serious scoring, air-gapped Docker parity, multiprocessing stress, and optional ORFS checks must run on cloud Ubuntu/GPU.

## Current Submission

The official final entry point is:

```bash
submissions/jaydenpiao/placer.py
```

Current real-benchmark smoke:

- `uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-legalizer-valid`
- `uv run python scripts/check_results.py results/all-ibm-legalizer-valid/summary.json --max-runtime 3300 --max-avg-proxy 1.4578`
- all 17 IBM benchmarks valid
- average proxy `1.4570`
- total hard overlaps `0`
- max local runtime `30.63s`
- `ibm01` proxy `1.0388`
- wirelength `0.064`
- density `0.813`
- congestion `1.137`
- overlaps `0`
- runtime `2.21s`

The current implementation is a deterministic legalizer-first baseline. Hypergraph local search exists behind `JAYDEN_SEARCH_ITERS`, but defaults to `0` because the legalizer-only path is currently the validated all-IBM baseline. Score improvements should be isolated in small PRs.

## Next Priorities

1. Push `infra/bootstrap-foundation` and open a draft PR for the bootstrap slice.
2. Watch GitHub Actions CI on the PR.
3. Reproduce the all-IBM run in a clean cloud Ubuntu/GPU evaluator.
4. Iterate on hybrid analytical placement plus local search to chase the top-7 cutoff.
