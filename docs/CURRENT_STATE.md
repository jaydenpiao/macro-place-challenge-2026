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

Initial real-benchmark smoke:

- `uv run evaluate submissions/jaydenpiao/placer.py -b ibm01`
- proxy `1.0560`
- wirelength `0.064`
- density `0.832`
- congestion `1.152`
- overlaps `0`
- runtime `2.14s`

The current implementation is a deterministic legalizer-first baseline with hypergraph local search. Score improvements should be isolated in small PRs.

## Next Priorities

1. Run all-IBM locally if feasible; otherwise run on cloud GPU/Ubuntu.
2. Push `infra/bootstrap-foundation` and open a draft PR for the bootstrap slice.
3. Watch GitHub Actions CI on the PR.
4. Iterate on hybrid analytical placement plus local search.
