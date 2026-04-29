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

The initial implementation should prioritize legality, determinism, and instrumentation over leaderboard score. Score improvements should be isolated in small PRs.

## Next Priorities

1. Finish agent docs and experiment harness.
2. Add deterministic legalizer-first baseline submission.
3. Add result schemas, promotion gates, and CI.
4. Run all-IBM locally if feasible; otherwise run on cloud GPU/Ubuntu.
5. Iterate on hybrid analytical placement plus local search.
