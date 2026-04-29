# Experiment Log

## 2026-04-29 - Bootstrap

Purpose: create the public competition fork and establish a reproducible, agent-friendly workflow.

Local environment:

- macOS arm64
- `uv 0.11.7`
- no Docker CLI
- no NVIDIA GPU

Baseline verification:

- `uv run --extra dev pytest` passed 7 tests.
- `uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01` produced proxy `2.0463` with `0` overlaps.

Decisions:

- Keep the repo public now.
- Use this Mac for development and smoke tests only.
- Use cloud Ubuntu/GPU for serious scoring and Docker parity.
- Do not use runtime LLM/VLM calls while official issue #55 remains unresolved.

Open hypotheses:

- Preserving and legalizing the initial placement should beat shelf packing while remaining robust.
- Hypergraph-local search can produce cheap proxy gains before investing in GPU analytical placement.
- Soft macro co-optimization matters for both proxy and real NG45 routability, but must be gated carefully.

## 2026-04-29 - Initial JaydenPiao placer smoke

Historical note: this smoke was recorded before the default `JAYDEN_SEARCH_ITERS` was changed to `0`. Treat the all-IBM legalizer-only run below as the current baseline.

Command:

```bash
uv run evaluate submissions/jaydenpiao/placer.py -b ibm01
```

Result:

- proxy: `1.0560`
- wirelength: `0.064`
- density: `0.832`
- congestion: `1.152`
- overlaps: `0`
- runtime: `2.14s`

Interpretation:

- The legalizer-first baseline already beats the upstream greedy row placer on `ibm01`.
- This single benchmark is not enough to claim leaderboard competitiveness; run all IBM benchmarks next through `scripts/run_experiment.py`.

Harness verification:

```bash
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --benchmarks ibm01 --run-id smoke-ibm01
uv run python scripts/check_results.py results/smoke-ibm01/summary.json --max-runtime 55 --max-avg-proxy 1.4578
```

The generated `summary.json` passed legality, runtime, and RePlAce-threshold checks for the single-benchmark smoke.

## 2026-04-29 - Legalizer-only all-IBM baseline

Command:

```bash
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-legalizer-valid
uv run python scripts/check_results.py results/all-ibm-legalizer-valid/summary.json --max-runtime 3300 --max-avg-proxy 1.4578
```

Aggregate result:

- average proxy: `1.4570`
- total hard overlaps: `0`
- max local runtime: `32.92s`
- benchmark validity: all 17 IBM benchmarks valid
- summary commit: `cb3a747`, dirty state `false`

Interpretation:

- The default legalizer-only configuration clears the first RePlAce-style promotion gate locally.
- It is still above the public top-7 proxy cutoff, so the next scoring work should target congestion-heavy cases such as `ibm17`, `ibm18`, `ibm06`, `ibm12`, `ibm15`, and `ibm14`.
- This result still needs cloud Ubuntu/GPU and official Docker parity before leaderboard submission.
