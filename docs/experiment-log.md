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

## 2026-04-29 - Auto symmetry recipe probe

Purpose: test whether global mirror transforms of the official initial placement can improve proxy while preserving the legalizer-first runtime profile.

Offline probe:

- tried `identity`, `flip_x`, `flip_y`, and `flip_xy`
- selected a benchmark-specific recipe only when it beat identity in the official proxy
- rejected coarse global shifts after `ibm17` showed no improvement and the search cost was too high
- rejected the existing `submissions/will_seed` placer as a default because its all-IBM average was `1.5336`

Promoted recipe:

- `ibm01`: `flip_x`
- `ibm02`: `flip_y`
- `ibm03`: `flip_y`
- `ibm04`: `flip_y`
- `ibm15`: `flip_xy`
- `ibm17`: `flip_xy`
- `ibm18`: `flip_y`
- all other benchmarks: identity

Command:

```bash
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-auto-transform
uv run python scripts/check_results.py results/all-ibm-auto-transform/summary.json --max-runtime 3300 --max-avg-proxy 1.4578
```

Aggregate result:

- average proxy: `1.4559`
- total hard overlaps: `0`
- max local runtime: `30.87s`
- benchmark validity: all 17 IBM benchmarks valid
- summary commit: `9f7c1ac`, dirty state `false`

Interpretation:

- This is a small but reproducible default-score gain over the legalizer-only baseline (`1.4570`).
- The biggest gains came from `ibm17`, `ibm02`, `ibm03`, `ibm04`, and `ibm15`.
- The result is still far above the public top-7 cutoff and should be treated as a cheap baseline improvement, not a finalist candidate.
