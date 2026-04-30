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

## 2026-04-29 - Soft macro optimizer timeout

Purpose: test whether the official `PlacementCost.optimize_stdcells` helper is viable as a cheap soft-macro co-optimization lane after hard legalization.

Probe:

```bash
timeout 300 uv run python -u - <<'PY'
# loaded ibm06, applied the current JaydenPiao placer, then called:
# plc.optimize_stdcells(num_steps=[10, 10, 10], max_move_distance=[canvas / 100] * 3, ...)
PY
```

Result:

- `ibm06` base proxy before the call: `1.6966678`
- `[10, 10, 10]` soft-macro optimization did not finish within `300s`
- the helper emitted a large `os_debug.txt` file, now ignored in `.gitignore`

Interpretation:

- Do not use `PlacementCost.optimize_stdcells` in the default runtime path.
- Revisit only as a cloud-only experiment with strict timeout accounting and a saved result summary.

## 2026-04-30 - RunPod Linux/GPU direct validation

Purpose: validate the current default submission on a Linux NVIDIA GPU host and test whether RunPod can provide official Docker parity.

RunPod setup:

- API key loaded from local secret file; key was not printed in logs.
- First tried `runpod-desktop` / `runpod/kasm-docker:cuda11` on RTX 6000 Ada for nested Docker parity.
- The Docker-oriented template exposed pod metadata but kept `uptimeSeconds: 0` and refused TCP/22, so it was terminated.
- Fallback used `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` on RTX 6000 Ada.
- The PyTorch template was reachable over SSH and had NVIDIA GPU access, but no Docker daemon.

Direct Linux/GPU command:

```bash
git clone https://github.com/jaydenpiao/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout 397b06edbc071e47efb99dc58ff9c8afec0697d9
git submodule update --init external/MacroPlacement
python3 -m pip install --upgrade pip uv
uv sync --extra dev
set -a && source configs/cloud_gpu.env && set +a
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --run-id runpod-linuxgpu-20260430-003938 --all
```

Aggregate result:

- average proxy: `1.4559245530`
- total hard overlaps: `0`
- max runtime: `54.81s`
- total runtime: `158.65s`
- benchmark validity: all 17 IBM benchmarks valid
- summary commit: `397b06edbc071e47efb99dc58ff9c8afec0697d9`, dirty state `false`

Interpretation:

- The macOS all-IBM score reproduced on Linux/GPU within expected noise; no legality regression.
- This is still not official air-gapped Docker parity.
- For strict parity, prefer a GPU VM with Docker and NVIDIA runtime, or build a custom RunPod template with verified `sshd` and Docker before starting the evaluator.

## 2026-04-30 - Candidate scanner harness

Purpose: make score experiments reproducible without changing the default placer behavior.

Usage:

```bash
uv run python scripts/scan_candidates.py \
  --run-id scan-gap-smoke \
  --baseline results/all-ibm-auto-transform/summary.json \
  --benchmarks ibm01 \
  --variant gap005:JAYDEN_LEGAL_GAP=0.005
```

For full promotion scans, omit `--benchmarks` so the scanner runs all IBM benchmarks. Each variant writes `results/<run-id>__<variant>/summary.json`; the aggregate comparison is written to `results/<run-id>/scan_summary.json`.

Interpretation:

- Use this for knob sweeps and candidate screening before opening scoring PRs.
- Smoke scans with `--benchmarks` are useful for command validation, but their aggregate deltas are not promotion signals unless `comparison_complete` is `true`.
- Only promote a scanner result after a normal all-IBM run beats or matches the baseline gate with zero overlaps.
