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

## 2026-04-30 - Auto profile schedule

Purpose: promote the best legal per-benchmark knob choices from `scan-existing-knobs-20260430` without hardcoding final coordinates.

Promoted profile:

- `ibm01`: `JAYDEN_LEGAL_GAP=0.005`
- `ibm02`: `JAYDEN_SEARCH_ITERS=100`
- `ibm03`, `ibm09`, `ibm11`, `ibm12`, `ibm13`, `ibm16`, `ibm18`: `JAYDEN_LEGAL_GAP=0.02`
- `ibm04`, `ibm06`, `ibm07`, `ibm14`: `JAYDEN_LEGAL_GAP=0.001`
- `ibm08`, `ibm10`: `JAYDEN_LEGAL_GAP=0.005`
- all other benchmarks: baseline defaults

Command:

```bash
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-auto-profile-schedule
uv run python scripts/check_results.py results/all-ibm-auto-profile-schedule/summary.json --max-runtime 3300 --max-avg-proxy 1.4559245531
uv run python scripts/compare_results.py results/all-ibm-auto-transform/summary.json results/all-ibm-auto-profile-schedule/summary.json --json results/all-ibm-auto-profile-schedule/comparison.json
```

Aggregate result:

- average proxy: `1.4555341426`
- total hard overlaps: `0`
- max local runtime: `30.39s`
- benchmark validity: all 17 IBM benchmarks valid
- comparison vs `all-ibm-auto-transform`: delta `-0.0003904104`, 15 benchmarks improved, 0 regressed

Interpretation:

- This is a small but clean improvement and should become the default if CI and review pass.
- It is not enough to approach top-7; the next scoring lane needs real placement moves, not just global knob schedules.

## 2026-05-01 - Auto profile RunPod direct validation

Purpose: validate the merged `JAYDEN_STRATEGY=auto` default on a Linux NVIDIA GPU host and record the strict parity preflight status.

RunPod setup:

- pod id: `if1pc3fdpisfqg`
- template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB, driver `570.195.03`
- vCPU allocation reported by RunPod: 16
- repo commit: `53390e1fff20017258b9967dda136fbc94f258f4`
- run id: `runpod-auto-profile-20260501-023124`
- result artifact: `results/runpod-auto-profile-20260501-023124/summary.json` (ignored by git)
- pod deleted after artifact collection; `runpodctl pod list` returned `[]`

Direct Linux/GPU command:

```bash
git clone https://github.com/jaydenpiao/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout 53390e1fff20017258b9967dda136fbc94f258f4
git submodule update --init external/MacroPlacement
uv sync --extra dev
set -a && source configs/cloud_gpu.env && set +a
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id runpod-auto-profile-20260501-023124
uv run python scripts/check_results.py results/runpod-auto-profile-20260501-023124/summary.json --max-runtime 3300 --max-avg-proxy 1.4559245531
```

Aggregate result:

- average proxy: `1.4555341426`
- total hard overlaps: `0`
- max runtime: `53.67s`
- total runtime: `168.31s`
- benchmark validity: all 17 IBM benchmarks valid

Strict parity preflight:

```text
cloud parity preflight failed: docker client/server failed with exit 127: [Errno 2] No such file or directory: 'docker'
```

Interpretation:

- The current `auto` default reproduced on Linux/GPU with the same score as local all-IBM validation and no legality regression.
- This is still not official air-gapped Docker parity because the RunPod PyTorch image has no Docker daemon.
- Do not spend more time on global knob sweeps; the next scoring lane should implement real macro moves, soft-density cleanup, or exact-proxy-screened local refinement for `ibm18`, `ibm17`, `ibm06`, `ibm12`, `ibm15`, and `ibm14`.

## 2026-05-01 - Density-aware local refinement profile

Purpose: test a narrow local-refinement improvement that reduces density hotspots without applying stochastic hard-macro search globally.

Implementation:

- `_net_hpwl` now uses pin-level hard-macro offsets from `Benchmark.net_pin_nodes` when available.
- `_density_surrogate_cost` approximates grid hotspot pressure from hard and soft macro area occupancy.
- `JAYDEN_DENSITY_WEIGHT` controls the density term in local search and defaults to `0`.
- `JAYDEN_STRATEGY=auto` enables `search_iters=100` and `density_weight=1000` only on scan-positive benchmarks: `ibm02`, `ibm06`, `ibm08`, `ibm10`, `ibm13`, and `ibm14`.

Scan result:

```bash
uv run python scripts/scan_candidates.py \
  --run-id scan-density-search-all-20260501 \
  --baseline /Users/jaydenpiao/Desktop/hrt_challenge/macro-place-challenge-2026/results/runpod-auto-profile-20260501-023124/summary.json \
  --variant d1000s100:JAYDEN_SEARCH_ITERS=100\;JAYDEN_DENSITY_WEIGHT=1000
```

Applying the density search globally was worse overall (`+0.000181` average delta), but it identified six net-positive benchmark profiles.

Promoted command:

```bash
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-density-profile-20260501
uv run python scripts/check_results.py results/all-ibm-density-profile-20260501/summary.json --max-runtime 3300 --max-avg-proxy 1.4555341427
uv run python scripts/compare_results.py /Users/jaydenpiao/Desktop/hrt_challenge/macro-place-challenge-2026/results/runpod-auto-profile-20260501-023124/summary.json results/all-ibm-density-profile-20260501/summary.json --json results/all-ibm-density-profile-20260501/comparison.json
```

Aggregate result:

- average proxy: `1.4553974306`
- total hard overlaps: `0`
- max local runtime: `29.31s`
- comparison vs `runpod-auto-profile-20260501-023124`: delta `-0.0001367120`, 6 benchmarks improved, 0 regressed

Improved benchmarks:

- `ibm02`: `-0.000828`
- `ibm06`: `-0.000601`
- `ibm08`: `-0.000038`
- `ibm10`: `-0.000680`
- `ibm13`: `-0.000159`
- `ibm14`: `-0.000018`

Interpretation:

- This is a small but clean score improvement and should be validated on RunPod Linux/GPU after merge.
- Generic hard-macro search remains unsafe for the weak benchmarks; density-aware search should remain benchmark-scheduled until a stronger exact-screened candidate selector exists.

## 2026-05-01 - Density profile RunPod direct validation

Purpose: validate the merged density-aware `auto` default on a Linux NVIDIA GPU host and record strict parity preflight status.

RunPod setup:

- pod id: `ipgg3pdzfmcvyn`
- template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB, driver `570.211.01`
- vCPU allocation reported by RunPod: 24
- repo commit: `fcfe2640d40b655b902921636f88d4f622d9e2aa`
- run id: `runpod-density-profile-20260501-043910`
- result artifact: `results/runpod-density-profile-20260501-043910/summary.json` (ignored by git)
- pod deleted after artifact collection; `runpodctl pod list` returned `[]`

Direct Linux/GPU command:

```bash
git clone https://github.com/jaydenpiao/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git checkout fcfe2640d40b655b902921636f88d4f622d9e2aa
git submodule update --init external/MacroPlacement
uv sync --extra dev
set -a && source configs/cloud_gpu.env && set +a
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id runpod-density-profile-20260501-043910
uv run python scripts/check_results.py results/runpod-density-profile-20260501-043910/summary.json --max-runtime 3300 --max-avg-proxy 1.4555341427
```

Aggregate result:

- average proxy: `1.4553974306`
- total hard overlaps: `0`
- max runtime: `77.31s`
- total runtime: `246.37s`
- benchmark validity: all 17 IBM benchmarks valid

Strict parity preflight:

```text
cloud parity preflight failed: docker client/server failed with exit 127: [Errno 2] No such file or directory: 'docker'
```

Interpretation:

- The density-aware `auto` default reproduced on Linux/GPU with the same average proxy as the local all-IBM run and no legality regression.
- This is still not official air-gapped Docker parity because the RunPod PyTorch image has no Docker daemon.
- Next scoring work should target a larger algorithmic move; this PR only harvests a small safe local-search schedule.

## 2026-05-04 - Exact-proxy structural candidate search harness

Purpose: add an offline lane for structural macro-move experiments that uses the official proxy as the judge without changing default submission behavior.

Implementation:

- `scripts/search_candidates.py` starts from the current placer output for each benchmark.
- Candidate families currently include single hard-macro moves, similar-size swaps, density push moves, and transform probes.
- Every candidate is hard-macro legal before scoring; fixed hard macros are preserved.
- The script writes `results/<run-id>/summary.json` and `results/<run-id>/candidate_trace.jsonl`.
- `summary.json` remains compatible with `scripts/check_results.py`; traces keep deterministic recipes for later promotion work.

Smoke command:

```bash
uv run python scripts/search_candidates.py --run-id exact-search-smoke --benchmarks ibm01 --families single --step-fractions 0.02 --max-candidates-per-benchmark 4
uv run python scripts/check_results.py results/exact-search-smoke/summary.json --max-runtime 3300 --max-avg-proxy 1.5
```

Smoke result:

- `ibm01` baseline proxy: `1.0381`
- best screened proxy: `1.0381`
- selected candidate: `baseline`
- candidate count: `4`
- total hard overlaps: `0`
- runtime: `15.53s`

Interpretation:

- This is infrastructure, not a promoted scoring change.
- Use this lane for weak-benchmark hard-macro LNS experiments before touching `submissions/jaydenpiao`.
- Candidate caps are enforced during generation because legalization can dominate runtime on large IBM designs.

## 2026-05-06 - Exact-search per-family cap

Purpose: keep broad exact-proxy sweeps from spending the entire benchmark candidate budget on the first enabled move family.

Implementation:

- `scripts/search_candidates.py` now accepts `--max-candidates-per-family`.
- The cap is applied independently as each family is generated, while `--max-candidates-per-benchmark` remains the total benchmark cap.
- `summary.json` records the per-family cap in `search_config` for reproducibility.
- No default submission behavior changed.

Validation:

```bash
uv run --extra dev pytest test/test_exact_candidate_search.py::test_candidate_generation_applies_per_family_candidate_cap test/test_exact_candidate_search.py::test_summary_records_search_metadata_and_aggregate_best_proxy
```

Result:

- `2 passed`

Search evidence:

```bash
uv run python -u scripts/search_candidates.py \
  --run-id exact-search-weak-v1-family2 \
  --benchmarks ibm18,ibm17,ibm06,ibm12,ibm15,ibm14,ibm02 \
  --families single,density,swap,transform \
  --step-fractions 0.01,0.02,0.05 \
  --max-candidates-per-benchmark 64 \
  --max-candidates-per-family 2
```

Aggregate result:

- candidate count: `56`
- improved benchmarks: `6`
- total hard overlaps: `0`
- best density candidates included `ibm06` `density-m23-0.02`, `ibm12` `density-m501-0.02`, and `ibm02` `density-m217-0.02`

Interpretation:

- This branch is infrastructure only.
- The useful follow-up is a separate scoring branch that replays general density-rank behavior from screened recipes, not this harness PR.
