# Current State

Last updated: 2026-05-01

## Repository

- Fork: `https://github.com/jaydenpiao/macro-place-challenge-2026`
- Upstream: `https://github.com/partcleda/macro-place-challenge-2026`
- Primary working branch: `main`
- Local primary clone: `/Users/jaydenpiao/Desktop/hrt_challenge/macro-place-challenge-2026`
- Bootstrap PR: `https://github.com/jaydenpiao/macro-place-challenge-2026/pull/1` merged into `main`
- Score recipe PR: `https://github.com/jaydenpiao/macro-place-challenge-2026/pull/2` merged into `main`
- Cloud parity wrapper PR: `https://github.com/jaydenpiao/macro-place-challenge-2026/pull/3` merged into `main`
- Result comparison helper PR: `https://github.com/jaydenpiao/macro-place-challenge-2026/pull/4` merged into `main`

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

## Cloud GPU Check

RunPod direct Linux/GPU validation completed on 2026-05-01 using the current `auto` default:

- RunPod secure cloud `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- NVIDIA RTX 6000 Ada Generation, 49140 MiB, driver `570.195.03`
- repo commit `53390e1fff20017258b9967dda136fbc94f258f4`
- result artifact: `results/runpod-auto-profile-20260501-023124/summary.json` (ignored by git)

Result:

- all 17 IBM benchmarks valid
- average proxy `1.4555341426`
- total hard overlaps `0`
- max runtime `53.67s`
- total runtime `168.31s`

Strict Docker parity was not run on this host because `scripts/check_cloud_parity_host.py` correctly refused the RunPod PyTorch image:

```text
cloud parity preflight failed: docker client/server failed with exit 127: [Errno 2] No such file or directory: 'docker'
```

Prior RunPod direct Linux/GPU validation completed on 2026-04-30 using:

- RunPod secure cloud `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- NVIDIA RTX 6000 Ada Generation, 49140 MiB, driver `570.124.06`
- repo commit `397b06edbc071e47efb99dc58ff9c8afec0697d9`
- result artifact: `results/runpod-linuxgpu-20260430-003938/summary.json` (ignored by git)

Result:

- all 17 IBM benchmarks valid
- average proxy `1.4559245530`
- total hard overlaps `0`
- max runtime `54.81s`
- total runtime `158.65s`

Important: neither RunPod PyTorch run was official Docker parity. The RunPod PyTorch image did not include Docker. The RunPod `runpod-desktop` / `runpod/kasm-docker:cuda11` attempt exposed pod metadata but never produced a usable SSH daemon on TCP 22 before termination. Use a true GPU VM, or a custom RunPod template with verified `sshd` plus Docker/NVIDIA runtime, for `scripts/run_cloud_parity.sh`.

## Current Submission

The official final entry point is:

```bash
submissions/jaydenpiao/placer.py
```

Current real-benchmark smoke:

- `uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --all --run-id all-ibm-auto-profile-schedule`
- `uv run python scripts/check_results.py results/all-ibm-auto-profile-schedule/summary.json --max-runtime 3300 --max-avg-proxy 1.4559245531`
- all 17 IBM benchmarks valid
- average proxy `1.4555341426`
- total hard overlaps `0`
- max local runtime `30.39s`
- comparison vs `all-ibm-auto-transform`: average delta `-0.0003904104`, 15 benchmarks improved, 0 regressed
- `ibm01` proxy `1.0381`
- wirelength `0.067`
- density `0.812`
- congestion `1.130`
- overlaps `0`
- runtime `1.24s`

The current implementation is a deterministic legalizer-first baseline with cheap benchmark-specific symmetry recipes behind `JAYDEN_TRANSFORM=auto` and learned benchmark-specific knob schedules behind `JAYDEN_STRATEGY=auto`. Global hypergraph local search remains off by default, except the learned profile enables a small bounded search on `ibm02`. Score improvements should be isolated in small PRs.

The same `auto` default has now reproduced on RunPod Linux/GPU at `1.4555341426` with zero overlaps. This is direct Linux/GPU validation only, not official Docker parity.

Candidate variant scans should use `scripts/scan_candidates.py` so every run produces per-variant `summary.json` files plus one aggregate `scan_summary.json` with deltas against the current baseline.

## Next Priorities

1. Run the official air-gapped Docker path before any leaderboard submission.
2. Use a GPU VM or a custom verified RunPod Docker template for `scripts/run_cloud_parity.sh`; the wrapper now preflights Docker, host NVIDIA, and Docker GPU visibility before evaluating.
3. Iterate on hybrid analytical placement plus local search to chase the top-7 cutoff.
4. Run NG45/OpenROAD-flow-scripts checks for finalist candidates.
