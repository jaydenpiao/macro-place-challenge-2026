# Cloud Parity

Use this only on a Linux host with Docker, NVIDIA runtime, and a GPU comparable to the official environment.

## One-Time Setup

```bash
git clone https://github.com/jaydenpiao/macro-place-challenge-2026.git
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement
uv sync --extra dev
docker --version
nvidia-smi
```

## Official Docker Parity

```bash
scripts/run_cloud_parity.sh cloud-auto-transform-001 jaydenpiao submissions/jaydenpiao/placer.py
```

The wrapper:

- sources `configs/cloud_gpu.env`
- runs `scripts/check_cloud_parity_host.py`
- refuses to continue unless `docker version`, `nvidia-smi`, and `docker run --rm --gpus all ... nvidia-smi` pass
- forwards `JAYDEN_*` knobs into `eval_docker/run_eval.sh`
- runs Docker with `--network none`, `--gpus all`, `--memory 64g`, and `--cpus 16`
- copies the Docker log to `results/<run-id>/run.log`
- writes reproducibility metadata to `results/<run-id>/metadata.json`

Override the preflight CUDA image only when the default cannot be pulled on the host:

```bash
JAYDEN_DOCKER_GPU_SMOKE_IMAGE=nvidia/cuda:12.4.1-base-ubuntu22.04 \
  scripts/run_cloud_parity.sh cloud-auto-transform-001 jaydenpiao submissions/jaydenpiao/placer.py
```

## Promotion Requirement

Before a leaderboard submission, record:

- `results/<run-id>/run.log`
- `results/<run-id>/metadata.json`
- exact Git commit
- cloud instance type
- max per-benchmark runtime
- average IBM proxy score
- total hard overlaps

Do not submit from a local macOS-only result.

## RunPod Notes

RunPod can be used for a direct Linux/GPU check, but the tested public templates were not enough for official Docker parity:

- `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` on RTX 6000 Ada worked over SSH and ran `scripts/run_experiment.py`, but did not include Docker.
- On 2026-05-01, the current `JAYDEN_STRATEGY=auto` default reproduced on that PyTorch template with run id `runpod-auto-profile-20260501-023124`, average proxy `1.4555341426`, total hard overlaps `0`, and max runtime `53.67s`.
- The strict parity preflight for that run failed as intended with `docker client/server failed with exit 127`, so `eval_docker/run_eval.sh` was not attempted.
- `runpod-desktop` / `runpod/kasm-docker:cuda11` looked like the best nested-Docker candidate, but in the 2026-04-30 attempt it exposed pod metadata while TCP/22 refused connections and `uptimeSeconds` stayed `0`.

Do not call a RunPod PyTorch result "cloud parity." Treat it as Linux/GPU direct validation only. For strict parity, use a real GPU VM with Docker/NVIDIA runtime or create a custom RunPod template whose startup command verifies:

```bash
sshd -T >/dev/null
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
