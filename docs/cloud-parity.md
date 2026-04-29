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
- forwards `JAYDEN_*` knobs into `eval_docker/run_eval.sh`
- runs Docker with `--network none`, `--gpus all`, `--memory 64g`, and `--cpus 16`
- copies the Docker log to `results/<run-id>/run.log`
- writes reproducibility metadata to `results/<run-id>/metadata.json`

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
