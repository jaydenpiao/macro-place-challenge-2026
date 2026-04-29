# Configs

These files document repeatable experiment modes. Source them from a shell before running `scripts/run_experiment.py`.

```bash
set -a
source configs/local_smoke.env
set +a
uv run python scripts/run_experiment.py --placer submissions/jaydenpiao/placer.py --benchmarks ibm01 --run-id local-smoke
```

Do not store secrets here.
