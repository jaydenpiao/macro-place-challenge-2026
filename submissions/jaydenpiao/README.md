# Jayden Piao Submission

Official entry point:

```bash
uv run evaluate submissions/jaydenpiao/placer.py -b ibm01
uv run evaluate submissions/jaydenpiao/placer.py --all
```

The placer is deterministic by default. Environment knobs:

- `JAYDEN_PLACER_SEED`: integer seed, default `20260429`
- `JAYDEN_SEARCH_ITERS`: local-search iterations, default `0`
- `JAYDEN_LEGAL_GAP`: hard-macro legalization gap in microns, default `0.01`
- `JAYDEN_TRANSFORM`: initial-placement transform, default `auto`
- `JAYDEN_STRATEGY`: benchmark-specific knob schedule, default `auto`; use `baseline` to disable learned per-benchmark profiles
- `JAYDEN_DENSITY_WEIGHT`: optional density-aware local-search weight, default `0`; the `auto` strategy enables it only for benchmarks where full-score scans showed a net gain
- `JAYDEN_RECIPE_PROFILE`: hard-macro density-rank recipe profile, default `exact_v2`; use `exact_v1` or `off` for ablations
- `JAYDEN_SOFT_PROFILE`: soft-macro recipe profile, default `soft_v1`; use `off` for ablations

The runtime algorithm avoids LLM/VLM/model calls and does not use external proprietary placement tools.
