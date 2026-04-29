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

The runtime algorithm avoids LLM/VLM/model calls and does not use external proprietary placement tools.
