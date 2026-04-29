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
