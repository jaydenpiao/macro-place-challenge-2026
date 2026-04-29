# Architecture

## Official Boundary

The challenge package supplies the evaluator:

- `macro_place/evaluate.py` loads a placer file and runs official benchmark evaluation.
- `macro_place/objective.py` computes proxy cost using the TILOS `PlacementCost` evaluator.
- `macro_place/utils.py` validates placement legality.

Challenge scoring code is treated as read-only. Project code may call it, but must not patch it to improve scores.

## Submission Boundary

`submissions/jaydenpiao/placer.py` is the only file the judges need as the entry point. It defines a deterministic placer class with:

```python
def place(self, benchmark: Benchmark) -> torch.Tensor:
    ...
```

Sibling modules under `submissions/jaydenpiao/` are allowed. The entry point adds its own directory to `sys.path` so the same code works when mounted into the air-gapped Docker evaluator.

## Experiment Boundary

Project experiment tooling lives in `scripts/` and writes machine-readable artifacts under `results/<run-id>/`.

Expected result files:

- `summary.json`: parsed benchmark metrics and reproducibility metadata.
- `run.log`: raw command output when generated through shell wrappers.
- optional visualizations or ORFS logs for finalist candidates.

The harness must record:

- repo commit and dirty state
- upstream commit
- placer path
- command and environment knobs
- hardware summary
- benchmark metrics

## Algorithm Direction

The first durable lane is a legalizer-first hybrid:

1. Start from official initial macro positions.
2. Clamp movable hard macros inside canvas bounds.
3. Repair hard-macro overlaps with a positive gap.
4. Use fast hypergraph surrogate search to improve wirelength without breaking legality.
5. Keep soft macros stable unless a tested optimizer improves proxy and routability.
6. Evaluate with the official proxy only at candidate boundaries.

Future lanes should be added behind explicit config knobs and promoted only with saved result summaries.
