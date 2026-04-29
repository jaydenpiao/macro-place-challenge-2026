"""Jayden Piao challenge submission entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from macro_place.benchmark import Benchmark

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from core import PlacerConfig, build_placement  # noqa: E402


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


class JaydenPiaoPlacer:
    """Deterministic legalizer-first hybrid placer."""

    def __init__(
        self,
        seed: int | None = None,
        search_iters: int | None = None,
        legal_gap: float | None = None,
    ) -> None:
        self.config = PlacerConfig(
            seed=seed if seed is not None else _env_int("JAYDEN_PLACER_SEED", 20260429),
            search_iters=(
                search_iters if search_iters is not None else _env_int("JAYDEN_SEARCH_ITERS", 4000)
            ),
            legal_gap=legal_gap if legal_gap is not None else _env_float("JAYDEN_LEGAL_GAP", 0.01),
        )

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        return build_placement(benchmark, self.config)
