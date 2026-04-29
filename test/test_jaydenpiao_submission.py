import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from macro_place.benchmark import Benchmark
from macro_place.evaluate import _load_placer
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_overlap_metrics
from macro_place.utils import validate_placement


def _benchmark(
    positions: torch.Tensor,
    sizes: torch.Tensor,
    fixed: torch.Tensor,
    *,
    num_hard: int,
) -> Benchmark:
    num_macros = positions.shape[0]
    return Benchmark(
        name="synthetic",
        canvas_width=10.0,
        canvas_height=10.0,
        num_macros=num_macros,
        num_hard_macros=num_hard,
        num_soft_macros=num_macros - num_hard,
        macro_positions=positions.clone().float(),
        macro_sizes=sizes.clone().float(),
        macro_fixed=fixed.clone().bool(),
        macro_names=[f"m{i}" for i in range(num_macros)],
        num_nets=0,
        net_nodes=[],
        net_weights=torch.zeros(0, dtype=torch.float32),
        grid_rows=4,
        grid_cols=4,
    )


def _load_submission_core():
    path = Path("submissions/jaydenpiao/core.py")
    spec = importlib.util.spec_from_file_location("jaydenpiao_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_submission_imports_with_official_loader():
    placer = _load_placer(Path("submissions/jaydenpiao/placer.py"))

    assert type(placer).__name__ == "JaydenPiaoPlacer"
    assert callable(placer.place)


def test_placer_repairs_hard_macro_overlaps_and_preserves_soft_macros():
    benchmark = _benchmark(
        positions=torch.tensor(
            [
                [2.0, 2.0],
                [2.25, 2.0],
                [8.0, 8.0],
            ]
        ),
        sizes=torch.tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
                [1.0, 1.0],
            ]
        ),
        fixed=torch.tensor([False, False, False]),
        num_hard=2,
    )
    placer = _load_placer(Path("submissions/jaydenpiao/placer.py"))

    placement = placer.place(benchmark)
    overlaps = compute_overlap_metrics(placement, benchmark)

    assert placement.shape == (benchmark.num_macros, 2)
    assert overlaps["overlap_count"] == 0
    assert torch.equal(placement[2], benchmark.macro_positions[2])


def test_placer_clamps_movable_soft_macros_inside_canvas_bounds():
    benchmark = _benchmark(
        positions=torch.tensor(
            [
                [2.0, 2.0],
                [11.0, 8.0],
            ]
        ),
        sizes=torch.tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
        fixed=torch.tensor([False, False]),
        num_hard=1,
    )
    placer = _load_placer(Path("submissions/jaydenpiao/placer.py"))

    placement = placer.place(benchmark)

    assert placement[1, 0] <= 9.0
    assert placement[1, 1] == benchmark.macro_positions[1, 1]


def test_placer_keeps_fixed_hard_macro_position_while_legalizing():
    benchmark = _benchmark(
        positions=torch.tensor(
            [
                [2.0, 2.0],
                [2.25, 2.0],
            ]
        ),
        sizes=torch.tensor(
            [
                [2.0, 2.0],
                [2.0, 2.0],
            ]
        ),
        fixed=torch.tensor([True, False]),
        num_hard=2,
    )
    placer = _load_placer(Path("submissions/jaydenpiao/placer.py"))

    placement = placer.place(benchmark)
    overlaps = compute_overlap_metrics(placement, benchmark)

    assert overlaps["overlap_count"] == 0
    assert torch.equal(placement[0], benchmark.macro_positions[0])


def test_auto_transform_flips_only_movable_macros_for_known_benchmark():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor(
            [
                [2.0, 2.0],
                [4.0, 4.0],
                [8.0, 8.0],
            ]
        ),
        sizes=torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        fixed=torch.tensor([True, False, False]),
        num_hard=2,
    )
    benchmark.name = "ibm01"

    placement = core.build_placement(benchmark, core.PlacerConfig(transform="auto"))

    assert torch.equal(placement[0], benchmark.macro_positions[0])
    assert placement[1, 0] == pytest.approx(6.0)
    assert placement[2, 0] == pytest.approx(2.0)
    assert placement[1, 1] == pytest.approx(4.0)
    assert placement[2, 1] == pytest.approx(8.0)


def test_placer_legalizer_only_is_valid_on_ibm06(monkeypatch):
    benchmark_dir = Path("external/MacroPlacement/Testcases/ICCAD04/ibm06")
    if not benchmark_dir.exists():
        pytest.skip("TILOS submodule not initialized")

    monkeypatch.setenv("JAYDEN_SEARCH_ITERS", "0")
    benchmark, _ = load_benchmark_from_dir(str(benchmark_dir))
    placer = _load_placer(Path("submissions/jaydenpiao/placer.py"))

    placement = placer.place(benchmark)
    overlaps = compute_overlap_metrics(placement, benchmark)
    valid, violations = validate_placement(placement, benchmark)

    assert overlaps["overlap_count"] == 0
    assert valid, violations
