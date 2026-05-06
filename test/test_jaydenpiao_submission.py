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


def _density_profile_benchmark(name: str, fixed: list[bool] | None = None) -> Benchmark:
    benchmark = _benchmark(
        positions=torch.tensor(
            [
                [3.0, 2.5],
                [7.5, 2.5],
                [8.0, 8.0],
                [2.5, 2.5],
                [2.6, 2.5],
            ]
        ),
        sizes=torch.tensor(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
                [2.0, 2.0],
                [1.0, 1.0],
            ]
        ),
        fixed=torch.tensor(fixed or [False, False, False, False, False]),
        num_hard=3,
    )
    benchmark.name = name
    benchmark.grid_rows = 2
    benchmark.grid_cols = 2
    return benchmark


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


def test_auto_strategy_uses_learned_profile_only_when_enabled():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )
    benchmark.name = "ibm02"

    auto = core.effective_config_for_benchmark(benchmark, core.PlacerConfig(strategy="auto"))
    baseline = core.effective_config_for_benchmark(
        benchmark, core.PlacerConfig(strategy="baseline")
    )

    assert auto.search_iters == 100
    assert auto.density_weight == pytest.approx(1000.0)
    assert auto.legal_gap == pytest.approx(0.01)
    assert baseline.search_iters == 0
    assert baseline.density_weight == pytest.approx(0.0)
    assert baseline.legal_gap == pytest.approx(0.01)


def test_auto_strategy_keeps_explicit_user_knobs():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [4.0, 4.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )
    benchmark.name = "ibm02"

    config = core.effective_config_for_benchmark(
        benchmark,
        core.PlacerConfig(strategy="auto", legal_gap=0.123, search_iters=7, density_weight=25.0),
    )

    assert config.legal_gap == pytest.approx(0.123)
    assert config.search_iters == 7
    assert config.density_weight == pytest.approx(25.0)


def test_placer_reads_density_weight_env(monkeypatch):
    monkeypatch.setenv("JAYDEN_DENSITY_WEIGHT", "25.5")
    module_path = Path("submissions/jaydenpiao/placer.py")
    spec = importlib.util.spec_from_file_location("density_env_placer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    placer = module.JaydenPiaoPlacer()

    assert placer.config.density_weight == pytest.approx(25.5)


def test_placer_reads_recipe_profile_env(monkeypatch):
    monkeypatch.setenv("JAYDEN_RECIPE_PROFILE", "exact_v1")
    module_path = Path("submissions/jaydenpiao/placer.py")
    spec = importlib.util.spec_from_file_location("recipe_env_placer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    placer = module.JaydenPiaoPlacer()

    assert placer.config.recipe_profile == "exact_v1"


def test_recipe_profile_defaults_to_exact_v1_and_off_preserves_baseline():
    core = _load_submission_core()
    benchmark = _density_profile_benchmark("ibm06")

    default = core.build_placement(
        benchmark, core.PlacerConfig(strategy="baseline", transform="identity")
    )
    explicit_off = core.build_placement(
        benchmark,
        core.PlacerConfig(strategy="baseline", transform="identity", recipe_profile="off"),
    )
    exact_v1 = core.build_placement(
        benchmark,
        core.PlacerConfig(strategy="baseline", transform="identity", recipe_profile="exact_v1"),
    )

    assert torch.allclose(default, exact_v1)
    assert not torch.allclose(
        explicit_off[: benchmark.num_hard_macros], default[: benchmark.num_hard_macros]
    )


def test_recipe_profile_rejects_unknown_profile():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor([[3.0, 2.5], [6.0, 2.5]]),
        sizes=torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )

    with pytest.raises(ValueError, match="unsupported recipe profile"):
        core.build_placement(benchmark, core.PlacerConfig(recipe_profile="not-a-profile"))


def test_exact_v1_density_profile_is_deterministic_and_legal():
    core = _load_submission_core()
    benchmark = _density_profile_benchmark("ibm06")

    config = core.PlacerConfig(strategy="baseline", transform="identity", recipe_profile="exact_v1")
    first = core.build_placement(benchmark, config)
    second = core.build_placement(benchmark, config)
    overlaps = compute_overlap_metrics(first, benchmark)

    assert torch.allclose(first, second)
    assert first[0, 0] == pytest.approx(6.2)
    assert first[0, 1] == pytest.approx(2.5)
    assert overlaps["overlap_count"] == 0


def test_exact_v1_density_profile_uses_configured_density_rank():
    core = _load_submission_core()
    benchmark = _density_profile_benchmark("ibm02")

    placement = core.build_placement(
        benchmark,
        core.PlacerConfig(strategy="baseline", transform="identity", recipe_profile="exact_v1"),
    )

    assert placement[0, 0] == pytest.approx(3.0)
    assert placement[0, 1] == pytest.approx(2.5)
    assert placement[1, 0] == pytest.approx(8.8)
    assert placement[1, 1] == pytest.approx(2.5)


def test_exact_v1_density_profile_preserves_fixed_hard_macros():
    core = _load_submission_core()
    benchmark = _density_profile_benchmark("ibm06", fixed=[True, False, False, False, False])

    placement = core.build_placement(
        benchmark,
        core.PlacerConfig(strategy="baseline", transform="identity", recipe_profile="exact_v1"),
    )
    overlaps = compute_overlap_metrics(placement, benchmark)

    assert torch.equal(placement[0], benchmark.macro_positions[0])
    assert overlaps["overlap_count"] == 0


def test_local_search_hpwl_uses_hard_macro_pin_offsets_when_available():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor([[5.0, 5.0], [9.0, 5.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )
    benchmark.net_nodes = [torch.tensor([0, 1])]
    benchmark.net_pin_nodes = [torch.tensor([[0, 0], [1, 0]])]
    benchmark.num_nets = 1
    benchmark.net_weights = torch.tensor([1.0])
    benchmark.macro_pin_offsets = [
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[-2.0, 0.0]]),
    ]

    center_hpwl = core._net_hpwl(0, benchmark.macro_positions.numpy(), benchmark)

    assert center_hpwl == pytest.approx(0.0)


def test_density_surrogate_penalizes_clustered_macros():
    core = _load_submission_core()
    benchmark = _benchmark(
        positions=torch.tensor([[1.25, 1.25], [8.75, 8.75]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )
    clustered = torch.tensor([[3.75, 3.75], [3.75, 3.75]]).numpy()
    spread = torch.tensor([[1.25, 1.25], [8.75, 8.75]]).numpy()

    clustered_cost = core._density_surrogate_cost(clustered, benchmark)
    spread_cost = core._density_surrogate_cost(spread, benchmark)

    assert clustered_cost > spread_cost


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
