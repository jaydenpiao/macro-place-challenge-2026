import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_overlap_metrics


def _load_searcher():
    path = Path("scripts/search_candidates.py")
    spec = importlib.util.spec_from_file_location("exact_candidate_search", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_single_move_candidates_are_legal_and_preserve_fixed_macros():
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([True, False, False]),
        num_hard=3,
    )
    config = searcher.SearchConfig(
        families=("single",),
        step_fractions=(0.2,),
        max_candidates_per_benchmark=16,
    )

    candidates = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert candidates
    assert {candidate.family for candidate in candidates} == {"single"}
    for candidate in candidates:
        assert torch.equal(candidate.placement[0], benchmark.macro_positions[0])
        assert compute_overlap_metrics(candidate.placement, benchmark)["overlap_count"] == 0
        assert candidate.recipe["family"] == "single"
        assert candidate.recipe["benchmark"] == "synthetic"


def test_candidate_generation_stops_when_max_candidates_is_reached(monkeypatch):
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False, False]),
        num_hard=3,
    )
    calls = 0

    def fake_make_candidate(benchmark, baseline, hard_positions, **kwargs):
        nonlocal calls
        calls += 1
        placement = baseline.clone()
        placement[0, 0] += calls * 0.001
        return searcher.Candidate(
            name=f"candidate-{calls}",
            family=kwargs["family"],
            recipe=kwargs["recipe"],
            placement=placement,
        )

    monkeypatch.setattr(searcher, "_make_candidate", fake_make_candidate)
    config = searcher.SearchConfig(
        families=("single",),
        step_fractions=(0.2,),
        max_candidates_per_benchmark=2,
    )

    candidates = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert len(candidates) == 2
    assert calls == 2


def test_candidate_generation_applies_per_family_candidate_cap(monkeypatch):
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False, False]),
        num_hard=3,
    )
    observed_limits = []

    def family_generator(family):
        def generate(benchmark, baseline, config, limit):
            observed_limits.append((family, limit))
            candidates = []
            for idx in range(limit):
                placement = baseline.clone()
                placement[0, 0] += (len(observed_limits) * 10 + idx) * 0.001
                candidates.append(
                    searcher.Candidate(
                        name=f"{family}-{idx}",
                        family=family,
                        recipe={"family": family, "benchmark": benchmark.name},
                        placement=placement,
                    )
                )
            return candidates

        return generate

    monkeypatch.setattr(searcher, "_single_move_candidates", family_generator("single"))
    monkeypatch.setattr(searcher, "_density_push_candidates", family_generator("density"))
    config = searcher.SearchConfig(
        families=("single", "density"),
        max_candidates_per_benchmark=10,
        max_candidates_per_family=2,
    )

    candidates = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert observed_limits == [("single", 2), ("density", 2)]
    assert [candidate.family for candidate in candidates] == [
        "single",
        "single",
        "density",
        "density",
    ]


def test_swap_candidates_are_deterministic_and_skip_dissimilar_macros():
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 1.0]]),
        fixed=torch.tensor([False, False, False]),
        num_hard=3,
    )
    config = searcher.SearchConfig(
        families=("swap",),
        max_candidates_per_benchmark=16,
        swap_area_ratio=1.25,
    )

    first = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)
    second = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert [candidate.name for candidate in first] == [candidate.name for candidate in second]
    assert [candidate.recipe["indices"] for candidate in first] == [[0, 1]]
    assert torch.equal(first[0].placement[0], benchmark.macro_positions[1])
    assert torch.equal(first[0].placement[1], benchmark.macro_positions[0])


def test_density_candidates_push_macros_away_from_dense_bin():
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [2.2, 4.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False, False]),
        num_hard=3,
    )
    config = searcher.SearchConfig(
        families=("density",),
        step_fractions=(0.2,),
        max_candidates_per_benchmark=4,
    )

    candidates = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert candidates
    candidate = candidates[0]
    moved_idx = candidate.recipe["macro_index"]
    dense_point = torch.tensor(candidate.recipe["away_from"])
    before = torch.linalg.vector_norm(benchmark.macro_positions[moved_idx] - dense_point)
    after = torch.linalg.vector_norm(candidate.placement[moved_idx] - dense_point)
    assert after > before
    assert compute_overlap_metrics(candidate.placement, benchmark)["overlap_count"] == 0


def test_transform_candidates_record_transform_recipe_and_preserve_fixed_macros():
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [4.0, 4.0], [8.0, 8.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([True, False, False]),
        num_hard=3,
    )
    config = searcher.SearchConfig(
        families=("transform",),
        max_candidates_per_benchmark=4,
    )

    candidates = searcher.generate_candidates(benchmark, benchmark.macro_positions, config)

    assert candidates
    assert {candidate.family for candidate in candidates} == {"transform"}
    assert {candidate.recipe["mode"] for candidate in candidates} <= {
        "flip_x",
        "flip_y",
        "flip_xy",
    }
    for candidate in candidates:
        assert torch.equal(candidate.placement[0], benchmark.macro_positions[0])


def test_screen_candidates_uses_exact_scores_and_writes_trace(tmp_path):
    searcher = _load_searcher()
    benchmark = _benchmark(
        positions=torch.tensor([[2.0, 2.0], [5.0, 5.0]]),
        sizes=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        fixed=torch.tensor([False, False]),
        num_hard=2,
    )
    baseline = benchmark.macro_positions.clone()
    better = searcher.Candidate(
        name="better",
        family="single",
        recipe={"family": "single", "benchmark": "synthetic"},
        placement=baseline + torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
    )
    worse = searcher.Candidate(
        name="worse",
        family="single",
        recipe={"family": "single", "benchmark": "synthetic"},
        placement=baseline + torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
    )

    def scorer(placement):
        return {
            "proxy_cost": float(placement[0, 0]),
            "wirelength_cost": 0.1,
            "density_cost": 0.2,
            "congestion_cost": 0.3,
            "overlap_count": 0,
        }

    trace_path = tmp_path / "candidate_trace.jsonl"
    result = searcher.screen_candidates(
        benchmark_name="synthetic",
        baseline_placement=baseline,
        candidates=[better, worse],
        score_placement=scorer,
        trace_path=trace_path,
    )

    assert result.best_name == "baseline"
    assert result.baseline_proxy == pytest.approx(2.0)
    assert result.best_proxy == pytest.approx(2.0)
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert [record["candidate"] for record in records] == ["baseline", "better", "worse"]
    assert records[0]["recipe"]["family"] == "baseline"


def test_summary_records_search_metadata_and_aggregate_best_proxy(tmp_path):
    searcher = _load_searcher()
    output_dir = tmp_path / "results" / "search-smoke"
    result = searcher.BenchmarkSearchResult(
        name="synthetic",
        baseline_proxy=2.0,
        best_proxy=1.5,
        best_name="candidate-a",
        best_recipe={"family": "single", "benchmark": "synthetic"},
        candidate_count=3,
        improved=True,
        runtime=0.25,
        overlaps=0,
        valid=True,
        wirelength=0.1,
        density=0.2,
        congestion=0.3,
    )

    summary_path = searcher.write_summary(
        run_id="search-smoke",
        placer_path=Path("submissions/jaydenpiao/placer.py"),
        command=["search"],
        config=searcher.SearchConfig(families=("single",), max_candidates_per_family=2),
        results=[result],
        output_dir=output_dir,
    )

    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == 1
    assert summary["environment"]["python"]
    assert summary["aggregate"]["average_proxy"] == pytest.approx(1.5)
    assert summary["aggregate"]["improved_count"] == 1
    assert summary["benchmarks"][0]["proxy_cost"] == pytest.approx(1.5)
    assert summary["benchmarks"][0]["best_recipe"]["family"] == "single"
    assert summary["search_config"]["max_candidates_per_family"] == 2
