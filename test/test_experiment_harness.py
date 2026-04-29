import importlib.util
import json
from pathlib import Path


def _load_script(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_summary(**overrides):
    summary = {
        "schema_version": 1,
        "run_id": "unit",
        "placer_path": "submissions/jaydenpiao/placer.py",
        "command": "unit",
        "git": {
            "commit": "abc123",
            "upstream_commit": "def456",
            "dirty": False,
        },
        "environment": {
            "python": "3.12",
            "platform": "unit",
            "machine": "unit",
        },
        "benchmarks": [
            {
                "name": "ibm01",
                "proxy_cost": 1.0,
                "wirelength": 0.1,
                "density": 0.2,
                "congestion": 0.3,
                "overlaps": 0,
                "runtime": 1.5,
                "valid": True,
            }
        ],
        "aggregate": {
            "average_proxy": 1.0,
            "total_overlaps": 0,
            "total_runtime": 1.5,
            "max_runtime": 1.5,
        },
    }
    summary.update(overrides)
    return summary


def test_build_summary_records_aggregate_metrics():
    run_experiment = _load_script("scripts/run_experiment.py")
    summary = run_experiment.build_summary(
        run_id="unit",
        placer_path=Path("submissions/jaydenpiao/placer.py"),
        command=["run"],
        benchmark_results=[
            {
                "name": "ibm01",
                "proxy_cost": 1.0,
                "wirelength": 0.1,
                "density": 0.2,
                "congestion": 0.3,
                "overlaps": 0,
                "runtime": 1.5,
                "valid": True,
            },
            {
                "name": "ibm02",
                "proxy_cost": 2.0,
                "wirelength": 0.2,
                "density": 0.4,
                "congestion": 0.6,
                "overlaps": 0,
                "runtime": 2.5,
                "valid": True,
            },
        ],
    )

    assert summary["schema_version"] == 1
    assert summary["aggregate"]["average_proxy"] == 1.5
    assert summary["aggregate"]["total_overlaps"] == 0
    assert summary["aggregate"]["max_runtime"] == 2.5
    assert len(summary["benchmarks"]) == 2


def test_check_results_accepts_valid_summary(tmp_path):
    check_results = _load_script("scripts/check_results.py")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_sample_summary()), encoding="utf-8")

    summary = check_results.load_summary(summary_path)
    errors = check_results.validate_summary(summary, max_runtime=55.0, max_avg_proxy=1.4578)

    assert errors == []


def test_check_results_flags_overlap_runtime_invalid_and_proxy_failures():
    check_results = _load_script("scripts/check_results.py")
    summary = _sample_summary(
        benchmarks=[
            {
                "name": "ibm01",
                "proxy_cost": 2.0,
                "wirelength": 0.1,
                "density": 0.2,
                "congestion": 0.3,
                "overlaps": 1,
                "runtime": 60.0,
                "valid": False,
            }
        ],
        aggregate={
            "average_proxy": 2.0,
            "total_overlaps": 1,
            "total_runtime": 60.0,
            "max_runtime": 60.0,
        },
    )

    errors = check_results.validate_summary(summary, max_runtime=55.0, max_avg_proxy=1.4578)

    assert "total overlaps 1 != 0" in errors
    assert "benchmark ibm01 is invalid" in errors
    assert "benchmark ibm01 runtime 60.00s exceeds 55.00s" in errors
    assert "average proxy 2.0000 exceeds 1.4578" in errors


def test_compare_results_reports_average_and_per_benchmark_deltas():
    compare_results = _load_script("scripts/compare_results.py")
    base = _sample_summary()
    candidate = _sample_summary(
        run_id="candidate",
        benchmarks=[
            {
                "name": "ibm01",
                "proxy_cost": 0.75,
                "wirelength": 0.1,
                "density": 0.2,
                "congestion": 0.3,
                "overlaps": 0,
                "runtime": 1.5,
                "valid": True,
            }
        ],
        aggregate={
            "average_proxy": 0.75,
            "total_overlaps": 0,
            "total_runtime": 1.5,
            "max_runtime": 1.5,
        },
    )

    comparison = compare_results.compare_summaries(base, candidate)

    assert comparison["average_delta"] == -0.25
    assert comparison["improved_count"] == 1
    assert comparison["regressed_count"] == 0
    assert comparison["benchmarks"][0]["delta"] == -0.25
