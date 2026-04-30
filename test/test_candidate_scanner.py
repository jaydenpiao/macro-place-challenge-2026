import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_scanner():
    path = Path("scripts/scan_candidates.py")
    spec = importlib.util.spec_from_file_location("candidate_scanner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(run_id: str, proxy: float) -> dict:
    return {
        "run_id": run_id,
        "benchmarks": [
            {
                "name": "ibm01",
                "proxy_cost": proxy,
                "overlaps": 0,
                "valid": True,
            }
        ],
        "aggregate": {
            "average_proxy": proxy,
            "total_overlaps": 0,
            "max_runtime": 1.0,
        },
    }


def test_parse_variant_accepts_jayden_env_knobs_only():
    scanner = _load_scanner()

    variant = scanner.parse_variant("gap005:JAYDEN_LEGAL_GAP=0.005;JAYDEN_TRANSFORM=auto")

    assert variant.name == "gap005"
    assert variant.env == {"JAYDEN_LEGAL_GAP": "0.005", "JAYDEN_TRANSFORM": "auto"}
    with pytest.raises(ValueError, match="JAYDEN_"):
        scanner.parse_variant("bad:PATH=/tmp")


def test_scan_runs_variants_and_writes_comparison_summary(tmp_path):
    scanner = _load_scanner()
    baseline_path = tmp_path / "baseline.json"
    output_root = tmp_path / "results"
    baseline_path.write_text(json.dumps(_summary("baseline", 1.5)), encoding="utf-8")
    calls = []

    def runner(command, *, env):
        calls.append((command, env))
        run_id = command[command.index("--run-id") + 1]
        candidate_dir = output_root / run_id
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "summary.json").write_text(
            json.dumps(_summary(run_id, 1.4)), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    rc = scanner.main(
        [
            "--run-id",
            "scan-smoke",
            "--baseline",
            str(baseline_path),
            "--output-root",
            str(output_root),
            "--benchmarks",
            "ibm01",
            "--variant",
            "gap005:JAYDEN_LEGAL_GAP=0.005",
        ],
        run_command=runner,
    )

    assert rc == 0
    command, env = calls[0]
    assert command[:3] == [sys.executable, "scripts/run_experiment.py", "--placer"]
    assert command[command.index("--run-id") + 1] == "scan-smoke__gap005"
    assert command[command.index("--benchmarks") + 1] == "ibm01"
    assert env["JAYDEN_LEGAL_GAP"] == "0.005"

    scan_summary = json.loads((output_root / "scan-smoke" / "scan_summary.json").read_text())
    assert scan_summary["run_id"] == "scan-smoke"
    assert scan_summary["baseline"] == str(baseline_path)
    assert scan_summary["variants"][0]["candidate_average_proxy"] == 1.4
    assert scan_summary["variants"][0]["average_delta"] == pytest.approx(-0.1)
    assert scan_summary["variants"][0]["comparison_complete"] is True
    assert scan_summary["variants"][0]["missing_benchmarks"] == []


def test_scan_stops_when_candidate_run_fails(tmp_path):
    scanner = _load_scanner()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_summary("baseline", 1.5)), encoding="utf-8")

    def runner(command, *, env):
        return SimpleNamespace(returncode=7)

    rc = scanner.main(
        [
            "--run-id",
            "scan-fail",
            "--baseline",
            str(baseline_path),
            "--output-root",
            str(tmp_path / "results"),
            "--variant",
            "gap005:JAYDEN_LEGAL_GAP=0.005",
        ],
        run_command=runner,
    )

    assert rc == 7


def test_scan_reports_missing_baseline_as_usage_error(tmp_path):
    scanner = _load_scanner()

    rc = scanner.main(
        [
            "--run-id",
            "scan-missing-baseline",
            "--baseline",
            str(tmp_path / "missing.json"),
            "--output-root",
            str(tmp_path / "results"),
            "--variant",
            "gap005:JAYDEN_LEGAL_GAP=0.005",
        ],
        run_command=lambda command, *, env: SimpleNamespace(returncode=0),
    )

    assert rc == 2
