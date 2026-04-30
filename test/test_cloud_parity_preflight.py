import importlib.util
import sys
from pathlib import Path

import pytest


def _load_preflight():
    path = Path("scripts/check_cloud_parity_host.py")
    spec = importlib.util.spec_from_file_location("cloud_parity_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_host_runs_required_parity_checks_in_order():
    preflight = _load_preflight()
    calls = []

    def runner(label, command):
        calls.append((label, command))
        return preflight.CommandResult(returncode=0, stdout="ok", stderr="")

    preflight.check_host(run=runner, gpu_smoke_image="nvidia/cuda:test")

    assert calls == [
        ("docker client/server", ["docker", "version"]),
        ("host NVIDIA driver", ["nvidia-smi"]),
        (
            "docker GPU container smoke",
            ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:test", "nvidia-smi"],
        ),
    ]


def test_check_host_reports_failed_gpu_container_smoke():
    preflight = _load_preflight()

    def runner(label, command):
        if label == "docker GPU container smoke":
            return preflight.CommandResult(returncode=125, stdout="", stderr="no gpu runtime")
        return preflight.CommandResult(returncode=0, stdout="ok", stderr="")

    with pytest.raises(preflight.PreflightError, match="docker GPU container smoke"):
        preflight.check_host(run=runner, gpu_smoke_image="nvidia/cuda:test")


def test_main_returns_error_when_preflight_fails():
    preflight = _load_preflight()

    def runner(label, command):
        return preflight.CommandResult(returncode=1, stdout="", stderr="docker unavailable")

    assert preflight.main(["--gpu-smoke-image", "nvidia/cuda:test"], run=runner) == 2
