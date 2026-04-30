"""Preflight checks for official Docker/GPU cloud parity runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

DEFAULT_GPU_SMOKE_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class PreflightError(RuntimeError):
    """Raised when a required parity host check fails."""


Runner = Callable[[str, list[str]], CommandResult]


def _subprocess_runner(label: str, command: list[str]) -> CommandResult:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        return CommandResult(returncode=127, stdout="", stderr=str(exc))
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _commands(gpu_smoke_image: str) -> list[tuple[str, list[str]]]:
    return [
        ("docker client/server", ["docker", "version"]),
        ("host NVIDIA driver", ["nvidia-smi"]),
        (
            "docker GPU container smoke",
            ["docker", "run", "--rm", "--gpus", "all", gpu_smoke_image, "nvidia-smi"],
        ),
    ]


def check_host(
    *,
    run: Runner = _subprocess_runner,
    gpu_smoke_image: str = DEFAULT_GPU_SMOKE_IMAGE,
) -> None:
    for label, command in _commands(gpu_smoke_image):
        result = run(label, command)
        if result.returncode == 0:
            continue
        detail = (result.stderr or result.stdout).strip()
        if detail:
            raise PreflightError(f"{label} failed with exit {result.returncode}: {detail}")
        raise PreflightError(f"{label} failed with exit {result.returncode}")


def main(argv: Sequence[str] | None = None, *, run: Runner = _subprocess_runner) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-smoke-image",
        default=os.environ.get("JAYDEN_DOCKER_GPU_SMOKE_IMAGE", DEFAULT_GPU_SMOKE_IMAGE),
        help="CUDA image used for docker --gpus smoke testing.",
    )
    args = parser.parse_args(argv)

    try:
        check_host(run=run, gpu_smoke_image=args.gpu_smoke_image)
    except PreflightError as exc:
        print(f"cloud parity preflight failed: {exc}", file=sys.stderr)
        return 2

    print("cloud parity preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
