"""Run and compare multiple environment-knob placement candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_results import compare_summaries, load_summary, print_report  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    env: dict[str, str]


RunCommand = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess[str]]


def parse_variant(raw: str) -> Variant:
    name, separator, env_spec = raw.partition(":")
    if not separator:
        raise ValueError("variant must use name:JAYDEN_KEY=value[;JAYDEN_KEY=value]")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"unsupported variant name: {name}")

    env: dict[str, str] = {}
    for entry in [part for part in env_spec.split(";") if part]:
        key, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"variant env entry must use KEY=value: {entry}")
        if not key.startswith("JAYDEN_"):
            raise ValueError(f"variant env key must start with JAYDEN_: {key}")
        env[key] = value
    if not env:
        raise ValueError(f"variant {name} does not set any env knobs")
    return Variant(name=name, env=env)


def _run_subprocess(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, env=env, check=False)


def _candidate_command(args: argparse.Namespace, candidate_run_id: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--placer",
        args.placer,
        "--run-id",
        candidate_run_id,
        "--output-root",
        str(args.output_root),
    ]
    if args.benchmarks:
        command.extend(["--benchmarks", args.benchmarks])
    else:
        command.append("--all")
    return command


def _write_scan_summary(
    *,
    run_id: str,
    baseline: Path,
    output_root: Path,
    rows: list[dict[str, object]],
) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scan_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "baseline": str(baseline),
                "variants": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_scan(
    args: argparse.Namespace,
    *,
    run_command: RunCommand = _run_subprocess,
) -> int:
    baseline = load_summary(args.baseline)
    rows: list[dict[str, object]] = []

    for variant in [parse_variant(raw) for raw in args.variant]:
        candidate_run_id = f"{args.run_id}__{variant.name}"
        command = _candidate_command(args, candidate_run_id)
        env = os.environ.copy()
        env.update(variant.env)

        print(f"variant {variant.name}: {' '.join(command)}", flush=True)
        result = run_command(command, env=env)
        if result.returncode != 0:
            return int(result.returncode)

        candidate_path = args.output_root / candidate_run_id / "summary.json"
        candidate = load_summary(candidate_path)
        comparison = compare_summaries(baseline, candidate)
        print_report(comparison)

        rows.append(
            {
                "name": variant.name,
                "env": variant.env,
                "run_id": candidate_run_id,
                "summary_path": str(candidate_path),
                "candidate_average_proxy": comparison["candidate_average_proxy"],
                "average_delta": comparison["average_delta"],
                "improved_count": comparison["improved_count"],
                "regressed_count": comparison["regressed_count"],
                "missing_benchmarks": comparison["missing_benchmarks"],
                "comparison_complete": not comparison["missing_benchmarks"],
            }
        )

    summary_path = _write_scan_summary(
        run_id=args.run_id,
        baseline=args.baseline,
        output_root=args.output_root,
        rows=rows,
    )
    print(f"scan summary written: {summary_path}")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    run_command: RunCommand = _run_subprocess,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Scan output directory under results/.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/all-ibm-auto-transform/summary.json"),
        help="Baseline summary.json to compare every candidate against.",
    )
    parser.add_argument(
        "--placer",
        default="submissions/jaydenpiao/placer.py",
        help="Placer entry point passed to scripts/run_experiment.py.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant spec: name:JAYDEN_KEY=value[;JAYDEN_KEY=value]. Repeatable.",
    )
    parser.add_argument("--benchmarks", help="Comma-separated benchmark names for smoke scans.")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    try:
        args = parser.parse_args(argv)
        return run_scan(args, run_command=run_command)
    except (FileNotFoundError, ValueError) as exc:
        print(f"candidate scan error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
