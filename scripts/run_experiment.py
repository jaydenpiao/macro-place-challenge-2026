"""Run official macro-placement evaluation and write a reproducible summary."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from macro_place.evaluate import IBM_BENCHMARKS, NG45_BENCHMARKS, _load_placer, evaluate_benchmark


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_metadata() -> dict[str, Any]:
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "upstream_commit": _git_output(["rev-parse", "upstream/main"]),
        "dirty": bool(_git_output(["status", "--porcelain"])),
    }


def _environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _clean_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(result["name"]),
        "proxy_cost": float(result["proxy_cost"]),
        "wirelength": float(result["wirelength"]),
        "density": float(result["density"]),
        "congestion": float(result["congestion"]),
        "overlaps": int(result["overlaps"]),
        "runtime": float(result["runtime"]),
        "valid": bool(result["valid"]),
    }


def build_summary(
    *,
    run_id: str,
    placer_path: Path,
    command: Iterable[str],
    benchmark_results: list[dict[str, Any]],
) -> dict[str, Any]:
    clean_results = [_clean_result(result) for result in benchmark_results]
    average_proxy = sum(result["proxy_cost"] for result in clean_results) / len(clean_results)
    total_overlaps = sum(result["overlaps"] for result in clean_results)
    total_runtime = sum(result["runtime"] for result in clean_results)
    max_runtime = max(result["runtime"] for result in clean_results)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "placer_path": str(placer_path),
        "command": " ".join(command),
        "git": _git_metadata(),
        "environment": _environment_metadata(),
        "env_knobs": {
            key: os.environ.get(key, "")
            for key in [
                "JAYDEN_PLACER_SEED",
                "JAYDEN_SEARCH_ITERS",
                "JAYDEN_LEGAL_GAP",
                "JAYDEN_TRANSFORM",
                "JAYDEN_STRATEGY",
                "JAYDEN_DENSITY_WEIGHT",
            ]
        },
        "benchmarks": clean_results,
        "aggregate": {
            "average_proxy": float(average_proxy),
            "total_overlaps": int(total_overlaps),
            "total_runtime": float(total_runtime),
            "max_runtime": float(max_runtime),
        },
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _select_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(IBM_BENCHMARKS)
    if args.ng45:
        return list(NG45_BENCHMARKS.keys())
    return [name.strip() for name in args.benchmarks.split(",") if name.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    benchmark_names = _select_benchmarks(args)
    testcase_root = Path("external/MacroPlacement/Testcases/ICCAD04")

    results: list[dict[str, Any]] = []
    for name in benchmark_names:
        print(f"{name}...", flush=True)
        ng45_dir = NG45_BENCHMARKS.get(name) if args.ng45 or name in NG45_BENCHMARKS else None
        result = evaluate_benchmark(placer, name, str(testcase_root), ng45_dir=ng45_dir)
        print(
            f"  proxy={result['proxy_cost']:.4f} "
            f"wl={result['wirelength']:.3f} "
            f"den={result['density']:.3f} "
            f"cong={result['congestion']:.3f} "
            f"overlaps={result['overlaps']} "
            f"runtime={result['runtime']:.2f}s",
            flush=True,
        )
        results.append(result)

    summary = build_summary(
        run_id=args.run_id,
        placer_path=placer_path,
        command=sys.argv,
        benchmark_results=results,
    )
    summary_path = write_summary(summary, Path(args.output_root) / args.run_id)
    print(f"summary written: {summary_path}")
    print(f"average proxy: {summary['aggregate']['average_proxy']:.4f}")
    print(f"total overlaps: {summary['aggregate']['total_overlaps']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placer", required=True, help="Path to placer .py file.")
    parser.add_argument("--run-id", required=True, help="Output directory name under results/.")
    parser.add_argument("--benchmarks", default="ibm01", help="Comma-separated benchmark names.")
    parser.add_argument("--all", action="store_true", help="Run all 17 IBM benchmarks.")
    parser.add_argument("--ng45", action="store_true", help="Run public NG45 benchmarks.")
    parser.add_argument("--output-root", default="results", help="Root output directory.")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
