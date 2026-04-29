"""Validate an experiment summary against promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_summary(
    summary: dict[str, Any],
    *,
    max_runtime: float = 3300.0,
    max_avg_proxy: float | None = None,
) -> list[str]:
    errors: list[str] = []
    aggregate = summary.get("aggregate", {})
    total_overlaps = int(aggregate.get("total_overlaps", 0))
    if total_overlaps != 0:
        errors.append(f"total overlaps {total_overlaps} != 0")

    benchmarks = summary.get("benchmarks", [])
    if not benchmarks:
        errors.append("no benchmark results found")

    for result in benchmarks:
        name = result.get("name", "<unknown>")
        if not result.get("valid", False):
            errors.append(f"benchmark {name} is invalid")
        runtime = float(result.get("runtime", 0.0))
        if runtime > max_runtime:
            errors.append(f"benchmark {name} runtime {runtime:.2f}s exceeds {max_runtime:.2f}s")
        overlaps = int(result.get("overlaps", 0))
        if overlaps != 0:
            errors.append(f"benchmark {name} overlaps {overlaps} != 0")

    if max_avg_proxy is not None:
        average_proxy = float(aggregate.get("average_proxy", float("inf")))
        if average_proxy > max_avg_proxy:
            errors.append(f"average proxy {average_proxy:.4f} exceeds {max_avg_proxy:.4f}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to results/<run-id>/summary.json")
    parser.add_argument("--max-runtime", type=float, default=3300.0)
    parser.add_argument("--max-avg-proxy", type=float, default=None)
    args = parser.parse_args()

    summary = load_summary(args.summary)
    errors = validate_summary(
        summary,
        max_runtime=args.max_runtime,
        max_avg_proxy=args.max_avg_proxy,
    )
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    aggregate = summary["aggregate"]
    print(
        "result check passed: "
        f"average_proxy={aggregate['average_proxy']:.4f} "
        f"total_overlaps={aggregate['total_overlaps']} "
        f"max_runtime={aggregate['max_runtime']:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
