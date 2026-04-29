"""Compare two experiment summary JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def compare_summaries(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_by_name = {row["name"]: row for row in base["benchmarks"]}
    candidate_by_name = {row["name"]: row for row in candidate["benchmarks"]}
    names = sorted(set(base_by_name) | set(candidate_by_name))

    rows = []
    improved = 0
    regressed = 0
    missing = []
    for name in names:
        base_row = base_by_name.get(name)
        candidate_row = candidate_by_name.get(name)
        if base_row is None or candidate_row is None:
            missing.append(name)
            continue

        delta = float(candidate_row["proxy_cost"]) - float(base_row["proxy_cost"])
        if delta < 0:
            improved += 1
        elif delta > 0:
            regressed += 1
        rows.append(
            {
                "name": name,
                "base_proxy": float(base_row["proxy_cost"]),
                "candidate_proxy": float(candidate_row["proxy_cost"]),
                "delta": delta,
                "base_overlaps": int(base_row["overlaps"]),
                "candidate_overlaps": int(candidate_row["overlaps"]),
                "base_valid": bool(base_row["valid"]),
                "candidate_valid": bool(candidate_row["valid"]),
            }
        )

    base_avg = float(base["aggregate"]["average_proxy"])
    candidate_avg = float(candidate["aggregate"]["average_proxy"])
    return {
        "base_run_id": base["run_id"],
        "candidate_run_id": candidate["run_id"],
        "base_average_proxy": base_avg,
        "candidate_average_proxy": candidate_avg,
        "average_delta": candidate_avg - base_avg,
        "improved_count": improved,
        "regressed_count": regressed,
        "missing_benchmarks": missing,
        "benchmarks": rows,
    }


def print_report(comparison: dict[str, Any]) -> None:
    print(
        f"{comparison['base_run_id']} -> {comparison['candidate_run_id']}: "
        f"avg {comparison['base_average_proxy']:.6f} -> "
        f"{comparison['candidate_average_proxy']:.6f} "
        f"(delta {comparison['average_delta']:+.6f})"
    )
    print(
        f"benchmarks improved={comparison['improved_count']} "
        f"regressed={comparison['regressed_count']}"
    )
    if comparison["missing_benchmarks"]:
        print("missing benchmarks: " + ", ".join(comparison["missing_benchmarks"]))
    print()
    print(f"{'benchmark':>10} {'base':>10} {'candidate':>10} {'delta':>10} {'valid':>9}")
    for row in comparison["benchmarks"]:
        valid = "yes" if row["candidate_valid"] and row["candidate_overlaps"] == 0 else "no"
        print(
            f"{row['name']:>10} "
            f"{row['base_proxy']:>10.6f} "
            f"{row['candidate_proxy']:>10.6f} "
            f"{row['delta']:>+10.6f} "
            f"{valid:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path, help="Baseline summary.json.")
    parser.add_argument("candidate", type=Path, help="Candidate summary.json.")
    parser.add_argument("--json", type=Path, help="Optional path to write comparison JSON.")
    args = parser.parse_args()

    comparison = compare_summaries(load_summary(args.base), load_summary(args.candidate))
    print_report(comparison)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
