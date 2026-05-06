"""Exact-proxy-screen structural placement candidates.

This tool is intentionally offline-only: it uses the official proxy evaluator
to screen generated placements, but it does not change the submission runtime
path or write learned coordinates into the placer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_place.evaluate import IBM_BENCHMARKS, _load_placer  # noqa: E402
from macro_place.loader import load_benchmark_from_dir  # noqa: E402
from macro_place.objective import compute_overlap_metrics, compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402

SUBMISSION_DIR = ROOT / "submissions" / "jaydenpiao"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

import core as jayden_core  # noqa: E402

WEAK_IBM_BENCHMARKS = ("ibm18", "ibm17", "ibm06", "ibm12", "ibm15", "ibm14", "ibm02")
TRANSFORM_MODES = ("identity", "flip_x", "flip_y", "flip_xy")


@dataclass(frozen=True)
class SearchConfig:
    families: tuple[str, ...] = ("single", "swap", "density", "transform")
    step_fractions: tuple[float, ...] = (0.02, 0.05, 0.1)
    max_candidates_per_benchmark: int = 128
    max_candidates_per_family: int | None = None
    legal_gap: float = 0.01
    swap_area_ratio: float = 1.5


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    recipe: dict[str, object]
    placement: torch.Tensor


@dataclass(frozen=True)
class BenchmarkSearchResult:
    name: str
    baseline_proxy: float
    best_proxy: float
    best_name: str
    best_recipe: dict[str, object]
    candidate_count: int
    improved: bool
    runtime: float
    overlaps: int
    valid: bool
    wirelength: float
    density: float
    congestion: float


ScorePlacement = Callable[[torch.Tensor], dict[str, object]]


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in _parse_csv(raw))
    if not values:
        raise ValueError("at least one step fraction is required")
    if any(value <= 0.0 for value in values):
        raise ValueError("step fractions must be positive")
    return values


def _normalize_families(families: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    allowed = {"single", "swap", "density", "transform"}
    for family in families:
        name = family.strip().lower()
        if not name:
            continue
        if name == "single_move":
            name = "single"
        if name not in allowed:
            raise ValueError(f"unsupported candidate family: {family}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("at least one candidate family is required")
    return tuple(normalized)


def generate_candidates(
    benchmark,
    baseline_placement: torch.Tensor,
    config: SearchConfig,
) -> list[Candidate]:
    """Generate deterministic legal structural candidates from a baseline placement."""
    config = replace(config, families=_normalize_families(config.families))
    candidates: list[Candidate] = []
    seen: set[tuple[float, ...]] = set()

    for family in config.families:
        remaining = int(config.max_candidates_per_benchmark) - len(candidates)
        if remaining <= 0:
            return candidates
        if config.max_candidates_per_family is not None:
            remaining = min(remaining, int(config.max_candidates_per_family))
        if family == "single":
            generated = _single_move_candidates(benchmark, baseline_placement, config, remaining)
        elif family == "swap":
            generated = _swap_candidates(benchmark, baseline_placement, config, remaining)
        elif family == "density":
            generated = _density_push_candidates(benchmark, baseline_placement, config, remaining)
        elif family == "transform":
            generated = _transform_candidates(benchmark, baseline_placement, config, remaining)
        else:  # pragma: no cover - guarded by _normalize_families
            generated = []

        for candidate in generated:
            key = _placement_key(candidate.placement, int(benchmark.num_hard_macros))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
            if len(candidates) >= int(config.max_candidates_per_benchmark):
                return candidates

    return candidates


def _single_move_candidates(
    benchmark,
    baseline: torch.Tensor,
    config: SearchConfig,
    limit: int,
) -> list[Candidate]:
    canvas = np.array([float(benchmark.canvas_width), float(benchmark.canvas_height)])
    directions = (
        ("left", np.array([-1.0, 0.0])),
        ("right", np.array([1.0, 0.0])),
        ("down", np.array([0.0, -1.0])),
        ("up", np.array([0.0, 1.0])),
    )
    candidates: list[Candidate] = []
    for idx in _movable_hard_indices(benchmark):
        for step_fraction in config.step_fractions:
            step = max(canvas) * float(step_fraction)
            for direction_name, direction in directions:
                hard = _hard_positions_np(baseline, benchmark)
                hard[idx] = hard[idx] + direction * step
                candidate = _make_candidate(
                    benchmark,
                    baseline,
                    hard,
                    family="single",
                    name=f"single-m{idx}-{direction_name}-{step_fraction:g}",
                    recipe={
                        "family": "single",
                        "benchmark": benchmark.name,
                        "macro_index": int(idx),
                        "direction": direction_name,
                        "step_fraction": float(step_fraction),
                        "delta": [float(direction[0] * step), float(direction[1] * step)],
                    },
                    gap=float(config.legal_gap),
                )
                if candidate is not None:
                    candidates.append(candidate)
                    if len(candidates) >= limit:
                        return candidates
    return candidates


def _swap_candidates(
    benchmark,
    baseline: torch.Tensor,
    config: SearchConfig,
    limit: int,
) -> list[Candidate]:
    sizes = benchmark.macro_sizes[: benchmark.num_hard_macros].detach().cpu().numpy()
    candidates: list[Candidate] = []
    movable = _movable_hard_indices(benchmark)
    for offset, i in enumerate(movable):
        for j in movable[offset + 1 :]:
            if not _similar_size(sizes[i], sizes[j], max_area_ratio=float(config.swap_area_ratio)):
                continue
            hard = _hard_positions_np(baseline, benchmark)
            hard[i], hard[j] = hard[j].copy(), hard[i].copy()
            candidate = _make_candidate(
                benchmark,
                baseline,
                hard,
                family="swap",
                name=f"swap-m{i}-m{j}",
                recipe={
                    "family": "swap",
                    "benchmark": benchmark.name,
                    "indices": [int(i), int(j)],
                    "swap_area_ratio": float(config.swap_area_ratio),
                },
                gap=float(config.legal_gap),
            )
            if candidate is not None:
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _density_push_candidates(
    benchmark,
    baseline: torch.Tensor,
    config: SearchConfig,
    limit: int,
) -> list[Candidate]:
    centers = baseline[: benchmark.num_hard_macros].detach().cpu().numpy()
    if centers.size == 0:
        return []

    dense_point = _densest_bin_center(benchmark, baseline)
    distances = np.linalg.norm(centers - dense_point, axis=1)
    ranked = [
        idx
        for idx in np.argsort(distances).tolist()
        if idx in set(_movable_hard_indices(benchmark))
    ]
    candidates: list[Candidate] = []
    for idx in ranked[: max(1, min(8, len(ranked)))]:
        vector = centers[idx] - dense_point
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-9:
            vector = centers[idx] - np.array(
                [float(benchmark.canvas_width) / 2.0, float(benchmark.canvas_height) / 2.0]
            )
            norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-9:
            vector = np.array([1.0, 0.0])
            norm = 1.0
        unit = vector / norm
        for step_fraction in config.step_fractions:
            step = max(float(benchmark.canvas_width), float(benchmark.canvas_height)) * float(
                step_fraction
            )
            hard = _hard_positions_np(baseline, benchmark)
            hard[idx] = hard[idx] + unit * step
            candidate = _make_candidate(
                benchmark,
                baseline,
                hard,
                family="density",
                name=f"density-m{idx}-{step_fraction:g}",
                recipe={
                    "family": "density",
                    "benchmark": benchmark.name,
                    "macro_index": int(idx),
                    "step_fraction": float(step_fraction),
                    "away_from": [float(dense_point[0]), float(dense_point[1])],
                },
                gap=float(config.legal_gap),
            )
            if candidate is not None:
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates
    return candidates


def _transform_candidates(
    benchmark,
    baseline: torch.Tensor,
    config: SearchConfig,
    limit: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for mode in TRANSFORM_MODES:
        transformed = jayden_core._initial_placement(benchmark, mode)
        hard = _hard_positions_np(transformed, benchmark)
        candidate = _make_candidate(
            benchmark,
            transformed,
            hard,
            family="transform",
            name=f"transform-{mode}",
            recipe={"family": "transform", "benchmark": benchmark.name, "mode": mode},
            gap=float(config.legal_gap),
        )
        if candidate is None:
            continue
        if torch.allclose(candidate.placement, baseline):
            continue
        candidates.append(candidate)
        if len(candidates) >= limit:
            return candidates
    return candidates


def _make_candidate(
    benchmark,
    baseline: torch.Tensor,
    hard_positions: np.ndarray,
    *,
    family: str,
    name: str,
    recipe: dict[str, object],
    gap: float,
) -> Candidate | None:
    n_hard = int(benchmark.num_hard_macros)
    full = baseline.detach().clone().cpu()
    if n_hard <= 0:
        return None

    hard_sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool)
    movable = ~fixed
    hard = hard_positions.astype(np.float64, copy=True)
    jayden_core._clamp_movable_to_canvas(
        hard,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )
    hard = jayden_core.legalize_hard_macros(
        hard,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
        gap=gap,
    )
    full[:n_hard] = torch.tensor(hard, dtype=full.dtype)
    if torch.allclose(full, baseline.detach().cpu()):
        return None
    if compute_overlap_metrics(full, benchmark)["overlap_count"] != 0:
        return None
    return Candidate(
        name=name, family=family, recipe=recipe, placement=full.to(dtype=baseline.dtype)
    )


def _hard_positions_np(placement: torch.Tensor, benchmark) -> np.ndarray:
    return (
        placement[: benchmark.num_hard_macros].detach().cpu().numpy().astype(np.float64, copy=True)
    )


def _movable_hard_indices(benchmark) -> list[int]:
    fixed = benchmark.macro_fixed[: benchmark.num_hard_macros].detach().cpu().numpy().astype(bool)
    return [idx for idx, is_fixed in enumerate(fixed.tolist()) if not is_fixed]


def _similar_size(lhs: np.ndarray, rhs: np.ndarray, *, max_area_ratio: float) -> bool:
    lhs_area = max(float(lhs[0] * lhs[1]), 1.0e-12)
    rhs_area = max(float(rhs[0] * rhs[1]), 1.0e-12)
    area_ratio = max(lhs_area, rhs_area) / min(lhs_area, rhs_area)
    lhs_aspect = max(float(lhs[0] / max(lhs[1], 1.0e-12)), float(lhs[1] / max(lhs[0], 1.0e-12)))
    rhs_aspect = max(float(rhs[0] / max(rhs[1], 1.0e-12)), float(rhs[1] / max(rhs[0], 1.0e-12)))
    aspect_ratio = max(lhs_aspect, rhs_aspect) / max(min(lhs_aspect, rhs_aspect), 1.0e-12)
    return area_ratio <= max_area_ratio and aspect_ratio <= max_area_ratio


def _densest_bin_center(benchmark, placement: torch.Tensor) -> np.ndarray:
    positions = placement.detach().cpu().numpy().astype(np.float64, copy=False)
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    rows = int(benchmark.grid_rows)
    cols = int(benchmark.grid_cols)
    grid = np.zeros((rows, cols), dtype=np.float64)
    bin_w = float(benchmark.canvas_width) / max(cols, 1)
    bin_h = float(benchmark.canvas_height) / max(rows, 1)
    bin_area = max(bin_w * bin_h, 1.0e-12)

    for center, size in zip(positions, sizes):
        min_x = max(0.0, center[0] - size[0] / 2.0)
        max_x = min(float(benchmark.canvas_width), center[0] + size[0] / 2.0)
        min_y = max(0.0, center[1] - size[1] / 2.0)
        max_y = min(float(benchmark.canvas_height), center[1] + size[1] / 2.0)
        if max_x <= min_x or max_y <= min_y:
            continue
        col0 = max(0, min(int(math.floor(min_x / bin_w)), cols - 1))
        col1 = max(0, min(int(math.floor((max_x - 1.0e-12) / bin_w)), cols - 1))
        row0 = max(0, min(int(math.floor(min_y / bin_h)), rows - 1))
        row1 = max(0, min(int(math.floor((max_y - 1.0e-12) / bin_h)), rows - 1))
        for row in range(row0, row1 + 1):
            cell_min_y = row * bin_h
            cell_max_y = cell_min_y + bin_h
            overlap_y = max(0.0, min(max_y, cell_max_y) - max(min_y, cell_min_y))
            for col in range(col0, col1 + 1):
                cell_min_x = col * bin_w
                cell_max_x = cell_min_x + bin_w
                overlap_x = max(0.0, min(max_x, cell_max_x) - max(min_x, cell_min_x))
                if overlap_x > 0.0 and overlap_y > 0.0:
                    grid[row, col] += (overlap_x * overlap_y) / bin_area

    row, col = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return np.array([(col + 0.5) * bin_w, (row + 0.5) * bin_h], dtype=np.float64)


def _placement_key(placement: torch.Tensor, n_hard: int) -> tuple[float, ...]:
    rounded = torch.round(placement[:n_hard].detach().cpu() * 1000.0) / 1000.0
    return tuple(float(value) for value in rounded.reshape(-1).tolist())


def screen_candidates(
    *,
    benchmark_name: str,
    baseline_placement: torch.Tensor,
    candidates: Sequence[Candidate],
    score_placement: ScorePlacement,
    trace_path: Path | None = None,
) -> BenchmarkSearchResult:
    baseline_metrics = _normalized_metrics(score_placement(baseline_placement))
    baseline_proxy = float(baseline_metrics["proxy_cost"])
    best_name = "baseline"
    best_recipe: dict[str, object] = {"family": "baseline", "benchmark": benchmark_name}
    best_metrics = baseline_metrics

    _write_trace(
        trace_path,
        {
            "benchmark": benchmark_name,
            "candidate": "baseline",
            "family": "baseline",
            "recipe": best_recipe,
            "metrics": baseline_metrics,
            "delta_vs_baseline": 0.0,
        },
    )

    for candidate in candidates:
        metrics = _normalized_metrics(score_placement(candidate.placement))
        proxy = float(metrics["proxy_cost"])
        delta = proxy - baseline_proxy
        legal = int(metrics["overlap_count"]) == 0 and bool(metrics["valid"])
        if legal and proxy < float(best_metrics["proxy_cost"]):
            best_name = candidate.name
            best_recipe = dict(candidate.recipe)
            best_metrics = metrics
        _write_trace(
            trace_path,
            {
                "benchmark": benchmark_name,
                "candidate": candidate.name,
                "family": candidate.family,
                "recipe": candidate.recipe,
                "metrics": metrics,
                "delta_vs_baseline": delta,
                "legal": legal,
            },
        )

    return BenchmarkSearchResult(
        name=benchmark_name,
        baseline_proxy=baseline_proxy,
        best_proxy=float(best_metrics["proxy_cost"]),
        best_name=best_name,
        best_recipe=best_recipe,
        candidate_count=len(candidates),
        improved=best_name != "baseline",
        runtime=0.0,
        overlaps=int(best_metrics["overlap_count"]),
        valid=bool(best_metrics["valid"]),
        wirelength=float(best_metrics["wirelength"]),
        density=float(best_metrics["density"]),
        congestion=float(best_metrics["congestion"]),
    )


def _normalized_metrics(raw: dict[str, object]) -> dict[str, object]:
    return {
        "proxy_cost": float(raw["proxy_cost"]),
        "wirelength": float(raw.get("wirelength", raw.get("wirelength_cost", 0.0))),
        "density": float(raw.get("density", raw.get("density_cost", 0.0))),
        "congestion": float(raw.get("congestion", raw.get("congestion_cost", 0.0))),
        "overlap_count": int(raw.get("overlap_count", raw.get("overlaps", 0))),
        "valid": bool(raw.get("valid", True)),
    }


def _write_trace(trace_path: Path | None, record: dict[str, object]) -> None:
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_benchmark_search(
    *,
    name: str,
    placer,
    config: SearchConfig,
    testcase_root: Path,
    trace_path: Path,
) -> BenchmarkSearchResult:
    start = time.time()
    benchmark, plc = load_benchmark_from_dir(str(testcase_root / name))
    baseline_placement = placer.place(benchmark)
    candidates = generate_candidates(benchmark, baseline_placement, config)

    def score_placement(placement: torch.Tensor) -> dict[str, object]:
        valid, violations = validate_placement(placement, benchmark)
        costs = compute_proxy_cost(placement, benchmark, plc)
        return {
            "proxy_cost": costs["proxy_cost"],
            "wirelength": costs["wirelength_cost"],
            "density": costs["density_cost"],
            "congestion": costs["congestion_cost"],
            "overlap_count": costs["overlap_count"],
            "valid": valid,
            "violations": violations,
        }

    result = screen_candidates(
        benchmark_name=name,
        baseline_placement=baseline_placement,
        candidates=candidates,
        score_placement=score_placement,
        trace_path=trace_path,
    )
    return replace(result, runtime=time.time() - start)


def write_summary(
    *,
    run_id: str,
    placer_path: Path,
    command: Iterable[str],
    config: SearchConfig,
    results: Sequence[BenchmarkSearchResult],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    average_proxy = sum(result.best_proxy for result in results) / len(results)
    total_overlaps = sum(result.overlaps for result in results)
    total_runtime = sum(result.runtime for result in results)
    max_runtime = max(result.runtime for result in results)
    benchmark_rows = []
    for result in results:
        row = asdict(result)
        row["proxy_cost"] = float(result.best_proxy)
        benchmark_rows.append(row)
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "placer_path": str(placer_path),
        "command": " ".join(command),
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "upstream_commit": _git_output(["rev-parse", "upstream/main"]),
            "dirty": bool(_git_output(["status", "--porcelain"])),
        },
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
        "search_config": asdict(config),
        "benchmarks": benchmark_rows,
        "aggregate": {
            "average_proxy": float(average_proxy),
            "total_overlaps": int(total_overlaps),
            "total_runtime": float(total_runtime),
            "max_runtime": float(max_runtime),
            "improved_count": sum(1 for result in results if result.improved),
            "candidate_count": sum(result.candidate_count for result in results),
        },
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _select_benchmarks(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(IBM_BENCHMARKS)
    if args.benchmarks:
        return list(_parse_csv(args.benchmarks))
    return list(WEAK_IBM_BENCHMARKS)


def run(args: argparse.Namespace) -> Path:
    config = SearchConfig(
        families=_normalize_families(_parse_csv(args.families)),
        step_fractions=_parse_float_csv(args.step_fractions),
        max_candidates_per_benchmark=int(args.max_candidates_per_benchmark),
        max_candidates_per_family=args.max_candidates_per_family,
        legal_gap=float(args.legal_gap),
        swap_area_ratio=float(args.swap_area_ratio),
    )
    placer_path = Path(args.placer)
    placer = _load_placer(placer_path)
    output_dir = Path(args.output_root) / args.run_id
    trace_path = output_dir / "candidate_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    results: list[BenchmarkSearchResult] = []
    for name in _select_benchmarks(args):
        print(f"{name}...", flush=True)
        result = run_benchmark_search(
            name=name,
            placer=placer,
            config=config,
            testcase_root=Path(args.testcase_root),
            trace_path=trace_path,
        )
        print(
            f"  baseline={result.baseline_proxy:.4f} best={result.best_proxy:.4f} "
            f"candidate={result.best_name} count={result.candidate_count} "
            f"overlaps={result.overlaps} runtime={result.runtime:.2f}s",
            flush=True,
        )
        results.append(result)

    summary_path = write_summary(
        run_id=args.run_id,
        placer_path=placer_path,
        command=sys.argv,
        config=config,
        results=results,
        output_dir=output_dir,
    )
    average_proxy = sum(result.best_proxy for result in results) / len(results)
    print(f"summary written: {summary_path}")
    print(f"candidate trace written: {trace_path}")
    print(f"average best proxy: {average_proxy:.4f}")
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Search output directory under results/.")
    parser.add_argument(
        "--placer",
        default="submissions/jaydenpiao/placer.py",
        help="Placer entry point used to create baseline placements.",
    )
    parser.add_argument(
        "--benchmarks", help="Comma-separated benchmark names. Defaults to weak IBM."
    )
    parser.add_argument("--all", action="store_true", help="Run all 17 IBM benchmarks.")
    parser.add_argument(
        "--families",
        default="single,swap,density,transform",
        help="Comma-separated candidate families: single,swap,density,transform.",
    )
    parser.add_argument(
        "--step-fractions",
        default="0.02,0.05,0.1",
        help="Comma-separated canvas-relative move step fractions.",
    )
    parser.add_argument("--max-candidates-per-benchmark", type=int, default=128)
    parser.add_argument(
        "--max-candidates-per-family",
        type=int,
        help="Optional cap for each candidate family before moving to the next family.",
    )
    parser.add_argument("--legal-gap", type=float, default=0.01)
    parser.add_argument("--swap-area-ratio", type=float, default=1.5)
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--testcase-root",
        default="external/MacroPlacement/Testcases/ICCAD04",
        help="Root containing IBM benchmark directories.",
    )
    try:
        run(parser.parse_args(argv))
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"candidate search error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
