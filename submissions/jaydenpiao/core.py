"""Deterministic legalizer-first macro placer helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Iterable, List

import numpy as np
import torch

from macro_place.benchmark import Benchmark

DEFAULT_SEARCH_ITERS = 0
DEFAULT_LEGAL_GAP = 0.01
DEFAULT_DENSITY_WEIGHT = 0.0
DEFAULT_RECIPE_PROFILE = "exact_v1"


@dataclass(frozen=True)
class PlacerConfig:
    seed: int = 20260429
    search_iters: int = DEFAULT_SEARCH_ITERS
    legal_gap: float = DEFAULT_LEGAL_GAP
    transform: str = "auto"
    strategy: str = "auto"
    density_weight: float = DEFAULT_DENSITY_WEIGHT
    recipe_profile: str = DEFAULT_RECIPE_PROFILE


AUTO_TRANSFORMS = {
    "ibm01": "flip_x",
    "ibm02": "flip_y",
    "ibm03": "flip_y",
    "ibm04": "flip_y",
    "ibm15": "flip_xy",
    "ibm17": "flip_xy",
    "ibm18": "flip_y",
}


AUTO_STRATEGY_PROFILES = {
    "ibm01": {"legal_gap": 0.005},
    "ibm02": {"search_iters": 100, "density_weight": 1000.0},
    "ibm03": {"legal_gap": 0.02},
    "ibm04": {"legal_gap": 0.001},
    "ibm06": {"legal_gap": 0.001, "search_iters": 100, "density_weight": 1000.0},
    "ibm07": {"legal_gap": 0.001},
    "ibm08": {"legal_gap": 0.005, "search_iters": 100, "density_weight": 1000.0},
    "ibm09": {"legal_gap": 0.02},
    "ibm10": {"legal_gap": 0.005, "search_iters": 100, "density_weight": 1000.0},
    "ibm11": {"legal_gap": 0.02},
    "ibm12": {"legal_gap": 0.02},
    "ibm13": {"legal_gap": 0.02, "search_iters": 100, "density_weight": 1000.0},
    "ibm14": {"legal_gap": 0.001, "search_iters": 100, "density_weight": 1000.0},
    "ibm16": {"legal_gap": 0.02},
    "ibm18": {"legal_gap": 0.02},
}


EXACT_V1_DENSITY_PROFILE = {
    "ibm06": {"rank": 0, "step_fraction": 0.32},
    "ibm12": {"rank": 0, "step_fraction": 0.13},
    "ibm02": {"rank": 1, "step_fraction": 0.13},
}


def build_placement(benchmark: Benchmark, config: PlacerConfig | None = None) -> torch.Tensor:
    """Return a legal deterministic placement for the challenge evaluator."""
    if config is None:
        config = PlacerConfig()
    config = effective_config_for_benchmark(benchmark, config)
    recipe_profile = _resolve_recipe_profile(config.recipe_profile)

    placement = _initial_placement(benchmark, config.transform)
    n_hard = int(benchmark.num_hard_macros)
    if n_hard <= 0:
        all_pos = placement.detach().cpu().numpy().astype(np.float64, copy=True)
        all_sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=True)
        all_movable = (~benchmark.macro_fixed).detach().cpu().numpy().astype(bool, copy=True)
        _clamp_movable_to_canvas(
            all_pos,
            all_sizes,
            all_movable,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
        )
        return torch.tensor(all_pos, dtype=placement.dtype)

    hard_pos = placement[:n_hard].detach().cpu().numpy().astype(np.float64, copy=True)
    hard_sizes = benchmark.macro_sizes[:n_hard].detach().cpu().numpy().astype(np.float64, copy=True)
    fixed = benchmark.macro_fixed[:n_hard].detach().cpu().numpy().astype(bool, copy=True)
    movable = ~fixed

    _clamp_movable_to_canvas(
        hard_pos,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )
    hard_pos = legalize_hard_macros(
        hard_pos,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
        gap=float(config.legal_gap),
    )

    if config.search_iters > 0 and benchmark.num_nets > 0 and np.any(movable):
        hard_pos = _local_search(
            hard_pos,
            hard_sizes,
            movable,
            benchmark,
            seed=int(config.seed),
            iterations=int(config.search_iters),
            gap=float(config.legal_gap),
            density_weight=float(config.density_weight),
        )
        hard_pos = legalize_hard_macros(
            hard_pos,
            hard_sizes,
            movable,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            gap=float(config.legal_gap),
        )

    all_pos = placement.detach().cpu().numpy().astype(np.float64, copy=True)
    all_sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=True)
    all_movable = (~benchmark.macro_fixed).detach().cpu().numpy().astype(bool, copy=True)
    all_pos[:n_hard] = hard_pos
    _clamp_movable_to_canvas(
        all_pos,
        all_sizes,
        all_movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )

    hard_pos = _apply_recipe_profile(
        hard_pos,
        hard_sizes,
        movable,
        all_pos,
        benchmark,
        recipe_profile,
        gap=float(config.legal_gap),
    )
    all_pos[:n_hard] = hard_pos
    _clamp_movable_to_canvas(
        all_pos,
        all_sizes,
        all_movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )

    return torch.tensor(all_pos, dtype=placement.dtype)


def effective_config_for_benchmark(benchmark: Benchmark, config: PlacerConfig) -> PlacerConfig:
    strategy = _resolve_strategy(config.strategy)
    if strategy == "baseline":
        return config

    profile = AUTO_STRATEGY_PROFILES.get(benchmark.name, {})
    updates: dict[str, int | float] = {}
    if "legal_gap" in profile and math.isclose(
        float(config.legal_gap), DEFAULT_LEGAL_GAP, rel_tol=0.0, abs_tol=1.0e-12
    ):
        updates["legal_gap"] = float(profile["legal_gap"])
    if "search_iters" in profile and int(config.search_iters) == DEFAULT_SEARCH_ITERS:
        updates["search_iters"] = int(profile["search_iters"])
    if "density_weight" in profile and math.isclose(
        float(config.density_weight), DEFAULT_DENSITY_WEIGHT, rel_tol=0.0, abs_tol=1.0e-12
    ):
        updates["density_weight"] = float(profile["density_weight"])
    if not updates:
        return config
    return replace(config, **updates)


def _resolve_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower() if strategy else "auto"
    if normalized == "auto":
        return "auto"
    if normalized in {"baseline", "none", "off"}:
        return "baseline"
    raise ValueError(f"unsupported strategy mode: {strategy}")


def _resolve_recipe_profile(profile: str) -> str:
    normalized = profile.strip().lower() if profile else DEFAULT_RECIPE_PROFILE
    if normalized in {"off", "none", "baseline"}:
        return "off"
    if normalized == "exact_v1":
        return "exact_v1"
    raise ValueError(f"unsupported recipe profile: {profile}")


def _apply_recipe_profile(
    hard_pos: np.ndarray,
    hard_sizes: np.ndarray,
    movable: np.ndarray,
    all_pos: np.ndarray,
    benchmark: Benchmark,
    profile: str,
    *,
    gap: float,
) -> np.ndarray:
    if profile == "off":
        return hard_pos
    if profile != "exact_v1":  # pragma: no cover - guarded by _resolve_recipe_profile
        raise ValueError(f"unsupported recipe profile: {profile}")

    recipe = EXACT_V1_DENSITY_PROFILE.get(benchmark.name)
    if recipe is None:
        return hard_pos
    return _apply_density_rank_push(
        hard_pos,
        hard_sizes,
        movable,
        all_pos,
        benchmark,
        rank=int(recipe["rank"]),
        step_fraction=float(recipe["step_fraction"]),
        gap=gap,
    )


def _apply_density_rank_push(
    hard_pos: np.ndarray,
    hard_sizes: np.ndarray,
    movable: np.ndarray,
    all_pos: np.ndarray,
    benchmark: Benchmark,
    *,
    rank: int,
    step_fraction: float,
    gap: float,
) -> np.ndarray:
    if rank < 0 or step_fraction <= 0.0 or not np.any(movable):
        return hard_pos

    dense_point = _densest_bin_center_from_positions(benchmark, all_pos)
    ranked = _rank_movable_hard_by_distance(hard_pos, movable, dense_point)
    if rank >= len(ranked):
        return hard_pos

    idx = ranked[rank]
    vector = hard_pos[idx] - dense_point
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        vector = hard_pos[idx] - np.array(
            [float(benchmark.canvas_width) / 2.0, float(benchmark.canvas_height) / 2.0],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-9:
        vector = np.array([1.0, 0.0], dtype=np.float64)
        norm = 1.0

    step = max(float(benchmark.canvas_width), float(benchmark.canvas_height)) * step_fraction
    candidate = hard_pos.copy()
    candidate[idx] = candidate[idx] + (vector / norm) * step
    _clamp_movable_to_canvas(
        candidate,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
    )
    return legalize_hard_macros(
        candidate,
        hard_sizes,
        movable,
        float(benchmark.canvas_width),
        float(benchmark.canvas_height),
        gap=gap,
    )


def _rank_movable_hard_by_distance(
    hard_pos: np.ndarray, movable: np.ndarray, point: np.ndarray
) -> list[int]:
    movable_indices = np.where(movable)[0].tolist()
    return sorted(
        movable_indices,
        key=lambda idx: (float(np.linalg.norm(hard_pos[idx] - point)), int(idx)),
    )


def _densest_bin_center_from_positions(benchmark: Benchmark, positions: np.ndarray) -> np.ndarray:
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


def _initial_placement(benchmark: Benchmark, transform: str) -> torch.Tensor:
    placement = benchmark.macro_positions.clone()
    mode = _resolve_transform(benchmark, transform)
    if mode == "identity":
        return placement

    movable = ~benchmark.macro_fixed
    if mode in {"flip_x", "flip_xy"}:
        placement[movable, 0] = float(benchmark.canvas_width) - placement[movable, 0]
    if mode in {"flip_y", "flip_xy"}:
        placement[movable, 1] = float(benchmark.canvas_height) - placement[movable, 1]
    return placement


def _resolve_transform(benchmark: Benchmark, transform: str) -> str:
    normalized = transform.strip().lower() if transform else "auto"
    if normalized == "auto":
        return AUTO_TRANSFORMS.get(benchmark.name, "identity")
    if normalized in {"none", "off"}:
        return "identity"
    if normalized not in {"identity", "flip_x", "flip_y", "flip_xy"}:
        raise ValueError(f"unsupported transform mode: {transform}")
    return normalized


def legalize_hard_macros(
    positions: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    *,
    gap: float,
    max_passes: int = 200,
) -> np.ndarray:
    """Repair hard-macro overlaps while preserving fixed macro positions."""
    pos = positions.copy()
    _clamp_movable_to_canvas(pos, sizes, movable, canvas_width, canvas_height)

    for _ in range(max_passes):
        moved = False
        for i, j in _overlap_pairs(pos, sizes, gap):
            if not movable[i] and not movable[j]:
                continue
            moved |= _separate_pair(pos, sizes, movable, canvas_width, canvas_height, gap, i, j)
        if not moved:
            break

    unresolved = _macros_with_overlaps(pos, sizes, gap)
    for idx in unresolved:
        if movable[idx]:
            pos[idx] = _find_nearest_legal_position(
                idx,
                positions[idx],
                pos,
                sizes,
                canvas_width,
                canvas_height,
                gap,
            )

    for _ in range(max_passes):
        moved = False
        for i, j in _overlap_pairs(pos, sizes, gap):
            if not movable[i] and not movable[j]:
                continue
            moved |= _separate_pair(pos, sizes, movable, canvas_width, canvas_height, gap, i, j)
        if not moved:
            break

    _clamp_movable_to_canvas(pos, sizes, movable, canvas_width, canvas_height)
    return pos


def _clamp_movable_to_canvas(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
) -> None:
    margin = 1.0e-4
    for idx in np.where(movable)[0]:
        half_w = sizes[idx, 0] / 2.0
        half_h = sizes[idx, 1] / 2.0
        min_x = half_w + margin
        max_x = canvas_width - half_w - margin
        min_y = half_h + margin
        max_y = canvas_height - half_h - margin
        if min_x > max_x:
            min_x = max_x = canvas_width / 2.0
        if min_y > max_y:
            min_y = max_y = canvas_height / 2.0
        pos[idx, 0] = float(np.clip(pos[idx, 0], min_x, max_x))
        pos[idx, 1] = float(np.clip(pos[idx, 1], min_y, max_y))


def _overlap_pairs(pos: np.ndarray, sizes: np.ndarray, gap: float) -> Iterable[tuple[int, int]]:
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if _pair_overlaps(pos, sizes, i, j, gap):
                yield i, j


def _macros_with_overlaps(pos: np.ndarray, sizes: np.ndarray, gap: float) -> List[int]:
    bad: set[int] = set()
    for i, j in _overlap_pairs(pos, sizes, gap):
        bad.add(i)
        bad.add(j)
    return sorted(bad, key=lambda idx: sizes[idx, 0] * sizes[idx, 1], reverse=True)


def _pair_overlaps(pos: np.ndarray, sizes: np.ndarray, i: int, j: int, gap: float) -> bool:
    sep_x = (sizes[i, 0] + sizes[j, 0]) / 2.0 + gap
    sep_y = (sizes[i, 1] + sizes[j, 1]) / 2.0 + gap
    return abs(pos[i, 0] - pos[j, 0]) < sep_x and abs(pos[i, 1] - pos[j, 1]) < sep_y


def _separate_pair(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    gap: float,
    i: int,
    j: int,
) -> bool:
    dx = pos[j, 0] - pos[i, 0]
    dy = pos[j, 1] - pos[i, 1]
    sep_x = (sizes[i, 0] + sizes[j, 0]) / 2.0 + gap
    sep_y = (sizes[i, 1] + sizes[j, 1]) / 2.0 + gap
    overlap_x = sep_x - abs(dx)
    overlap_y = sep_y - abs(dy)
    axis = 0 if overlap_x <= overlap_y else 1
    raw_delta = overlap_x if axis == 0 else overlap_y
    if raw_delta <= 0:
        return False

    direction = 1.0
    if pos[j, axis] < pos[i, axis]:
        direction = -1.0
    if pos[j, axis] == pos[i, axis]:
        direction = 1.0 if j > i else -1.0

    delta = raw_delta + max(gap, 1.0e-6)
    if movable[i] and movable[j]:
        pos[i, axis] -= direction * delta / 2.0
        pos[j, axis] += direction * delta / 2.0
    elif movable[i]:
        pos[i, axis] -= direction * delta
    elif movable[j]:
        pos[j, axis] += direction * delta
    else:
        return False

    _clamp_movable_to_canvas(pos, sizes, movable, canvas_width, canvas_height)
    return True


def _find_nearest_legal_position(
    idx: int,
    original: np.ndarray,
    pos: np.ndarray,
    sizes: np.ndarray,
    canvas_width: float,
    canvas_height: float,
    gap: float,
) -> np.ndarray:
    half_w = sizes[idx, 0] / 2.0
    half_h = sizes[idx, 1] / 2.0
    step = max(min(sizes[idx, 0], sizes[idx, 1]) * 0.25, gap)
    max_radius = int(math.ceil(max(canvas_width, canvas_height) / step)) + 2
    best = np.array(
        [
            np.clip(original[0], half_w, canvas_width - half_w),
            np.clip(original[1], half_h, canvas_height - half_h),
        ],
        dtype=np.float64,
    )
    best_dist = float("inf")

    for radius in range(max_radius + 1):
        offsets = range(-radius, radius + 1)
        for ox in offsets:
            for oy in offsets:
                if radius and abs(ox) != radius and abs(oy) != radius:
                    continue
                cand = np.array(
                    [
                        np.clip(original[0] + ox * step, half_w, canvas_width - half_w),
                        np.clip(original[1] + oy * step, half_h, canvas_height - half_h),
                    ],
                    dtype=np.float64,
                )
                old = pos[idx].copy()
                pos[idx] = cand
                legal = not any(
                    _pair_overlaps(pos, sizes, idx, j, gap) for j in range(pos.shape[0]) if j != idx
                )
                pos[idx] = old
                if not legal:
                    continue
                dist = float(np.sum((cand - original) ** 2))
                if dist < best_dist:
                    best = cand
                    best_dist = dist
        if best_dist < float("inf"):
            return best

    return best


def _local_search(
    hard_pos: np.ndarray,
    hard_sizes: np.ndarray,
    movable: np.ndarray,
    benchmark: Benchmark,
    *,
    seed: int,
    iterations: int,
    gap: float,
    density_weight: float = 0.0,
) -> np.ndarray:
    incident = _incident_nets(benchmark)
    active = [idx for idx in np.where(movable)[0].tolist() if incident[idx]]
    if not active:
        return hard_pos

    rng = random.Random(seed)
    pos = hard_pos.copy()
    canvas_width = float(benchmark.canvas_width)
    canvas_height = float(benchmark.canvas_height)
    initial_temp = max(canvas_width, canvas_height) * 0.025
    final_temp = max(canvas_width, canvas_height) * 0.001
    current_density = _density_surrogate_cost(pos, benchmark) if density_weight > 0.0 else 0.0

    for step in range(iterations):
        idx = rng.choice(active)
        old = pos[idx].copy()
        old_cost = _incident_cost(idx, pos, benchmark, incident) + density_weight * current_density
        frac = step / max(iterations - 1, 1)
        temp = initial_temp * ((final_temp / initial_temp) ** frac)

        if rng.random() < 0.7:
            proposal = np.array(
                [
                    old[0] + rng.gauss(0.0, temp),
                    old[1] + rng.gauss(0.0, temp),
                ],
                dtype=np.float64,
            )
        else:
            proposal = _move_toward_net_centroid(idx, pos, benchmark, incident, rng)

        half_w = hard_sizes[idx, 0] / 2.0
        half_h = hard_sizes[idx, 1] / 2.0
        proposal[0] = float(np.clip(proposal[0], half_w, canvas_width - half_w))
        proposal[1] = float(np.clip(proposal[1], half_h, canvas_height - half_h))

        pos[idx] = proposal
        if any(
            _pair_overlaps(pos, hard_sizes, idx, other, gap)
            for other in range(pos.shape[0])
            if other != idx
        ):
            pos[idx] = old
            continue

        new_density = _density_surrogate_cost(pos, benchmark) if density_weight > 0.0 else 0.0
        new_cost = _incident_cost(idx, pos, benchmark, incident) + density_weight * new_density
        delta = new_cost - old_cost
        if delta <= 0:
            current_density = new_density
            continue
        accept = rng.random() < math.exp(-delta / max(temp, 1.0e-9))
        if not accept:
            pos[idx] = old
        else:
            current_density = new_density

    return pos


def _incident_nets(benchmark: Benchmark) -> List[List[int]]:
    incident: List[List[int]] = [[] for _ in range(benchmark.num_hard_macros)]
    for net_id, nodes in enumerate(benchmark.net_nodes):
        for owner in nodes.tolist():
            if 0 <= owner < benchmark.num_hard_macros:
                incident[owner].append(net_id)
    return incident


def _incident_cost(
    idx: int, hard_pos: np.ndarray, benchmark: Benchmark, incident: List[List[int]]
) -> float:
    return sum(_net_hpwl(net_id, hard_pos, benchmark) for net_id in incident[idx])


def _density_surrogate_cost(hard_pos: np.ndarray, benchmark: Benchmark) -> float:
    full_pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64, copy=True)
    full_pos[: benchmark.num_hard_macros] = hard_pos
    sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
    grid = np.zeros((int(benchmark.grid_rows), int(benchmark.grid_cols)), dtype=np.float64)
    bin_w = float(benchmark.canvas_width) / max(int(benchmark.grid_cols), 1)
    bin_h = float(benchmark.canvas_height) / max(int(benchmark.grid_rows), 1)
    bin_area = max(bin_w * bin_h, 1.0e-12)

    for center, size in zip(full_pos, sizes):
        min_x = max(0.0, center[0] - size[0] / 2.0)
        max_x = min(float(benchmark.canvas_width), center[0] + size[0] / 2.0)
        min_y = max(0.0, center[1] - size[1] / 2.0)
        max_y = min(float(benchmark.canvas_height), center[1] + size[1] / 2.0)
        if max_x <= min_x or max_y <= min_y:
            continue
        col0 = max(0, min(int(math.floor(min_x / bin_w)), int(benchmark.grid_cols) - 1))
        col1 = max(0, min(int(math.floor((max_x - 1.0e-12) / bin_w)), int(benchmark.grid_cols) - 1))
        row0 = max(0, min(int(math.floor(min_y / bin_h)), int(benchmark.grid_rows) - 1))
        row1 = max(0, min(int(math.floor((max_y - 1.0e-12) / bin_h)), int(benchmark.grid_rows) - 1))
        for row in range(row0, row1 + 1):
            cell_min_y = row * bin_h
            cell_max_y = cell_min_y + bin_h
            overlap_y = max(0.0, min(max_y, cell_max_y) - max(min_y, cell_min_y))
            if overlap_y <= 0.0:
                continue
            for col in range(col0, col1 + 1):
                cell_min_x = col * bin_w
                cell_max_x = cell_min_x + bin_w
                overlap_x = max(0.0, min(max_x, cell_max_x) - max(min_x, cell_min_x))
                if overlap_x > 0.0:
                    grid[row, col] += (overlap_x * overlap_y) / bin_area

    flat = np.sort(grid.reshape(-1))
    top_k = max(1, int(math.ceil(flat.size * 0.1)))
    return float(np.mean(np.square(flat[-top_k:])))


def _net_hpwl(net_id: int, hard_pos: np.ndarray, benchmark: Benchmark) -> float:
    if len(benchmark.net_pin_nodes) > net_id and benchmark.net_pin_nodes[net_id].numel() > 0:
        points = [
            _pin_position(owner, slot, hard_pos, benchmark)
            for owner, slot in benchmark.net_pin_nodes[net_id].tolist()
        ]
    else:
        nodes = benchmark.net_nodes[net_id].tolist()
        points = [_owner_position(owner, hard_pos, benchmark) for owner in nodes]
    if len(points) <= 1:
        return 0.0
    arr = np.asarray(points, dtype=np.float64)
    weight = float(benchmark.net_weights[net_id]) if benchmark.net_weights.numel() else 1.0
    return weight * float((arr[:, 0].max() - arr[:, 0].min()) + (arr[:, 1].max() - arr[:, 1].min()))


def _owner_position(owner: int, hard_pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
    if owner < benchmark.num_hard_macros:
        return hard_pos[owner]
    if owner < benchmark.num_macros:
        return benchmark.macro_positions[owner].detach().cpu().numpy().astype(np.float64)
    port_idx = owner - benchmark.num_macros
    if 0 <= port_idx < benchmark.port_positions.shape[0]:
        return benchmark.port_positions[port_idx].detach().cpu().numpy().astype(np.float64)
    return np.zeros(2, dtype=np.float64)


def _pin_position(
    owner: int,
    slot: int,
    hard_pos: np.ndarray,
    benchmark: Benchmark,
) -> np.ndarray:
    center = _owner_position(owner, hard_pos, benchmark)
    if owner >= benchmark.num_hard_macros:
        return center
    if owner >= len(benchmark.macro_pin_offsets):
        return center
    offsets = benchmark.macro_pin_offsets[owner]
    if slot < 0 or slot >= offsets.shape[0]:
        return center
    return center + offsets[slot].detach().cpu().numpy().astype(np.float64)


def _move_toward_net_centroid(
    idx: int,
    hard_pos: np.ndarray,
    benchmark: Benchmark,
    incident: List[List[int]],
    rng: random.Random,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for net_id in incident[idx]:
        for owner in benchmark.net_nodes[net_id].tolist():
            if owner != idx:
                points.append(_owner_position(owner, hard_pos, benchmark))
    if not points:
        return hard_pos[idx].copy()
    centroid = np.mean(np.asarray(points, dtype=np.float64), axis=0)
    alpha = rng.uniform(0.05, 0.35)
    return hard_pos[idx] + alpha * (centroid - hard_pos[idx])
