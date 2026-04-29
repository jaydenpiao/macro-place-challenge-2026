"""Deterministic legalizer-first macro placer helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch

from macro_place.benchmark import Benchmark


@dataclass(frozen=True)
class PlacerConfig:
    seed: int = 20260429
    search_iters: int = 4000
    legal_gap: float = 0.01


def build_placement(benchmark: Benchmark, config: PlacerConfig | None = None) -> torch.Tensor:
    """Return a legal deterministic placement for the challenge evaluator."""
    if config is None:
        config = PlacerConfig()

    placement = benchmark.macro_positions.clone()
    n_hard = int(benchmark.num_hard_macros)
    if n_hard <= 0:
        return placement

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
        )
        hard_pos = legalize_hard_macros(
            hard_pos,
            hard_sizes,
            movable,
            float(benchmark.canvas_width),
            float(benchmark.canvas_height),
            gap=float(config.legal_gap),
        )

    placement[:n_hard] = torch.tensor(hard_pos, dtype=placement.dtype)
    return placement


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

    _clamp_movable_to_canvas(pos, sizes, movable, canvas_width, canvas_height)
    return pos


def _clamp_movable_to_canvas(
    pos: np.ndarray,
    sizes: np.ndarray,
    movable: np.ndarray,
    canvas_width: float,
    canvas_height: float,
) -> None:
    for idx in np.where(movable)[0]:
        half_w = sizes[idx, 0] / 2.0
        half_h = sizes[idx, 1] / 2.0
        pos[idx, 0] = float(np.clip(pos[idx, 0], half_w, canvas_width - half_w))
        pos[idx, 1] = float(np.clip(pos[idx, 1], half_h, canvas_height - half_h))


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

    for step in range(iterations):
        idx = rng.choice(active)
        old = pos[idx].copy()
        old_cost = _incident_cost(idx, pos, benchmark, incident)
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

        new_cost = _incident_cost(idx, pos, benchmark, incident)
        delta = new_cost - old_cost
        if delta <= 0:
            continue
        accept = rng.random() < math.exp(-delta / max(temp, 1.0e-9))
        if not accept:
            pos[idx] = old

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


def _net_hpwl(net_id: int, hard_pos: np.ndarray, benchmark: Benchmark) -> float:
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
