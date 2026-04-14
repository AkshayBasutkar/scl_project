"""
Baseline algorithms for comparison.

1. Dijkstra's shortest path (multi-objective composite cost)
2. CPU-only ACO (single-objective, basic pheromone logic)
3. Simple GPU ACO baseline (fixed weights, no adaptation)
"""

import heapq
import numpy as np
import time
import logging
from typing import List, Optional, Dict, Tuple

from config import ACOConfig
from graph_utils import CSRGraph
from cuda_kernels import cpu_construct_routes, cpu_evaporate_pheromones, cpu_deposit_pheromones

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Result container (shared with aco.py format)
# ──────────────────────────────────────────────────────────────

class BaselineResult:
    """Stores baseline algorithm output."""
    def __init__(self):
        self.best_route: List[int] = []
        self.best_cost: float = float("inf")
        self.best_travel_time: float = 0.0
        self.best_distance: float = 0.0
        self.runtime_ms: float = 0.0
        self.convergence_history: List[float] = []
        self.iterations_run: int = 0
        self.mode: str = "cpu"
        self.algorithm: str = ""

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "best_route": self.best_route,
            "best_cost": round(self.best_cost, 6),
            "best_travel_time": round(self.best_travel_time, 4),
            "best_distance": round(self.best_distance, 4),
            "runtime_ms": round(self.runtime_ms, 2),
            "iterations_run": self.iterations_run,
            "mode": self.mode,
            "path_length": len(self.best_route),
        }


# ══════════════════════════════════════════════════════════════
#  1. Dijkstra (multi-objective composite cost)
# ══════════════════════════════════════════════════════════════

def dijkstra(
    graph: CSRGraph,
    source: int,
    destination: int,
    w_tt: float = 0.25,
    w_dist: float = 0.20,
    w_cong: float = 0.20,
    w_sig: float = 0.15,
    w_cap: float = 0.10,
) -> BaselineResult:
    """
    Dijkstra's algorithm using the same composite cost function
    as the ACO, so results are directly comparable.
    """
    result = BaselineResult()
    result.algorithm = "Dijkstra"

    t0 = time.perf_counter()

    INF = float("inf")
    dist = np.full(graph.num_nodes, INF)
    prev = np.full(graph.num_nodes, -1, dtype=np.int32)
    dist[source] = 0.0

    # Min-heap: (cost, node)
    heap = [(0.0, source)]
    visited = set()

    while heap:
        d_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        if u == destination:
            break

        start = graph.row_ptr[u]
        end = graph.row_ptr[u + 1]
        for k in range(start, end):
            v = graph.col_idx[k]
            if v in visited:
                continue
            cap_val = max(float(graph.capacities[k]), 1.0)
            edge_cost = (
                w_tt * graph.travel_times[k]
                + w_dist * graph.distances[k]
                + w_cong * graph.congestions[k]
                + w_sig * graph.signal_delays[k]
                + w_cap * (1000.0 / cap_val)
            )
            alt = d_u + edge_cost
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    t_elapsed = (time.perf_counter() - t0) * 1000.0

    # Reconstruct path
    if dist[destination] < INF:
        path = []
        node = destination
        while node != -1:
            path.append(int(node))
            node = prev[node]
        path.reverse()
        result.best_route = path
        result.best_cost = dist[destination]
        result.best_travel_time = _sum_attr(graph, path, graph.travel_times)
        result.best_distance = _sum_attr(graph, path, graph.distances)

    result.runtime_ms = t_elapsed
    logger.info(
        f"Dijkstra: cost={result.best_cost:.4f}, "
        f"tt={result.best_travel_time:.2f}, "
        f"dist={result.best_distance:.2f}, "
        f"time={result.runtime_ms:.2f} ms"
    )
    return result


# ══════════════════════════════════════════════════════════════
#  2. CPU ACO (basic, single-objective-ish, no adaptation)
# ══════════════════════════════════════════════════════════════

def cpu_aco(
    graph: CSRGraph,
    source: int,
    destination: int,
    cfg: Optional[ACOConfig] = None,
    num_ants: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> BaselineResult:
    """
    Vanilla CPU-based ACO without adaptive behaviour or
    dynamic traffic, for fair baseline comparison.
    """
    if cfg is None:
        cfg = ACOConfig()
    num_ants = num_ants or cfg.num_ants
    max_iterations = max_iterations or cfg.max_iterations
    rng = np.random.RandomState(cfg.random_seed)

    result = BaselineResult()
    result.algorithm = "CPU-ACO"
    result.mode = "cpu"

    # Reset pheromones
    graph.reset_pheromones(cfg.pheromone_init)

    global_best_route = None
    global_best_cost = float("inf")
    convergence_history = []

    t0 = time.perf_counter()

    for iteration in range(max_iterations):
        routes, route_lengths, route_costs = cpu_construct_routes(
            graph.row_ptr, graph.col_idx,
            graph.travel_times, graph.distances,
            graph.congestions, graph.signal_delays, graph.capacities,
            graph.pheromones,
            cfg.alpha, cfg.beta,
            cfg.w_travel_time, cfg.w_distance,
            cfg.w_congestion, cfg.w_signal_delay, cfg.w_capacity,
            num_ants, graph.num_nodes, source, destination,
            cfg.max_path_length, rng,
        )

        # Find iteration best
        valid = route_costs < 1e9
        if valid.any():
            idx = np.argmin(route_costs)
            if route_costs[idx] < global_best_cost:
                global_best_cost = route_costs[idx]
                plen = route_lengths[idx]
                global_best_route = routes[idx, :plen].copy()
                result.convergence_iteration = iteration

        convergence_history.append(global_best_cost)

        # Evaporate + deposit (basic)
        cpu_evaporate_pheromones(
            graph.pheromones, cfg.evaporation_rate,
            cfg.pheromone_min, cfg.pheromone_max,
        )
        cpu_deposit_pheromones(
            graph.row_ptr, graph.col_idx,
            routes, route_lengths, route_costs,
            graph.pheromones, cfg.q_deposit, cfg.pheromone_max,
            num_ants,
        )

    t_elapsed = (time.perf_counter() - t0) * 1000.0

    result.best_cost = global_best_cost
    result.convergence_history = convergence_history
    result.runtime_ms = t_elapsed
    result.iterations_run = len(convergence_history)

    if global_best_route is not None:
        result.best_route = global_best_route.tolist()
        result.best_travel_time = _sum_attr(graph, result.best_route, graph.travel_times)
        result.best_distance = _sum_attr(graph, result.best_route, graph.distances)

    logger.info(
        f"CPU-ACO: cost={result.best_cost:.4f}, "
        f"tt={result.best_travel_time:.2f}, "
        f"dist={result.best_distance:.2f}, "
        f"time={result.runtime_ms:.2f} ms, "
        f"iters={result.iterations_run}"
    )
    return result


# ══════════════════════════════════════════════════════════════
#  3. Simple GPU ACO baseline (no adaptation, no dynamic traffic)
# ══════════════════════════════════════════════════════════════

def gpu_aco_baseline(
    graph: CSRGraph,
    source: int,
    destination: int,
    cfg: Optional[ACOConfig] = None,
    num_ants: Optional[int] = None,
    max_iterations: Optional[int] = None,
) -> BaselineResult:
    """
    GPU-based ACO *without* adaptive evaporation or dynamic
    traffic — isolates the CUDA speedup from algorithmic improvements.
    """
    from cuda_kernels import CUDAEngine

    if cfg is None:
        cfg = ACOConfig()
    num_ants = num_ants or cfg.num_ants
    max_iterations = max_iterations or cfg.max_iterations
    rng = np.random.RandomState(cfg.random_seed)

    engine = CUDAEngine(use_cuda=cfg.use_cuda, block_size=cfg.cuda_block_size)
    graph.reset_pheromones(cfg.pheromone_init)
    engine.upload_graph(graph)

    result = BaselineResult()
    result.algorithm = "GPU-ACO-Baseline"
    result.mode = "cuda" if engine.use_cuda else "cpu"

    global_best_route = None
    global_best_cost = float("inf")
    convergence_history = []

    t0 = time.perf_counter()

    for iteration in range(max_iterations):
        routes, route_lengths, route_costs = engine.construct_routes(
            graph, num_ants, source, destination,
            cfg.max_path_length,
            cfg.alpha, cfg.beta,
            cfg.w_travel_time, cfg.w_distance,
            cfg.w_congestion, cfg.w_signal_delay, cfg.w_capacity,
            rng,
        )

        valid = route_costs < 1e9
        if valid.any():
            idx = np.argmin(route_costs)
            if route_costs[idx] < global_best_cost:
                global_best_cost = route_costs[idx]
                plen = route_lengths[idx]
                global_best_route = routes[idx, :plen].copy()

        convergence_history.append(global_best_cost)

        engine.evaporate(graph, cfg.evaporation_rate)
        engine.sync_pheromones_to_device(graph.pheromones)
        engine.deposit(graph, routes, route_lengths, route_costs,
                       cfg.q_deposit, num_ants)

    t_elapsed = (time.perf_counter() - t0) * 1000.0

    result.best_cost = global_best_cost
    result.convergence_history = convergence_history
    result.runtime_ms = t_elapsed
    result.iterations_run = len(convergence_history)

    if global_best_route is not None:
        result.best_route = global_best_route.tolist()
        result.best_travel_time = _sum_attr(graph, result.best_route, graph.travel_times)
        result.best_distance = _sum_attr(graph, result.best_route, graph.distances)

    logger.info(
        f"GPU-ACO-Baseline: cost={result.best_cost:.4f}, "
        f"tt={result.best_travel_time:.2f}, "
        f"dist={result.best_distance:.2f}, "
        f"time={result.runtime_ms:.2f} ms"
    )
    return result


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────

def _sum_attr(graph: CSRGraph, route: List[int], attr: np.ndarray) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        eidx = graph.get_edge_index(route[i], route[i + 1])
        if eidx >= 0:
            total += attr[eidx]
    return total
