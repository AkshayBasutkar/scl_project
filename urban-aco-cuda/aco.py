"""
Dynamic Multi-Objective Ant Colony Optimization engine.

Orchestrates the ACO iteration loop, adaptive behaviour, dynamic
traffic updates, and convergence monitoring. Uses CUDAEngine for
GPU-accelerated sub-routines when available.
"""

import numpy as np
import logging
import time
from typing import List, Optional, Tuple, Dict

from config import ACOConfig
from graph_utils import CSRGraph, update_congestion
from cuda_kernels import CUDAEngine

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────

class ACOResult:
    """Stores the outcome of an ACO run."""

    def __init__(self):
        self.best_route: List[int] = []
        self.best_cost: float = float("inf")
        self.best_travel_time: float = 0.0
        self.best_distance: float = 0.0
        self.convergence_history: List[float] = []
        self.runtime_ms: float = 0.0
        self.convergence_iteration: int = -1
        self.iterations_run: int = 0
        self.mode: str = "unknown"  # "cuda" or "cpu"

    def to_dict(self) -> dict:
        return {
            "best_route": self.best_route,
            "best_cost": round(self.best_cost, 6),
            "best_travel_time": round(self.best_travel_time, 4),
            "best_distance": round(self.best_distance, 4),
            "runtime_ms": round(self.runtime_ms, 2),
            "convergence_iteration": self.convergence_iteration,
            "iterations_run": self.iterations_run,
            "mode": self.mode,
            "path_length": len(self.best_route),
        }


# ──────────────────────────────────────────────────────────────
# Main ACO solver
# ──────────────────────────────────────────────────────────────

class DynamicMultiObjectiveACO:
    """
    GPU-accelerated, multi-objective, adaptive ACO with
    dynamic traffic modelling.
    """

    def __init__(self, graph: CSRGraph, cfg: ACOConfig):
        self.graph = graph
        self.cfg = cfg
        self.engine = CUDAEngine(
            use_cuda=cfg.use_cuda, block_size=cfg.cuda_block_size
        )
        self.rng = np.random.RandomState(cfg.random_seed)

        # Adaptive state
        self._evaporation_rate = cfg.evaporation_rate
        self._stagnation_counter = 0
        self._prev_best_cost = float("inf")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def solve(
        self,
        source: int,
        destination: int,
        num_ants: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> ACOResult:
        """
        Run the full ACO loop and return an ACOResult.
        """
        cfg = self.cfg
        num_ants = num_ants or cfg.num_ants
        max_iterations = max_iterations or cfg.max_iterations

        # Reset pheromones
        self.graph.reset_pheromones(cfg.pheromone_init)
        self._evaporation_rate = cfg.evaporation_rate
        self._stagnation_counter = 0
        self._prev_best_cost = float("inf")

        # Upload graph to GPU (if CUDA available)
        self.engine.upload_graph(self.graph)

        result = ACOResult()
        result.mode = "cuda" if self.engine.use_cuda else "cpu"

        global_best_route = None
        global_best_cost = float("inf")
        global_best_len = 0
        convergence_history = []

        t_start = time.perf_counter()

        for iteration in range(max_iterations):
            # ── 1. Construct routes ──────────────────────────
            routes, route_lengths, route_costs = self.engine.construct_routes(
                self.graph, num_ants, source, destination,
                cfg.max_path_length,
                cfg.alpha, cfg.beta,
                cfg.w_travel_time, cfg.w_distance,
                cfg.w_congestion, cfg.w_signal_delay, cfg.w_capacity,
                self.rng,
            )

            # ── 2. Find iteration-best ──────────────────────
            valid_mask = route_costs < 1e9
            if valid_mask.any():
                iter_best_idx = np.argmin(route_costs)
                iter_best_cost = route_costs[iter_best_idx]
                iter_best_len = route_lengths[iter_best_idx]
                iter_best_route = routes[iter_best_idx, :iter_best_len].copy()

                if iter_best_cost < global_best_cost:
                    global_best_cost = iter_best_cost
                    global_best_route = iter_best_route.copy()
                    global_best_len = iter_best_len
                    result.convergence_iteration = iteration

            convergence_history.append(global_best_cost)

            # ── 3. Pheromone evaporation ─────────────────────
            self.engine.evaporate(self.graph, self._evaporation_rate)

            # ── 4. Pheromone deposit ─────────────────────────
            # Sync pheromones to device before deposit if needed
            self.engine.sync_pheromones_to_device(self.graph.pheromones)
            self.engine.deposit(
                self.graph, routes, route_lengths, route_costs,
                cfg.q_deposit, num_ants,
            )

            # ── 5. Best-ant elitist boost ────────────────────
            if global_best_route is not None:
                self.engine.boost_best(
                    self.graph,
                    global_best_route, global_best_len,
                    global_best_cost, cfg.q_deposit, boost_factor=2.0,
                )

            # ── 6. Clamp pheromones ──────────────────────────
            np.clip(self.graph.pheromones, cfg.pheromone_min,
                    cfg.pheromone_max, out=self.graph.pheromones)
            self.engine.sync_pheromones_to_device(self.graph.pheromones)

            # ── 7. Adaptive behaviour ────────────────────────
            if cfg.adaptive_evaporation:
                self._adapt(global_best_cost, iteration, cfg)

            # ── 8. Dynamic traffic update ────────────────────
            if cfg.dynamic_traffic and (iteration + 1) % cfg.traffic_update_interval == 0:
                best_list = (
                    global_best_route.tolist() if global_best_route is not None else None
                )
                update_congestion(
                    self.graph, best_list,
                    cfg.congestion_increase_rate,
                    cfg.congestion_decrease_rate,
                    cfg.congestion_noise_std,
                    self.rng,
                )
                self.engine.sync_congestion_to_device(self.graph.congestions)

            # ── 9. Early stopping ────────────────────────────
            if self._check_convergence(global_best_cost, cfg):
                logger.info(
                    f"Converged at iteration {iteration} "
                    f"(best cost = {global_best_cost:.4f})"
                )
                break

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        # ── Pack result ──────────────────────────────────────
        result.best_cost = global_best_cost
        result.convergence_history = convergence_history
        result.runtime_ms = t_elapsed
        result.iterations_run = len(convergence_history)

        if global_best_route is not None:
            result.best_route = global_best_route.tolist()
            result.best_travel_time = self._sum_attribute(
                result.best_route, self.graph.travel_times
            )
            result.best_distance = self._sum_attribute(
                result.best_route, self.graph.distances
            )

        logger.info(
            f"ACO [{result.mode.upper()}] finished: "
            f"cost={result.best_cost:.4f}, "
            f"tt={result.best_travel_time:.2f} min, "
            f"dist={result.best_distance:.2f} km, "
            f"time={result.runtime_ms:.1f} ms, "
            f"iters={result.iterations_run}"
        )
        return result

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _adapt(self, current_best: float, iteration: int, cfg: ACOConfig):
        """Adjust evaporation rate based on improvement trend."""
        if iteration < cfg.stagnation_window:
            self._prev_best_cost = current_best
            return

        improvement = self._prev_best_cost - current_best
        if improvement < cfg.convergence_threshold:
            self._stagnation_counter += 1
            if self._stagnation_counter >= cfg.stagnation_window:
                # Increase evaporation to boost exploration
                self._evaporation_rate = min(
                    self._evaporation_rate + cfg.evaporation_increase, 0.5
                )
                self._stagnation_counter = 0
                logger.debug(
                    f"Iter {iteration}: stagnation → evap rate = "
                    f"{self._evaporation_rate:.3f}"
                )
        else:
            self._stagnation_counter = 0
            # Decrease evaporation to exploit good solutions
            self._evaporation_rate = max(
                self._evaporation_rate - cfg.evaporation_decrease, 0.05
            )

        self._prev_best_cost = current_best

    def _check_convergence(self, current_best: float, cfg: ACOConfig) -> bool:
        """Return True if solution has converged (no improvement)."""
        if self._stagnation_counter >= cfg.convergence_patience:
            return True
        return False

    def _sum_attribute(self, route: List[int], attr_array: np.ndarray) -> float:
        """Sum an edge attribute along a route."""
        total = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            eidx = self.graph.get_edge_index(u, v)
            if eidx >= 0:
                total += attr_array[eidx]
        return total
