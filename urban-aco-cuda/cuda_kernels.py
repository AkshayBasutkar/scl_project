"""
CUDA kernels for Dynamic Multi-Objective Ant Colony Optimization.

All GPU-accelerated routines are implemented here using Numba CUDA.
If CUDA is not available the module exposes CPU fallback functions
with identical signatures so the rest of the codebase is device-agnostic.

Kernels
-------
1. route_construction_kernel  – one thread per ant builds a full route
2. route_scoring_kernel       – one thread per ant scores its route
3. pheromone_deposit_kernel   – one thread per ant deposits pheromone
4. pheromone_evaporate_kernel – one thread per edge applies evaporation
5. probability_precompute_kernel – pre-compute heuristic desirability
"""

import math
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Try to import Numba CUDA
# ──────────────────────────────────────────────────────────────
try:
    from numba import cuda
    import numba
    NUMBA_CUDA_AVAILABLE = True
    logger.info("Numba CUDA module loaded successfully.")
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    logger.warning("Numba CUDA not available. GPU kernels disabled.")


# ══════════════════════════════════════════════════════════════
#  GPU KERNELS (Numba CUDA)
# ══════════════════════════════════════════════════════════════

if NUMBA_CUDA_AVAILABLE:

    # ----------------------------------------------------------
    # Kernel 1 – Route Construction
    # ----------------------------------------------------------
    @cuda.jit
    def route_construction_kernel(
        row_ptr,          # int32[num_nodes+1]
        col_idx,          # int32[num_edges]
        travel_times,     # float32[num_edges]
        distances,        # float32[num_edges]
        congestions,      # float32[num_edges]
        signal_delays,    # float32[num_edges]
        pheromones,       # float32[num_edges]
        alpha,            # float32 scalar
        beta,             # float32 scalar
        w_tt, w_dist, w_cong, w_sig,  # float32 weight scalars
        random_vals,      # float32[num_ants, max_path_length]
        routes,           # int32[num_ants, max_path_length]
        route_lengths,    # int32[num_ants]
        route_costs,      # float32[num_ants]
        num_nodes,        # int32 scalar
        source,           # int32 scalar
        destination,      # int32 scalar
        max_path_len,     # int32 scalar
    ):
        """
        Each thread constructs one ant's route from source → destination.

        Uses probabilistic selection based on pheromone^α × heuristic^β
        where heuristic = 1 / composite_edge_cost.
        """
        ant_id = cuda.grid(1)
        if ant_id >= routes.shape[0]:
            return

        # --- local visited array (bit-field replacement: direct bool array) ---
        # Numba CUDA doesn't support dynamic local arrays well, so we use
        # a fixed-size scratch in the route output itself and a visited flag
        # stored at the tail of `routes` temporarily. We'll use the last
        # portion of the routes row as visited markers during construction,
        # but it's cleaner to iterate neighbors each time.

        current = source
        routes[ant_id, 0] = current
        path_len = 1
        total_cost = 0.0

        for step in range(1, max_path_len):
            if current == destination:
                break

            # Enumerate neighbours of `current`
            start = row_ptr[current]
            end = row_ptr[current + 1]
            num_neighbors = end - start

            if num_neighbors == 0:
                break  # dead end

            # --- compute desirabilities for all neighbours ----
            total_prob = 0.0
            best_nb_idx = -1
            best_prob = -1.0

            # First pass: sum of desirabilities (un-normalised probabilities)
            # We'll store partial sums implicitly and do roulette selection.
            # Because we can't allocate a dynamic array per thread, we do
            # a two-pass approach: first compute total, then select.

            for k in range(start, end):
                nb = col_idx[k]
                # Check if already visited (linear scan – acceptable for
                # moderate path lengths on GPU)
                visited = False
                for v in range(path_len):
                    if routes[ant_id, v] == nb:
                        visited = True
                        break
                if visited:
                    continue

                # Composite edge cost
                edge_cost = (
                    w_tt * travel_times[k]
                    + w_dist * distances[k]
                    + w_cong * congestions[k]
                    + w_sig * signal_delays[k]
                )
                if edge_cost < 1e-8:
                    edge_cost = 1e-8

                heuristic = 1.0 / edge_cost
                tau = pheromones[k]
                if tau < 1e-10:
                    tau = 1e-10

                desirability = math.pow(tau, alpha) * math.pow(heuristic, beta)
                total_prob += desirability

            if total_prob < 1e-12:
                break  # stuck – no unvisited neighbours

            # Second pass: roulette wheel selection
            rnd = random_vals[ant_id, step] * total_prob
            cum = 0.0
            selected_edge = -1
            selected_nb = -1
            for k in range(start, end):
                nb = col_idx[k]
                visited = False
                for v in range(path_len):
                    if routes[ant_id, v] == nb:
                        visited = True
                        break
                if visited:
                    continue

                edge_cost = (
                    w_tt * travel_times[k]
                    + w_dist * distances[k]
                    + w_cong * congestions[k]
                    + w_sig * signal_delays[k]
                )
                if edge_cost < 1e-8:
                    edge_cost = 1e-8
                heuristic = 1.0 / edge_cost
                tau = pheromones[k]
                if tau < 1e-10:
                    tau = 1e-10

                desirability = math.pow(tau, alpha) * math.pow(heuristic, beta)
                cum += desirability
                if cum >= rnd:
                    selected_edge = k
                    selected_nb = nb
                    break

            if selected_nb == -1:
                # Fallback: pick first unvisited neighbour
                for k in range(start, end):
                    nb = col_idx[k]
                    visited = False
                    for v in range(path_len):
                        if routes[ant_id, v] == nb:
                            visited = True
                            break
                    if not visited:
                        selected_edge = k
                        selected_nb = nb
                        break

            if selected_nb == -1:
                break  # completely stuck

            # Move to selected neighbour
            routes[ant_id, path_len] = selected_nb
            # Accumulate cost
            edge_cost = (
                w_tt * travel_times[selected_edge]
                + w_dist * distances[selected_edge]
                + w_cong * congestions[selected_edge]
                + w_sig * signal_delays[selected_edge]
            )
            total_cost += edge_cost
            current = selected_nb
            path_len += 1

        route_lengths[ant_id] = path_len
        # If ant did NOT reach destination, set cost to a large penalty
        if current != destination:
            route_costs[ant_id] = 1e10
        else:
            route_costs[ant_id] = total_cost

    # ----------------------------------------------------------
    # Kernel 2 – Route Scoring (re-score with updated congestion)
    # ----------------------------------------------------------
    @cuda.jit
    def route_scoring_kernel(
        row_ptr, col_idx,
        travel_times, distances, congestions, signal_delays,
        w_tt, w_dist, w_cong, w_sig,
        routes, route_lengths, route_costs,
        num_nodes,
    ):
        """Re-compute route costs (e.g. after congestion update)."""
        ant_id = cuda.grid(1)
        if ant_id >= routes.shape[0]:
            return

        plen = route_lengths[ant_id]
        if plen <= 1:
            route_costs[ant_id] = 1e10
            return

        total = 0.0
        for i in range(plen - 1):
            u = routes[ant_id, i]
            v = routes[ant_id, i + 1]
            # find edge index
            start = row_ptr[u]
            end = row_ptr[u + 1]
            for k in range(start, end):
                if col_idx[k] == v:
                    total += (
                        w_tt * travel_times[k]
                        + w_dist * distances[k]
                        + w_cong * congestions[k]
                        + w_sig * signal_delays[k]
                    )
                    break
        route_costs[ant_id] = total

    # ----------------------------------------------------------
    # Kernel 3 – Pheromone Deposit
    # ----------------------------------------------------------
    @cuda.jit
    def pheromone_deposit_kernel(
        row_ptr, col_idx,
        routes, route_lengths, route_costs,
        pheromones,
        q_deposit,
        num_ants,
        pheromone_max,
    ):
        """
        Each thread handles one ant and deposits pheromone along its route.
        Uses atomicAdd for thread-safe updates.
        """
        ant_id = cuda.grid(1)
        if ant_id >= num_ants:
            return

        cost = route_costs[ant_id]
        if cost >= 1e9:
            return  # invalid route

        deposit = q_deposit / cost
        plen = route_lengths[ant_id]

        for i in range(plen - 1):
            u = routes[ant_id, i]
            v = routes[ant_id, i + 1]
            start = row_ptr[u]
            end = row_ptr[u + 1]
            for k in range(start, end):
                if col_idx[k] == v:
                    cuda.atomic.add(pheromones, k, deposit)
                    # Clamp (approximate – exact clamp via separate kernel)
                    break

    # ----------------------------------------------------------
    # Kernel 4 – Pheromone Evaporation
    # ----------------------------------------------------------
    @cuda.jit
    def pheromone_evaporate_kernel(
        pheromones,
        evaporation_rate,
        pheromone_min,
        pheromone_max,
        num_edges,
    ):
        """One thread per edge: apply evaporation and clamp."""
        eidx = cuda.grid(1)
        if eidx >= num_edges:
            return
        val = pheromones[eidx] * (1.0 - evaporation_rate)
        if val < pheromone_min:
            val = pheromone_min
        if val > pheromone_max:
            val = pheromone_max
        pheromones[eidx] = val

    # ----------------------------------------------------------
    # Kernel 5 – Best-ant pheromone boost
    # ----------------------------------------------------------
    @cuda.jit
    def best_ant_boost_kernel(
        row_ptr, col_idx,
        best_route, best_route_len, best_cost,
        pheromones,
        q_deposit,
        boost_factor,
        pheromone_max,
    ):
        """Extra pheromone deposit on the globally-best route."""
        idx = cuda.grid(1)
        if idx >= best_route_len - 1:
            return
        u = best_route[idx]
        v = best_route[idx + 1]
        start = row_ptr[u]
        end = row_ptr[u + 1]
        for k in range(start, end):
            if col_idx[k] == v:
                deposit = boost_factor * q_deposit / best_cost
                cuda.atomic.add(pheromones, k, deposit)
                break


# ══════════════════════════════════════════════════════════════
#  CPU FALLBACK FUNCTIONS (identical logic, pure NumPy)
# ══════════════════════════════════════════════════════════════

def cpu_construct_routes(
    row_ptr, col_idx,
    travel_times, distances, congestions, signal_delays,
    pheromones,
    alpha, beta,
    w_tt, w_dist, w_cong, w_sig,
    num_ants, num_nodes, source, destination, max_path_len,
    rng=None,
):
    """CPU reference implementation of route construction."""
    if rng is None:
        rng = np.random.RandomState()

    routes = np.full((num_ants, max_path_len), -1, dtype=np.int32)
    route_lengths = np.zeros(num_ants, dtype=np.int32)
    route_costs = np.full(num_ants, 1e10, dtype=np.float32)

    for ant in range(num_ants):
        visited = set()
        current = source
        routes[ant, 0] = current
        visited.add(current)
        path_len = 1
        total_cost = 0.0

        for step in range(1, max_path_len):
            if current == destination:
                break

            start = row_ptr[current]
            end = row_ptr[current + 1]

            # Gather unvisited neighbours and desirabilities
            candidates = []
            desirabilities = []
            for k in range(start, end):
                nb = col_idx[k]
                if nb in visited:
                    continue
                edge_cost = (
                    w_tt * travel_times[k]
                    + w_dist * distances[k]
                    + w_cong * congestions[k]
                    + w_sig * signal_delays[k]
                )
                edge_cost = max(edge_cost, 1e-8)
                heuristic = 1.0 / edge_cost
                tau = max(pheromones[k], 1e-10)
                d = (tau ** alpha) * (heuristic ** beta)
                candidates.append((k, nb))
                desirabilities.append(d)

            if not candidates:
                break

            # Roulette-wheel selection
            total = sum(desirabilities)
            probs = np.array([d / total for d in desirabilities])
            probs /= probs.sum()
            choice_idx = rng.choice(len(candidates), p=probs)
            edge_k, selected_nb = candidates[choice_idx]

            routes[ant, path_len] = selected_nb
            edge_cost = (
                w_tt * travel_times[edge_k]
                + w_dist * distances[edge_k]
                + w_cong * congestions[edge_k]
                + w_sig * signal_delays[edge_k]
            )
            total_cost += edge_cost
            visited.add(selected_nb)
            current = selected_nb
            path_len += 1

        route_lengths[ant] = path_len
        if current == destination:
            route_costs[ant] = total_cost

    return routes, route_lengths, route_costs


def cpu_evaporate_pheromones(pheromones, evaporation_rate, pmin, pmax):
    """CPU pheromone evaporation."""
    pheromones *= (1.0 - evaporation_rate)
    np.clip(pheromones, pmin, pmax, out=pheromones)


def cpu_deposit_pheromones(
    row_ptr, col_idx,
    routes, route_lengths, route_costs,
    pheromones, q_deposit, pmax,
    num_ants,
):
    """CPU pheromone deposit."""
    for ant in range(num_ants):
        cost = route_costs[ant]
        if cost >= 1e9:
            continue
        deposit = q_deposit / cost
        plen = route_lengths[ant]
        for i in range(plen - 1):
            u = routes[ant, i]
            v = routes[ant, i + 1]
            start = row_ptr[u]
            end = row_ptr[u + 1]
            for k in range(start, end):
                if col_idx[k] == v:
                    pheromones[k] = min(pheromones[k] + deposit, pmax)
                    break


# ══════════════════════════════════════════════════════════════
#  High-level wrapper – device-agnostic interface
# ══════════════════════════════════════════════════════════════

class CUDAEngine:
    """
    Wraps CUDA kernels with automatic CPU fallback.

    All public methods accept plain NumPy arrays and handle
    host↔device transfers internally for minimum caller complexity.
    """

    def __init__(self, use_cuda: bool = True, block_size: int = 128):
        self.use_cuda = use_cuda and NUMBA_CUDA_AVAILABLE
        self.block_size = block_size
        if self.use_cuda:
            logger.info("CUDAEngine initialised with GPU acceleration.")
        else:
            logger.info("CUDAEngine running in CPU-only mode.")

        # Persistent device arrays (populated on first call)
        self._d_row_ptr = None
        self._d_col_idx = None
        self._d_travel_times = None
        self._d_distances = None
        self._d_congestions = None
        self._d_signal_delays = None
        self._d_pheromones = None
        self._graph_uploaded = False

    # ----------------------------------------------------------
    # Upload graph to GPU (once, then update only pheromones/congestion)
    # ----------------------------------------------------------
    def upload_graph(self, graph) -> None:
        """Transfer static CSR arrays to device memory."""
        if not self.use_cuda:
            return
        self._d_row_ptr = cuda.to_device(graph.row_ptr)
        self._d_col_idx = cuda.to_device(graph.col_idx)
        self._d_travel_times = cuda.to_device(graph.travel_times)
        self._d_distances = cuda.to_device(graph.distances)
        self._d_congestions = cuda.to_device(graph.congestions)
        self._d_signal_delays = cuda.to_device(graph.signal_delays)
        self._d_pheromones = cuda.to_device(graph.pheromones)
        self._graph_uploaded = True
        logger.debug("Graph uploaded to GPU.")

    def sync_pheromones_to_device(self, pheromones: np.ndarray) -> None:
        if self.use_cuda and self._graph_uploaded:
            self._d_pheromones.copy_to_device(pheromones)

    def sync_pheromones_from_device(self, pheromones: np.ndarray) -> None:
        if self.use_cuda and self._graph_uploaded:
            self._d_pheromones.copy_to_host(pheromones)

    def sync_congestion_to_device(self, congestions: np.ndarray) -> None:
        if self.use_cuda and self._graph_uploaded:
            self._d_congestions.copy_to_device(congestions)

    # ----------------------------------------------------------
    # Construct routes
    # ----------------------------------------------------------
    def construct_routes(
        self, graph, num_ants, source, destination, max_path_len,
        alpha, beta, w_tt, w_dist, w_cong, w_sig, rng=None,
    ):
        """
        Returns (routes, route_lengths, route_costs) as NumPy arrays.
        """
        if rng is None:
            rng = np.random.RandomState()

        if self.use_cuda and self._graph_uploaded:
            return self._gpu_construct(
                graph, num_ants, source, destination, max_path_len,
                alpha, beta, w_tt, w_dist, w_cong, w_sig, rng,
            )
        else:
            return cpu_construct_routes(
                graph.row_ptr, graph.col_idx,
                graph.travel_times, graph.distances,
                graph.congestions, graph.signal_delays, graph.pheromones,
                alpha, beta, w_tt, w_dist, w_cong, w_sig,
                num_ants, graph.num_nodes, source, destination,
                max_path_len, rng,
            )

    def _gpu_construct(self, graph, num_ants, source, destination,
                       max_path_len, alpha, beta, w_tt, w_dist,
                       w_cong, w_sig, rng):
        # Pre-generate random values on CPU → transfer to GPU
        random_vals = rng.random((num_ants, max_path_len)).astype(np.float32)
        d_random = cuda.to_device(random_vals)

        # Allocate output arrays on device
        routes = np.full((num_ants, max_path_len), -1, dtype=np.int32)
        route_lengths = np.zeros(num_ants, dtype=np.int32)
        route_costs = np.full(num_ants, 1e10, dtype=np.float32)
        d_routes = cuda.to_device(routes)
        d_rlens = cuda.to_device(route_lengths)
        d_rcosts = cuda.to_device(route_costs)

        threads = self.block_size
        blocks = (num_ants + threads - 1) // threads

        route_construction_kernel[blocks, threads](
            self._d_row_ptr, self._d_col_idx,
            self._d_travel_times, self._d_distances,
            self._d_congestions, self._d_signal_delays,
            self._d_pheromones,
            np.float32(alpha), np.float32(beta),
            np.float32(w_tt), np.float32(w_dist),
            np.float32(w_cong), np.float32(w_sig),
            d_random,
            d_routes, d_rlens, d_rcosts,
            np.int32(graph.num_nodes),
            np.int32(source), np.int32(destination),
            np.int32(max_path_len),
        )
        cuda.synchronize()

        d_routes.copy_to_host(routes)
        d_rlens.copy_to_host(route_lengths)
        d_rcosts.copy_to_host(route_costs)
        return routes, route_lengths, route_costs

    # ----------------------------------------------------------
    # Pheromone evaporation
    # ----------------------------------------------------------
    def evaporate(self, graph, evaporation_rate):
        if self.use_cuda and self._graph_uploaded:
            threads = self.block_size
            blocks = (graph.num_edges + threads - 1) // threads
            pheromone_evaporate_kernel[blocks, threads](
                self._d_pheromones,
                np.float32(evaporation_rate),
                np.float32(graph.pheromones.min()),  # pmin placeholder
                np.float32(10.0),
                np.int32(graph.num_edges),
            )
            cuda.synchronize()
            self._d_pheromones.copy_to_host(graph.pheromones)
        else:
            cpu_evaporate_pheromones(
                graph.pheromones, evaporation_rate, 0.001, 10.0
            )

    # ----------------------------------------------------------
    # Pheromone deposit
    # ----------------------------------------------------------
    def deposit(self, graph, routes, route_lengths, route_costs,
                q_deposit, num_ants):
        if self.use_cuda and self._graph_uploaded:
            d_routes = cuda.to_device(routes)
            d_rlens = cuda.to_device(route_lengths)
            d_rcosts = cuda.to_device(route_costs)

            threads = self.block_size
            blocks = (num_ants + threads - 1) // threads
            pheromone_deposit_kernel[blocks, threads](
                self._d_row_ptr, self._d_col_idx,
                d_routes, d_rlens, d_rcosts,
                self._d_pheromones,
                np.float32(q_deposit),
                np.int32(num_ants),
                np.float32(10.0),
            )
            cuda.synchronize()
            self._d_pheromones.copy_to_host(graph.pheromones)
        else:
            cpu_deposit_pheromones(
                graph.row_ptr, graph.col_idx,
                routes, route_lengths, route_costs,
                graph.pheromones, q_deposit, 10.0, num_ants,
            )

    # ----------------------------------------------------------
    # Best-ant boost
    # ----------------------------------------------------------
    def boost_best(self, graph, best_route, best_len, best_cost,
                   q_deposit, boost_factor=2.0):
        if best_cost >= 1e9 or best_len <= 1:
            return
        if self.use_cuda and self._graph_uploaded:
            d_best = cuda.to_device(best_route.astype(np.int32))
            threads = self.block_size
            blocks = (best_len + threads - 1) // threads
            best_ant_boost_kernel[blocks, threads](
                self._d_row_ptr, self._d_col_idx,
                d_best, np.int32(best_len), np.float32(best_cost),
                self._d_pheromones,
                np.float32(q_deposit),
                np.float32(boost_factor),
                np.float32(10.0),
            )
            cuda.synchronize()
            self._d_pheromones.copy_to_host(graph.pheromones)
        else:
            # CPU boost
            deposit = boost_factor * q_deposit / best_cost
            for i in range(best_len - 1):
                eidx = graph.get_edge_index(best_route[i], best_route[i + 1])
                if eidx >= 0:
                    graph.pheromones[eidx] = min(
                        graph.pheromones[eidx] + deposit, 10.0
                    )
