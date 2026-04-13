"""
Graph utilities for Urban ACO-CUDA project.

Converts DataFrames into GPU-friendly CSR (Compressed Sparse Row)
representation and provides adjacency-dict helpers for CPU algorithms.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CSR graph representation (GPU-friendly)
# ──────────────────────────────────────────────────────────────

class CSRGraph:
    """
    Compressed Sparse Row representation of a directed graph.

    Attributes
    ----------
    num_nodes : int
    num_edges : int
    row_ptr   : np.ndarray[int32]   (num_nodes + 1,)
    col_idx   : np.ndarray[int32]   (num_edges,)
    travel_times   : np.ndarray[float32] (num_edges,)
    distances      : np.ndarray[float32] (num_edges,)
    congestions    : np.ndarray[float32] (num_edges,)
    signal_delays  : np.ndarray[float32] (num_edges,)
    capacities     : np.ndarray[float32] (num_edges,)
    pheromones     : np.ndarray[float32] (num_edges,)  – mutable
    """

    def __init__(
        self,
        num_nodes: int,
        row_ptr: np.ndarray,
        col_idx: np.ndarray,
        travel_times: np.ndarray,
        distances: np.ndarray,
        congestions: np.ndarray,
        signal_delays: np.ndarray,
        capacities: np.ndarray,
        pheromone_init: float = 0.1,
    ):
        self.num_nodes = num_nodes
        self.num_edges = len(col_idx)
        self.row_ptr = row_ptr.astype(np.int32)
        self.col_idx = col_idx.astype(np.int32)
        self.travel_times = travel_times.astype(np.float32)
        self.distances = distances.astype(np.float32)
        self.congestions = congestions.astype(np.float32)
        self.signal_delays = signal_delays.astype(np.float32)
        self.capacities = capacities.astype(np.float32)
        self.pheromones = np.full(self.num_edges, pheromone_init, dtype=np.float32)

    def get_neighbors(self, node: int) -> np.ndarray:
        """Return array of neighbor node IDs for *node*."""
        start = self.row_ptr[node]
        end = self.row_ptr[node + 1]
        return self.col_idx[start:end]

    def get_edge_index(self, u: int, v: int) -> int:
        """Return the CSR edge index for edge (u, v), or -1 if not found."""
        start = self.row_ptr[u]
        end = self.row_ptr[u + 1]
        for idx in range(start, end):
            if self.col_idx[idx] == v:
                return idx
        return -1

    def reset_pheromones(self, value: float = 0.1) -> None:
        self.pheromones[:] = value

    def summary(self) -> str:
        return (
            f"CSRGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"avg_degree={self.num_edges / self.num_nodes:.1f})"
        )


# ──────────────────────────────────────────────────────────────
# Build CSR from edges DataFrame
# ──────────────────────────────────────────────────────────────

def build_csr_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    pheromone_init: float = 0.1,
) -> CSRGraph:
    """
    Convert nodes/edges DataFrames into a CSRGraph.

    edges_df **must** have columns:
        source, target, distance, travel_time, capacity, signal_delay, congestion
    """
    num_nodes = int(nodes_df["node_id"].max()) + 1

    # Sort edges by source to build CSR correctly
    edges_sorted = edges_df.sort_values("source").reset_index(drop=True)

    sources = edges_sorted["source"].values.astype(np.int32)
    targets = edges_sorted["target"].values.astype(np.int32)

    # Build row_ptr
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int32)
    for s in sources:
        row_ptr[s + 1] += 1
    row_ptr = np.cumsum(row_ptr)

    graph = CSRGraph(
        num_nodes=num_nodes,
        row_ptr=row_ptr,
        col_idx=targets,
        travel_times=edges_sorted["travel_time"].values,
        distances=edges_sorted["distance"].values,
        congestions=edges_sorted["congestion"].values,
        signal_delays=edges_sorted["signal_delay"].values,
        capacities=edges_sorted["capacity"].values,
        pheromone_init=pheromone_init,
    )
    logger.info(f"Built {graph.summary()}")
    return graph


# ──────────────────────────────────────────────────────────────
# Adjacency-dict representation (CPU algorithms)
# ──────────────────────────────────────────────────────────────

def build_adjacency_dict(
    edges_df: pd.DataFrame,
) -> Dict[int, Dict[int, dict]]:
    """
    Build an adjacency dictionary from edges DataFrame.

    Returns adjacency[u][v] = {travel_time, distance, congestion,
                                signal_delay, capacity}
    """
    adj: Dict[int, Dict[int, dict]] = {}
    for _, row in edges_df.iterrows():
        u, v = int(row["source"]), int(row["target"])
        props = {
            "travel_time": float(row["travel_time"]),
            "distance": float(row["distance"]),
            "congestion": float(row["congestion"]),
            "signal_delay": float(row["signal_delay"]),
            "capacity": float(row["capacity"]),
        }
        adj.setdefault(u, {})[v] = props
    return adj


# ──────────────────────────────────────────────────────────────
# Dynamic traffic update
# ──────────────────────────────────────────────────────────────

def update_congestion(
    graph: CSRGraph,
    best_route: Optional[List[int]],
    increase_rate: float = 0.05,
    decrease_rate: float = 0.03,
    noise_std: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> None:
    """
    Simulate dynamic traffic by adjusting congestion values.

    - Edges on the best route get slightly *more* congested (popular corridor).
    - All other edges drift toward a baseline with small random noise.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Global noise drift
    noise = rng.normal(0.0, noise_std, size=graph.num_edges).astype(np.float32)
    graph.congestions += noise

    # Random decay toward 0.5 baseline
    graph.congestions += decrease_rate * (0.5 - graph.congestions)

    # Increase congestion on best-route edges
    if best_route is not None and len(best_route) > 1:
        for i in range(len(best_route) - 1):
            eidx = graph.get_edge_index(best_route[i], best_route[i + 1])
            if eidx >= 0:
                graph.congestions[eidx] += increase_rate

    # Clip to [0, 1]
    np.clip(graph.congestions, 0.0, 1.0, out=graph.congestions)


def update_adjacency_congestion(
    adj: Dict[int, Dict[int, dict]],
    congestion_array: np.ndarray,
    edges_df: pd.DataFrame,
) -> None:
    """Sync the adjacency dict's congestion values from a flat array."""
    edges_sorted = edges_df.sort_values("source").reset_index(drop=True)
    for idx, (_, row) in enumerate(edges_sorted.iterrows()):
        u, v = int(row["source"]), int(row["target"])
        if u in adj and v in adj[u]:
            adj[u][v]["congestion"] = float(congestion_array[idx])
