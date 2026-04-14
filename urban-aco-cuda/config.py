"""
Configuration module for Urban Road Network Optimization using
Dynamic Multi-Objective CUDA Ant Colony Optimization.

All tunable hyper-parameters are centralised here so experiments
can be reproduced by persisting a single JSON/dict snapshot.
"""

from dataclasses import dataclass, field, asdict
import json
import os

# ──────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────

@dataclass
class ACOConfig:
    """Ant Colony Optimization hyper-parameters."""

    # ── Graph / IO ───────────────────────────────────────────
    nodes_file: str = "data/nodes.csv"
    edges_file: str = "data/edges.csv"
    output_dir: str = "outputs"
    plots_dir: str = "plots"

    # ── ACO core ─────────────────────────────────────────────
    num_ants: int = 256
    max_iterations: int = 200
    alpha: float = 1.0       # pheromone influence
    beta: float = 2.5        # heuristic influence
    evaporation_rate: float = 0.15
    pheromone_init: float = 0.1
    pheromone_min: float = 0.001
    pheromone_max: float = 10.0
    q_deposit: float = 100.0  # pheromone deposit constant Q

    # ── Multi-objective cost weights ─────────────────────────
    w_travel_time: float = 0.25
    w_distance: float = 0.20
    w_congestion: float = 0.20
    w_signal_delay: float = 0.15
    w_capacity: float = 0.10       # inverse capacity penalty
    w_pheromone_bonus: float = 0.10

    # ── Adaptive behaviour ───────────────────────────────────
    adaptive_evaporation: bool = True
    evaporation_increase: float = 0.02   # when improvement stalls
    evaporation_decrease: float = 0.01   # when improving
    stagnation_window: int = 10          # iterations to detect stagnation
    exploration_boost_threshold: float = 0.7  # congestion level to boost exploration
    exploration_boost_factor: float = 1.5

    # ── Dynamic traffic ──────────────────────────────────────
    dynamic_traffic: bool = True
    traffic_update_interval: int = 10    # update congestion every N iterations
    congestion_increase_rate: float = 0.05
    congestion_decrease_rate: float = 0.03
    congestion_noise_std: float = 0.02

    # ── CUDA settings ────────────────────────────────────────
    use_cuda: bool = True
    cuda_block_size: int = 128
    max_path_length: int = 200           # maximum nodes in a single route

    # ── Experiment ───────────────────────────────────────────
    source_node: int = 0
    destination_node: int = 99
    random_seed: int = 42

    # ── Convergence ──────────────────────────────────────────
    convergence_threshold: float = 1e-4  # min improvement to consider converged
    convergence_patience: int = 20       # stop after N iterations with no improvement

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ACOConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self) -> str:
        lines = [f"  {k}={v}" for k, v in self.to_dict().items()]
        return "ACOConfig(\n" + "\n".join(lines) + "\n)"
