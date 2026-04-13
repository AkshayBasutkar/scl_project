"""
Utility helpers for the Urban ACO-CUDA project.

Provides:
- CUDA availability detection with graceful fallback
- Timing context manager
- Path-reconstruction helpers
- Logging setup
- Reproducibility helpers
"""

import time
import logging
import sys
import os
import numpy as np
from contextlib import contextmanager
from typing import List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# CUDA availability
# ──────────────────────────────────────────────────────────────

_CUDA_AVAILABLE: Optional[bool] = None


def check_cuda_available() -> bool:
    """Check whether a CUDA-capable GPU is reachable via Numba."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    try:
        from numba import cuda
        cuda.detect()
        # try a trivial device query
        dev = cuda.get_current_device()
        _CUDA_AVAILABLE = True
        logging.info(f"CUDA available – {dev.name.decode()}")
    except Exception as exc:
        _CUDA_AVAILABLE = False
        logging.warning(f"CUDA NOT available ({exc}). Falling back to CPU.")
    return _CUDA_AVAILABLE


def get_cuda_device_info() -> dict:
    """Return a dict with GPU device properties, or empty dict if unavailable."""
    if not check_cuda_available():
        return {}
    from numba import cuda
    dev = cuda.get_current_device()
    return {
        "name": dev.name.decode() if isinstance(dev.name, bytes) else str(dev.name),
        "compute_capability": f"{dev.compute_capability[0]}.{dev.compute_capability[1]}",
        "max_threads_per_block": dev.MAX_THREADS_PER_BLOCK,
        "max_shared_memory_per_block": dev.MAX_SHARED_MEMORY_PER_BLOCK,
    }


# ──────────────────────────────────────────────────────────────
# Timing
# ──────────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer."""
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self) -> float:
        if self.start_time is not None:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            self.start_time = None
        return self.elapsed_ms


@contextmanager
def timed(label: str = ""):
    """Context manager that prints elapsed time."""
    t = Timer()
    t.start()
    yield t
    t.stop()
    if label:
        logging.info(f"[TIMER] {label}: {t.elapsed_ms:.2f} ms")


# ──────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────

def reconstruct_path(route_array: np.ndarray, route_length: int) -> List[int]:
    """Convert a fixed-length route array into a Python list of node IDs."""
    return route_array[:route_length].tolist()


def compute_route_cost(
    route: List[int],
    adjacency: dict,
    weights: Tuple[float, float, float, float],
) -> float:
    """
    Compute composite cost of a route using multi-objective weights.

    Parameters
    ----------
    route : list[int]
        Sequence of node IDs.
    adjacency : dict
        adjacency[u][v] = dict with 'travel_time', 'distance', 'congestion', 'signal_delay'.
    weights : tuple(w_tt, w_dist, w_cong, w_sig)

    Returns
    -------
    float  total cost (lower is better).
    """
    w_tt, w_dist, w_cong, w_sig = weights
    total = 0.0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        e = adjacency[u][v]
        total += (
            w_tt * e["travel_time"]
            + w_dist * e["distance"]
            + w_cong * e["congestion"]
            + w_sig * e["signal_delay"]
        )
    return total


def compute_route_travel_time(route: List[int], adjacency: dict) -> float:
    """Sum of travel_time along a route."""
    return sum(adjacency[route[i]][route[i + 1]]["travel_time"]
               for i in range(len(route) - 1))


def compute_route_distance(route: List[int], adjacency: dict) -> float:
    """Sum of distance along a route."""
    return sum(adjacency[route[i]][route[i + 1]]["distance"]
               for i in range(len(route) - 1))


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger."""
    fmt = "%(asctime)s | %(levelname)-7s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ──────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────

def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
