"""
Data loader for Urban ACO-CUDA project.

Reads road-network data from CSV files and returns structured
NumPy arrays ready for graph construction.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from config import ACOConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CSV loaders
# ──────────────────────────────────────────────────────────────

def load_nodes(filepath: str) -> pd.DataFrame:
    """
    Load nodes CSV.

    Expected columns: node_id, latitude, longitude
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nodes file not found: {filepath}")
    df = pd.read_csv(filepath)
    required = {"node_id", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Nodes CSV missing columns: {missing}")
    logger.info(f"Loaded {len(df)} nodes from {filepath}")
    return df


def load_edges(filepath: str) -> pd.DataFrame:
    """
    Load edges CSV.

    Expected columns: source, target, distance, travel_time,
                      capacity, signal_delay, congestion
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Edges file not found: {filepath}")
    df = pd.read_csv(filepath)
    required = {"source", "target", "distance", "travel_time",
                "capacity", "signal_delay", "congestion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Edges CSV missing columns: {missing}")
    logger.info(f"Loaded {len(df)} edges from {filepath}")
    return df


def load_graph_data(cfg: ACOConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both nodes and edges from paths specified in *cfg*."""
    nodes_df = load_nodes(cfg.nodes_file)
    edges_df = load_edges(cfg.edges_file)
    return nodes_df, edges_df


# ──────────────────────────────────────────────────────────────
# Sample-graph generator (for out-of-the-box testing)
# ──────────────────────────────────────────────────────────────

def generate_sample_graph(
    rows: int = 10,
    cols: int = 10,
    save_dir: str = "data",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a grid-based urban road network with realistic properties.

    Creates a rows×cols grid where each node is connected to its 4-neighbours
    (+ some diagonal shortcuts) with randomised travel_time, distance,
    capacity, signal_delay, and congestion values.

    Returns (nodes_df, edges_df) and also saves to CSV.
    """
    rng = np.random.RandomState(seed)
    num_nodes = rows * cols

    # ── Nodes ────────────────────────────────────────────────
    node_ids = np.arange(num_nodes)
    # Place on a lat/lon-style grid centred around a fictitious city
    base_lat, base_lon = 28.6139, 77.2090  # New Delhi approximate
    lats, lons = [], []
    for r in range(rows):
        for c in range(cols):
            lats.append(base_lat + r * 0.005 + rng.normal(0, 0.0005))
            lons.append(base_lon + c * 0.005 + rng.normal(0, 0.0005))

    nodes_df = pd.DataFrame({
        "node_id": node_ids,
        "latitude": lats,
        "longitude": lons,
    })

    # ── Edges (bidirectional) ────────────────────────────────
    edges = []

    def _add_edge(src, tgt):
        dist = rng.uniform(0.3, 2.0)                  # km
        speed = rng.uniform(20, 60)                     # km/h
        tt = dist / speed * 60                          # minutes
        cap = rng.choice([1, 2, 3, 4]) * 500            # vehicles/hour
        sig = rng.uniform(0.0, 2.0)                     # minutes delay
        cong = rng.uniform(0.0, 1.0)                    # 0=free, 1=jammed
        edges.append((src, tgt, round(dist, 3), round(tt, 3),
                       int(cap), round(sig, 3), round(cong, 3)))

    for r in range(rows):
        for c in range(cols):
            nid = r * cols + c
            # right
            if c + 1 < cols:
                _add_edge(nid, nid + 1)
                _add_edge(nid + 1, nid)
            # down
            if r + 1 < rows:
                _add_edge(nid, nid + cols)
                _add_edge(nid + cols, nid)
            # diagonal (right-down) – with 40 % probability for variety
            if r + 1 < rows and c + 1 < cols and rng.random() < 0.4:
                _add_edge(nid, nid + cols + 1)
                _add_edge(nid + cols + 1, nid)
            # diagonal (left-down)
            if r + 1 < rows and c - 1 >= 0 and rng.random() < 0.25:
                _add_edge(nid, nid + cols - 1)
                _add_edge(nid + cols - 1, nid)

    edges_df = pd.DataFrame(edges, columns=[
        "source", "target", "distance", "travel_time",
        "capacity", "signal_delay", "congestion",
    ])

    # ── Save ────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    nodes_path = os.path.join(save_dir, "nodes.csv")
    edges_path = os.path.join(save_dir, "edges.csv")
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)
    logger.info(f"Sample graph saved: {num_nodes} nodes, {len(edges_df)} edges  ->  {save_dir}/")
    return nodes_df, edges_df
