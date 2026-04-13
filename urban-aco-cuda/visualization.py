"""
Visualization module for research paper plots.

Produces publication-quality figures for:
- Convergence curves
- Runtime comparison (bar chart)
- Cost comparison (bar chart)
- Speedup chart
- Route visualisation on the node grid
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Matplotlib style ─────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

COLORS = {
    "Dijkstra": "#2196F3",
    "CPU-ACO": "#FF9800",
    "GPU-ACO-Baseline": "#9C27B0",
    "DM-CUDA-ACO": "#4CAF50",
}


def _color_for(name: str) -> str:
    for key, c in COLORS.items():
        if key.lower() in name.lower():
            return c
    return "#607D8B"


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  1. Convergence curves
# ══════════════════════════════════════════════════════════════

def plot_convergence(
    histories: Dict[str, List[float]],
    save_path: str = "plots/convergence.png",
    title: str = "Convergence Comparison",
) -> None:
    """
    Plot convergence curves for multiple algorithms on one figure.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots()

    for name, hist in histories.items():
        ax.plot(hist, label=name, color=_color_for(name))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Cost")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Convergence plot -> {save_path}")


# ══════════════════════════════════════════════════════════════
#  2. Runtime comparison
# ══════════════════════════════════════════════════════════════

def plot_runtime_comparison(
    results: Dict[str, dict],
    save_path: str = "plots/runtime_comparison.png",
) -> None:
    """Bar chart of runtime (ms) for each algorithm."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots()

    names = list(results.keys())
    times = [results[n].get("runtime_ms", 0) for n in names]
    colors = [_color_for(n) for n in names]

    bars = ax.bar(names, times, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Runtime Comparison")

    # Value labels on bars
    for bar, val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}", ha="center", va="bottom", fontsize=9,
        )

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Runtime plot -> {save_path}")


# ══════════════════════════════════════════════════════════════
#  3. Cost comparison
# ══════════════════════════════════════════════════════════════

def plot_cost_comparison(
    results: Dict[str, dict],
    save_path: str = "plots/cost_comparison.png",
) -> None:
    """Bar chart of best route cost for each algorithm."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots()

    names = list(results.keys())
    costs = [results[n].get("best_cost", 0) for n in names]
    colors = [_color_for(n) for n in names]

    bars = ax.bar(names, costs, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Best Route Cost")
    ax.set_title("Cost Comparison")

    for bar, val in zip(bars, costs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.3f}", ha="center", va="bottom", fontsize=9,
        )

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Cost plot -> {save_path}")


# ══════════════════════════════════════════════════════════════
#  4. Speedup chart
# ══════════════════════════════════════════════════════════════

def plot_speedup(
    results: Dict[str, dict],
    reference_key: str = "CPU-ACO",
    save_path: str = "plots/speedup.png",
) -> None:
    """Bar chart of speedup relative to a reference algorithm."""
    _ensure_dir(save_path)

    ref_time = results.get(reference_key, {}).get("runtime_ms", 1)
    if ref_time <= 0:
        ref_time = 1

    names = [n for n in results if n != reference_key]
    speedups = [ref_time / max(results[n].get("runtime_ms", 1), 0.01) for n in names]
    colors = [_color_for(n) for n in names]

    fig, ax = plt.subplots()
    bars = ax.bar(names, speedups, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_ylabel(f"Speedup vs {reference_key}")
    ax.set_title("Speedup Comparison")

    for bar, val in zip(bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.2f}x", ha="center", va="bottom", fontsize=9,
        )

    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Speedup plot -> {save_path}")


# ══════════════════════════════════════════════════════════════
#  5. Route visualisation
# ══════════════════════════════════════════════════════════════

def plot_route_on_grid(
    nodes_df: pd.DataFrame,
    routes: Dict[str, List[int]],
    save_path: str = "plots/route_map.png",
    title: str = "Route Comparison on Road Network",
) -> None:
    """
    Plot node positions and overlay routes from different algorithms.
    """
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 10))

    lons = nodes_df["longitude"].values
    lats = nodes_df["latitude"].values

    # Draw all nodes
    ax.scatter(lons, lats, s=15, color="#B0BEC5", zorder=1, alpha=0.6)

    # Overlay routes
    for name, route in routes.items():
        if not route:
            continue
        rx = [lons[n] for n in route]
        ry = [lats[n] for n in route]
        ax.plot(rx, ry, label=name, color=_color_for(name),
                marker="o", markersize=4, zorder=3, alpha=0.85)

    # Mark source/destination
    all_routes = list(routes.values())
    if all_routes and all_routes[0]:
        src, dst = all_routes[0][0], all_routes[0][-1]
        ax.scatter([lons[src]], [lats[src]], s=120, color="#F44336",
                   marker="^", zorder=5, label="Source")
        ax.scatter([lons[dst]], [lats[dst]], s=120, color="#4CAF50",
                   marker="s", zorder=5, label="Destination")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Route map -> {save_path}")


# ══════════════════════════════════════════════════════════════
#  6. Experiment-parameter sweep plots
# ══════════════════════════════════════════════════════════════

def plot_parameter_sweep(
    x_values: List,
    y_dict: Dict[str, List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    """Generic line plot for parameter-sweep experiments."""
    _ensure_dir(save_path)
    fig, ax = plt.subplots()

    for name, ys in y_dict.items():
        ax.plot(x_values[:len(ys)], ys, marker="o", label=name,
                color=_color_for(name))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.savefig(save_path)
    plt.close(fig)
    logger.info(f"Param-sweep plot -> {save_path}")


def plot_ant_count_experiment(
    ant_counts: List[int],
    costs: Dict[str, List[float]],
    runtimes: Dict[str, List[float]],
    save_dir: str = "plots",
) -> None:
    """Plots for varying ant-count experiments."""
    plot_parameter_sweep(
        ant_counts, costs, "Number of Ants", "Best Cost",
        "Effect of Ant Count on Solution Quality",
        os.path.join(save_dir, "ant_count_cost.png"),
    )
    plot_parameter_sweep(
        ant_counts, runtimes, "Number of Ants", "Runtime (ms)",
        "Effect of Ant Count on Runtime",
        os.path.join(save_dir, "ant_count_runtime.png"),
    )


def plot_iteration_experiment(
    iter_counts: List[int],
    costs: Dict[str, List[float]],
    save_dir: str = "plots",
) -> None:
    """Plot convergence quality vs iteration budget."""
    plot_parameter_sweep(
        iter_counts, costs, "Max Iterations", "Best Cost",
        "Effect of Iteration Budget on Solution Quality",
        os.path.join(save_dir, "iteration_cost.png"),
    )
