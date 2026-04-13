"""
Metrics computation for research evaluation.

Computes and formats performance metrics including runtime,
speedup, route quality, convergence statistics, and comparison tables.
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Single-run metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    result_dict: dict,
    baseline_runtime_ms: float = 0.0,
    baseline_cost: float = 0.0,
) -> dict:
    """
    Enrich a result dict with derived metrics.

    Parameters
    ----------
    result_dict : dict from ACOResult.to_dict() or BaselineResult.to_dict()
    baseline_runtime_ms : runtime of reference algorithm for speedup calc
    baseline_cost : cost of reference algorithm for improvement calc
    """
    metrics = dict(result_dict)  # shallow copy

    # Speedup vs baseline
    if baseline_runtime_ms > 0 and result_dict.get("runtime_ms", 0) > 0:
        metrics["speedup"] = round(
            baseline_runtime_ms / result_dict["runtime_ms"], 2
        )
    else:
        metrics["speedup"] = 1.0

    # Cost improvement (%) relative to baseline
    if baseline_cost > 0:
        improvement = (baseline_cost - result_dict["best_cost"]) / baseline_cost * 100
        metrics["cost_improvement_pct"] = round(improvement, 2)
    else:
        metrics["cost_improvement_pct"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────
# Comparison table
# ──────────────────────────────────────────────────────────────

def build_comparison_table(
    results: Dict[str, dict],
    reference_key: str = "Dijkstra",
) -> pd.DataFrame:
    """
    Build a DataFrame comparing multiple algorithm results.

    Parameters
    ----------
    results : {algorithm_name: result_dict}
    reference_key : which result to use as the speedup/improvement baseline

    Returns
    -------
    pd.DataFrame with one row per algorithm.
    """
    ref = results.get(reference_key, {})
    ref_runtime = ref.get("runtime_ms", 0)
    ref_cost = ref.get("best_cost", 0)

    rows = []
    for name, rd in results.items():
        m = compute_metrics(rd, ref_runtime, ref_cost)
        rows.append({
            "Algorithm": name,
            "Best Cost": m.get("best_cost", None),
            "Travel Time (min)": m.get("best_travel_time", None),
            "Distance (km)": m.get("best_distance", None),
            "Runtime (ms)": m.get("runtime_ms", None),
            "Speedup vs Ref": m.get("speedup", None),
            "Cost Improv. (%)": m.get("cost_improvement_pct", None),
            "Path Length": m.get("path_length", None),
            "Iterations": m.get("iterations_run", None),
            "Mode": m.get("mode", ""),
        })

    df = pd.DataFrame(rows)
    return df


# ──────────────────────────────────────────────────────────────
# Convergence statistics
# ──────────────────────────────────────────────────────────────

def convergence_stats(history: List[float]) -> dict:
    """Extract statistics from a convergence history."""
    arr = np.array(history)
    if len(arr) == 0:
        return {}

    # Find first iteration where cost stops improving by > 0.1 %
    final_cost = arr[-1]
    conv_iter = len(arr) - 1
    for i in range(len(arr)):
        if arr[i] <= final_cost * 1.001:
            conv_iter = i
            break

    return {
        "initial_cost": round(float(arr[0]), 4),
        "final_cost": round(float(final_cost), 4),
        "improvement_pct": round(
            (float(arr[0]) - float(final_cost)) / float(arr[0]) * 100, 2
        ) if arr[0] > 0 else 0.0,
        "convergence_iteration": conv_iter,
        "total_iterations": len(arr),
        "std_last_10": round(float(np.std(arr[-10:])), 6) if len(arr) >= 10 else 0.0,
    }


# ──────────────────────────────────────────────────────────────
# Experiment-wide summary
# ──────────────────────────────────────────────────────────────

def summarise_experiments(
    experiment_results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Combine results from multiple experiment runs into a single DataFrame.
    Each entry should have keys: experiment, algorithm, and standard metrics.
    """
    return pd.DataFrame(experiment_results)


# ──────────────────────────────────────────────────────────────
# Save helpers
# ──────────────────────────────────────────────────────────────

def save_results_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Results saved -> {path}")


def save_results_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results saved -> {path}")


def save_convergence_history(
    histories: Dict[str, List[float]],
    path: str,
) -> None:
    """Save convergence histories for all algorithms to CSV."""
    max_len = max(len(h) for h in histories.values()) if histories else 0
    data = {"iteration": list(range(max_len))}
    for name, hist in histories.items():
        padded = list(hist) + [hist[-1]] * (max_len - len(hist)) if hist else [0] * max_len
        data[name] = padded
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Convergence history saved -> {path}")


# ──────────────────────────────────────────────────────────────
# Pretty print
# ──────────────────────────────────────────────────────────────

def print_comparison_table(df: pd.DataFrame) -> None:
    """Print the comparison table nicely to stdout."""
    print("\n" + "=" * 100)
    print("  ALGORITHM COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100 + "\n")
