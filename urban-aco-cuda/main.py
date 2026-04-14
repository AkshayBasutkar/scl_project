import argparse
import sys
import logging
import os
import numpy as np

from config import ACOConfig
from utils import setup_logging, check_cuda_available, ensure_dirs, timed, set_seed
from data_loader import load_graph_data, generate_sample_graph
from graph_utils import build_csr_graph
from aco import DynamicMultiObjectiveACO
from baselines import dijkstra, cpu_aco, gpu_aco_baseline
from metrics import build_comparison_table, print_comparison_table, save_results_csv, save_results_json, save_convergence_history
from visualization import plot_convergence, plot_runtime_comparison, plot_cost_comparison, plot_speedup, plot_route_on_grid

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Urban Road Network Optimization using DM-CUDA-ACO")
    
    parser.add_argument("--source", type=int, default=0, help="Source node ID")
    parser.add_argument("--destination", type=int, default=99, help="Destination node ID")
    parser.add_argument("--nodes-file", type=str, default="data/nodes.csv", help="Path to nodes CSV")
    parser.add_argument("--edges-file", type=str, default="data/edges.csv", help="Path to edges CSV")
    parser.add_argument("--ants", type=int, default=256, help="Number of ants")
    parser.add_argument("--iterations", type=int, default=100, help="Max iterations")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample grid network")
    parser.add_argument("--grid-size", type=int, default=10, help="Grid size (N x N) for sample generation")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA explicitly")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    
    logger.info("="*50)
    logger.info("   Urban DM-CUDA-ACO Research Prototype")
    logger.info("="*50)

    # 1. Config & Setup
    cfg = ACOConfig()
    cfg.source_node = args.source
    cfg.destination_node = args.destination
    cfg.num_ants = args.ants
    cfg.max_iterations = args.iterations
    cfg.nodes_file = args.nodes_file
    cfg.edges_file = args.edges_file
    cfg.random_seed = args.seed
    
    if args.no_cuda:
        cfg.use_cuda = False
        
    set_seed(cfg.random_seed)
    ensure_dirs(cfg.output_dir, cfg.plots_dir, "data")
    
    # Check actual CUDA visibility
    has_cuda = check_cuda_available()
    if cfg.use_cuda and not has_cuda:
        logger.warning("CUDA requested but not available. Falling back to CPU for all runs.")
        cfg.use_cuda = False

    # 2. Data Loading
    if args.generate_sample or not (os.path.exists(cfg.nodes_file) and os.path.exists(cfg.edges_file)):
        logger.info(f"Generating {(args.grid_size)}x{(args.grid_size)} sample grid graph...")
        nodes_df, edges_df = generate_sample_graph(
            rows=args.grid_size, 
            cols=args.grid_size, 
            save_dir="data", 
            seed=cfg.random_seed
        )
        if args.source == 0 and args.destination == 99 and args.grid_size != 10:
            cfg.destination_node = (args.grid_size * args.grid_size) - 1
            logger.info(f"Adjusted destination to {cfg.destination_node} for new grid size")
    else:
        logger.info("Loading graph from files...")
        nodes_df, edges_df = load_graph_data(cfg)

    # Validate source/dest
    num_nodes = len(nodes_df)
    if cfg.source_node < 0 or cfg.source_node >= num_nodes:
        logger.error(f"Invalid source node {cfg.source_node}")
        sys.exit(1)
    if cfg.destination_node < 0 or cfg.destination_node >= num_nodes:
        logger.error(f"Invalid destination node {cfg.destination_node}")
        sys.exit(1)
        
    logger.info(f"Task: Route from {cfg.source_node} to {cfg.destination_node}")

    # Build GPU-friendly graph
    csr_graph = build_csr_graph(nodes_df, edges_df)

    # 3. Run Experiments (Comparisons)
    results_dict = {}
    convergence_histories = {}
    routes_dict = {}
    
    # ── A. Dijkstra (Baseline 1)
    logger.info("\n--- Running Dijkstra ---")
    res_dijkstra = dijkstra(
        csr_graph, cfg.source_node, cfg.destination_node,
        cfg.w_travel_time, cfg.w_distance, cfg.w_congestion, cfg.w_signal_delay, cfg.w_capacity
    )
    results_dict["Dijkstra"] = res_dijkstra.to_dict()
    routes_dict["Dijkstra"] = res_dijkstra.best_route
    
    # Check if a path even exists
    if not res_dijkstra.best_route:
        logger.error(f"No path exists between {cfg.source_node} and {cfg.destination_node}.")
        sys.exit(1)

    # ── B. CPU ACO (Baseline 2)
    logger.info("\n--- Running CPU ACO ---")
    res_cpu = cpu_aco(csr_graph, cfg.source_node, cfg.destination_node, cfg)
    results_dict["CPU-ACO"] = res_cpu.to_dict()
    convergence_histories["CPU-ACO"] = res_cpu.convergence_history
    routes_dict["CPU-ACO"] = res_cpu.best_route

    # ── C. GPU ACO Baseline (Baseline 3)
    if cfg.use_cuda:
        logger.info("\n--- Running GPU ACO Baseline ---")
        res_gpu_base = gpu_aco_baseline(csr_graph, cfg.source_node, cfg.destination_node, cfg)
        results_dict["GPU-ACO-Baseline"] = res_gpu_base.to_dict()
        convergence_histories["GPU-ACO-Baseline"] = res_gpu_base.convergence_history
        routes_dict["GPU-ACO-Baseline"] = res_gpu_base.best_route

    # ── D. Proposed DM-CUDA-ACO
    logger.info("\n--- Running DM-CUDA-ACO ---")
    dm_aco = DynamicMultiObjectiveACO(csr_graph, cfg)
    res_dm = dm_aco.solve(cfg.source_node, cfg.destination_node)
    
    results_dict["DM-CUDA-ACO"] = res_dm.to_dict()
    convergence_histories["DM-CUDA-ACO"] = res_dm.convergence_history
    routes_dict["DM-CUDA-ACO"] = res_dm.best_route

    # 4. Metrics & Evaluation
    logger.info("\nGenerating Metrics and Tables...")
    df_comparison = build_comparison_table(results_dict, reference_key="Dijkstra")
    print_comparison_table(df_comparison)
    
    # Save results
    save_results_csv(df_comparison, os.path.join(cfg.output_dir, "comparison.csv"))
    save_results_json(results_dict, os.path.join(cfg.output_dir, "results.json"))
    save_convergence_history(convergence_histories, os.path.join(cfg.output_dir, "convergence.csv"))
    
    # Save config mapping
    cfg.save(os.path.join(cfg.output_dir, "experiment_config.json"))

    # 5. Visualisation
    logger.info("Generating Paper Plots...")
    plot_convergence(convergence_histories, os.path.join(cfg.plots_dir, "convergence.png"))
    plot_runtime_comparison(results_dict, os.path.join(cfg.plots_dir, "runtime_comparison.png"))
    plot_cost_comparison(results_dict, os.path.join(cfg.plots_dir, "cost_comparison.png"))
    plot_speedup(results_dict, reference_key="CPU-ACO", save_path=os.path.join(cfg.plots_dir, "speedup.png"))
    plot_route_on_grid(nodes_df, routes_dict, os.path.join(cfg.plots_dir, "route_map.png"))

    # ══════════════════════════════════════════════════════════
    # 6. Traffic Scenario Experiments (low / normal / peak)
    # ══════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 50)
    logger.info("  TRAFFIC SCENARIO EXPERIMENTS")
    logger.info("=" * 50)

    import copy
    scenario_results = []
    traffic_scenarios = {
        "Low Traffic":   0.15,   # avg congestion ~0.15
        "Normal Traffic": 0.45,  # avg congestion ~0.45
        "Peak Traffic":  0.85,   # avg congestion ~0.85
    }

    # Save original congestion
    original_congestion = csr_graph.congestions.copy()

    for scenario_name, cong_level in traffic_scenarios.items():
        logger.info(f"\n--- Scenario: {scenario_name} (congestion ~{cong_level}) ---")
        
        # Set congestion to scenario level with small variance
        rng_scenario = np.random.RandomState(cfg.random_seed)
        csr_graph.congestions[:] = np.clip(
            rng_scenario.normal(cong_level, 0.1, size=csr_graph.num_edges),
            0.0, 1.0
        ).astype(np.float32)

        # Run DM-CUDA-ACO under this scenario
        scenario_aco = DynamicMultiObjectiveACO(csr_graph, cfg)
        res_scenario = scenario_aco.solve(cfg.source_node, cfg.destination_node)
        
        # Run Dijkstra under this scenario for reference
        res_dij = dijkstra(
            csr_graph, cfg.source_node, cfg.destination_node,
            cfg.w_travel_time, cfg.w_distance, cfg.w_congestion, cfg.w_signal_delay, cfg.w_capacity
        )

        scenario_results.append({
            "Scenario": scenario_name,
            "Congestion Level": cong_level,
            "Dijkstra Cost": round(res_dij.best_cost, 4),
            "DM-CUDA-ACO Cost": round(res_scenario.best_cost, 4),
            "ACO Travel Time": round(res_scenario.best_travel_time, 2),
            "ACO Distance": round(res_scenario.best_distance, 2),
            "ACO Runtime (ms)": round(res_scenario.runtime_ms, 1),
            "ACO Path Length": len(res_scenario.best_route),
            "Iterations": res_scenario.iterations_run,
        })

    # Restore original congestion
    csr_graph.congestions[:] = original_congestion

    # Print & save traffic scenario results
    import pandas as pd
    df_scenarios = pd.DataFrame(scenario_results)
    print("\n" + "=" * 90)
    print("  TRAFFIC SCENARIO RESULTS")
    print("=" * 90)
    print(df_scenarios.to_string(index=False))
    print("=" * 90)
    save_results_csv(df_scenarios, os.path.join(cfg.output_dir, "traffic_scenarios.csv"))

    # Plot: Cost vs Congestion Level
    from visualization import plot_parameter_sweep
    cong_levels = [s["Congestion Level"] for s in scenario_results]
    plot_parameter_sweep(
        cong_levels,
        {
            "Dijkstra": [s["Dijkstra Cost"] for s in scenario_results],
            "DM-CUDA-ACO": [s["DM-CUDA-ACO Cost"] for s in scenario_results],
        },
        "Congestion Level", "Best Route Cost",
        "Route Cost vs Traffic Congestion Level",
        os.path.join(cfg.plots_dir, "cost_vs_congestion.png"),
    )
    
    logger.info("\nAll Experiments Complete! Results are in /outputs and /plots.")

if __name__ == "__main__":
    main()
