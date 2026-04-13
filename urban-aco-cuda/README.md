# Urban Road Network Optimization using DM-CUDA-ACO

This repository contains the prototype implementation for the research paper: **"Urban Road Network Optimization using Dynamic Multi-Objective CUDA Ant Colony Optimization"**.

It implements a parallel, GPU-accelerated Ant Colony Optimization route-finding algorithm handling multiple objectives (travel time, distance, congestion, and signal delay) with dynamic traffic adjustments.

## Features

1. **Algorithm Improvements**: Multi-objective routing (time, distance, congestion, signal delay), adaptive pheromone evaporation, elitist best-route boosting.
2. **CUDA Acceleration**: Parallel route construction, score evaluation, and pheromone updates per edge/ant using Numba CUDA.
3. **Dynamic Traffic**: Traffic conditions simulated to change during routing, boosting congestion on popular paths.
4. **Research Infrastructure**: Complete comparison pipeline against CPU Dijkstra, CPU ACO, and Vanilla GPU ACO. Automated CSV telemetry and graph plotting.

## Installation

Ensure you have Python 3.10+ installed.

```bash
pip install -r requirements.txt
```

If you have an NVIDIA GPU, ensure the appropriate CUDA toolkit is installed so Numba can access it. If no GPU is available, the codebase safely falls back to pure CPU computation.

## Project Structure

- `main.py`: Entrypoint for running the experiments.
- `aco.py`: The proposed Dynamic Multi-Objective ACO algorithm structure.
- `cuda_kernels.py`: The Numba CUDA kernels and device-agnostic fallback abstractions.
- `graph_utils.py`: GPU-friendly CSR graph representation and tools.
- `metrics.py` & `visualization.py`: Logging, output generation, and publication plotting.
- `config.py`: Hyperparameters and weight settings.
- `data_loader.py`: Node/Edge CSV loading and sample-graph generator.
- `baselines.py`: Benchmark algorithms (Dijkstra, CPU-ACO, Baseline-GPU-ACO).

## Quick Start (Demo Network)

Run the script generating a randomized 10x10 city grid automatically to demonstrate functionality.

```bash
python main.py --generate-sample --grid-size 10 --source 0 --destination 99
```

## Running with the Full Urban Network Dataset

Prepare `data/nodes.csv` and `data/edges.csv` and run:

```bash
python main.py --nodes-file data/nodes.csv --edges-file data/edges.csv --source 0 --destination 50 --ants 512 --iterations 200
```

### Data Format

**nodes.csv**:
```csv
node_id,latitude,longitude
0,28.61,77.20
1,28.62,77.21
...
```

**edges.csv**:
```csv
source,target,distance,travel_time,capacity,signal_delay,congestion
0,1,1.5,2.0,1500,0.5,0.2
...
```

## Outputs

All experiment results stream to `/outputs` (for CSV tables and JSON results) and `/plots` (for convergence graphs, runtime comparison bar charts, and layout visualizations).
