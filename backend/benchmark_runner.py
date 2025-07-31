# backend/benchmark_runner.py

import json
import time
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
from types import SimpleNamespace

from backend.optimizer import optimize_routes, evaluate_with_details
from backend.aco_optimizer import aco_optimize_routes
from backend.hybrid_optimizer import hybrid_optimize_routes

def load_json_points(filepath):
    with open(filepath, 'r') as f:
        raw = json.load(f)
    # Assume raw is a list of customer dicts
    base = SimpleNamespace(id=0, lat=43.651070, lon=-79.347015, label="Depot", time_window=None)
    points = [SimpleNamespace(**p) for p in raw]
    return base, points

def run_benchmark(dataset_path, algorithm, output_dir="benchmark_results", run_id=None, return_history=False):
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_point, delivery_points = load_json_points(dataset_path)

    all_points = delivery_points + [base_point]
    point_map = {p.id: vars(p) for p in all_points}

    if algorithm == "ga":
        # Convert everything to dicts to match Streamlit behavior
        delivery_points_dicts = [vars(p) for p in delivery_points]
        base_point_dict = vars(base_point)
        result = optimize_routes(delivery_points_dicts, base_point_dict)
        best_route = result["best_route"]
        breakdown = result["route_breakdown"]
    elif algorithm == "aco":
        if return_history:
            best_route, total_cost, history, fitness_over_time, pheromone_history, metrics_over_time = aco_optimize_routes(
                delivery_points, base_point, point_map, return_history=True
            )
        else:
            best_route, total_cost = aco_optimize_routes(
                delivery_points, base_point, point_map, return_history=False
            )

        _, breakdown, _ = evaluate_with_details(best_route, point_map)
    elif algorithm == "hybrid":
        best_route, _, _, _, _ = hybrid_optimize_routes(delivery_points, base_point, point_map, return_history=False)
        _, breakdown, _ = evaluate_with_details(best_route, point_map)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    fitness, _, metrics = evaluate_with_details(best_route, point_map)

    row = {
        "dataset": os.path.basename(dataset_path),
        "optimizer": algorithm,
        "fitness": fitness,
        "distance_km": metrics["total_distance_km"],
        "travel_time_sec": metrics["total_time"],
        "cost": metrics["total_cost"],
        "runtime_sec": 0,
        "run_id": run_id
    }

    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "results.csv")
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
