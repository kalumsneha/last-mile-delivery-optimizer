
import os
import json
import time
import csv

from backend.optimizer import optimize_routes, evaluate_with_details
from backend.aco_optimizer import aco_optimize_routes
from backend.hybrid_optimizer import hybrid_optimize_routes
from types import SimpleNamespace

def load_benchmark(filepath):
    with open(filepath, 'r') as f:
        raw = json.load(f)
    base = SimpleNamespace(**raw['base'])
    points = [SimpleNamespace(**p) for p in raw['points']]
    return base, points

def point_dict(points, base):
    all_points = [base] + points + [base]
    return {p.id: vars(p) for p in all_points}

def run_and_evaluate(label, func, *args, **kwargs):
    start = time.time()
    route, _ = func(*args, **kwargs)
    duration = time.time() - start
    point_map = kwargs['point_map']
    fitness, breakdown = evaluate_with_details(route, point_map)
    total_distance = sum(seg['distance_km'] for seg in breakdown)
    total_travel_time = sum(seg['travel_time_sec'] for seg in breakdown)
    total_cost = sum(seg['segment_cost'] for seg in breakdown)
    return {
        "optimizer": label,
        "fitness": fitness,
        "distance_km": total_distance,
        "travel_time_sec": total_travel_time,
        "cost": total_cost,
        "runtime_sec": duration
    }

def batch_run(data_dir='backend/../data', output_csv='backend/results.csv'):
    results = []
    for file in os.listdir(data_dir):
        if not file.endswith('.json'):
            continue
        benchmark_path = os.path.join(data_dir, file)
        base, points = load_benchmark(benchmark_path)
        pmap = point_dict(points, base)

        print(f"Running benchmark: {file}")
        for label, func in [
            ("GA", optimize_routes),
            ("ACO", aco_optimize_routes),
            ("Hybrid", hybrid_optimize_routes)
        ]:
            print(f"  -> {label}")
            result = run_and_evaluate(label, func, points, base_point=base, point_map=pmap)
            result["dataset"] = file
            results.append(result)

    keys = ["dataset", "optimizer", "fitness", "distance_km", "travel_time_sec", "cost", "runtime_sec"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"✅ Batch benchmarking complete. Results saved to {output_csv}")

if __name__ == "__main__":
    batch_run()
