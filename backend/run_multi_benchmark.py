# backend/run_multi_benchmark.py

from backend.benchmark_runner import run_benchmark
from backend.report_generator import generate_report

import os

DATASETS = [
    "data/C101.json",
    "data/RC101.json",
    "data/R101.json"
]

ALGORITHMS = ["ga", "aco", "hybrid"]
REPEATS = 10
OUTPUT_FOLDER = "benchmark_results"

for dataset in DATASETS:
    for algo in ALGORITHMS:
        for run in range(REPEATS):
            print(f"▶️ Running {algo.upper()} on {dataset} (Run {run + 1}/{REPEATS})")
            run_benchmark(dataset_path=dataset, algorithm=algo, output_dir=OUTPUT_FOLDER)

# Final report generation
generate_report(csv_path=f"{OUTPUT_FOLDER}/results.csv", output_dir=f"{OUTPUT_FOLDER}/reports")
