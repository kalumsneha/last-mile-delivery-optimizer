import os
from backend.benchmark_runner import run_benchmark
from backend.report_generator import generate_report

# JSON Solomon datasets
dataset_files = [
    "data/R101.json",
    "data/C101.json",
    "data/RC101.json"
]

# Algorithms to benchmark
algorithms = ["ga", "aco", "hybrid"]

# Output folder for saving results
output_folder = "benchmark_results"

# Run benchmarks
for algo in algorithms:
    for file in dataset_files:
        print(f"\u25B6\ufe0f Running {algo.upper()} on {file}")
        run_benchmark(
            dataset_path=file,
            algorithm=algo,
            output_dir=output_folder,
            run_id=None  # uses timestamp
        )

# Generate summary plots + tables
generate_report(
    csv_path=os.path.join(output_folder, "results.csv"),
    output_dir=os.path.join(output_folder, "reports")
)