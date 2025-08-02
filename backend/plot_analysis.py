# backend/plot_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_boxplots(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    metrics = ["fitness", "distance_km", "travel_time_sec", "cost"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="optimizer", y=metric, palette="Set2")
        plt.title(f"Distribution of {metric} by Optimizer")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/boxplot_{metric}.png")
        plt.close()
        print(f"✅ Boxplot saved for {metric}")

if __name__ == "__main__":
    generate_boxplots("benchmark_results/results.csv", "benchmark_results/plots")
