
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_report(csv_path='backend/results.csv', output_dir='backend/reports'):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Summary table by algorithm
    summary = df.groupby("optimizer").agg({
        "fitness": "mean",
        "distance_km": "mean",
        "travel_time_sec": "mean",
        "cost": "mean",
        "runtime_sec": "mean"
    }).reset_index()
    print("\nAverage Metrics by Optimizer:")
    print(summary)

    # Save summary to CSV
    summary.to_csv(os.path.join(output_dir, "summary_by_optimizer.csv"), index=False)

    # Plot: Bar chart for distance, time, cost
    plt.figure(figsize=(12, 6))
    df["travel_time_min"] = df["travel_time_sec"] / 60
    melt_df = df.melt(id_vars=["optimizer"], value_vars=["distance_km", "travel_time_min", "cost"])

    sns.barplot(data=melt_df, x="optimizer", y="value", hue="variable")
    plt.title("Comparison of Distance (km), Travel Time (min), and Cost ($)")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=0)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "optimizer_comparison.png")
    plt.savefig(bar_path)
    plt.close()

    print(f"✅ Plot saved to {bar_path}")

    # Optional: per-dataset radar chart
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset]
        if len(subset) < 3:
            continue

        radar_data = subset.groupby("optimizer")[["distance_km", "travel_time_sec", "cost"]].mean()

        radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-6)

        labels = radar_data.columns.tolist()
        angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
        angles += angles[:1]

        plt.figure(figsize=(6, 6))
        for optimizer in radar_data.index:
            values = radar_data.loc[optimizer].values.flatten().tolist()
            values += values[:1]
            plt.polar(angles, values, label=optimizer)
        plt.xticks(angles[:-1], labels)
        plt.title(f"Radar Chart – {dataset}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        radar_path = os.path.join(output_dir, f"radar_{dataset.replace('.json','')}.png")
        plt.savefig(radar_path)
        plt.close()

    print(f"✅ Radar charts and summaries saved to {output_dir}")

if __name__ == "__main__":
    generate_report()
