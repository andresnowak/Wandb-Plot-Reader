"""
Example usage of the W&B Plot Reader.

Make sure you have wandb installed and are logged in:
    pip install wandb matplotlib pandas numpy
    wandb login
"""

from wandb_plot_reader import WandbPlotReader
from plots import (
    plot_line,
    plot_bar,
    plot_scatter,
    plot_histogram,
    plot_heatmap,
)
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Initialize the reader with your entity and project
    # Replace these with your actual W&B entity and project names
    reader = WandbPlotReader(entity="your-entity", project="your-project")

    # Example 1: Line plot of training loss over steps
    print("Example 1: Line plot")
    run = reader.get_run("your-run-id")
    history = reader.get_run_history(run, keys=["loss"])

    data = [{"x": history["_step"].values, "y": history["loss"].values, "label": "loss"}]
    ax = plot_line(
        data,
        title="Training Loss",
        xlabel="Step",
        ylabel="Loss",
    )
    plt.savefig("line_plot.png")
    plt.show()

    # Example 2: Multiple metrics on one line plot
    print("Example 2: Multi-metric line plot")
    history = reader.get_run_history(run, keys=["train_loss", "val_loss"])

    data = [
        {"x": history["_step"].values, "y": history["train_loss"].values, "label": "train_loss"},
        {"x": history["_step"].values, "y": history["val_loss"].values, "label": "val_loss"},
    ]
    ax = plot_line(data, title="Training vs Validation Loss")
    plt.savefig("multi_line_plot.png")
    plt.show()

    # Example 3: Bar plot comparing accuracy across runs
    print("Example 3: Bar plot")
    runs = reader.get_runs()[:5]  # Get first 5 runs
    labels = [r.name for r in runs]
    values = [dict(r.summary).get("accuracy", 0) for r in runs]

    ax = plot_bar(
        labels=labels,
        values=values,
        title="Accuracy Comparison Across Runs",
        ylabel="Accuracy",
        color="steelblue",
    )
    plt.savefig("bar_plot.png")
    plt.show()

    # Example 4: Scatter plot
    print("Example 4: Scatter plot")
    history = reader.get_run_history(run, keys=["learning_rate", "loss"])

    ax = plot_scatter(
        x=history["learning_rate"].values,
        y=history["loss"].values,
        title="Learning Rate vs Loss",
        xlabel="Learning Rate",
        ylabel="Loss",
        alpha=0.6,
    )
    plt.savefig("scatter_plot.png")
    plt.show()

    # Example 5: Histogram of a metric distribution
    print("Example 5: Histogram")
    history = reader.get_run_history(run, keys=["loss"])

    ax = plot_histogram(
        values=history["loss"].values,
        title="Loss Distribution",
        xlabel="Loss",
        bins=30,
        color="coral",
        edgecolor="black",
    )
    plt.savefig("histogram.png")
    plt.show()

    # Example 6: Compare multiple runs on the same plot
    print("Example 6: Multi-run comparison")
    runs = reader.get_runs()[:3]
    data = []
    for r in runs:
        h = reader.get_run_history(r, keys=["loss"])
        data.append({"x": h["_step"].values, "y": h["loss"].values, "label": r.name})

    ax = plot_line(data, title="Loss Comparison Across Runs", ylabel="Loss")
    plt.savefig("multi_run_lines.png")
    plt.show()

    # Example 7: Heatmap of metrics across runs
    print("Example 7: Heatmap")
    runs = reader.get_runs()[:5]
    metrics = ["accuracy", "loss", "f1_score"]
    run_names = [r.name for r in runs]

    heatmap_data = []
    for r in runs:
        summary = dict(r.summary)
        row = [summary.get(m, np.nan) for m in metrics]
        heatmap_data.append(row)

    ax = plot_heatmap(
        data=heatmap_data,
        row_labels=run_names,
        col_labels=metrics,
        title="Metrics Heatmap",
        cmap="coolwarm",
    )
    plt.savefig("heatmap.png")
    plt.show()

    # Example 8: Creating subplots
    print("Example 8: Subplots")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Line plot
    history = reader.get_run_history(run, keys=["loss"])
    data = [{"x": history["_step"].values, "y": history["loss"].values}]
    plot_line(data, ax=axes[0, 0], title="Training Loss", ylabel="Loss", cmap=None)

    # Another line plot
    history = reader.get_run_history(run, keys=["accuracy"])
    data = [{"x": history["_step"].values, "y": history["accuracy"].values}]
    plot_line(data, ax=axes[0, 1], title="Accuracy", ylabel="Accuracy", cmap=None)

    # Histogram
    history = reader.get_run_history(run, keys=["loss"])
    plot_histogram(history["loss"].values, ax=axes[1, 0], title="Loss Distribution")

    # Bar plot
    runs = reader.get_runs()[:4]
    labels = [r.name for r in runs]
    values = [dict(r.summary).get("accuracy", 0) for r in runs]
    plot_bar(labels, values, ax=axes[1, 1], title="Run Comparison", ylabel="Accuracy")

    plt.tight_layout()
    plt.savefig("subplots.png")
    plt.show()


if __name__ == "__main__":
    main()
