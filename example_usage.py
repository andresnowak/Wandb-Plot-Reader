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
    plot_multi_run_lines,
    plot_heatmap,
)
import matplotlib.pyplot as plt


def main():
    # Initialize the reader with your entity and project
    # Replace these with your actual W&B entity and project names
    reader = WandbPlotReader(entity="your-entity", project="your-project")

    # Example 1: Line plot of training loss over steps
    print("Example 1: Line plot")
    ax = plot_line(
        reader,
        run="your-run-id",
        y_keys="loss",
        title="Training Loss",
        xlabel="Step",
        ylabel="Loss",
    )
    plt.savefig("line_plot.png")
    plt.show()

    # Example 2: Multiple metrics on one line plot
    print("Example 2: Multi-metric line plot")
    ax = plot_line(
        reader,
        run="your-run-id",
        y_keys=["train_loss", "val_loss"],
        title="Training vs Validation Loss",
    )
    plt.savefig("multi_line_plot.png")
    plt.show()

    # Example 3: Bar plot comparing accuracy across runs
    print("Example 3: Bar plot")
    runs = reader.get_runs()[:5]  # Get first 5 runs
    ax = plot_bar(
        reader,
        runs=runs,
        metric="accuracy",
        title="Accuracy Comparison Across Runs",
        color="steelblue",
    )
    plt.savefig("bar_plot.png")
    plt.show()

    # Example 4: Scatter plot
    print("Example 4: Scatter plot")
    ax = plot_scatter(
        reader,
        run="your-run-id",
        x_key="learning_rate",
        y_key="loss",
        title="Learning Rate vs Loss",
        alpha=0.6,
    )
    plt.savefig("scatter_plot.png")
    plt.show()

    # Example 5: Histogram of a metric distribution
    print("Example 5: Histogram")
    ax = plot_histogram(
        reader,
        run="your-run-id",
        key="loss",
        title="Loss Distribution",
        bins=30,
        color="coral",
        edgecolor="black",
    )
    plt.savefig("histogram.png")
    plt.show()

    # Example 6: Compare multiple runs on the same plot
    print("Example 6: Multi-run comparison")
    runs = reader.get_runs()[:3]
    ax = plot_multi_run_lines(
        reader,
        runs=runs,
        y_key="loss",
        title="Loss Comparison Across Runs",
    )
    plt.savefig("multi_run_lines.png")
    plt.show()

    # Example 7: Heatmap of metrics across runs
    print("Example 7: Heatmap")
    runs = reader.get_runs()[:5]
    ax = plot_heatmap(
        reader,
        runs=runs,
        metrics=["accuracy", "loss", "f1_score"],
        title="Metrics Heatmap",
        cmap="coolwarm",
    )
    plt.savefig("heatmap.png")
    plt.show()

    # Example 8: Creating subplots
    print("Example 8: Subplots")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    plot_line(
        reader,
        run="your-run-id",
        y_keys="loss",
        ax=axes[0, 0],
        title="Training Loss",
    )

    plot_line(
        reader,
        run="your-run-id",
        y_keys="accuracy",
        ax=axes[0, 1],
        title="Accuracy",
    )

    plot_histogram(
        reader,
        run="your-run-id",
        key="loss",
        ax=axes[1, 0],
        title="Loss Distribution",
    )

    runs = reader.get_runs()[:4]
    plot_bar(
        reader,
        runs=runs,
        metric="accuracy",
        ax=axes[1, 1],
        title="Run Comparison",
    )

    plt.tight_layout()
    plt.savefig("subplots.png")
    plt.show()


if __name__ == "__main__":
    main()
