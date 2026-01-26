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
import os

from env import ENTITY, PROJECT


def main():
    output_dir = "plots/"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the reader with your entity and project
    # Replace these with your actual W&B entity and project names
    reader = WandbPlotReader(entity=ENTITY, project=PROJECT)

    run_ids = [
        "Qwen3-7B-A1B-ZE8-TAU0.2-20260111_213819",
        "Qwen3-7B-A1B-ZE8-TAU0.3-20260104_011314",
        "Qwen3-7B-A1B-ZE0-TAU0-20260103_234430",
    ]

    # Load run data once
    runs_data = []
    for run_id in run_ids:
        run = reader.get_run(run_id)
        history = reader.get_run_history(
            run, keys=["lm loss", "expert_max_violation", "load_balancing_loss", "TFLOPs-per-GPU"]
        )
        runs_data.append({"run": run, "history": history})

    # Example 1: Line plot of training loss over steps
    print("Example 1: Line plot - LM Loss")
    data = [
        {"x": rd["history"]["_step"].values, "y": rd["history"]["lm loss"].values, "label": rd["run"].name}
        for rd in runs_data
    ]
    ax = plot_line(
        data,
        title="Training language modelling Loss",
        xlabel="Step",
        ylabel="Loss",
    )
    plt.savefig(os.path.join(output_dir, "lm_loss_line_plot.png"))
    # plt.show()

    print("Example 2: Line plot - Expert Max Violation")
    data = [
        {"x": rd["history"]["_step"].values, "y": rd["history"]["expert_max_violation"].values, "label": rd["run"].name}
        for rd in runs_data
    ]
    ax = plot_line(
        data,
        title="Expert Max Violation",
        xlabel="Step",
        ylabel="Expert Max Violation",
        y_step=0.5,
        ylim=(0, 12),
    )
    plt.savefig(os.path.join(output_dir, "expert_max_violation_line_plot.png"))
    # plt.show()

    print("Example 3: Line plot - Load Balancing Loss")
    data = [
        {"x": rd["history"]["_step"].values, "y": rd["history"]["load_balancing_loss"].values, "label": rd["run"].name}
        for rd in runs_data
    ]
    ax = plot_line(
        data,
        title="Load Balancing Loss",
        xlabel="Step",
        ylabel="Load Balancing Loss",
    )
    plt.savefig(os.path.join(output_dir, "load_balancing_loss_line_plot.png"))
    # plt.show()

    print("Example 4: Line plot - TFLOPs per GPU")
    data = [
        {"x": rd["history"]["_step"].values, "y": rd["history"]["TFLOPs-per-GPU"].values, "label": rd["run"].name}
        for rd in runs_data
    ]
    ax = plot_line(
        data,
        title="TFLOPs per GPU",
        xlabel="Step",
        ylabel="TFLOPs per GPU",
    )
    plt.savefig(os.path.join(output_dir, "tflops_per_gpu_line_plot.png"))
    # plt.show()


if __name__ == "__main__":
    main()
