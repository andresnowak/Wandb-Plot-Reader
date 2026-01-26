"""
Plotting functions for W&B data visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import wandb
    from wandb_plot_reader import WandbPlotReader


def plot_line(
    reader: "WandbPlotReader",
    run: Union[str, "wandb.apis.public.Run"],
    y_keys: Union[str, List[str]],
    x_key: str = "_step",
    project: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a line plot from run history.

    Args:
        reader: WandbPlotReader instance.
        run: Run ID or Run object.
        y_keys: Key(s) to plot on y-axis.
        x_key: Key for x-axis (default: _step).
        project: Project name.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        legend: Whether to show legend.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.plot().

    Returns:
        Matplotlib Axes object.
    """
    if isinstance(y_keys, str):
        y_keys = [y_keys]

    keys_to_fetch = [x_key] + y_keys if x_key != "_step" else y_keys
    history = reader.get_run_history(run, keys=keys_to_fetch, project=project)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for y_key in y_keys:
        if y_key in history.columns:
            ax.plot(history[x_key], history[y_key], label=y_key, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or (y_keys[0] if len(y_keys) == 1 else "Value"))
    if legend and len(y_keys) > 1:
        ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_bar(
    reader: "WandbPlotReader",
    runs: Optional[List[Union[str, "wandb.apis.public.Run"]]] = None,
    metric: str = "accuracy",
    project: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    use_run_names: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Create a bar plot comparing a metric across runs.

    Args:
        reader: WandbPlotReader instance.
        runs: List of Run IDs or Run objects. If None, fetches all runs.
        metric: Metric key to compare.
        project: Project name.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        color: Bar color.
        ax: Existing axes to plot on.
        use_run_names: Use run names instead of IDs for labels.
        **kwargs: Additional arguments passed to plt.bar().

    Returns:
        Matplotlib Axes object.
    """
    if runs is None:
        runs = reader.get_runs(project=project)

    values = []
    labels = []

    for run in runs:
        if isinstance(run, str):
            run = reader.get_run(run, project=project)

        summary = dict(run.summary)
        if metric in summary:
            values.append(summary[metric])
            labels.append(run.name if use_run_names else run.id)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.bar(labels, values, color=color, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "Run")
    ax.set_ylabel(ylabel or metric)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    return ax


def plot_scatter(
    reader: "WandbPlotReader",
    run: Union[str, "wandb.apis.public.Run"],
    x_key: str,
    y_key: str,
    project: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    color_key: Optional[str] = None,
    size_key: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a scatter plot from run history.

    Args:
        reader: WandbPlotReader instance.
        run: Run ID or Run object.
        x_key: Key for x-axis values.
        y_key: Key for y-axis values.
        project: Project name.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        color_key: Optional key for color mapping.
        size_key: Optional key for size mapping.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.scatter().

    Returns:
        Matplotlib Axes object.
    """
    keys = [x_key, y_key]
    if color_key:
        keys.append(color_key)
    if size_key:
        keys.append(size_key)

    history = reader.get_run_history(run, keys=keys, project=project)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    scatter_kwargs = kwargs.copy()
    if color_key and color_key in history.columns:
        scatter_kwargs["c"] = history[color_key]
        scatter_kwargs["cmap"] = scatter_kwargs.get("cmap", "viridis")
    if size_key and size_key in history.columns:
        scatter_kwargs["s"] = history[size_key]

    scatter = ax.scatter(history[x_key], history[y_key], **scatter_kwargs)

    if color_key and color_key in history.columns:
        plt.colorbar(scatter, ax=ax, label=color_key)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.grid(True, alpha=0.3)

    return ax


def plot_histogram(
    reader: "WandbPlotReader",
    run: Union[str, "wandb.apis.public.Run"],
    key: str,
    project: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    bins: Union[int, str] = "auto",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a histogram from run history.

    Args:
        reader: WandbPlotReader instance.
        run: Run ID or Run object.
        key: Key for values to histogram.
        project: Project name.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        bins: Number of bins or binning strategy.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.hist().

    Returns:
        Matplotlib Axes object.
    """
    history = reader.get_run_history(run, keys=[key], project=project)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.hist(history[key].dropna(), bins=bins, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or key)
    ax.set_ylabel(ylabel or "Frequency")
    ax.grid(True, alpha=0.3)

    return ax


def plot_multi_run_lines(
    reader: "WandbPlotReader",
    runs: Optional[List[Union[str, "wandb.apis.public.Run"]]] = None,
    y_key: str = "loss",
    x_key: str = "_step",
    project: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create overlaid line plots for multiple runs.

    Args:
        reader: WandbPlotReader instance.
        runs: List of Run IDs or Run objects. If None, fetches all runs.
        y_key: Key for y-axis values.
        x_key: Key for x-axis values.
        project: Project name.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.plot().

    Returns:
        Matplotlib Axes object.
    """
    if runs is None:
        runs = reader.get_runs(project=project)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for run in runs:
        if isinstance(run, str):
            run_obj = reader.get_run(run, project=project)
        else:
            run_obj = run

        history = run_obj.history(keys=[y_key] if x_key == "_step" else [x_key, y_key])
        if y_key in history.columns:
            ax.plot(history[x_key], history[y_key], label=run_obj.name, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_heatmap(
    reader: "WandbPlotReader",
    runs: Optional[List[Union[str, "wandb.apis.public.Run"]]] = None,
    metrics: Optional[List[str]] = None,
    project: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a heatmap of metrics across runs.

    Args:
        reader: WandbPlotReader instance.
        runs: List of Run IDs or Run objects.
        metrics: List of metric keys to include.
        project: Project name.
        title: Plot title.
        figsize: Figure size tuple.
        cmap: Colormap name.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib Axes object.
    """
    if runs is None:
        runs = reader.get_runs(project=project)

    data = []
    run_names = []

    for run in runs:
        if isinstance(run, str):
            run_obj = reader.get_run(run, project=project)
        else:
            run_obj = run

        summary = dict(run_obj.summary)
        run_names.append(run_obj.name)

        if metrics is None:
            metrics = [
                k
                for k in summary.keys()
                if isinstance(summary.get(k), (int, float)) and not k.startswith("_")
            ]

        row = [summary.get(m, np.nan) for m in metrics]
        data.append(row)

    data_array = np.array(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data_array, cmap=cmap, aspect="auto", **kwargs)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(run_names)))
    ax.set_yticklabels(run_names)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax
