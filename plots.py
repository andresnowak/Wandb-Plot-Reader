"""
Plotting functions for data visualization.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Union, Dict
from numpy.typing import ArrayLike


def plot_line(
    data: List[Dict[str, ArrayLike]],
    x_key: str = "x",
    y_key: str = "y",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    legend: bool = True,
    smooth: Optional[float] = None,
    ylim: Optional[tuple] = None,
    y_step: Optional[float] = None,
    cmap: Optional[str] = "Paired",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a line plot from data.

    Args:
        data: List of dicts, each containing x, y arrays and optional 'label'.
              Example: [{'x': [1,2,3], 'y': [4,5,6], 'label': 'run1'}, ...]
        x_key: Key for x values in data dicts (default: 'x').
        y_key: Key for y values in data dicts (default: 'y').
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        legend: Whether to show legend.
        smooth: Smoothing factor (0-1). Higher values = smoother lines.
                Uses exponential moving average. None for no smoothing.
        ylim: Y-axis limits as (min, max) tuple.
        y_step: Step size between y-axis ticks.
        cmap: Colormap name to use for line colors (e.g., 'viridis', 'plasma').
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.plot().

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Calculate total number of lines for colormap
    total_lines = len(data)
    colors = None
    if cmap is not None:
        colormap = plt.cm.get_cmap(cmap)
        colors = [colormap(i / max(total_lines - 1, 1)) for i in range(total_lines)]

    for idx, item in enumerate(data):
        x_values = np.asarray(item.get(x_key, item.get("x", [])))
        y_values = np.asarray(item.get(y_key, item.get("y", [])))
        label = item.get("label", None)

        if smooth is not None and 0 < smooth < 1:
            # Exponential moving average smoothing
            smoothed = np.zeros_like(y_values, dtype=float)
            smoothed[0] = y_values[0]
            for i in range(1, len(y_values)):
                if np.isnan(y_values[i]):
                    smoothed[i] = smoothed[i - 1]
                else:
                    smoothed[i] = smooth * smoothed[i - 1] + (1 - smooth) * y_values[i]
            y_values = smoothed

        plot_kwargs = kwargs.copy()
        if colors is not None:
            plot_kwargs["color"] = colors[idx]
        ax.plot(x_values, y_values, label=label, **plot_kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)

    if ylim is not None and len(ylim) == 2:
        ax.set_ylim(*ylim)
    if y_step is not None:
        y_min, y_max = ax.get_ylim()
        ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))

    if legend and any(item.get("label") for item in data):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize="small", ncol=1)
        ax.figure.subplots_adjust(bottom=0.25)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_bar(
    labels: List[str],
    values: List[float],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    color: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a bar plot.

    Args:
        labels: List of labels for each bar.
        values: List of values for each bar.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        color: Bar color.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.bar().

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.bar(labels, values, color=color, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    return ax


def plot_scatter(
    x: ArrayLike,
    y: ArrayLike,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    c: Optional[ArrayLike] = None,
    s: Optional[ArrayLike] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a scatter plot.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple.
        c: Values for color mapping.
        s: Values for size mapping.
        cmap: Colormap name.
        colorbar_label: Label for colorbar.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.scatter().

    Returns:
        Matplotlib Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    scatter_kwargs = kwargs.copy()
    if c is not None:
        scatter_kwargs["c"] = c
        scatter_kwargs["cmap"] = cmap
    if s is not None:
        scatter_kwargs["s"] = s

    scatter = ax.scatter(x, y, **scatter_kwargs)

    if c is not None:
        plt.colorbar(scatter, ax=ax, label=colorbar_label)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.grid(True, alpha=0.3)

    return ax


def plot_histogram(
    values: ArrayLike,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: tuple = (10, 6),
    bins: Union[int, str] = "auto",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a histogram.

    Args:
        values: Values to histogram.
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
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Remove NaN values
    values = np.asarray(values)
    values = values[~np.isnan(values)]

    ax.hist(values, bins=bins, **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "Frequency")
    ax.grid(True, alpha=0.3)

    return ax


def plot_heatmap(
    data: ArrayLike,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 8),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a heatmap.

    Args:
        data: 2D array of values.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        title: Plot title.
        figsize: Figure size tuple.
        cmap: Colormap name.
        ax: Existing axes to plot on.
        **kwargs: Additional arguments passed to plt.imshow().

    Returns:
        Matplotlib Axes object.
    """
    data = np.asarray(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
    plt.colorbar(im, ax=ax)

    if col_labels is not None:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    if row_labels is not None:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax
