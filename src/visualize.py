from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from tqdm import tqdm

import pandas as pd

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - optional dependency
    px = None


def plot_timers(timer: Dict[str, float]) -> None:
    x = list(timer.keys())
    y = list(timer.values())
    sns.barplot(x=x, y=y)


def plot_grid(
    img: np.ndarray,
    filepath: Path,
    k: int = 3,
    figsize: Tuple = (10, 10),
    title: str = "",
) -> None:
    fig, axs = plt.subplots(k, k, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    axs = axs.ravel()
    for i in tqdm(range(k * k)):
        axs[i].imshow(img[i], cmap="gray")
        axs[i].axis("off")
    fig.savefig(filepath)
    logger.success(f"saved grid to {filepath}")


# Function to plot images
def plot_categories(
    images, class_names, figsize: Tuple = (16, 15), filepath: Optional[Path] = None
):
    fig, axes = plt.subplots(1, 11, figsize=figsize)
    axes = axes.flatten()

    # Plot an empty canvas
    ax = axes[0]
    dummy_array = np.array([[[0, 0, 0, 0]]], dtype="uint8")
    ax.set_title("reference")
    ax.set_axis_off()
    ax.imshow(dummy_array, interpolation="nearest")

    # Plot an image for every category
    for k, v in images.items():
        ax = axes[k + 1]
        ax.imshow(v, cmap=plt.cm.binary)
        ax.set_title(f"{class_names[k]}")
        ax.set_axis_off()

    if filepath is not None:
        fig.savefig(filepath)
        logger.success(f"saved grid to {filepath}")
    else:
        plt.tight_layout()
        plt.show()


def parallel_plot(analysis, columns: list[str]):
    if px is None:
        raise ImportError("plotly is required for parallel_plot")
    plot = analysis.results_df
    p = plot[columns].reset_index()
    return px.parallel_coordinates(p, color="accuracy")


def plot_heatmap_from_csv(
    csv_path: Path | str,
    *,
    value_col: str,
    row: str,
    column: str,
    agg: str = "mean",
    filters: Optional[Dict[str, object]] = None,
    title: Optional[str] = None,
    annotate: bool = True,
    decimals: int = 3,
    cmap: str = "magma",
    figsize: Optional[Tuple[int, int]] = None,
    output_path: Optional[Path] = None,
) -> sns.Axes:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if filters:
        for key, value in filters.items():
            df = df[df[key] == value]
    if df.empty:
        raise ValueError("No rows left after applying filters")
    pivot = df.pivot_table(index=row, columns=column, values=value_col, aggfunc=agg)
    if pivot.empty:
        raise ValueError("Pivot table produced no data; check row/column arguments")
    pivot = pivot.sort_index().sort_index(axis=1)
    if figsize is None:
        width = max(6, 1 + 1.2 * len(pivot.columns))
        height = max(4, 1 + 1.0 * len(pivot.index))
        figsize = (int(width), int(height))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=annotate,
        fmt=f".{decimals}f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": value_col},
    )
    ax.set_title(title or f"{value_col} heatmap")
    ax.set_xlabel(column)
    ax.set_ylabel(row)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        logger.success(f"saved heatmap to {output_path}")
    return ax