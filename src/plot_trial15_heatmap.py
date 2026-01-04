"""Generate a heatmap comparing validation accuracies for six trial-15 variants."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from loguru import logger

# Reuse training helpers for consistent preprocessing and evaluation.
from hypertuning import (  # type: ignore import-not-found
    build_model,
    build_transforms,
    evaluate_dual_modes,
    prepare_streamers,
    select_device,
)


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HEATMAP_DIR = ROOT / "data" / "processed" / "heatmaps"
OUTPUT_CSV = HEATMAP_DIR / "trial15_epoch100_patience15_metrics.csv"
OUTPUT_PNG = HEATMAP_DIR / "trial15_epoch100_patience15_heatmap.png"

# Hyperparameters replicated from trial 15.
HEAD_DEPTH = 2
HEAD_WIDTH_FACTOR = 0.9388607111568091
HEAD_DROPOUT = 0.26087065725099706
UNFREEZE_BLOCKS = 1
BATCH_SIZE = 32
NUM_CLASSES = 5

VariantSpec = dict[str, object]

VARIANTS: list[VariantSpec] = [
    {
        "label": "RGB",
        "checkpoint": MODELS_DIR / "flowers_resnet18_color_trial15_rgb_epoch100_patience15.pt",
        "grayscale": False,
        "color_space": "rgb",
        "edge_shift": False,
        "edge_shift_pixels": None,
    },
    {
        "label": "LAB",
        "checkpoint": MODELS_DIR / "flowers_resnet18_color_trial15_lab_epoch100_patience15.pt",
        "grayscale": False,
        "color_space": "lab",
        "edge_shift": False,
        "edge_shift_pixels": None,
    },
    {
        "label": "Grayscale",
        "checkpoint": MODELS_DIR / "flowers_resnet18_grayscale_trial15_epoch100_patience15.pt",
        "grayscale": True,
        "color_space": "rgb",
        "edge_shift": False,
        "edge_shift_pixels": None,
    },
    {
        "label": "EdgeShift px1",
        "checkpoint": MODELS_DIR / "flowers_resnet18_color_trial15_edge_shift_px1_epoch100_patience15.pt",
        "grayscale": False,
        "color_space": "rgb",
        "edge_shift": True,
        "edge_shift_pixels": [1],
    },
    {
        "label": "EdgeShift px2",
        "checkpoint": MODELS_DIR / "flowers_resnet18_color_trial15_edge_shift_px2_epoch100_patience15.pt",
        "grayscale": False,
        "color_space": "rgb",
        "edge_shift": True,
        "edge_shift_pixels": [2],
    },
    {
        "label": "EdgeShift px3",
        "checkpoint": MODELS_DIR / "flowers_resnet18_color_trial15_edge_shift_px3_epoch100_patience15.pt",
        "grayscale": False,
        "color_space": "rgb",
        "edge_shift": True,
        "edge_shift_pixels": [3],
    },
]


def _as_iterable(value: object) -> Iterable[int] | None:
    if value is None:
        return None
    if isinstance(value, Iterable):
        return value  # type: ignore[return-value]
    return [int(value)]


def evaluate_variant(spec: VariantSpec, device: torch.device) -> dict[str, float]:
    checkpoint_path = Path(spec["checkpoint"]).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    grayscale = bool(spec["grayscale"])
    transforms_map = build_transforms(
        grayscale,
        edge_shift=bool(spec["edge_shift"]),
        edge_shift_pixels=_as_iterable(spec["edge_shift_pixels"]),
        color_space=str(spec["color_space"]),
    )

    streamers = prepare_streamers(BATCH_SIZE, transforms_map, device, grayscale)
    valid_streamer = streamers["valid"]

    model = build_model(
        NUM_CLASSES,
        grayscale=grayscale,
        head_depth=HEAD_DEPTH,
        head_width_factor=HEAD_WIDTH_FACTOR,
        head_dropout=HEAD_DROPOUT,
        unfreeze_blocks=UNFREEZE_BLOCKS,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    results = evaluate_dual_modes(
        model=model,
        device=device,
        transforms_map=transforms_map,
        valid_streamer=valid_streamer,
        grayscale=grayscale,
    )
    return results


def main() -> None:
    device = select_device()
    logger.info(f"Evaluating checkpoints on {device}")

    records: list[dict[str, object]] = []
    for spec in VARIANTS:
        label = str(spec["label"])
        logger.info(f"Evaluating {label}")
        results = evaluate_variant(spec, device)
        for metric, value in results.items():
            display_metric = "Color" if metric == "val_color" else "Grayscale"
            records.append({"variant": label, "metric": display_metric, "accuracy": value})

    df = pd.DataFrame(records)
    df.sort_values(["metric", "variant"], inplace=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Saved metrics to {OUTPUT_CSV}")

    pivot = df.pivot(index="variant", columns="metric", values="accuracy")
    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="magma",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Validation Accuracy"},
    )
    ax.set_xlabel("Evaluation Mode")
    ax.set_ylabel("Variant")
    ax.set_title("Trial 15 Variants (100 epochs, patience 15)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    logger.success(f"Saved heatmap to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
