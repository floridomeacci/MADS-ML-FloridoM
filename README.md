# MADS-ML-FloridoM

Flowers classification experiments for the MADS programme by Florido Meacci.

## Project Overview

We fine-tune ResNet18 on the Oxford 5-flowers dataset to compare how different input pre-processing strategies affect convergence and final accuracy. The focus is on:

- Standard RGB vs LAB colour space.
- Converting images to grayscale.
- Edge-shifted RGB variants that emphasise spatial gradients.

All variants reuse the same hyperparameters sampled during sweep trial 15 and are trained for up to 100 epochs with early stopping (patience 15, delta 0.001). The best RGB checkpoint now reaches **0.990 validation accuracy** (327/330 images) on the validation split.

## Quick Start

```bash
# Create virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install project
pip install -e .

# Optional: enable MPS on macOS for faster runs
python -c "import torch; print(torch.backends.mps.is_available())"
```

The dataset downloads automatically on first use via `mads_datasets`.

## Reproducing Trial 15 Runs

Each variant is launched with the same command-line overrides; adjust flags for the desired preprocessing:

```bash
# RGB
.venv/bin/python src/hypertuning.py --mode single --epochs 100 \
    --seed 42 --lr 0.015515907574180044 --momentum 0.7617721012508225 \
    --weight-decay 2.1656873184936155e-06 --step-size 1 --gamma 0.3301724543143726 \
    --head-depth 2 --head-width-factor 0.9388607111568091 --head-dropout 0.26087065725099706 \
    --unfreeze-blocks 1 --early-stop-patience 15 --early-stop-delta 0.001

# LAB colour space
.venv/bin/python src/hypertuning.py ... --color-space lab

# Grayscale
.venv/bin/python src/hypertuning.py ... --grayscale

# Edge shifts (swap ... with the common hyperparameters)
.venv/bin/python src/hypertuning.py ... --edge-shift --edge-shift-pixels 1
.venv/bin/python src/hypertuning.py ... --edge-shift --edge-shift-pixels 2
.venv/bin/python src/hypertuning.py ... --edge-shift --edge-shift-pixels 3
```

TensorBoard logs are stored under `Les1_modellogs/selected/*` and checkpoints under `models/`.

## Latest Validation Results (Trial 15)

| Variant | Val (Colour) | Val (Grayscale) |
|---------|---------------|-----------------|
| RGB | 0.9901 | 0.8466 |
| LAB | 0.9872 | 0.8267 |
| Grayscale | 0.9702 | 0.9702 |
| EdgeShift px1 | 0.9631 | 0.7003 |
| EdgeShift px2 | 0.9673 | 0.6790 |
| EdgeShift px3 | 0.9673 | 0.7173 |

The summary table above is exported by `src/plot_trial15_heatmap.py` to `data/processed/heatmaps/trial15_epoch100_patience15_metrics.csv`. The companion heatmap is saved as `data/processed/heatmaps/trial15_epoch100_patience15_heatmap.png`.

## Visualisation Utilities

- `src/plot_trial15_heatmap.py`: Loads the six checkpoints, re-evaluates them on the validation streamer, writes the metrics CSV, and renders the heatmap.
- `src/visualize.py`: Shared plotting helpers (seaborn heatmaps, sample grids).

Run the heatmap script with:

```bash
.venv/bin/python src/plot_trial15_heatmap.py
```

## Repository Layout

```
MADS-ML-FloridoM-master/
├── data/
│   ├── RAW/                  # Placeholder for raw data (auto-downloaded)
│   └── processed/heatmaps/   # Exported figures & CSV summaries
├── Les1_modellogs/           # TensorBoard runs (ignored by Git)
├── models/                   # ResNet18 checkpoints (ignored by Git)
├── src/
│   ├── hypertuning.py        # Sweep + single-run entry point
│   ├── plot_trial15_heatmap.py
│   └── visualize.py          # Plotting helpers
├── pyproject.toml
├── README.md
└── settings.toml             # Saved TrainerSettings for latest runs
```

## Notes & Next Steps

- The validation set is small (330 images); confirm generalisation by adding a separate test evaluation.
- LAB and grayscale both converge faster than RGB but still trail in final colour accuracy while improving stability.
- Edge-shift preprocessing does not consistently boost performance and increases variance; treat as exploratory.

## Author

**Florido Meacci** – MADS Machine Learning track (January 2026)
