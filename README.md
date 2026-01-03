# MADS-ML-FloridoM

CNN Hyperparameter Optimization for Fashion MNIST Classification
Machine Learning project for MADS (Post-Bachelor of AI & ML specialist) by Florido Meacci.

## 🎯 Project Overview

This project explores deep learning architecture optimization through systematic experimentation on the Fashion MNIST dataset. Using MLflow for experiment tracking and Hyperopt for hyperparameter optimization, we achieved **94.18% accuracy** through careful architectural design and training strategies.

### Key Achievements
- **94.18% best accuracy** (exp12_optimized_training_full)
- **12+ controlled experiments** testing dropout, depth, width, training duration
- **Complete MLflow tracking** with 350+ training runs
- **Statistical validation** of architectural decisions

## 🚀 Quick Start

### Setup

This project uses `uv` for dependency management:

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -e .

# Activate environment
source .venv/bin/activate  # macOS/Linux
```

### Run Training

```bash
# Start MLflow UI (in separate terminal)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Run training with current configuration
python src/train_mlflow.py

# Analyze results
python analyze_results.py
```

## 📊 Experiment Results

### Round 1: Architecture Search (5 experiments, 5 epochs each)

| Experiment | Best Acc | Key Finding |
|------------|----------|-------------|
| exp1_baseline_batchnorm | 89.57% | BatchNorm essential |
| exp2_no_batchnorm | 86.52% | -10.55% without BatchNorm |
| exp3_aggressive_dropout | 87.85% | Too much dropout hurts |
| exp4_minimal_dropout | 90.00% | Sweet spot found |
| exp5_deep_network | 90.04% | 4 blocks > 3 blocks ✓ |

### Round 2: Deep Training (7 experiments, 25-100 epochs)

| Experiment | Best Acc | Avg Acc | Key Innovation |
|------------|----------|---------|----------------|
| exp6_deep_long_training | 91.76% | 91.26% | 25 epochs baseline |
| exp7_lr_schedule | 93.44% | 93.08% | + LR decay (StepLR) |
| exp8_wider_network | 92.93% | 92.11% | 512 filters slower, not better |
| exp9_residual_connections | 91.91% | 90.43% | High variance |
| exp10_ensemble_architecture | 92.54% | 91.70% | Dual-path stable |
| exp11_long_training_earlystop | 93.20% | 92.83% | 100 epochs = overkill |
| exp12_optimized_training_full | **94.18%** | **93.53%** | **40 epochs + patience=15** ⭐ |

### Round 3: Ongoing Experiments

| Experiment | Status | Goal |
|------------|--------|------|
| exp13_dropout_placement_extended | Running | Test dropout placement strategies (30 runs) |

## 🔬 Key Findings

### 1. BatchNorm is Essential
- **Impact**: +10.55% accuracy improvement
- Enables higher learning rates (0.001 vs 0.0001)
- Stabilizes training across all architectures

### 2. Optimal Architecture
```python
# Winner configuration (exp12)
filters = 256          # Not 128 (too small) or 512 (diminishing returns)
units1 = 192          # Dense layer 1
units2 = 96           # Dense layer 2
dropout = [0.05, 0.15, 0.1]  # Minimal dropout
num_conv_blocks = 3   # 3 blocks optimal for 28×28 images
```

### 3. Training Duration Sweet Spot
- ❌ 5 epochs: Underfitting (90.04%)
- ✅ **25-40 epochs: Optimal** (93-94%)
- ❌ 100 epochs: Overfitting risk (93.20%)

### 4. Parameter Efficiency
| Params | Accuracy | Training Time | Verdict |
|--------|----------|---------------|---------|
| 300K-800K | **93-94%** | 2-4 min | ✅ **Optimal** |
| 1.2M-4.1M | 92-93% | 5-8 min | ❌ Diminishing returns |

### 5. Learning Rate Schedule Matters
- No schedule: 91.76%
- **StepLR (step=10, gamma=0.5)**: +1.7% improvement
- Critical for convergence beyond epoch 20

## 📁 Project Structure

```
MADS-ML-FloridoM/
├── src/
│   ├── train_mlflow.py          # Main training script with Hyperopt
│   └── main.py                   # Legacy training script
├── data/
│   ├── external/                 # External datasets
│   ├── processed/                # Preprocessed data
│   └── RAW/                      # Raw data
├── models/                       # Saved model checkpoints
├── temp_logs/                    # Training logs (temp)
├── mlflow.db                     # MLflow experiment database
├── analyze_results.py            # Result analysis script
├── ACTIONPLAN.md                 # Round 1 experiment plan
├── ACTIONPLAN_ROUND2.md          # Round 2 experiment plan + results
└── README.md                     # This file
```

## 🛠️ Dependencies

### Core ML Stack
- **PyTorch**: Deep learning framework (MPS/CUDA support)
- **MLflow**: Experiment tracking and model registry
- **Hyperopt**: Bayesian hyperparameter optimization

### Data Science
- numpy, pandas, matplotlib, seaborn
- scikit-learn

### Project-specific
- **mads_datasets**: Fashion MNIST data loader
- **mltrainer**: Training utilities and metrics
- loguru: Logging

### Development
```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Includes: jupyter, tensorboard, torch-tb-profiler
```

## 📈 Running Experiments

### Configuration
Edit `src/train_mlflow.py` to configure experiments:

```python
# Training settings
BATCH_SIZE = 64
EPOCHS = 40
MAX_EVALS = 10  # Number of hyperparameter trials

# Search space
SEARCH_SPACE = {
    "filters": hp.choice("filters", [256, 192, 128]),
    "units1": hp.choice("units1", [256, 192, 128]),
    "units2": hp.choice("units2", [128, 96, 64]),
}

# Experiment name
EXPERIMENT_NAME = "exp14_your_experiment"
```

### View Results

```bash
# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# Open: http://localhost:5001

# Command-line analysis
python analyze_results.py
```

## 🎓 Methodology

### Systematic Experimentation
1. **Hypothesis-driven**: Each experiment tests a specific hypothesis
2. **Controlled variables**: Change one aspect at a time
3. **Statistical validation**: Multiple runs per configuration
4. **MLflow tracking**: All experiments logged automatically

### Best Practices Applied
- Early stopping (patience=15) prevents overfitting
- LR scheduling (StepLR) improves convergence
- BatchNorm after every conv layer
- Minimal dropout (0.05-0.15) for regularization
- Data augmentation via dropout, not transforms

## 📝 Documentation

- **ACTIONPLAN.md**: Round 1 experiments (architecture search)
- **ACTIONPLAN_ROUND2.md**: Round 2 experiments (deep training) + complete results
- **MLflow UI**: Interactive exploration of all runs

## 🏆 Production Model

**Recommended configuration** (from exp12):
```python
# Architecture
CNN(
    filters=256,
    units1=192,
    units2=96,
    num_blocks=3,
    dropout_rates=[0.05, 0.15, 0.1]
)

# Training
optimizer = Adam(lr=0.001)
scheduler = StepLR(step_size=10, gamma=0.5)
epochs = 40
batch_size = 64
early_stopping = {"patience": 15}
```

**Expected performance**: 93-94% validation accuracy

## 📚 References

- Fashion MNIST: [github.com/zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
- MLflow: [mlflow.org](https://mlflow.org)
- Hyperopt: [github.com/hyperopt/hyperopt](https://github.com/hyperopt/hyperopt)

## 👤 Author

**Florido Meacci**  
Master of Applied Data Science (MADS)  
December 2025
