"""
Fashion MNIST CNN Training with MLflow & Hyperopt
Hyperparameter optimization with MLflow experiment tracking
"""

from datetime import datetime
from pathlib import Path
from typing import Iterator
import random

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


# ============================================================================
# TRAINING SETTINGS - CONFIGURE HERE
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 25  # Extended to test dropout placement effects
TRAIN_STEPS = 235  # Full Fashion MNIST train: ~60k / 256
VALID_STEPS = 40   # Full Fashion MNIST valid: ~10k / 256

# Hyperparameter search space - Test dropout placement strategies
# Running multiple trials per strategy for statistical significance
SEARCH_SPACE = {
    "filters": hp.choice("filters", [256]),  # Fixed at optimal
    "units1": hp.choice("units1", [256]),    # Fixed at optimal
    "units2": hp.choice("units2", [128]),    # Fixed at optimal
    "dropout_placement": hp.choice("dropout_placement", [
        "early",      # Dropout only at first conv block
        "late",       # Dropout only at last conv block
        "both",       # Dropout at first and last conv blocks
        "middle",     # Dropout only at middle conv block
        "all",        # Dropout at all 3 conv blocks (baseline)
        "none"        # No dropout in conv blocks (only dense)
    ])
}

# Hyperopt settings
MAX_EVALS = 30  # 5 runs per strategy (6 strategies × 5 = 30 total runs)

# MLflow settings
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "exp13_dropout_placement_extended"  # 5 runs per strategy for statistical power
MODEL_TAG = "convnet"
DEVELOPER_TAG = "FloridoM"

# Model save directory
MODEL_DIR = Path("models").resolve()

# ============================================================================
# END OF SETTINGS
# ============================================================================


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return trainstreamer, validstreamer


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device


# CNN with normalization layers, dropout, and modular design using ModuleList
class CNN(nn.Module):
    def __init__(self, filters, units1, units2, dropout_placement="all", input_size=(32, 1, 28, 28)):
        super().__init__()
        self.in_channels = input_size[1]
        self.input_size = input_size
        self.filters = filters
        self.units1 = units1
        self.units2 = units2
        self.dropout_placement = dropout_placement

        # Determine which blocks get dropout (0.1 rate for testing)
        dropout_rate = 0.1
        dropout_flags = {
            "early": [True, False, False],   # Only first block
            "middle": [False, True, False],  # Only middle block
            "late": [False, False, True],    # Only last block
            "both": [True, False, True],     # First and last
            "all": [True, True, True],       # All blocks (baseline)
            "none": [False, False, False]    # No conv dropout
        }
        
        use_dropout = dropout_flags.get(dropout_placement, [True, True, True])

        # Build convolutional blocks using ModuleList for flexibility
        # Each block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> [Optional Dropout]
        self.conv_blocks = nn.ModuleList()
        
        # First conv block (input channels -> filters)
        layers = [
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ]
        if use_dropout[0]:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_blocks.append(nn.Sequential(*layers))
        
        # Second conv block (filters -> filters)
        layers = [
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ]
        if use_dropout[1]:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_blocks.append(nn.Sequential(*layers))
        
        # Third conv block (filters -> filters)
        layers = [
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        ]
        if use_dropout[2]:
            layers.append(nn.Dropout2d(p=dropout_rate))
        self.conv_blocks.append(nn.Sequential(*layers))

        activation_map_size = self._conv_test(input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        # Dense layers with BatchNorm and minimal dropout
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.BatchNorm1d(units1),
            nn.ReLU(),
            nn.Dropout(p=0.15),  # Minimal dropout
            nn.Linear(units1, units2),
            nn.BatchNorm1d(units2),
            nn.ReLU(),
            nn.Dropout(p=0.1),  # Minimal dropout
            nn.Linear(units2, 10),
        )

    def _conv_test(self, input_size=(32, 1, 28, 28)):
        """Test forward pass through conv blocks to calculate activation map size"""
        x = torch.ones(input_size)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        return x.shape[-2:]

    def forward(self, x):
        # Pass through all convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Aggregate and pass through dense layers
        x = self.agg(x)
        logits = self.dense(x)
        return logits


def generate_experiment_name() -> str:
    """Generate experiment name in format YYMMDD-RANDOM_ANIMAL"""
    animals = [
        "LION", "TIGER", "BEAR", "WOLF", "EAGLE", "HAWK", "PANDA", "KOALA",
        "DOLPHIN", "SHARK", "WHALE", "OCTOPUS", "PENGUIN", "FALCON", "OWL",
        "FOX", "DEER", "MOOSE", "BISON", "RHINO", "HIPPO", "GIRAFFE", "ZEBRA",
        "CHEETAH", "JAGUAR", "LEOPARD", "PANTHER", "COUGAR", "LYNX", "OTTER",
        "BADGER", "RACCOON", "SQUIRREL", "BEAVER", "RABBIT", "HARE", "LEMUR",
        "MONKEY", "GORILLA", "CHIMP", "ORANGUTAN", "SLOTH", "ARMADILLO",
        "ANTEATER", "PLATYPUS", "WOMBAT", "KANGAROO", "WALLABY", "COBRA",
        "PYTHON", "VIPER", "GECKO", "IGUANA", "DRAGON", "TURTLE", "TORTOISE"
    ]
    date_str = datetime.now().strftime("%y%m%d")
    animal = random.choice(animals)
    return f"{date_str}-{animal}"


def setup_mlflow(experiment_path: str) -> None:
    """Initialize MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_path)


def objective(params):
    """Objective function for Hyperopt optimization"""
    # Setup model directory
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        logger.info(f"Created {MODEL_DIR}")
    
    # Load data
    trainstreamer, validstreamer = get_fashion_streamers(BATCH_SIZE)
    
    # Setup metrics and training settings
    accuracy = metrics.Accuracy()
    
    # Create temp logdir (required by TrainerSettings, but not used with MLflow)
    temp_logdir = Path(f"temp_logs/hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    settings = TrainerSettings(
        epochs=EPOCHS,
        metrics=[accuracy],
        logdir=temp_logdir,
        train_steps=TRAIN_STEPS,
        valid_steps=VALID_STEPS,
        reporttypes=[ReportTypes.MLFLOW],
        scheduler_kwargs={},  # No kwargs for StepLR (set in lambda)
        earlystop_kwargs={"patience": 15, "verbose": True},  # More patient
    )
    # Start a new MLflow run for tracking the experiment
    device = get_device()
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", MODEL_TAG)
        mlflow.set_tag("dev", DEVELOPER_TAG)
        
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("train_steps", TRAIN_STEPS)
        mlflow.log_param("valid_steps", VALID_STEPS)
        
        # Log architecture details
        mlflow.log_param("num_conv_blocks", 3)
        mlflow.log_param("dropout_rate", 0.1)
        mlflow.log_param("dropout_placement", params.get("dropout_placement", "all"))
        mlflow.log_param("dropout_dense_1", 0.15)
        mlflow.log_param("dropout_dense_2", 0.1)
        mlflow.log_param("normalization", "BatchNorm")
        mlflow.log_param("lr_schedule", "StepLR(step=10,gamma=0.5)")
        mlflow.log_param("initial_lr", 0.001)
        mlflow.log_param("max_epochs", 50)
        mlflow.log_param("early_stopping", True)
        mlflow.log_param("early_stop_patience", 15)

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Instantiate the CNN model with the given hyperparameters
        model = CNN(**params)
        model.to(device)
        
        # Train the model using a custom train loop with StepLR scheduler
        # LR: 0.001 → 0.0005 (epoch 10) → 0.00025 (epoch 20) → 0.000125 (epoch 30)
        # Early stopping will halt training if no improvement for 15 epochs
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=lambda opt: optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5),
            device=device,
        )
        trainer.loop()
        
        # Log final metrics for comparison
        # Get the last epoch's losses from the metric container
        final_test_loss = float(trainer.test_loss)  # This attribute exists
        
        # Get accuracy from metrics container
        if hasattr(trainer, 'metric_container') and len(trainer.metric_container.metrics) > 0:
            accuracy_metric = trainer.metric_container.metrics[0]
            final_train_acc = float(accuracy_metric.train)
            final_test_acc = float(accuracy_metric.test)
            
            mlflow.log_metric("final_train_accuracy", final_train_acc)
            mlflow.log_metric("final_test_accuracy", final_test_acc)
            
            # Calculate and log generalization gap (overfitting indicator)
            generalization_gap = final_train_acc - final_test_acc
            mlflow.log_metric("generalization_gap", generalization_gap)
        
        mlflow.log_metric("final_test_loss", final_test_loss)
        
        # Log model complexity
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_metric("total_parameters", total_params)
        mlflow.log_metric("trainable_parameters", trainable_params)
        
        # Log experiment name for easier filtering
        mlflow.set_tag("experiment_group", EXPERIMENT_NAME)

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = MODEL_DIR / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")
        
        # Return for Hyperopt (minimize test_loss)
        return {"loss": final_test_loss, "status": STATUS_OK}


def main():
    """Main training loop with hyperparameter optimization"""
    logger.info(f"Starting experiment: {EXPERIMENT_NAME}")
    logger.info(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    logger.info(f"Max evaluations: {MAX_EVALS}")
    
    setup_mlflow(EXPERIMENT_NAME)

    # Run hyperparameter optimization
    best_result = fmin(
        fn=objective, 
        space=SEARCH_SPACE, 
        algo=tpe.suggest, 
        max_evals=MAX_EVALS, 
        trials=Trials()
    )

    logger.info(f"Best result: {best_result}")
    logger.info(f"View results at: http://localhost:5001")


if __name__ == "__main__":
    main()
