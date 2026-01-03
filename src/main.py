
import torch.optim as optim
import torch
import mads_datasets
from mads_datasets import DatasetFactoryProvider, DatasetType

from mltrainer.preprocessors import BasePreprocessor
from mltrainer import imagemodels, Trainer, TrainerSettings, ReportTypes, metrics

from torch import nn
from tomlserializer import TOMLSerializer


# Settings - Optimized for >90% accuracy

accuracy = metrics.Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()
batchsize = 256  # Larger batch = more stable gradients
learning_rate = 0.003  # Higher LR (sweet spot between 0.001 and 0.01)

settings = TrainerSettings(
    epochs=150, # More epochs for convergence
    metrics=[accuracy],
    logdir="modellogs",
    train_steps=235,  # ~60000 / 256 = full Fashion MNIST train set
    valid_steps=40,   # ~10000 / 256 = full Fashion MNIST valid set
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    optimizer_kwargs={"lr": learning_rate, "weight_decay": 1e-4},  # Add weight decay for regularization
    earlystop_kwargs={"patience": 20, "verbose": True},  # More patience
    scheduler_kwargs={"factor": 0.5, "patience": 8},  # Reduce LR when plateauing
)


# Data loading

fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(batchsize=batchsize, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

# Basic model 3 layers

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int, units1: int, units2: int, units3: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, units3),
            nn.ReLU(),
            nn.Linear(units3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork(
    num_classes=10, units1=256, units2=256, units3=256)

# This package will 1. check if there is a `__dict__` attribute available, and if so, it will use that to extract the parameters that do not start with an underscore, like this:
{k: v for k, v in model.__dict__.items() if not k.startswith("_")}

# Save settings and model configuration to TOML files
tomlserializer = TOMLSerializer()
tomlserializer.save(settings, "settings.toml")
tomlserializer.save(model, "model.toml")

# Training

trainer = Trainer(
    model=model,
    settings=settings,
    loss_fn=loss_fn,
    optimizer=optim.Adam,
    traindataloader=trainstreamer,
    validdataloader=validstreamer,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau
)
trainer.loop()

# grid search for units

units = [256, 64, 16]
for unit1 in units:
    for unit2 in units:
        for unit3 in units:

            model = NeuralNetwork(num_classes=10, units1=unit1, units2=unit2, units3=unit3)

            trainer = Trainer(
                model=model,
                settings=settings,
                loss_fn=loss_fn,
                optimizer=optim.Adam,
                traindataloader=trainstreamer,
                validdataloader=validstreamer,
                scheduler=optim.lr_scheduler.ReduceLROnPlateau
            )
            trainer.loop()

