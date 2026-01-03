import argparse
import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from tomlserializer import TOMLSerializer
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet

import mltrainer.trainer as trainer_module

try:  # torch>=2.6 switches torch.load default to weights_only=True
    from torch.serialization import add_safe_globals
except ImportError:  # pragma: no cover - fallback for older torch
    add_safe_globals = None  # type: ignore

if add_safe_globals is not None:
    add_safe_globals([ResNet])  # allow early-stopping checkpoint reloads


def _load_best_checkpoint(self):
    try:
        return torch.load(self.path, weights_only=False)
    except TypeError:  # pragma: no cover - support older torch versions
        return torch.load(self.path)


trainer_module.EarlyStopping.get_best = _load_best_checkpoint  # type: ignore[attr-defined]

Batch = Iterable[Tuple[torch.Tensor, torch.Tensor]]


class AugmentPreprocessor:
    def __init__(
        self,
        transform: transforms.Compose,
        device: torch.device | None = None,
        target_channels: int | None = None,
    ):
        self.transform = transform
        self.device = device
        self.target_channels = target_channels

    def __call__(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        features, targets = zip(*batch)
        augmented = [self.transform(sample) for sample in features]
        stacked = torch.stack(augmented)
        if self.target_channels is not None and stacked.shape[1] != self.target_channels:
            if self.target_channels == 1 and stacked.shape[1] == 3:
                weights = torch.tensor([0.299, 0.587, 0.114], device=stacked.device)
                stacked = (stacked.permute(0, 2, 3, 1) @ weights).unsqueeze(1)
            elif self.target_channels == 3 and stacked.shape[1] == 1:
                stacked = stacked.repeat(1, 3, 1, 1)
        labels = torch.stack([torch.as_tensor(label, dtype=torch.long) for label in targets])
        if self.device is not None:
            stacked = stacked.to(self.device)
            labels = labels.to(self.device)
        return stacked, labels


class ShiftedChannelEdgeEnhancer:
    def __init__(self, pixel_distances: Iterable[int] | None = None) -> None:
        distances = []
        if pixel_distances is not None:
            for value in pixel_distances:
                try:
                    distance = abs(int(value))
                except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                    raise ValueError("Edge shift distances must be integers") from exc
                if distance == 0:
                    continue
                if distance not in distances:
                    distances.append(distance)
        if not distances:
            distances.append(1)
        distances.sort()
        offsets: list[Tuple[int, int]] = []
        for distance in distances:
            offsets.extend(
                [
                    (-distance, 0),
                    (distance, 0),
                    (0, -distance),
                    (0, distance),
                ]
            )
        self.offsets: Tuple[Tuple[int, int], ...] = tuple(offsets)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3:
            raise ValueError("Expected a tensor with shape [C, H, W]")
        base = tensor
        edges = torch.zeros_like(base)
        for dy, dx in self.offsets:
            shifted = torch.roll(base, shifts=(dy, dx), dims=(1, 2))
            edges = torch.maximum(edges, (base - shifted).abs())
        return edges


class RgbToLab:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError("Expected a [3, H, W] tensor for RGB to LAB conversion")

        rgb = tensor.clamp(0.0, 1.0)
        mask = rgb > 0.04045
        linear_rgb = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

        r, g, b = linear_rgb[0], linear_rgb[1], linear_rgb[2]

        x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
        y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
        z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

        x /= 0.95047
        z /= 1.08883

        epsilon = 216 / 24389  # (6/29)^3
        kappa = 24389 / 27  # (29/3)^3

        def _f(channel: torch.Tensor) -> torch.Tensor:
            return torch.where(
                channel > epsilon,
                channel.pow(1.0 / 3.0),
                (kappa * channel + 16.0) / 116.0,
            )

        fx = _f(x)
        fy = _f(y)
        fz = _f(z)

        l = (116.0 * fy) - 16.0
        a = 500.0 * (fx - fy)
        b_channel = 200.0 * (fy - fz)

        lab = torch.stack((l, a, b_channel), dim=0)
        lab[0] = lab[0].clamp(0.0, 100.0)
        lab[1] = lab[1].clamp(-128.0, 127.0)
        lab[2] = lab[2].clamp(-128.0, 127.0)
        return lab


def build_transforms(
    grayscale: bool,
    edge_shift: bool = False,
    edge_shift_pixels: Iterable[int] | None = None,
    color_space: str = "rgb",
) -> Dict[str, transforms.Compose]:
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]
    gray_mean = [0.449]
    gray_std = [0.226]
    lab_mean = [50.0, 0.0, 0.0]
    lab_std = [50.0, 128.0, 128.0]

    def make_pipeline(train: bool, force_gray: bool | None = None) -> transforms.Compose:
        ops: list = [transforms.ConvertImageDtype(torch.float)]
        if edge_shift:
            ops.append(ShiftedChannelEdgeEnhancer(edge_shift_pixels))
        if train:
            ops.extend(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            ops.extend(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ]
            )

        if force_gray is None:
            force_gray = grayscale

        if force_gray:
            num_channels = 1 if grayscale else 3
            ops.append(transforms.Grayscale(num_output_channels=num_channels))

        if color_space == "lab" and not grayscale:
            ops.append(RgbToLab())

        if grayscale and force_gray:
            mean, std = (gray_mean, gray_std)
        elif color_space == "lab" and not grayscale:
            mean, std = (lab_mean, lab_std)
        else:
            mean, std = (color_mean, color_std)
        ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)

    transforms_map: Dict[str, transforms.Compose] = {
        "train": make_pipeline(train=True),
        "val": make_pipeline(train=False),
    }

    if grayscale:
        transforms_map["eval_gray"] = make_pipeline(train=False)
        transforms_map["eval_color"] = make_pipeline(train=False, force_gray=False)
    else:
        transforms_map["eval_color"] = make_pipeline(train=False, force_gray=False)
        transforms_map["eval_gray"] = make_pipeline(train=False, force_gray=True)

    return transforms_map


def select_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    logger.warning(
        "Running on CPU. Expect 15-20 minutes per epoch; consider using hardware acceleration."
    )
    return torch.device("cpu")


def build_model(
    num_classes: int,
    grayscale: bool,
    *,
    head_depth: int = 1,
    head_width_factor: float = 1.0,
    head_dropout: float = 0.0,
    unfreeze_blocks: int = 0,
) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if grayscale:
        conv1 = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias is not None,
        )
        with torch.no_grad():
            model.conv1.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
            if conv1.bias is not None:
                model.conv1.bias.copy_(conv1.bias)
    for param in model.parameters():
        param.requires_grad = False

    allowed_blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
    unfreeze_count = max(0, min(len(allowed_blocks), int(unfreeze_blocks)))
    for block in allowed_blocks[:unfreeze_count]:
        for param in block.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features  # type: ignore[attr-defined]

    depth = max(1, min(6, int(head_depth)))
    width_factor = max(0.25, float(head_width_factor))
    dropout = min(max(float(head_dropout), 0.0), 0.9)

    hidden_features = max(num_classes * 2, int(in_features * width_factor))
    layers: List[nn.Module] = []
    input_dim = in_features

    for layer_idx in range(depth - 1):
        layers.append(nn.Linear(input_dim, hidden_features))
        layers.append(nn.BatchNorm1d(hidden_features))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        input_dim = hidden_features

    layers.append(nn.Linear(input_dim, num_classes))
    model.fc = nn.Sequential(*layers)  # type: ignore[assignment]
    return model


def prepare_streamers(
    batch_size: int,
    transforms_map: Dict[str, transforms.Compose],
    device: torch.device,
    grayscale: bool,
) -> Dict[str, Any]:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FLOWERS)
    factory.settings.img_size = (500, 500)
    streamers = factory.create_datastreamer(batchsize=batch_size)

    train_streamer = streamers["train"]
    valid_streamer = streamers["valid"]

    target_channels = 1 if grayscale else 3
    train_streamer.preprocessor = AugmentPreprocessor(  # type: ignore[attr-defined]
        transforms_map["train"], device=device, target_channels=target_channels
    )
    valid_streamer.preprocessor = AugmentPreprocessor(  # type: ignore[attr-defined]
        transforms_map["val"], device=device, target_channels=target_channels
    )

    return {
        "train": train_streamer,
        "valid": valid_streamer,
    }


def evaluate_accuracy(
    model: nn.Module,
    device: torch.device,
    streamer,
    steps: int,
) -> float:
    generator = streamer.stream()
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(steps):
            inputs, targets = next(generator)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    if was_training:
        model.train()
    return correct / total if total else 0.0


def sample_hyperparameters() -> Dict[str, float]:
    return {
        "lr": 10 ** random.uniform(-4, -1),
        "momentum": random.uniform(0.5, 0.95),
        "weight_decay": 10 ** random.uniform(-6, -3),
        "step_size": random.choice([1, 2, 3]),
        "gamma": random.uniform(0.1, 0.6),
        "head_depth": random.randint(1, 6),
        "head_width_factor": random.uniform(0.5, 2.0),
        "head_dropout": random.uniform(0.0, 0.5),
        "unfreeze_blocks": random.randint(0, 2),
    }


def ensure_logdir(base: Path, mode: str, trial_idx: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_dir = base / mode / f"trial_{trial_idx:02d}_{timestamp}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def evaluate_dual_modes(
    *,
    model: nn.Module,
    device: torch.device,
    transforms_map: Dict[str, transforms.Compose],
    valid_streamer,
    grayscale: bool,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    original_preprocessor = valid_streamer.preprocessor

    if "eval_color" in transforms_map:
        color_target = 1 if grayscale else 3
        color_eval = AugmentPreprocessor(
            transforms_map["eval_color"], device=device, target_channels=color_target
        )
        valid_streamer.preprocessor = color_eval  # type: ignore[attr-defined]
        results["val_color"] = evaluate_accuracy(
            model=model,
            device=device,
            streamer=valid_streamer,
            steps=len(valid_streamer),
        )

    if "eval_gray" in transforms_map:
        gray_target = 1 if grayscale else 3
        gray_eval = AugmentPreprocessor(
            transforms_map["eval_gray"], device=device, target_channels=gray_target
        )
        valid_streamer.preprocessor = gray_eval  # type: ignore[attr-defined]
        results["val_grayscale"] = evaluate_accuracy(
            model=model,
            device=device,
            streamer=valid_streamer,
            steps=len(valid_streamer),
        )

    valid_streamer.preprocessor = original_preprocessor  # type: ignore[attr-defined]
    return results


def run_single_trial(
    *,
    device: torch.device,
    num_classes: int,
    batch_size: int,
    epochs: int,
    grayscale: bool,
    hyperparams: Dict[str, float],
    logdir: Path,
    edge_shift: bool,
    edge_shift_pixels: Iterable[int] | None,
    color_space: str = "rgb",
) -> Dict[str, float]:
    transforms_map = build_transforms(
        grayscale,
        edge_shift=edge_shift,
        edge_shift_pixels=edge_shift_pixels,
        color_space=color_space,
    )
    streamers = prepare_streamers(batch_size, transforms_map, device, grayscale)
    train_streamer = streamers["train"]
    valid_streamer = streamers["valid"]

    model = build_model(
        num_classes,
        grayscale=grayscale,
        head_depth=hyperparams.get("head_depth", 1),
        head_width_factor=hyperparams.get("head_width_factor", 1.0),
        head_dropout=hyperparams.get("head_dropout", 0.0),
        unfreeze_blocks=hyperparams.get("unfreeze_blocks", 0),
    )
    model.to(device)

    accuracy_metric = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=epochs,
        metrics=[accuracy_metric],
        logdir=logdir,
        train_steps=len(train_streamer),
        valid_steps=len(valid_streamer),
        reporttypes=[ReportTypes.TENSORBOARD],
        optimizer_kwargs={
            "lr": hyperparams["lr"],
            "momentum": hyperparams["momentum"],
            "weight_decay": hyperparams["weight_decay"],
        },
        scheduler_kwargs={
            "step_size": max(1, int(hyperparams["step_size"])),
            "gamma": float(hyperparams["gamma"]),
        },
        earlystop_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        settings=settings,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD,
        traindataloader=train_streamer.stream(),
        validdataloader=valid_streamer.stream(),
        scheduler=torch.optim.lr_scheduler.StepLR,
    )

    trainer.loop()

    eval_results = evaluate_dual_modes(
        model=model,
        device=device,
        transforms_map=transforms_map,
        valid_streamer=valid_streamer,
        grayscale=grayscale,
    )

    return {
        **hyperparams,
        **eval_results,
    }


def run_sweep(
    *,
    device: torch.device,
    trials_per_mode: int,
    epochs: int,
    batch_size: int,
    num_classes: int,
    log_base: Path,
    edge_shift: bool = False,
    edge_shift_pixels: Iterable[int] | None = None,
    sweep_modes: Iterable[str] | None = None,
    color_space: str = "rgb",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    mode_lookup = {"color": False, "grayscale": True}
    if sweep_modes is None:
        modes = list(mode_lookup.items())
    else:
        modes = []
        for mode_name in sweep_modes:
            if mode_name not in mode_lookup:
                raise ValueError(f"Unsupported sweep mode: {mode_name}")
            modes.append((mode_name, mode_lookup[mode_name]))

    for mode_name, grayscale in modes:
        logger.info(f"Starting sweep for {mode_name} trials")
        for trial_idx in range(1, trials_per_mode + 1):
            hyperparams = sample_hyperparameters()
            trial_logdir = ensure_logdir(log_base, mode_name, trial_idx)
            logger.info(
                f"Trial {mode_name} #{trial_idx:02d} – lr={hyperparams['lr']:.5f} "
                f"momentum={hyperparams['momentum']:.3f} weight_decay={hyperparams['weight_decay']:.6f} "
                f"step_size={int(hyperparams['step_size'])} gamma={hyperparams['gamma']:.3f}"
            )

            trial_result = run_single_trial(
                device=device,
                num_classes=num_classes,
                batch_size=batch_size,
                epochs=epochs,
                grayscale=grayscale,
                hyperparams=hyperparams,
                logdir=trial_logdir,
                edge_shift=edge_shift,
                edge_shift_pixels=edge_shift_pixels,
                color_space=color_space,
            )

            trial_record: Dict[str, Any] = {
                "mode": mode_name,
                "trial": trial_idx,
                "color_space": color_space,
                "edge_shift_pixels": " ".join(str(value) for value in edge_shift_pixels)
                if edge_shift_pixels is not None
                else "",
                **trial_result,
            }
            results.append(trial_record)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def persist_results(results: List[Dict[str, Any]], output_path: Path) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Flowers hyperparameter sweeps")
    parser.add_argument("--mode", choices=["single", "sweep"], default="single")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument(
        "--color-space",
        choices=["rgb", "lab"],
        default="rgb",
        help="Color space to use for color-mode training and evaluation (default: rgb).",
    )
    parser.add_argument(
        "--edge-shift",
        action="store_true",
        help="Apply channel-shift edge enhancement before converting inputs to grayscale.",
    )
    parser.add_argument(
        "--edge-shift-pixels",
        type=int,
        nargs="+",
        help="Pixel distances for edge enhancement shifts (default: 1).",
    )
    parser.add_argument("--trials-per-mode", type=int, default=15)
    parser.add_argument("--sweep-logdir", default="Les1_modellogs/hyper_sweep")
    parser.add_argument(
        "--sweep-modes",
        choices=["color", "grayscale"],
        nargs="+",
        help="Subset of data modes to sweep (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed controlling hyperparameter sampling and trainer RNGs.",
    )
    parser.add_argument("--lr", type=float, help="Override learning rate for single run")
    parser.add_argument("--momentum", type=float, help="Override momentum for single run")
    parser.add_argument(
        "--weight-decay", dest="weight_decay", type=float, help="Override weight decay for single run"
    )
    parser.add_argument(
        "--step-size", dest="step_size", type=int, help="Override LR scheduler step size for single run"
    )
    parser.add_argument("--gamma", type=float, help="Override LR scheduler gamma for single run")
    parser.add_argument(
        "--head-depth",
        type=int,
        help="Number of linear blocks in the classifier head (default: 1)",
    )
    parser.add_argument(
        "--head-width-factor",
        type=float,
        help="Multiplier for classifier hidden width relative to backbone features (default: 1.0)",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        help="Dropout probability applied between head blocks (default: 0.0)",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        help="Number of residual blocks to unfreeze starting from layer4 (default: 0)",
    )
    parser.add_argument(
        "--single-logdir",
        type=str,
        help="Target directory for tensorboard logs in single mode (defaults to modellogs/flowers)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        help="Filename (relative or absolute) for saving the trained checkpoint in single mode",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        help="Enable early stopping with the given patience (epochs without improvement)",
    )
    parser.add_argument(
        "--early-stop-delta",
        type=float,
        default=0.0,
        help="Minimum improvement required to reset early stopping patience (default: 0.0)",
    )
    args = parser.parse_args()

    batch_size = args.batch_size
    num_classes = 5
    epochs = args.epochs
    use_grayscale = args.grayscale
    color_space = args.color_space
    evaluate_color = True

    device = select_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Color space: {color_space}")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(args.seed)
    else:
        random.seed()

    if use_grayscale and color_space != "rgb":
        logger.warning("Color space selection is ignored for grayscale mode.")

    if args.mode == "sweep":
        sweep_results = run_sweep(
            device=device,
            trials_per_mode=args.trials_per_mode,
            epochs=epochs,
            batch_size=batch_size,
            num_classes=num_classes,
            log_base=Path(args.sweep_logdir),
            edge_shift=args.edge_shift,
            edge_shift_pixels=args.edge_shift_pixels,
            sweep_modes=args.sweep_modes,
            color_space=color_space,
        )
        results_path = Path(args.sweep_logdir) / "sweep_results.csv"
        persist_results(sweep_results, results_path)
        logger.info(f"Saved sweep summary to {results_path}")
        return

    transforms_map = build_transforms(
        use_grayscale,
        edge_shift=args.edge_shift,
        edge_shift_pixels=args.edge_shift_pixels,
        color_space=color_space,
    )
    streamers = prepare_streamers(batch_size, transforms_map, device, use_grayscale)
    train_streamer = streamers["train"]
    valid_streamer = streamers["valid"]

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    default_checkpoint = "flowers_resnet18_gray.pt" if use_grayscale else "flowers_resnet18_color.pt"
    if args.checkpoint_name:
        checkpoint_path = Path(args.checkpoint_name)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    else:
        checkpoint_path = (model_dir / default_checkpoint).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    run_fine_tuning = True

    hyperparams = {
        "lr": args.lr if args.lr is not None else 0.1,
        "momentum": args.momentum if args.momentum is not None else 0.9,
        "weight_decay": args.weight_decay if args.weight_decay is not None else 1e-5,
        "step_size": args.step_size if args.step_size is not None else 10,
        "gamma": args.gamma if args.gamma is not None else 0.1,
        "head_depth": args.head_depth if args.head_depth is not None else 1,
        "head_width_factor": args.head_width_factor if args.head_width_factor is not None else 1.0,
        "head_dropout": args.head_dropout if args.head_dropout is not None else 0.0,
        "unfreeze_blocks": args.unfreeze_blocks if args.unfreeze_blocks is not None else 0,
    }

    model = build_model(
        num_classes,
        grayscale=use_grayscale,
        head_depth=hyperparams.get("head_depth", 1),
        head_width_factor=hyperparams.get("head_width_factor", 1.0),
        head_dropout=hyperparams.get("head_dropout", 0.0),
        unfreeze_blocks=hyperparams.get("unfreeze_blocks", 0),
    )
    model.to(device)

    earlystop_kwargs: Dict[str, Any] | None = None
    if args.early_stop_patience is not None:
        earlystop_kwargs = {
            "patience": int(args.early_stop_patience),
            "delta": float(args.early_stop_delta),
            "verbose": True,
            "save": True,
        }

    if run_fine_tuning:
        accuracy = metrics.Accuracy()
        if args.single_logdir:
            logdir = Path(args.single_logdir)
            if not logdir.is_absolute():
                logdir = (Path.cwd() / logdir).resolve()
        else:
            logdir = Path("modellogs/flowers").resolve()
        logdir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Single run hyperparameters: "
            f"lr={hyperparams['lr']:.6f} "
            f"momentum={hyperparams['momentum']:.3f} "
            f"weight_decay={hyperparams['weight_decay']:.6f} "
            f"step_size={hyperparams['step_size']} "
            f"gamma={hyperparams['gamma']:.3f} "
            f"head_depth={hyperparams['head_depth']} "
            f"head_width_factor={hyperparams['head_width_factor']:.2f} "
            f"head_dropout={hyperparams['head_dropout']:.2f} "
            f"unfreeze_blocks={hyperparams['unfreeze_blocks']}"
        )
        if earlystop_kwargs is not None:
            logger.info(
                "Early stopping enabled (patience=%d, delta=%.4f)",
                earlystop_kwargs["patience"],
                earlystop_kwargs["delta"],
            )

        settings = TrainerSettings(
            epochs=epochs,
            metrics=[accuracy],
            logdir=logdir,
            train_steps=len(train_streamer),
            valid_steps=len(valid_streamer),
            reporttypes=[ReportTypes.TENSORBOARD],
            optimizer_kwargs={
                "lr": hyperparams["lr"],
                "weight_decay": hyperparams["weight_decay"],
                "momentum": hyperparams["momentum"],
            },
            scheduler_kwargs={
                "step_size": max(1, int(hyperparams["step_size"])),
                "gamma": float(hyperparams["gamma"]),
            },
            earlystop_kwargs=earlystop_kwargs,
        )

        toml_serializer = TOMLSerializer()
        toml_serializer.save(settings, "settings.toml")
        toml_serializer.save(model, "model.toml")

        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD,
            traindataloader=train_streamer.stream(),
            validdataloader=valid_streamer.stream(),
            scheduler=torch.optim.lr_scheduler.StepLR,
        )

        trainer.loop()
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    else:
        logger.info("Skipping fine-tuning; evaluating pretrained weights only.")
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}; using randomly initialized head.")

    eval_results = evaluate_dual_modes(
        model=model,
        device=device,
        transforms_map=transforms_map,
        valid_streamer=valid_streamer,
        grayscale=use_grayscale,
    )

    if evaluate_color and "val_color" in eval_results:
        logger.info(f"Validation accuracy (color): {eval_results['val_color']:.4f}")

    if "val_grayscale" in eval_results:
        logger.info(f"Validation accuracy (grayscale): {eval_results['val_grayscale']:.4f}")


if __name__ == "__main__":
    main()