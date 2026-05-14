from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import timm
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from timm.data import create_transform, resolve_model_data_config


matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int


class PklImageDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        root_dir: Path,
        transform=None,
        path_replace_from: str = "",
        path_replace_to: str = "",
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.path_replace_from = path_replace_from
        self.path_replace_to = path_replace_to
        self.samples: list[Sample] = []
        missing_paths: list[str] = []

        for index, record in enumerate(records):
            if "img_root" not in record or "label" not in record:
                raise KeyError(
                    f"Record {index} must contain 'img_root' and 'label' keys. Got: {sorted(record.keys())}"
                )

            raw_image_path = str(record["img_root"])
            if self.path_replace_from and raw_image_path.startswith(self.path_replace_from):
                raw_image_path = self.path_replace_to + raw_image_path[len(self.path_replace_from) :]

            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                image_path = (root_dir / image_path).resolve()
            label = int(record["label"])
            self.samples.append(Sample(image_path=image_path, label=label))

            if not image_path.exists():
                missing_paths.append(str(image_path))

        if missing_paths:
            preview = "\n".join(missing_paths[:10])
            raise FileNotFoundError(
                f"{len(missing_paths)} image files referenced by the PKL were not found.\n"
                f"First missing paths:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


def load_records(pkl_path: Path) -> list[dict]:
    with pkl_path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Expected {pkl_path} to contain a list, got {type(records)!r}")
    return records


def build_model(num_classes: int, model_name: str = "tiny_vit_5m_224", pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def build_transforms(model: nn.Module):
    data_config = resolve_model_data_config(model)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    return train_transform, eval_transform


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> int:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).sum().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def make_dataloaders(
    train_records: list[dict],
    test_records: list[dict],
    root_dir: Path,
    train_transform,
    eval_transform,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    path_replace_from: str,
    path_replace_to: str,
):
    full_train_dataset = PklImageDataset(
        train_records,
        root_dir=root_dir,
        transform=train_transform,
        path_replace_from=path_replace_from,
        path_replace_to=path_replace_to,
    )
    test_dataset = PklImageDataset(
        test_records,
        root_dir=root_dir,
        transform=eval_transform,
        path_replace_from=path_replace_from,
        path_replace_to=path_replace_to,
    )

    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in the range [0.0, 1.0)")

    if val_split > 0:
        val_size = max(1, int(len(full_train_dataset) * val_split))
        train_size = len(full_train_dataset) - val_size
        if train_size <= 0:
            raise ValueError("val_split is too large for the available training samples")
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = full_train_dataset
        val_dataset = None

    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pretrained TinyViT classifier from PKL file lists.")
    parser.add_argument("--train-pkl", type=Path, default=Path("cpsmi2025_train_list.pkl"))
    parser.add_argument("--test-pkl", type=Path, default=Path("cpsmi2025_test_list.pkl"))
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Base directory for relative img_root paths.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tiny_vit_5m"))
    parser.add_argument("--model-name", type=str, default="tiny_vit_5m_224")
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--path-replace-from", type=str, default="")
    parser.add_argument("--path-replace-to", type=str, default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    return parser.parse_args()


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tiny_vit_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)
    return logger


def save_metrics_csv(output_dir: Path, history: list[dict[str, float | int]]) -> None:
    csv_path = output_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_training_plot(output_dir: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    train_accuracy = [float(row["train_accuracy"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history if "val_loss" in row]
    val_accuracy = [float(row["val_accuracy"]) for row in history if "val_accuracy" in row]
    val_epochs = [int(row["epoch"]) for row in history if "val_accuracy" in row]

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(14, 5))

    loss_ax.plot(epochs, train_loss, label="Train loss", color="#1f77b4", linewidth=2)
    if val_loss:
        loss_ax.plot(val_epochs, val_loss, label="Val loss", color="#ff7f0e", linewidth=2)
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    acc_ax.plot(epochs, train_accuracy, label="Train accuracy", color="#2ca02c", linewidth=2)
    if val_accuracy:
        acc_ax.plot(val_epochs, val_accuracy, label="Val accuracy", color="#d62728", linewidth=2)
    acc_ax.set_title("Accuracy")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.grid(True, alpha=0.3)
    acc_ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    train_records = load_records(args.train_pkl)
    test_records = load_records(args.test_pkl)

    labels = sorted({int(record["label"]) for record in train_records + test_records})
    if labels != list(range(len(labels))):
        print(f"Warning: labels are not zero-based contiguous integers: {labels}")
    if len(labels) != args.num_classes:
        raise ValueError(
            f"Expected {args.num_classes} classes, but found {len(labels)} unique labels: {labels}"
        )
    num_classes = args.num_classes

    model = build_model(
        num_classes=num_classes,
        model_name=args.model_name,
        pretrained=not args.no_pretrained,
    ).to(get_device())
    train_transform, eval_transform = build_transforms(model)

    train_loader, val_loader, test_loader = make_dataloaders(
        train_records=train_records,
        test_records=test_records,
        root_dir=args.data_root.resolve(),
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        path_replace_from=args.path_replace_from,
        path_replace_to=args.path_replace_to,
    )

    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Model: %s", args.model_name)
    params_m = sum(p.numel() for p in model.parameters()) / 1_000_000
    logger.info("Model parameters: %.2fM", params_m)
    logger.info("Number of classes: %s", num_classes)
    logger.info("Train samples: %s", len(train_loader.dataset))
    if val_loader is not None:
        logger.info("Validation samples: %s", len(val_loader.dataset))
    logger.info("Test samples: %s", len(test_loader.dataset))

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"

    best_val_accuracy = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, criterion, device)

        scheduler.step()

        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }

        message = (
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f}"
        )

        if val_metrics is not None:
            record["val_loss"] = val_metrics["loss"]
            record["val_accuracy"] = val_metrics["accuracy"]
            message += f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f}"

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                save_checkpoint(
                    best_checkpoint_path,
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "num_classes": num_classes,
                        "best_val_accuracy": best_val_accuracy,
                        "args": vars(args),
                    },
                )

        logger.info(message)
        history.append(record)

        save_checkpoint(
            last_checkpoint_path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "num_classes": num_classes,
                "args": vars(args),
            },
        )

    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, device)
    logger.info("Test loss %.4f acc %.4f", test_metrics["loss"], test_metrics["accuracy"])

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "history": history,
                "test": test_metrics,
                "best_val_accuracy": best_val_accuracy,
                "device": str(device),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    save_metrics_csv(output_dir, history)
    save_training_plot(output_dir, history)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved CSV log to %s", output_dir / "metrics.csv")
    logger.info("Saved training plot to %s", output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
