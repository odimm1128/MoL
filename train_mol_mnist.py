import argparse
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Large_Weights

from mol_core import (
    MixtureOfLayersClassifier,
    evaluate,
    format_gate_distribution,
    maybe_plot_learning_curve,
    save_checkpoint,
    train_one_epoch,
)


def create_dataloaders(
    data_root: str, batch_size: int, num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    mean, std = preprocess.mean, preprocess.std

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=test_transform
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader




def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE-like MNIST classifier using frozen MobileNetV3 layers"
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--learning-curve-path",
        type=str,
        default=None,
        help="Optional path to save a train/validation learning curve plot.",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        type=str,
        default=None,
        help="Optional path to always save the last-epoch checkpoint.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = create_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = MixtureOfLayersClassifier(
        num_classes=10,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_gate = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_gate = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d}/{args.epochs:03d}")
        print(
            f"  Train loss: {train_loss:.4f}, acc: {train_acc*100:.2f}% | "
            f"Gate: {format_gate_distribution(model.layer_names, train_gate)}"
        )
        print(
            f"  Valid loss: {val_loss:.4f}, acc: {val_acc*100:.2f}% | "
            f"Gate: {format_gate_distribution(model.layer_names, val_gate)}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                args.save_path,
                model,
                epoch,
                val_acc,
                args,
                tag="best",
            )

    print(f"Finished training. Best accuracy: {best_acc*100:.2f}%")
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
    save_checkpoint(
        args.final_checkpoint_path,
        model,
        args.epochs,
        final_val_acc,
        args,
        tag="final",
    )
    maybe_plot_learning_curve(history, args.learning_curve_path)


if __name__ == "__main__":
    main()
