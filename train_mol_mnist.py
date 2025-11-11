import argparse
from typing import Tuple, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class MixtureOfLayersClassifier(nn.Module):
    """Frozen MobileNetV3 + MoE selector + three-layer MLP classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_large(weights=weights)

        self.backbone = backbone.features
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)

        # capture points across the MobileNetV3 feature stack
        self.layer_specs = [
            ("stage1", 1, 16),
            ("stage2", 3, 24),
            ("stage3", 6, 40),
            ("stage4", 10, 80),
            ("stage5", 15, 160),
        ]
        self.layer_names = [name for name, _, _ in self.layer_specs]
        self.capture_map = {idx: name for name, idx, _ in self.layer_specs}
        channel_dims: Dict[str, int] = {
            name: channels for name, _, channels in self.layer_specs
        }
        self.final_feature_dim = backbone.classifier[0].in_features
        self.final_pool = nn.AdaptiveAvgPool2d(1)

        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channel_dims[name], feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU(),
                )
                for name in self.layer_names
            }
        )

        self.top_k = max(1, min(top_k, len(self.layer_names)))
        gate_hidden = hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(self.final_feature_dim, gate_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gate_hidden, gate_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(gate_hidden // 2, len(self.layer_names)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def train(self, mode: bool = True) -> "MixtureOfLayersClassifier":
        super().train(mode)
        # keep the frozen backbone in eval mode so BN stats stay fixed
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        captured_features: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            out = x
            for idx, layer in enumerate(self.backbone):
                out = layer(out)
                if idx in self.capture_map:
                    captured_features[self.capture_map[idx]] = out
            final_features = out

        gating_input = torch.flatten(self.final_pool(final_features), 1)
        gate_logits = self.gate(gating_input)

        batch_size = x.size(0)
        projected_stack = []
        for name in self.layer_names:
            projection = self.projections[name](captured_features[name])
            projected_stack.append(projection.unsqueeze(1))
        projected_stack = torch.cat(projected_stack, dim=1)

        topk_values, topk_indices = torch.topk(gate_logits, k=self.top_k, dim=1)
        weights = torch.softmax(topk_values, dim=1)

        batch_idx = (
            torch.arange(batch_size, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
        )
        selected = projected_stack[batch_idx, topk_indices, :]
        mixture = (selected * weights.unsqueeze(-1)).sum(dim=1)

        logits = self.classifier(mixture)
        gate_info = {"logits": gate_logits, "indices": topk_indices, "weights": weights}
        return logits, gate_info


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


def train_one_epoch(
    model: MixtureOfLayersClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_hist = torch.zeros(len(model.layer_names), dtype=torch.float32)

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, gate_info = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

        with torch.no_grad():
            counts = torch.bincount(
                gate_info["indices"].view(-1).cpu(),
                minlength=len(model.layer_names),
            ).float()
            gate_hist += counts

    gate_hist = gate_hist / gate_hist.sum().clamp_min(1.0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, gate_hist


def evaluate(
    model: MixtureOfLayersClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_hist = torch.zeros(len(model.layer_names), dtype=torch.float32)

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            logits, gate_info = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

            counts = torch.bincount(
                gate_info["indices"].view(-1).cpu(),
                minlength=len(model.layer_names),
            ).float()
            gate_hist += counts

    gate_hist = gate_hist / gate_hist.sum().clamp_min(1.0)
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc, gate_hist


def format_gate_distribution(layer_names, distribution: torch.Tensor) -> str:
    return ", ".join(
        f"{name}:{prob:.2f}"
        for name, prob in zip(layer_names, distribution.tolist())
    )


def save_checkpoint(
    path: Optional[str],
    model: MixtureOfLayersClassifier,
    epoch: int,
    val_acc: float,
    args: argparse.Namespace,
    tag: str,
) -> None:
    if not path:
        return

    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "config": {
            "num_classes": getattr(model, "num_classes", 10),
            "feature_dim": getattr(args, "feature_dim", 256),
            "hidden_dim": getattr(args, "hidden_dim", 512),
            "top_k": getattr(args, "top_k", 2),
        },
        "args": dict(vars(args)),
        "tag": tag,
    }
    torch.save(payload, path)
    print(f"  Saved {tag} checkpoint to {path}")


def maybe_plot_learning_curve(history: Dict[str, list], output_path: Optional[str]) -> None:
    if not output_path:
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        print(
            "matplotlib is required for plotting learning curves. "
            "Please install it (e.g., `pip install matplotlib`)."
        )
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Validation", marker="o")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Learning Curve")

    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["train_acc"]],
        label="Train",
        marker="o",
    )
    axes[1].plot(
        epochs,
        [acc * 100 for acc in history["val_acc"]],
        label="Validation",
        marker="o",
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved learning curve to {output_path}")


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
