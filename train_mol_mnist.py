import argparse
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class MixtureOfLayersClassifier(nn.Module):
    """Frozen ResNet-18 + MoE selector + three-layer MLP classifier."""

    def __init__(
        self,
        num_classes: int = 10,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        backbone = resnet18(weights=weights)

        # freeze backbone parameters
        for param in backbone.parameters():
            param.requires_grad_(False)

        return_nodes = {
            "relu": "conv1",
            "layer1": "layer1",
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
            "avgpool": "avgpool",
        }
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.extractor.eval()
        for param in self.extractor.parameters():
            param.requires_grad_(False)

        self.layer_names = ["conv1", "layer1", "layer2", "layer3", "layer4"]
        channel_dims: Dict[str, int] = {
            "conv1": 64,
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
        }

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
            nn.Linear(channel_dims["layer4"], gate_hidden),
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

    def train(self, mode: bool = True) -> "MixtureOfLayersClassifier":
        super().train(mode)
        # keep the frozen backbone in eval mode so BN stats stay fixed
        self.extractor.eval()
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            features = self.extractor(x)

        gating_input = torch.flatten(features["avgpool"], 1)
        gate_logits = self.gate(gating_input)

        batch_size = x.size(0)
        projected_stack = []
        for name in self.layer_names:
            projection = self.projections[name](features[name])
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
    weights = ResNet18_Weights.IMAGENET1K_V1
    mean, std = weights.meta["mean"], weights.meta["std"]

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE-like MNIST classifier using frozen ResNet layers"
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
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_gate = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_gate = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

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
            if args.save_path:
                torch.save(
                    {"model_state": model.state_dict(), "val_acc": val_acc},
                    args.save_path,
                )
                print(f"  Saved new best checkpoint to {args.save_path}")

    print(f"Finished training. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
