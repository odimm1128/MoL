import argparse
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


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
    confusion_matrix: Optional[torch.Tensor] = None,
    prediction_recorder: Optional[
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            None,
        ]
    ] = None,
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

            if confusion_matrix is not None:
                num_classes = confusion_matrix.size(0)
                targets_cpu = targets.view(-1).to(device="cpu", dtype=torch.long)
                preds_cpu = preds.view(-1).to(device="cpu", dtype=torch.long)
                flat_indices = targets_cpu * num_classes + preds_cpu
                batch_counts = torch.bincount(
                    flat_indices, minlength=num_classes * num_classes
                )
                confusion_matrix += batch_counts.view(num_classes, num_classes)

            if prediction_recorder is not None:
                prediction_recorder(
                    targets.detach().cpu(),
                    preds.detach().cpu(),
                    gate_info["indices"].detach().cpu(),
                    gate_info["weights"].detach().cpu(),
                )

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
