#!/usr/bin/env python3
"""MoL training / evaluation entrypoint for WM-811K wafer maps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
except Exception as exc:  # pragma: no cover - guardrail, torchvision is required
    raise ImportError("torchvision is required for image transforms") from exc

try:  # Optional plotting support
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except Exception:  # pragma: no cover - plotting optional
    plt = None

try:
    import timm
except Exception as exc:  # pragma: no cover - guardrail, timm is required
    raise ImportError("timm must be installed (pip install timm)") from exc


SWIFTFORMER_CHECKPOINTS = {
    "swiftformer_xs": "12RchxzyiJrtZS-2Bur9k4wcRQMItA43S",
    "swiftformer_s": "1awpcXAaHH38WaHrOmUM8updxQazUZ3Nb",
    "swiftformer_l1": "1SDzauVmpR5uExkOv3ajxdwFnP-Buj9Uo",
    "swiftformer_l3": "1DAxMe6FlnZBBIpR-HYIDfFLWJzIgiF0Y",
}


# ---------------------------------------------------------------------------
# Data utilities


def normalize_failure_label(value: Any) -> Optional[str]:
    """Flatten nested WM-811K label representations into a plain string."""

    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return str(value).strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            label = normalize_failure_label(item)
            if label:
                return label
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        for item in value.flatten():
            label = normalize_failure_label(item)
            if label:
                return label
        return None

    label = str(value).strip()
    if not label:
        return None
    if label.startswith("[") and label.endswith("]"):
        inner = label[1:-1]
        return normalize_failure_label(inner)
    label = label.strip("'\" ")
    if not label or label.lower() == "nan":
        return None
    return label


def load_wm811k_dataframe(data_root: str | os.PathLike[str]) -> pd.DataFrame:
    """Load the pickled WM-811K dataframe, handling zip bundles as well."""
    path = Path(data_root)
    if path.is_dir():
        for candidate in ("LSWMD.pkl", "LSWMD.pkl.zip", "LSWMD.zip"):
            candidate_path = path / candidate
            if candidate_path.exists():
                path = candidate_path
                break
        else:
            raise FileNotFoundError(
                f"Could not locate LSWMD.pkl or zip archive under {path}"
            )

    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            candidates = [name for name in zf.namelist() if name.lower().endswith(".pkl")]
            if not candidates:
                raise ValueError(f"No pickle files found inside {path}")
            target = None
            for name in candidates:
                if "lswmd" in name.lower():
                    target = name
                    break
            if target is None:
                target = candidates[0]
            with zf.open(target, "r") as stream:
                return pd.read_pickle(stream)
    raise ValueError(f"Unsupported data extension for {path}")


@dataclass
class WaferRecord:
    """Lightweight container for wafer metadata."""

    wafer_map: Any
    label: str
    identifier: str
    split_tag: str


def build_records(
    df: pd.DataFrame,
    limit_per_class: Optional[int] = None,
    seed: int = 0,
    include_none: bool = True,
) -> List[WaferRecord]:
    """Convert the WM-811K dataframe rows into WaferRecord structures.

    When ``limit_per_class`` is provided the function shuffles once and only keeps
    the requested number of wafers per class to avoid materializing the entire
    800K-record dataframe when the caller only needs a tiny subset.
    """

    if limit_per_class and limit_per_class > 0:
        order = list(range(len(df)))
        random.Random(seed).shuffle(order)

        def iterator() -> Iterable[Tuple[int, pd.Series]]:
            for pos in order:
                yield df.index[pos], df.iloc[pos]
    else:
        iterator = df.iterrows

    counts: Dict[str, int] = defaultdict(int)
    records: List[WaferRecord] = []
    for idx, row in iterator():
        wafer_map = row.get("waferMap")
        if wafer_map is None:
            continue
        failure_label = normalize_failure_label(row.get("failureType"))
        if failure_label is None:
            continue
        label = failure_label
        if not include_none and label.lower() == "none":
            continue
        if limit_per_class and limit_per_class > 0 and counts[label] >= limit_per_class:
            continue
        identifier = row.get("waferIndex")
        split_tag = str(row.get("trainTestLabel") or "").strip().lower()
        rec = WaferRecord(
            wafer_map=wafer_map,
            label=label,
            identifier=str(identifier if identifier is not None else idx),
            split_tag=split_tag,
        )
        records.append(rec)
        counts[label] += 1
    if not records:
        raise RuntimeError("Dataset appears empty after filtering invalid wafer maps.")
    return records


def build_label_mapping(records: Sequence[WaferRecord]) -> Tuple[Dict[str, int], List[str]]:
    """Create a deterministic label-to-index mapping, ensuring 'none' exists."""
    labels = sorted({rec.label for rec in records})
    if "none" not in labels:
        labels.append("none")
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = [label for label in labels]
    return label_to_idx, idx_to_label


def apply_limit_per_class(
    records: Sequence[WaferRecord], limit_per_class: Optional[int], seed: int
) -> List[WaferRecord]:
    """Optionally sub-sample a fixed number of wafers per failure type."""
    if not limit_per_class or limit_per_class <= 0:
        return list(records)
    grouped: Dict[str, List[WaferRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.label].append(rec)
    rng = random.Random(seed)
    trimmed: List[WaferRecord] = []
    for label, group in grouped.items():
        group_copy = list(group)
        rng.shuffle(group_copy)
        trimmed.extend(group_copy[:limit_per_class])
    rng.shuffle(trimmed)
    return trimmed


def stratified_split(
    records: Sequence[WaferRecord], val_split: float, seed: int
) -> Tuple[List[WaferRecord], List[WaferRecord]]:
    """Stratified train/val split to keep per-class distributions intact."""
    if val_split <= 0.0:
        return list(records), []
    grouped: Dict[str, List[WaferRecord]] = defaultdict(list)
    for rec in records:
        grouped[rec.label].append(rec)
    rng = random.Random(seed)
    train: List[WaferRecord] = []
    val: List[WaferRecord] = []
    for label, group in grouped.items():
        items = list(group)
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        val_count = max(1, int(round(len(items) * val_split)))
        val_count = min(len(items) - 1, val_count)
        val.extend(items[:val_count])
        train.extend(items[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def describe_class_distribution(records: Sequence[WaferRecord]) -> str:
    counts = Counter(rec.label for rec in records)
    total = sum(counts.values())
    parts = [
        f"{label}: {count} ({(count / total * 100):.1f}%)"
        for label, count in sorted(counts.items(), key=lambda item: item[0])
    ]
    return ", ".join(parts)


def compute_class_weights(
    records: Sequence[WaferRecord], label_to_idx: Dict[str, int]
) -> torch.Tensor:
    """Inverse-frequency class balancing tensor."""
    counts = Counter(rec.label for rec in records)
    weights = torch.ones(len(label_to_idx), dtype=torch.float32)
    for label, idx in label_to_idx.items():
        freq = counts.get(label, 0)
        if freq == 0:
            weights[idx] = 0.0
        else:
            weights[idx] = 1.0 / freq
    total = weights.sum().item()
    if total > 0:
        weights = weights * (len(weights) / total)
    return weights


# ---------------------------------------------------------------------------
# Dataset


class WaferMapDataset(Dataset):
    """Torch dataset that turns WM-811K wafer maps into RGB tensors."""

    def __init__(
        self,
        records: Sequence[WaferRecord],
        label_to_idx: Dict[str, int],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.records = list(records)
        self.label_to_idx = label_to_idx
        self.transform = self._build_transform(image_size, augment)

    @staticmethod
    def _build_transform(image_size: int, augment: bool) -> transforms.Compose:
        interpolation = InterpolationMode.BILINEAR
        ops: List[Any] = []
        if augment:
            ops.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(25),
                ]
            )
        ops.extend(
            [
                transforms.Resize((image_size, image_size), interpolation=interpolation),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        return transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _wafer_to_pil(wafer: Any) -> Image.Image:
        arr = np.asarray(wafer)
        if arr.ndim == 3:  # collapse redundant channel dimension if present
            arr = arr[:, :, 0]
        arr = arr.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.records[index]
        image = self._wafer_to_pil(rec.wafer_map)
        tensor = self.transform(image)
        label_idx = self.label_to_idx[rec.label]
        return {
            "image": tensor,
            "target": label_idx,
            "id": rec.identifier,
            "label": rec.label,
        }


# ---------------------------------------------------------------------------
# Model


def convert_swiftformer_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map official SwiftFormer checkpoints to timm's naming scheme."""

    if not any(key.startswith(("patch_embed", "network")) for key in state_dict):
        return state_dict

    def add_gamma_suffix(name: str, suffix: str) -> str:
        return f"{name}.gamma" if name.endswith(suffix) else name

    converted: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key
        if key.startswith("patch_embed."):
            new_key = "stem." + key.split(".", 1)[1]
        elif key.startswith("network."):
            parts = key.split(".")
            net_idx = int(parts[1])
            remainder = parts[2:]
            if net_idx % 2 == 0:
                stage_idx = net_idx // 2
                if not remainder:
                    continue
                block_idx = remainder[0]
                rest = remainder[1:]
                new_key = ".".join(["stages", str(stage_idx), "blocks", block_idx] + rest)
            else:
                stage_idx = (net_idx + 1) // 2
                new_key = ".".join(["stages", str(stage_idx), "downsample"] + remainder)
        elif key.startswith("dist_head"):
            new_key = key.replace("dist_head", "head_dist", 1)
        new_key = new_key.replace("attn.Proj", "attn.proj")
        new_key = add_gamma_suffix(new_key, "layer_scale")
        new_key = add_gamma_suffix(new_key, "layer_scale_1")
        new_key = add_gamma_suffix(new_key, "layer_scale_2")
        converted[new_key] = tensor
    return converted


class FrozenSwiftFormer(nn.Module):
    """SwiftFormer backbone that exposes intermediate stages and stays frozen."""

    def __init__(
        self,
        backbone_name: str,
        image_size: int,
        checkpoint_path: Optional[str] = None,
        auto_download_dir: Optional[str] = None,
        allow_download: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.image_size = image_size
        checkpoint = checkpoint_path or maybe_download_swiftformer(
            backbone_name, auto_download_dir, allow_download
        )
        pretrained = checkpoint is None
        self.model = timm.create_model(backbone_name, pretrained=pretrained)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            state_dict = state.get("model", state)
            state_dict = convert_swiftformer_state_dict(state_dict)
            missing = self.model.load_state_dict(state_dict, strict=False)
            if missing.missing_keys:
                print(
                    f"[backbone] Loaded checkpoint with missing keys: {missing.missing_keys[:5]}..."
                )
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.stage_names = [f"stage_{idx + 1}" for idx in range(len(self.model.stages))]
        self.feature_dims = self._infer_feature_dims()

    def _infer_feature_dims(self) -> List[int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            features = self._forward_stages(dummy)
        return [feat.shape[1] for feat in features]

    def _forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        out = self.model.stem(x)
        for idx, stage in enumerate(self.model.stages):
            out = stage(out)
            if idx == len(self.model.stages) - 1:
                feats.append(self.model.norm(out))
            else:
                feats.append(out)
        return feats

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            feats = self._forward_stages(x)
        return [feat.detach() for feat in feats]


class SwiftFormerMoL(nn.Module):
    """Mixture-of-Layers gating head stacked on top of a frozen SwiftFormer."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        image_size: int,
        proj_dim: int = 256,
        head_hidden_dim: int = 512,
        gate_top_k: int = 2,
        checkpoint_path: Optional[str] = None,
        swiftformer_cache_dir: Optional[str] = None,
        allow_backbone_download: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = FrozenSwiftFormer(
            backbone_name=backbone_name,
            image_size=image_size,
            checkpoint_path=checkpoint_path,
            auto_download_dir=swiftformer_cache_dir,
            allow_download=allow_backbone_download,
        )
        feature_dims = self.backbone.feature_dims
        self.layer_names = self.backbone.stage_names
        self.num_layers = len(feature_dims)
        self.gate_top_k = max(1, min(gate_top_k, self.num_layers))

        projections = []
        for channels in feature_dims:
            projections.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                )
            )
        self.projections = nn.ModuleList(projections)
        self.gate_controller = nn.Sequential(
            nn.LayerNorm(proj_dim * self.num_layers),
            nn.GELU(),
            nn.Linear(proj_dim * self.num_layers, self.num_layers),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def _compute_gate_weights(
        self, gate_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gate_top_k >= self.num_layers:
            weights = gate_logits.softmax(dim=-1)
            topk_indices = torch.arange(
                self.num_layers, device=gate_logits.device
            ).expand(gate_logits.size(0), self.num_layers)
            return weights, topk_indices
        topk_vals, topk_idx = gate_logits.topk(self.gate_top_k, dim=-1)
        probs = topk_vals.softmax(dim=-1)
        weights = torch.zeros_like(gate_logits)
        weights.scatter_(dim=-1, index=topk_idx, src=probs)
        return weights, topk_idx

    def forward(
        self, x: torch.Tensor, return_gates: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        features = self.backbone(x)
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]
        stacked = torch.stack(projected, dim=1)  # [B, L, D]
        gate_input = stacked.flatten(start_dim=1)
        logits = self.gate_controller(gate_input)
        gate_weights, topk_idx = self._compute_gate_weights(logits)
        mixture = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)
        out = self.head(mixture)
        if return_gates:
            return out, gate_weights, topk_idx
        return out


# ---------------------------------------------------------------------------
# Training / evaluation helpers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GateTracker:
    """Accumulates gate selection statistics across an epoch."""

    def __init__(self, num_layers: int) -> None:
        self.counts = torch.zeros(num_layers, dtype=torch.long)

    def update(self, topk_indices: torch.Tensor) -> None:
        flat = topk_indices.reshape(-1)
        for idx in flat.tolist():
            self.counts[idx] += 1

    def summary(self) -> torch.Tensor:
        total = self.counts.sum().item()
        if total == 0:
            return torch.zeros_like(self.counts, dtype=torch.float32)
        return self.counts.float() / total


def gate_summary_to_str(
    gate_hist: torch.Tensor, layer_names: Sequence[str]
) -> str:
    return ", ".join(
        f"{name}: {prob * 100:.1f}%"
        for name, prob in zip(layer_names, gate_hist.tolist())
    )


def move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    images = batch["image"].to(device, non_blocking=True)
    targets = batch["target"]
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device, non_blocking=True)
    else:
        targets = torch.as_tensor(targets, device=device)
    ids: List[str] = batch.get("id", [])
    labels: List[str] = batch.get("label", [])
    return images, targets, ids, labels


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def train_one_epoch(
    model: SwiftFormerMoL,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    log_every: int,
    amp: bool,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_tracker = GateTracker(model.num_layers)
    start_time = time.time()
    for step, batch in enumerate(dataloader, 1):
        images, targets, _, _ = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits, gate_weights, gate_idx = model(images, return_gates=True)
            loss = criterion(logits, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        gate_tracker.update(gate_idx.detach().cpu())
        if step % log_every == 0:
            elapsed = time.time() - start_time
            acc = total_correct / total_samples if total_samples else 0.0
            print(
                f"[train] step {step}/{len(dataloader)} "
                f"loss={total_loss / total_samples:.4f} "
                f"acc={acc * 100:.2f}% ({elapsed:.1f}s)"
            )
    epoch_loss = total_loss / max(1, total_samples)
    epoch_acc = total_correct / max(1, total_samples)
    gate_hist = gate_tracker.summary()
    return {"loss": epoch_loss, "accuracy": epoch_acc, "gate_hist": gate_hist}


@torch.no_grad()
def evaluate(
    model: SwiftFormerMoL,
    dataloader: DataLoader,
    criterion: Optional[nn.Module],
    device: torch.device,
    amp: bool,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    gate_tracker = GateTracker(model.num_layers)
    all_targets: List[int] = []
    all_preds: List[int] = []
    for batch in dataloader:
        images, targets, _, _ = move_batch_to_device(batch, device)
        with autocast_context(device, amp):
            logits, _, gate_idx = model(images, return_gates=True)
            loss = criterion(logits, targets) if criterion else torch.tensor(0.0, device=device)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        batch_size = targets.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        gate_tracker.update(gate_idx.detach().cpu())
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "gate_hist": gate_tracker.summary(),
        "targets": all_targets,
        "preds": all_preds,
    }


def confusion_matrix(
    preds: Sequence[int], targets: Sequence[int], num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
    return matrix


def confusion_to_string(
    matrix: np.ndarray, label_names: Sequence[str]
) -> str:
    header = "          " + " ".join(f"{name[:6]:>7}" for name in label_names)
    lines = [header]
    for idx, row in enumerate(matrix):
        cells = " ".join(f"{value:7d}" for value in row)
        lines.append(f"{label_names[idx][:8]:>8} {cells}")
    return "\n".join(lines)


def save_learning_curve(history: List[Dict[str, float]], path: str) -> None:
    if not plt:
        print("matplotlib unavailable, skipping learning curve plot.")
        return
    epochs = list(range(1, len(history) + 1))
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_losses, label="train")
    axes[0].plot(epochs, val_losses, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, train_accs, label="train")
    axes[1].plot(epochs, val_accs, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_checkpoint(
    path: str,
    model: SwiftFormerMoL,
    label_to_idx: Dict[str, int],
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "label_to_idx": label_to_idx,
        "layer_names": model.layer_names,
        "model_config": {
            "backbone": model.backbone.backbone_name,
            "image_size": model.backbone.image_size,
            "proj_dim": model.projections[0][2].out_features
            if model.projections
            else metadata.get("proj_dim", 256),
            "head_hidden_dim": metadata.get("head_hidden_dim"),
            "gate_top_k": model.gate_top_k,
        },
        "metadata": metadata,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"[checkpoint] Saved to {path}")


def load_checkpoint(path: str) -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# SwiftFormer weight helpers


def maybe_download_swiftformer(
    backbone_name: str,
    cache_dir: Optional[str],
    allow_download: bool,
) -> Optional[str]:
    if not cache_dir:
        cache_dir = os.path.join("pretrained", "swiftformer")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_path / f"{backbone_name}_ckpt.pth"
    if ckpt_path.exists():
        return str(ckpt_path)
    if not allow_download:
        return None
    drive_id = SWIFTFORMER_CHECKPOINTS.get(backbone_name)
    if not drive_id:
        print(f"[backbone] No known checkpoint id for {backbone_name}, skipping download.")
        return None
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download SwiftFormer weights. "
            "Install via `pip install gdown` or provide --swiftformer-cache-dir with weights."
        ) from exc
    url = f"https://drive.google.com/uc?id={drive_id}"
    print(f"[backbone] Downloading {backbone_name} weights via gdown...")
    gdown.download(url, str(ckpt_path), quiet=False)
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# CLI commands


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[setup] Using device {device}")
    df = load_wm811k_dataframe(args.data_root)
    all_records = build_records(df, include_none=not args.exclude_none)
    label_to_idx, _ = build_label_mapping(all_records)
    usable_records = apply_limit_per_class(all_records, args.limit_per_class, args.seed)
    train_records, val_records = stratified_split(usable_records, args.val_split, args.seed)
    if not train_records:
        raise RuntimeError("No training samples found after filtering.")
    print(f"[data] Train samples: {len(train_records)}")
    print(f"[data] Train distribution: {describe_class_distribution(train_records)}")
    if val_records:
        print(f"[data] Val samples: {len(val_records)}")
        print(f"[data] Val distribution: {describe_class_distribution(val_records)}")
    else:
        print("[data] Validation split disabled.")

    train_dataset = WaferMapDataset(
        train_records,
        label_to_idx=label_to_idx,
        image_size=args.image_size,
        augment=not args.no_augment,
    )
    val_dataset = (
        WaferMapDataset(
            val_records,
            label_to_idx=label_to_idx,
            image_size=args.image_size,
            augment=False,
        )
        if val_records
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = SwiftFormerMoL(
        backbone_name=args.backbone,
        num_classes=len(label_to_idx),
        image_size=args.image_size,
        proj_dim=args.proj_dim,
        head_hidden_dim=args.head_hidden_dim,
        gate_top_k=args.gate_top_k,
        checkpoint_path=args.backbone_checkpoint,
        swiftformer_cache_dir=args.swiftformer_cache_dir,
        allow_backbone_download=not args.offline_backbone,
    ).to(device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_records, label_to_idx)
        print(f"[data] Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=args.label_smoothing,
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.98)
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    history: List[Dict[str, float]] = []
    best_val_acc = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    for epoch in range(1, args.epochs + 1):
        print(f"\n[epoch {epoch}/{args.epochs}]")
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler if args.amp else None,
            log_every=args.log_every,
            amp=args.amp,
        )
        print(
            f"[train] loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy'] * 100:.2f}% "
            f"gate={gate_summary_to_str(train_metrics['gate_hist'], model.layer_names)}"
        )
        val_metrics = None
        if val_loader:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                amp=args.amp,
            )
            print(
                f"[val] loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['accuracy'] * 100:.2f}% "
                f"gate={gate_summary_to_str(val_metrics['gate_hist'], model.layer_names)}"
            )
        if scheduler:
            scheduler.step()

        history.append(
            {
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"] if val_metrics else 0.0,
                "val_acc": val_metrics["accuracy"] if val_metrics else 0.0,
            }
        )

        if val_metrics and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = model.state_dict()
            if args.save_path:
                save_checkpoint(
                    args.save_path,
                    model,
                    label_to_idx,
                    metadata={
                        "epoch": epoch,
                        "val_acc": best_val_acc,
                        "train_acc": train_metrics["accuracy"],
                        "proj_dim": args.proj_dim,
                        "head_hidden_dim": args.head_hidden_dim,
                    },
                )

    if args.final_checkpoint_path:
        state_dict = best_state if best_state is not None else model.state_dict()
        torch.save(
            {
                "model_state": state_dict,
                "label_to_idx": label_to_idx,
                "layer_names": model.layer_names,
            },
            args.final_checkpoint_path,
        )
        print(f"[checkpoint] Final weights stored in {args.final_checkpoint_path}")

    if args.learning_curve_path and history:
        save_learning_curve(history, args.learning_curve_path)


def run_command(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    label_to_idx = checkpoint.get("label_to_idx")
    if not label_to_idx:
        raise RuntimeError("Checkpoint missing label_to_idx mapping.")
    idx_to_label = [label for label, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    model_cfg = checkpoint.get("model_config", {})
    model = SwiftFormerMoL(
        backbone_name=model_cfg.get("backbone", args.backbone),
        num_classes=len(label_to_idx),
        image_size=model_cfg.get("image_size", args.image_size),
        proj_dim=model_cfg.get("proj_dim", args.proj_dim),
        head_hidden_dim=model_cfg.get("head_hidden_dim", args.head_hidden_dim),
        gate_top_k=model_cfg.get("gate_top_k", args.gate_top_k),
        checkpoint_path=args.backbone_checkpoint or None,
        swiftformer_cache_dir=args.swiftformer_cache_dir,
        allow_backbone_download=not args.offline_backbone,
    )
    missing = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"[warn] State dict mismatch: {missing}")
    model.to(device)

    df = load_wm811k_dataframe(args.data_root)
    eval_records = build_records(
        df,
        limit_per_class=args.limit_per_class,
        seed=args.seed,
        include_none=not args.exclude_none,
    )
    dataset = WaferMapDataset(
        eval_records,
        label_to_idx=label_to_idx,
        image_size=model.backbone.image_size,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate(
        model=model,
        dataloader=loader,
        criterion=None,
        device=device,
        amp=args.amp,
    )
    print(
        f"[run] accuracy={metrics['accuracy'] * 100:.2f}% "
        f"gate={gate_summary_to_str(metrics['gate_hist'], model.layer_names)}"
    )
    cm = confusion_matrix(
        preds=metrics["preds"],
        targets=metrics["targets"],
        num_classes=len(idx_to_label),
    )
    print("[run] Confusion matrix (target rows, prediction columns):")
    print(confusion_to_string(cm, idx_to_label))

    preview = min(args.preview_samples, len(dataset))
    predictions_csv = []
    collected = 0
    for batch in loader:
        images, targets, ids, _ = move_batch_to_device(batch, device)
        logits, _, gate_idx = model(images, return_gates=True)
        preds = logits.argmax(dim=-1).cpu().tolist()
        gate_idx = gate_idx.cpu().tolist()
        target_list = targets.detach().cpu().tolist()
        batch_ids = (
            ids
            if ids
            else [f"sample_{i}" for i in range(collected, collected + len(preds))]
        )
        for sample_id, tgt_idx, pred_idx, routed in zip(
            batch_ids, target_list, preds, gate_idx
        ):
            layer_names = [model.layer_names[i] for i in routed]
            predictions_csv.append(
                {
                    "id": sample_id,
                    "target": idx_to_label[tgt_idx],
                    "pred": idx_to_label[pred_idx],
                    "layers": json.dumps(layer_names),
                }
            )
            if collected < preview:
                print(
                    f"[sample] id={sample_id} target={idx_to_label[tgt_idx]} "
                    f"pred={idx_to_label[pred_idx]} layers={layer_names}"
                )
                collected += 1
    csv_path = args.predictions_csv or "wm811k_predictions.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["id", "target", "pred", "layers"])
        writer.writeheader()
        writer.writerows(predictions_csv)
    print(f"[run] Predictions written to {csv_path}")


# ---------------------------------------------------------------------------
# Argument parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or evaluate the Mixture-of-Layers classifier on WM-811K."
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train on WM-811K.")
    train_parser.add_argument(
        "--data-root",
        type=str,
        default="data/LSWMD.pkl",
        help="Path to LSWMD.pkl or the zip archive containing it.",
    )
    train_parser.add_argument("--backbone", type=str, default="swiftformer_s")
    train_parser.add_argument("--backbone-checkpoint", type=str, default=None)
    train_parser.add_argument("--swiftformer-cache-dir", type=str, default=None)
    train_parser.add_argument("--offline-backbone", action="store_true")
    train_parser.add_argument("--limit-per-class", type=int, default=None)
    train_parser.add_argument("--exclude-none", action="store_true")
    train_parser.add_argument("--val-split", type=float, default=0.1)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--learning-rate", type=float, default=5e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--proj-dim", type=int, default=256)
    train_parser.add_argument("--head-hidden-dim", type=int, default=512)
    train_parser.add_argument("--gate-top-k", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--log-every", type=int, default=25)
    train_parser.add_argument("--no-augment", action="store_true")
    train_parser.add_argument("--no-class-weights", action="store_true")
    train_parser.add_argument("--label-smoothing", type=float, default=0.0)
    train_parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    train_parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    train_parser.add_argument("--device", type=str, default=None)
    train_parser.add_argument("--save-path", type=str, default="chkpts/wm811k_best.pth")
    train_parser.add_argument("--final-checkpoint-path", type=str, default=None)
    train_parser.add_argument("--learning-curve-path", type=str, default=None)

    run_parser = subparsers.add_parser("run", help="Evaluate a trained checkpoint.")
    run_parser.add_argument("--checkpoint", type=str, required=True)
    run_parser.add_argument(
        "--data-root",
        type=str,
        default="data/LSWMD.pkl",
    )
    run_parser.add_argument("--backbone", type=str, default="swiftformer_s")
    run_parser.add_argument("--backbone-checkpoint", type=str, default=None)
    run_parser.add_argument("--swiftformer-cache-dir", type=str, default=None)
    run_parser.add_argument("--offline-backbone", action="store_true")
    run_parser.add_argument("--limit-per-class", type=int, default=None)
    run_parser.add_argument("--exclude-none", action="store_true")
    run_parser.add_argument("--batch-size", type=int, default=64)
    run_parser.add_argument("--num-workers", type=int, default=4)
    run_parser.add_argument("--preview-samples", type=int, default=10)
    run_parser.add_argument("--predictions-csv", type=str, default="wm811k_predictions.csv")
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument("--amp", action="store_true")
    run_parser.add_argument("--image-size", type=int, default=224)
    run_parser.add_argument("--proj-dim", type=int, default=256)
    run_parser.add_argument("--head-hidden-dim", type=int, default=512)
    run_parser.add_argument("--gate-top-k", type=int, default=2)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        train_command(args)
    elif args.command == "run":
        run_command(args)
    else:  # pragma: no cover - argparse ensures this won't happen
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
