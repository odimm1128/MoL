import argparse
import zipfile
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from mol_core import (
    MixtureOfLayersClassifier,
    evaluate,
    format_gate_distribution,
    maybe_plot_learning_curve,
    save_checkpoint,
    train_one_epoch,
)


def normalize_failure_type(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and value:
        return normalize_failure_type(value[0])
    if isinstance(value, np.ndarray) and value.size > 0:
        return normalize_failure_type(value.flatten()[0])
    return None


def derive_class_names(df: pd.DataFrame) -> List[str]:
    if "failureType" not in df.columns:
        raise ValueError("WM-811K dataframe must contain a 'failureType' column.")

    ordered: List[str] = []
    seen = set()
    for raw in df["failureType"]:
        label = normalize_failure_type(raw)
        if label is None or label in seen:
            continue
        seen.add(label)
        ordered.append(label)

    if not ordered:
        raise ValueError("Could not infer any failure types from the dataset.")

    if "none" in seen:
        ordered = [name for name in ordered if name != "none"] + ["none"]

    return ordered


def locate_wm811k_pickle(data_root: str) -> Path:
    """Return the path to LSWMD.pkl, extracting it from a zip if needed."""
    root = Path(data_root)
    if root.is_file():
        return _resolve_file_path(root)

    candidates = [root / "LSWMD.pkl"]
    candidates.extend(root.glob("*.pkl"))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    zip_candidates = [root / "LSWMD.zip"]
    zip_candidates.extend(root.glob("*.zip"))
    for archive in zip_candidates:
        extracted = _extract_pickle_from_zip(archive, root)
        if extracted:
            return extracted

    raise FileNotFoundError(
        f"Could not find LSWMD.pkl under {root}. "
        "Place the Kaggle file there or point --data-root directly to it."
    )


def _resolve_file_path(path: Path) -> Path:
    if path.suffix.lower() == ".pkl" and path.exists():
        return path
    if path.suffix.lower() == ".zip":
        extracted = _extract_pickle_from_zip(path, path.parent)
        if extracted:
            return extracted
    raise FileNotFoundError(f"{path} is not a readable pickle file.")


def _extract_pickle_from_zip(zip_path: Path, output_dir: Path) -> Optional[Path]:
    if not zip_path.exists():
        return None

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            if member.lower().endswith(".pkl"):
                output_dir.mkdir(parents=True, exist_ok=True)
                target = output_dir / Path(member).name
                if target.exists():
                    return target

                archive.extract(member, path=output_dir)
                extracted = output_dir / member
                if extracted != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted.rename(target)
                return target
    return None


def build_wm811k_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def wafer_map_to_image(wafer_map) -> Image.Image:
    arr = np.asarray(wafer_map)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.nan_to_num(arr, nan=0.0)
    arr = arr.astype(np.float32)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


class WaferMapDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_names: List[str],
        transform: Optional[transforms.Compose] = None,
        limit_per_class: Optional[int] = None,
        drop_failure_types: Optional[set] = None,
    ) -> None:
        df = dataframe.copy()
        df = df[df["waferMap"].notna()]
        if not class_names:
            raise ValueError("class_names must include at least one label.")

        class_to_index = {name: idx for idx, name in enumerate(class_names)}

        df["failureType"] = (
            df["failureType"]
            .apply(normalize_failure_type)
            .apply(lambda val: val if val in class_to_index else None)
        )
        df = df[df["failureType"].notna()]
        if drop_failure_types:
            df = df[~df["failureType"].isin(drop_failure_types)]
        df = df.reset_index(drop=True)

        if limit_per_class:
            df = (
                df.groupby("failureType", group_keys=False)
                .head(limit_per_class)
                .reset_index(drop=True)
            )

        self.maps: List = df["waferMap"].tolist()
        self.labels: List[int] = df["failureType"].map(class_to_index).tolist()
        self.transform = transform or build_wm811k_transform(224)
        self.class_names = list(class_names)
        self.class_counts = Counter(self.labels)

        if not self.maps:
            raise ValueError("No wafer maps available after filtering.")

    def __len__(self) -> int:
        return len(self.maps)

    def __getitem__(self, idx: int):
        wafer_map = self.maps[idx]
        label = self.labels[idx]
        image = wafer_map_to_image(wafer_map)
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_weights_tensor(self) -> torch.Tensor:
        counts = torch.tensor(
            [self.class_counts.get(idx, 0) for idx in range(len(self.class_names))],
            dtype=torch.float32,
        )
        total = counts.sum().clamp_min(1.0)
        weights = torch.zeros_like(counts)
        non_zero = counts > 0
        weights[non_zero] = total / (counts[non_zero] * len(self.class_names))
        return weights


def load_wm811k_dataframe(data_root: str) -> Tuple[pd.DataFrame, Path]:
    pkl_path = locate_wm811k_pickle(data_root)
    df = pd.read_pickle(pkl_path)
    if "waferMap" not in df.columns or "failureType" not in df.columns:
        raise ValueError(
            f"{pkl_path} does not contain the expected WM-811K dataframe columns."
        )
    return df, pkl_path


def create_wm811k_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    limit_per_class: Optional[int],
    image_size: int,
) -> Tuple[DataLoader, DataLoader, WaferMapDataset, Path]:
    df, data_path = load_wm811k_dataframe(data_root)
    class_names = derive_class_names(df)
    dataset = WaferMapDataset(
        df,
        class_names=class_names,
        transform=build_wm811k_transform(image_size),
        limit_per_class=limit_per_class,
        drop_failure_types={"none"},
    )
    val_split = max(0.01, min(val_split, 0.5))
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_len, val_len], generator=generator
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, dataset, data_path


def create_inference_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    limit_per_class: Optional[int],
    image_size: int,
) -> Tuple[DataLoader, WaferMapDataset, Path]:
    df, data_path = load_wm811k_dataframe(data_root)
    dataset = WaferMapDataset(
        df,
        class_names=derive_class_names(df),
        transform=build_wm811k_transform(image_size),
        limit_per_class=limit_per_class,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset, data_path


def describe_layer_selection(
    layer_names: List[str],
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> str:
    if not layer_names or indices.numel() == 0:
        return "N/A"

    idx_values = indices.view(-1).tolist()
    weight_values = weights.view(-1).tolist() if weights.numel() > 0 else []
    selection = []
    for pos, layer_idx in enumerate(idx_values):
        if layer_idx < 0 or layer_idx >= len(layer_names):
            continue
        weight = weight_values[pos] if pos < len(weight_values) else 0.0
        selection.append(f"{layer_names[layer_idx]}({weight:.2f})")
    return ", ".join(selection) if selection else "N/A"


def print_class_distribution(counts: Counter, class_names: List[str]) -> None:
    total = sum(counts.values())
    print("\nClass distribution:")
    for idx, name in enumerate(class_names):
        count = counts.get(idx, 0)
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {name:10s}: {count:8d} ({pct:5.2f}%)")


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_classes: Optional[int] = None,
) -> Tuple[MixtureOfLayersClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    effective_num_classes = num_classes or config.get("num_classes")
    if effective_num_classes is None:
        raise ValueError(
            "Checkpoint is missing num_classes info; please provide num_classes explicitly."
        )
    model = MixtureOfLayersClassifier(
        num_classes=effective_num_classes,
        feature_dim=config.get("feature_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        top_k=config.get("top_k", 2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def preview_predictions(
    model: MixtureOfLayersClassifier,
    dataloader: DataLoader,
    device: torch.device,
    limit: int,
    class_names: List[str],
) -> None:
    if limit <= 0:
        return

    shown = 0
    print("\nSample predictions:")
    layer_names = getattr(model, "layer_names", [])
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits, gate_info = model(images)
            preds = logits.argmax(dim=1).cpu()
            layer_indices = gate_info["indices"].cpu()
            layer_weights = gate_info["weights"].cpu()

            for i, (target, pred) in enumerate(zip(targets, preds)):
                shown += 1
                selection = describe_layer_selection(
                    layer_names, layer_indices[i], layer_weights[i]
                )
                print(
                    f"  Sample {shown:03d}: "
                    f"target={class_names[target.item()]} "
                    f"pred={class_names[pred.item()]} "
                    f"| layers: {selection}"
                )
                if shown >= limit:
                    return


def print_confusion_matrix(matrix: torch.Tensor, class_names: List[str]) -> None:
    matrix = matrix.cpu()
    label_width = max(10, max(len(name) for name in class_names) + 2)
    header = " " * label_width + "".join(
        f"{name:>{label_width}}" for name in class_names
    )
    print(header)
    for idx, name in enumerate(class_names):
        row_values = "".join(
            f"{matrix[idx, j].item():>{label_width}d}"
            for j in range(len(class_names))
        )
        print(f"{name:>{label_width}}{row_values}")


def run_training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, dataset, data_path = create_wm811k_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        limit_per_class=args.limit_per_class,
        image_size=args.image_size,
    )
    print(f"Loaded {len(dataset)} wafer maps from {data_path}")
    print_class_distribution(dataset.class_counts, dataset.class_names)

    model = MixtureOfLayersClassifier(
        num_classes=len(dataset.class_names),
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        top_k=args.top_k,
    ).to(device)

    class_weights = (
        dataset.class_weights_tensor().to(device)
        if args.use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            model, val_loader, criterion, device
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


def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, dataset, data_path = create_inference_loader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_per_class=args.limit_per_class,
        image_size=args.image_size,
    )
    print(f"Loaded {len(dataset)} wafer maps for evaluation from {data_path}")
    model, checkpoint = load_model_from_checkpoint(
        args.checkpoint, device, num_classes=len(dataset.class_names)
    )
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(
        f"  Stored epoch: {checkpoint.get('epoch', 'N/A')}, "
        f"val_acc: {checkpoint.get('val_acc', 'N/A')}"
    )

    criterion = nn.CrossEntropyLoss()
    prediction_rows: List[dict] = []
    sample_counter = 0

    def record_predictions(
        targets_cpu: torch.Tensor,
        preds_cpu: torch.Tensor,
        layer_indices_cpu: torch.Tensor,
        layer_weights_cpu: torch.Tensor,
    ) -> None:
        nonlocal sample_counter
        batch_size = targets_cpu.size(0)
        for i in range(batch_size):
            target_idx = targets_cpu[i].item()
            pred_idx = preds_cpu[i].item()
            layer_desc = describe_layer_selection(
                model.layer_names,
                layer_indices_cpu[i],
                layer_weights_cpu[i],
            )
            prediction_rows.append(
                {
                    "id": sample_counter,
                    "target": dataset.class_names[target_idx],
                    "pred": dataset.class_names[pred_idx],
                    "layers": layer_desc,
                }
            )
            sample_counter += 1

    confusion = torch.zeros(
        (len(dataset.class_names), len(dataset.class_names)), dtype=torch.int64
    )
    val_loss, val_acc, gate_hist = evaluate(
        model,
        dataloader,
        criterion,
        device,
        confusion_matrix=confusion,
        prediction_recorder=record_predictions,
    )
    print("\nEvaluation on WM-811K split:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc*100:.2f}%")
    print(
        f"  Gate distribution: "
        f"{format_gate_distribution(model.layer_names, gate_hist)}"
    )
    print("\nConfusion matrix (rows=true, cols=pred):")
    print_confusion_matrix(confusion, dataset.class_names)

    if prediction_rows:
        csv_path = Path(args.predictions_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(prediction_rows).to_csv(csv_path, index=False)
        print(f"\nSaved detailed predictions to {csv_path}")

    preview_predictions(
        model,
        dataloader,
        device,
        args.preview_samples,
        dataset.class_names,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoL training & inference pipeline for the WM-811K wafer map dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train the MoL classifier on WM-811K."
    )
    train_parser.add_argument("--data-root", type=str, default="./data/wm811k")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--feature-dim", type=int, default=256)
    train_parser.add_argument("--hidden-dim", type=int, default=512)
    train_parser.add_argument("--top-k", type=int, default=2)
    train_parser.add_argument("--val-split", type=float, default=0.1)
    train_parser.add_argument("--limit-per-class", type=int, default=None)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--save-path", type=str, default=None)
    train_parser.add_argument("--final-checkpoint-path", type=str, default=None)
    train_parser.add_argument("--learning-curve-path", type=str, default=None)
    train_parser.add_argument(
        "--no-class-weights",
        action="store_false",
        dest="use_class_weights",
        help="Disable inverse-frequency class weighting.",
    )
    train_parser.set_defaults(use_class_weights=True)

    run_parser = subparsers.add_parser(
        "run", help="Evaluate a checkpoint on WM-811K and preview predictions."
    )
    run_parser.add_argument("--checkpoint", type=str, required=True)
    run_parser.add_argument("--data-root", type=str, default="./data/wm811k")
    run_parser.add_argument("--batch-size", type=int, default=128)
    run_parser.add_argument("--num-workers", type=int, default=4)
    run_parser.add_argument("--limit-per-class", type=int, default=2000)
    run_parser.add_argument("--image-size", type=int, default=224)
    run_parser.add_argument("--preview-samples", type=int, default=5)
    run_parser.add_argument(
        "--predictions-csv",
        type=str,
        default="wm811k_predictions.csv",
        help="Write per-sample id/target/pred/layers to this CSV path.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "train":
        run_training(args)
    elif args.command == "run":
        run_inference(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
