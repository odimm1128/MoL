import argparse
from typing import Tuple

import torch
from torch import nn

from mol_core import MixtureOfLayersClassifier, evaluate, format_gate_distribution
from train_mol_mnist import create_dataloaders


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[MixtureOfLayersClassifier, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = MixtureOfLayersClassifier(
        num_classes=config.get("num_classes", 10),
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
    dataloader,
    device: torch.device,
    limit: int,
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
                selection = []
                if layer_names:
                    for layer_idx, weight in zip(
                        layer_indices[i], layer_weights[i]
                    ):
                        name = layer_names[layer_idx.item()]
                        selection.append(f"{name}({weight.item():.2f})")
                selection_str = ", ".join(selection) if selection else "N/A"
                print(
                    f"  Sample {shown:03d}: target={target.item()} "
                    f"pred={pred.item()} | layers: {selection_str}"
                )
                if shown >= limit:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a MoL checkpoint and run MNIST inference."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=5,
        help="Print this many example predictions after evaluation.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = create_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(
        f"  Stored epoch: {checkpoint.get('epoch', 'N/A')}, "
        f"val_acc: {checkpoint.get('val_acc', 'N/A')}"
    )

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, gate_hist = evaluate(model, test_loader, criterion, device)
    print("\nEvaluation on MNIST test split:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc*100:.2f}%")
    print(
        f"  Gate distribution: "
        f"{format_gate_distribution(model.layer_names, gate_hist)}"
    )

    preview_predictions(model, test_loader, device, args.preview_samples)


if __name__ == "__main__":
    main()
